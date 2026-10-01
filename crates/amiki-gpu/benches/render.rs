//! M3 基准套件：criterion 端到端计时 + GPU 时间戳逐 pass 分析。
//!
//! 运行：
//! ```bash
//! cargo bench -p amiki-gpu                     # 全部基准
//! cargo bench -p amiki-gpu -- render           # 只跑端到端组
//! ```
//!
//! 指标（RENDERING_STRATEGY §4.5）：1080p、10k 曲线路径、逐帧 < 2ms（RTX 级 GPU）。
//!
//! 两代分箱并存，报告同时给出两边的容量上限：
//! - **稠密网格（M2f）**：显存 ∝ tiles × draws，且 clear 单维 dispatch 受
//!   WebGPU 65535 工作组上限约束 ⇒ 720p 约 71 draw、1080p 约 31 draw 即到顶。
//! - **稀疏桶表（M3b，默认）**：显存 ∝ 画面实际内容（条带/单元/桶/条目/成员），
//!   目标场景 1080p × 10k 曲线由报告给出稠密需求（约 21 GB）与实际稀疏占用对比。

use amiki_gpu::{build_gpu_scene, GpuRenderer, SparseStats};
use amiki_math::{Affine, Vec2};
use amiki_scene::{Color, Paint, PathBuilder, Scene};
use criterion::Criterion;
use std::hint::black_box;
use std::time::Duration;

const EPS: f32 = 0.1;

/// 种子化 LCG（基准可复现，不引入 rand 依赖）。
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// n 个圆（曲线路径；模拟图标/贴图/粒子的典型内容）。
fn curves_scene(w: u32, h: u32, n: usize, r: f32) -> Scene {
    let mut scene = Scene::new(w, h);
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    for i in 0..n {
        let c = Vec2::new(rng.range(r, w as f32 - r), rng.range(r, h as f32 - r));
        let mut pb = PathBuilder::new();
        pb.circle(c, r * (0.6 + 0.8 * rng.next_f32()), false);
        let t = if i % 3 == 0 { Affine::rotate(rng.range(0.0, 1.5)) } else { Affine::IDENTITY };
        let col = Color::rgba8(
            (rng.next_f32() * 255.0) as u8,
            (rng.next_f32() * 255.0) as u8,
            (rng.next_f32() * 255.0) as u8,
            200,
        );
        scene.push(t, pb.build(), Paint::Solid(col));
    }
    scene
}

/// n 个矩形（UI/线框类内容的近似负载）。
fn rects_scene(w: u32, h: u32, n: usize) -> Scene {
    let mut scene = Scene::new(w, h);
    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    for _ in 0..n {
        let x = rng.range(0.0, w as f32 * 0.9);
        let y = rng.range(0.0, h as f32 * 0.9);
        let mut pb = PathBuilder::new();
        pb.rect(x, y, x + rng.range(8.0, 120.0), y + rng.range(8.0, 90.0));
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 160, 90, 180)),
        );
    }
    scene
}

/// 稠密网格显存需求（字节）：counters + grid_aux + carries + bucket_sums。
fn dense_bytes(w: u32, h: u32, draws: u64) -> u64 {
    let tx = (w as u64).div_ceil(16);
    let ty = (h as u64).div_ceil(16);
    let grid = tx * ty * draws * 16;
    let bucket = draws * ty * 16 * (tx + 1);
    grid * 4 * 3 + bucket * 4
}

/// 稠密 clear dispatch 上限（65535 工作组 × 64 线程）对应的最大 draw 数。
/// clear 一次派发覆盖 `max(n_grid, n_bucket)` 个元素（后者 = draws×ty×16×(tx+1)）。
fn dense_draw_ceiling(w: u32, h: u32) -> u64 {
    let tx = (w as u64).div_ceil(16);
    let ty = (h as u64).div_ceil(16);
    let per_draw = (tx * ty * 16).max(ty * 16 * (tx + 1));
    65_535 * 64 / per_draw
}

/// 稀疏分箱的实际分配字节数（含固定结构与内容规模两部分）。
fn sparse_bytes(w: u32, h: u32, draws: u64, s: &SparseStats) -> u64 {
    let tx = (w as u64).div_ceil(16);
    let ty = (h as u64).div_ceil(16);
    let strips = draws * ty;
    let fixed = strips * 16                      // strip_meta（4×u32）
        + strips * 4 * 4                         // strip_cell_off / strip_bucket_off / tr_prefix / tr_list
        + ty * 8                                 // tr_offsets
        + tx * ty * 4 * 2; // tile_count / tile_start
    let content = s.n_cells as u64 * 8            // cell_aux（计数 + 游标）
        + (s.n_cells as u64 + 1) * 4              // cell CSR 基址
        + s.n_buckets as u64 * 4                  // carry 定点桶
        + s.n_cells as u64 * 16                   // cell_meta（carry + 段基址/数量）
        + s.n_tile_entries as u64 * 16            // 逐 tile 桶表条目
        + s.n_items as u64 * 16; // 段列表（端点直存）
    fixed + content
}

/// 目标与上限报告（每次基准运行前打印一次）。
fn report(renderer: &GpuRenderer) {
    println!("\n=== Amiki2D 基准报告 ===");
    println!("适配器：{}", renderer.adapter_name);
    println!(
        "GPU 时间戳：{}（period {:.3} ns/tick）",
        if renderer.has_timestamps { "支持" } else { "不可用（回退端到端计时）" },
        renderer.timestamp_period_ns
    );
    for (label, w, h) in [("720p", 1280u32, 720u32), ("1080p", 1920, 1080)] {
        println!(
            "{label} 稠密网格上限：draw {}（clear dispatch 限制），每 draw 显存 {:.2} MB",
            dense_draw_ceiling(w, h),
            dense_bytes(w, h, 1) as f64 / 1e6
        );
    }

    // §4.5 目标场景：稠密不可能 → 稀疏实测
    for (label, w, h, n, r) in [
        ("§4.5 目标场景 1080p × 10k 曲线", 1920u32, 1080u32, 10_000usize, 5.0f32),
        ("1080p × 2k 曲线", 1920, 1080, 2_000, 8.0),
        ("1080p × 1k 矩形", 1920, 1080, 1_000, 0.0),
    ] {
        let scene = if r > 0.0 { curves_scene(w, h, n, r) } else { rects_scene(w, h, n) };
        let dense = dense_bytes(w, h, n as u64) as f64;
        match renderer.sparse_stats(&scene, EPS) {
            Ok(Some(s)) => println!(
                "  [{label}] 稠密需 {:.2} GB → 稀疏实占 {:.1} MB\
                 （条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}）",
                dense / 1e9,
                sparse_bytes(w, h, n as u64, &s) as f64 / 1e6,
                s.n_strips,
                s.n_cells,
                s.n_buckets,
                s.n_tile_entries,
                s.n_items
            ),
            Ok(None) => println!("  [{label}] 稀疏统计不可用"),
            Err(e) => println!("  [{label}] 失败：{e}"),
        }
    }

    // 逐 pass GPU 时间戳
    for (label, scene) in [
        ("720p×64 曲线", curves_scene(1280, 720, 64, 10.0)),
        ("1080p×10k 曲线", curves_scene(1920, 1080, 10_000, 5.0)),
    ] {
        match renderer.profile(&scene, EPS) {
            Ok(t) => {
                // 主机侧分段 + 逐 pass GPU 时长（两者都给才能定位瓶颈在 CPU 还是 GPU）
                println!("  [{label}] {}", t.summary());
                if t.timestamps.supported {
                    println!("  [{label}] {}", t.timestamps.summary());
                }
            }
            Err(e) => println!("  [{label}] profile 失败：{e}"),
        }
    }
    println!("========================\n");
}

fn bench_extract(c: &mut Criterion) {
    // CPU 每帧成本：场景 → 控制点/直线提取（不含上传）
    let mut group = c.benchmark_group("cpu_extract");
    group.bench_function("720p_64_curves", |b| {
        let scene = curves_scene(1280, 720, 64, 10.0);
        b.iter(|| black_box(build_gpu_scene(black_box(&scene), EPS)));
    });
    group.bench_function("1080p_10k_curves", |b| {
        let scene = curves_scene(1920, 1080, 10_000, 5.0);
        b.iter(|| black_box(build_gpu_scene(black_box(&scene), EPS)));
    });
    // **判据看的是这一条**：引擎逐帧走的是 `prepare_into` → `build_gpu_scene_into`，
    // 缓冲跨帧复用，于是没有每帧全新分配（大块分配走 mmap + 缺页，实测差 5.3 倍）。
    // 上面那条 `1080p_10k_curves` 保留作对比：它是**一次性调用**的成本。
    group.bench_function("1080p_10k_curves_reused", |b| {
        let scene = curves_scene(1920, 1080, 10_000, 5.0);
        let mut gs = amiki_gpu::GpuScene::default();
        amiki_gpu::build_gpu_scene_into(black_box(&scene), EPS, &mut gs); // 热身，容量长到位
        b.iter(|| {
            amiki_gpu::build_gpu_scene_into(black_box(&scene), EPS, &mut gs);
            black_box(gs.total_segments)
        });
    });
    group.bench_function("1080p_1000_curves", |b| {
        let scene = curves_scene(1920, 1080, 1000, 6.0);
        b.iter(|| black_box(build_gpu_scene(black_box(&scene), EPS)));
    });
    group.finish();
}

fn bench_render(c: &mut Criterion) {
    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过 GPU 基准：无可用适配器");
        return;
    };
    report(&renderer);

    // 帧路径（游戏/动效场景）：每帧重建场景 → prepare → render_into → GPU 同步。
    // 不含像素读回；与 end_to_end 的差值即读回/同步开销。
    let mut group = c.benchmark_group("gpu_frame_path");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    for (label, scene) in [
        ("720p_64_curves", curves_scene(1280, 720, 64, 10.0)),
        ("1080p_10k_curves", curves_scene(1920, 1080, 10_000, 5.0)),
    ] {
        let tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bench_out"),
            size: wgpu::Extent3d { width: scene.width, height: scene.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        group.bench_function(label, |b| {
            b.iter(|| {
                let frame = renderer.prepare(black_box(&scene), EPS).expect("prepare 失败");
                renderer.render_into(&tex, &frame).expect("渲染失败");
                renderer.device.poll(wgpu::PollType::Wait).expect("poll 失败");
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("gpu_render_end_to_end");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for (label, scene) in [
        ("720p_64_rects", rects_scene(1280, 720, 64)),
        ("720p_64_curves", curves_scene(1280, 720, 64, 10.0)),
        ("1080p_1k_rects", rects_scene(1920, 1080, 1_000)),
        ("1080p_2k_curves", curves_scene(1920, 1080, 2_000, 8.0)),
        ("1080p_10k_curves", curves_scene(1920, 1080, 10_000, 5.0)),
    ] {
        group.bench_function(label, |b| {
            b.iter(|| black_box(renderer.render_blocking(black_box(&scene), EPS).expect("渲染失败")));
        });
    }
    group.finish();
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_extract(&mut c);
    bench_render(&mut c);
    c.final_summary();
}