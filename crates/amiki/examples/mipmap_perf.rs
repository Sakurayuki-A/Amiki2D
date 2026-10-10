//! B4 验收的性能判据（任务书 §6）：
//!
//! 1. mip 链生成 ≤ 原图大小的 2 倍上传时间
//! 2. 采样路径无 >5% 性能回退
//! 3. 内存 +33%（实测）
//!
//! 运行：`cargo run --release -p amiki --example mipmap_perf`
//!
//! 口径说明（写死在这里，防止将来口径漂移）：
//! - "上传时间"取 `GpuRenderer::profile` 的 `upload_ms` **增量**
//!   （带链场景 − 不带链场景；场景只含一张图，增量 ≈ mip 层的
//!   `write_texture` 代价）。性能数字取多次最小值（并行/负载下会失真）。
//! - "采样路径"取**逐 pass GPU 时间戳**里 fine 段的时长（真正的
//!   逐像素采样发生在那里），而非 render_blocking 墙钟 —— 后者混入了
//!   每帧重建纹理数组的上传与读回同步，不是采样本身的口径。

use amiki::scene::{Color, Image, Paint, PathBuilder, Scene};
use amiki_math::Affine;

/// 计时 helper：捕获起点，返回距起点的微秒
struct Stopwatch(std::time::Instant);
impl Stopwatch {
    fn start() -> Stopwatch {
        Stopwatch(std::time::Instant::now())
    }
    fn us(&self) -> u128 {
        self.0.elapsed().as_nanos() / 1_000
    }
}

fn main() {
    let (w, h) = (1024u32, 1024u32);
    let mut px = vec![0u8; (w * h * 4) as usize];
    for (i, b) in px.iter_mut().enumerate() {
        *b = (i * 7 % 251) as u8;
    }

    // ── 判据 3：内存 +33% ──
    let img = Image::from_rgba8_with_mipmap(w, h, px.clone()).unwrap();
    let chain_bytes: usize = img.mip.as_ref().unwrap().levels.iter().map(|l| l.data.len()).sum();
    let orig = (w * h * 4) as usize;
    println!("判据 3（内存）：原图 {} KB，链 {} KB（理论 = 原图的 1/3 = {} KB），比率 {:.1}%",
        orig / 1024, chain_bytes / 1024, orig / 3 / 1024,
        chain_bytes as f64 / orig as f64 * 100.0);

    // ── 判据 1：链生成时间 vs 上传增量 ──
    let mut gen_min = u128::MAX;
    for _ in 0..10 {
        let t = Stopwatch::start();
        let _ = amiki_scene::generate_mip_chain(w, h, &px);
        gen_min = gen_min.min(t.us());
    }
    println!("链生成（1024²，多次取最小）：{gen_min} µs");

    let Ok(renderer) = pollster::block_on(amiki_gpu::GpuRenderer::new()) else {
        println!("无 GPU 适配器，判据 1/2 跳过");
        return;
    };

    // 场景：一张 1024² 图 + 一个 1px 矩形（其余路径两口径完全相同）
    let mk = |mip: bool| {
        let img = std::sync::Arc::new(if mip {
            Image::from_rgba8_with_mipmap(w, h, px.clone()).unwrap()
        } else {
            Image::from_rgba8(w, h, px.clone()).unwrap()
        });
        let mut sc = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 1.0, 1.0);
        sc.push(Affine::IDENTITY, pb.build(), Paint::Image {
            image: img,
            placement: amiki::scene::ImagePlacement::rect(0.0, 0.0, 1.0, 1.0),
        });
        sc
    };
    let upload_delta_of = |scene: &Scene| -> f64 {
        let _ = renderer.profile(scene, 0.05); // 预热
        let mut m = f64::MAX;
        for _ in 0..10 {
            let p = renderer.profile(scene, 0.05).expect("profile");
            m = m.min(p.upload_ms);
        }
        m
    };
    let up_plain = upload_delta_of(&mk(false));
    let up_mip = upload_delta_of(&mk(true));
    let upload_delta = (up_mip - up_plain).max(0.0) * 1000.0; // → µs
    println!("判据 1（生成 vs 上传）：原图上传 {:.0} µs；mip 层上传增量 {upload_delta:.0} µs",
        up_plain * 1000.0);
    // 任务书原文分母是"原图大小的 2 倍上传时间"（原图上传，不是增量）
    println!("  判据：生成 ≤ 2×原图上传 → {}",
        if (gen_min as f64) <= up_plain * 2000.0 { "✅ 过" } else { "❌ 未过" });

    // ── 判据 2：采样路径（fine pass 的 GPU 时间戳）──
    // 480×360 画布，1024² 图缩小到 120×90（≈1/8.5，LOD 3.1）
    let mk_scene = |mip: bool| {
        let img = std::sync::Arc::new(if mip {
            Image::from_rgba8_with_mipmap(w, h, px.clone()).unwrap()
        } else {
            Image::from_rgba8(w, h, px.clone()).unwrap()
        });
        let mut sc = Scene::new(480, 360);
        let mut bg = PathBuilder::new();
        bg.rect(0.0, 0.0, 480.0, 360.0);
        sc.push(Affine::IDENTITY, bg.build(), Paint::Solid(Color::rgb8(30, 30, 40)));
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 120.0, 90.0);
        sc.push(Affine::IDENTITY, pb.build(), Paint::Image {
            image: img,
            placement: amiki::scene::ImagePlacement::rect(0.0, 0.0, 120.0, 90.0),
        });
        sc
    };
    let fine_of = |scene: &Scene| -> f64 {
        let _ = renderer.profile(scene, 0.05); // 预热
        let mut m = f64::MAX;
        for _ in 0..20 {
            let p = renderer.profile(scene, 0.05).expect("profile");
            if !p.timestamps.supported {
                continue;
            }
            // fine 段 = 真正逐像素采样的 pass
            for (label, ns) in &p.timestamps.passes {
                if label.to_ascii_lowercase().contains("fine") {
                    m = m.min(*ns / 1000.0);
                }
            }
        }
        m
    };
    let f_plain = fine_of(&mk_scene(false));
    let f_mip = fine_of(&mk_scene(true));
    if f_plain.is_infinite() {
        println!("判据 2（采样回退）：时间戳查询不可用（B8 的已知降级），无法测 GPU 逐 pass 口径");
    } else {
        let pct = (f_mip - f_plain) / f_plain * 100.0;
        println!("判据 2（采样回退，fine pass GPU 口径）：无 mip {f_plain:.0} µs；带 mip {f_mip:.0} µs（{pct:+.1}%）");
        println!("  判据：回退 ≤ 5% → {}", if pct <= 5.0 { "✅ 过" } else { "❌ 未过" });
    }
}
