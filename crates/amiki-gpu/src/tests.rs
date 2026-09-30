//! amiki-gpu 的集成/对拍测试。
//!
//! 从 `lib.rs` 拆出：这部分占原文件约 43%（2466/5719 行），
//! 且与实现代码的改动节奏完全不同（实现改一行，测试经常改几十行）。

// 测试通过 `crate::` 全路径引用被测项，不需要 `use super::*`。

use super::*;

/// GPU/CPU 渲染对拍（donut + 旋转）。
/// M2f 已知问题：曲线边缘像素覆盖率偏差（圆极端点行约 -0.28，且不随 ε 变化，
/// 怀疑 flatten 段端点或 seg_ends 数据链路），调查恢复后回归 tolerance 2。
#[test]
fn gpu_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
    pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
    let mut scene = Scene::new(96, 96);
    let t = Affine::translate(Vec2::new(48.0, 48.0))
        .then(Affine::rotate(0.3))
        .then(Affine::translate(Vec2::new(-48.0, -48.0)));
    scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));

    let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

    let mut max_diff = 0u32;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
    }
    assert!(max_diff <= 2, "GPU 与 CPU 最大通道差 {max_diff} 超出容差");
}

/// M2f 管线核心对拍：纯直线/矩形场景（不含曲线，当前已验证）。
#[test]
fn gpu_rect_matches_cpu() {        use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    // 多 draw + 半透明 + 旋转矩形 + 跨 tile 边界
    let mut scene = Scene::new(72, 72);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 72.0, 72.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
    let mut pb = PathBuilder::new();
    pb.rect(8.0, 8.0, 40.0, 56.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
    let mut pb = PathBuilder::new();
    pb.rect(-16.0, -10.0, 16.0, 10.0);
    let t = Affine::translate(Vec2::new(44.0, 36.0)).then(Affine::rotate(0.5));
    scene.push_blend(t, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 180)), BlendMode::Multiply);

    let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
    let mut max_diff = 0u32;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
    }
    assert!(max_diff <= 2, "GPU 与 CPU 最大通道差 {max_diff} 超出容差");
}

/// M6：**GPU 渐变着色必须与 CPU 参考实现一致**。
///
/// 这是渐变在 GPU 侧的唯一正确性依据：WGSL 的 `grad_color` 与 CPU 的
/// `Gradient::sample` 是两份独立实现，只有逐像素对拍才能证明语义一致。
/// 覆盖三种几何（线性/径向/扫描）与多色标（含中间色标，以验证区间查找），
/// 以及 repeat 范围行为。
#[test]
fn gpu_gradient_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Gradient, GradientStop, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    let stops = || {
        vec![
            GradientStop::new(0.0, Color::rgba8(20, 40, 200, 255)),
            GradientStop::new(0.35, Color::rgba8(240, 200, 40, 255)),
            GradientStop::new(1.0, Color::rgba8(220, 30, 90, 255)),
        ]
    };

    let cases: Vec<(&str, Gradient)> = vec![
        (
            "线性",
            Gradient::linear(Vec2::new(6.0, 6.0), Vec2::new(66.0, 42.0), stops()).unwrap(),
        ),
        (
            "径向",
            Gradient::radial(Vec2::new(36.0, 36.0), 34.0, stops()).unwrap(),
        ),
        (
            "扫描",
            Gradient::sweep(Vec2::new(36.0, 36.0), 0.7, stops()).unwrap(),
        ),
    ];

    for (name, g) in cases {
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(4.0, 4.0, 68.0, 68.0);
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Gradient(std::sync::Arc::new(g)),
        );

        let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

        let mut max_diff = 0u32;
        let mut worst = (0usize, 0usize);
        let mut n_opaque = 0usize;
        for i in 0..(72 * 72) {
            let (a, b) = (gpu[i * 4], cpu[i * 4]);
            let d = (a as i32 - b as i32).unsigned_abs();
            if d > max_diff {
                max_diff = d;
                worst = (i % 72, i / 72);
            }
            if gpu[i * 4 + 3] > 0 {
                n_opaque += 1;
            }
        }
        assert!(
            n_opaque > 2000,
            "{name}渐变应产生大量不透明像素（否则对拍无意义），实际 {n_opaque}"
        );
        assert!(
            max_diff <= 2,
            "{name}渐变：GPU 与 CPU 最大通道差 {max_diff}（位于 {worst:?}）超出容差\n\
             该点 GPU={:?} CPU={:?}",
            {
                let o = (worst.1 * 72 + worst.0) * 4;
                [gpu[o], gpu[o + 1], gpu[o + 2], gpu[o + 3]]
            },
            {
                let o = (worst.1 * 72 + worst.0) * 4;
                [cpu[o], cpu[o + 1], cpu[o + 2], cpu[o + 3]]
            }
        );
        eprintln!("GPU {name}渐变与 CPU 一致（最大通道差 {max_diff}，不透明像素 {n_opaque}）");
    }

    // repeat 模式：范围外应重复而非 clamp
    let mut rep =
        Gradient::linear(Vec2::new(10.0, 10.0), Vec2::new(30.0, 10.0), stops()).unwrap();
    rep.repeat = true;
    let mut scene = Scene::new(72, 72);
    let mut pb = PathBuilder::new();
    pb.rect(4.0, 4.0, 68.0, 68.0);
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Gradient(std::sync::Arc::new(rep)),
    );
    let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
    let max_diff = gpu
        .iter()
        .zip(cpu.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 2, "repeat 渐变：最大通道差 {max_diff} 超出容差");
    eprintln!("GPU repeat 渐变与 CPU 一致（最大通道差 {max_diff}）");
}
/// M6：**GPU 位图采样必须与 CPU 参考实现一致**。
///
/// 用一张**合成图案**（而非纯色）作测试图：纯色无法区分"采样正确"与
/// "把整张图当成一个颜色"这两种实现；棋盘 + 渐变的组合能同时检验
/// 坐标映射、双线性过滤与缩放。
#[test]
fn gpu_image_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Image, ImagePlacement, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    // 32×24 合成图：R/G 为渐变（检验坐标映射），B 为 4×4 棋盘（检验过滤）
    let (iw, ih) = (32u32, 24u32);
    let mut data = Vec::with_capacity((iw * ih * 4) as usize);
    for y in 0..ih {
        for x in 0..iw {
            let checker = if ((x / 4) + (y / 4)) % 2 == 0 { 40u8 } else { 215u8 };
            data.push((x * 255 / (iw - 1)) as u8);
            data.push((y * 255 / (ih - 1)) as u8);
            data.push(checker);
            data.push(255);
        }
    }
    let img = std::sync::Arc::new(Image::from_rgba8(iw, ih, data).expect("图像数据合法"));

    // 三种放置：等比、放大（检验双线性）、旋转（检验仿射逆映射）
    let cases: Vec<(&str, ImagePlacement, bool)> = vec![
        ("等比", ImagePlacement::rect(8.0, 8.0, 40.0, 32.0), false),
        ("放大", ImagePlacement::rect(4.0, 4.0, 68.0, 68.0), false),
        (
            "旋转",
            ImagePlacement {
                origin: Vec2::new(36.0, 6.0),
                u_axis: Vec2::new(28.0, 14.0),
                v_axis: Vec2::new(-14.0, 28.0),
                repeat: false,
            },
            false,
        ),
        ("平铺", ImagePlacement { repeat: true, ..ImagePlacement::rect(10.0, 10.0, 30.0, 22.0) }, false),
    ];

    for (name, place, _) in cases {
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        // 覆盖范围按放置四角的包围盒（与 Canvas::image 一致）
        let p = |u: f32, v: f32| place.origin + place.u_axis * u + place.v_axis * v;
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (u, v) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            let q = p(u, v);
            x0 = x0.min(q.x);
            y0 = y0.min(q.y);
            x1 = x1.max(q.x);
            y1 = y1.max(q.y);
        }
        pb.rect(x0, y0, x1, y1);
        // 底：不透明色，便于观察图像是否真的覆盖上去
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(250, 250, 250)));
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Image { image: img.clone(), placement: place },
        );

        // GL 后端读不出位图，引擎**显式报错**而不是画出空白
        // （见 `mk_image_texture_array` 的说明）。该情形是已知且明确的。
        let gpu = match renderer.render_blocking(&scene, 0.05) {
            Ok(v) => v,
            Err(e) if e.to_string().contains("GL 后端无法正确读取位图") => {
                eprintln!("GL 后端：位图按预期显式拒绝");
                return;
            }
            Err(e) => panic!("GPU 渲染失败：{e}"),
        };
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

        let mut max_diff = 0u32;
        let mut worst = 0usize;
        for (i, (a, b)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            if d > max_diff {
                max_diff = d;
                worst = i;
            }
        }
        // 双线性在 CPU 与 GPU 上的舍入不同，容差给 3（比纯几何的 2 略宽）
        assert!(
            max_diff <= 3,
            "{name}放置：GPU 与 CPU 最大通道差 {max_diff}（字节偏移 {worst}）超出容差"
        );
        eprintln!("GPU 位图（{name}放置）与 CPU 一致（最大通道差 {max_diff}）");
    }
}

/// M6：**GPU 离屏层（滤镜）必须与 CPU 参考实现逐像素一致**。
///
/// 这是"滤镜双端一致"的唯一依据。两端的模糊是独立的两份实现
/// （WGSL 可分离盒式 vs Rust 可分离盒式），只有对拍才能证明语义一致。
///
/// 容差 2/255：与纯几何对拍同级。模糊是多次加权平均，两端在盒边界与
/// 舍入上有微小差异，实测三种滤镜的最大差均为 0~1。
#[test]
fn gpu_layer_matches_cpu() {
    use amiki_math::Affine;
    use amiki_scene::{BlendMode, Color, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    let cases: Vec<(&str, Filter)> = vec![
        ("分组", Filter::None),
        ("模糊", Filter::Blur { sigma: 6.0 }),
        (
            "投影",
            Filter::DropShadow {
                dx: 10.0,
                dy: 12.0,
                sigma: 9.0,
                color: Color::rgba8(30, 26, 22, 190),
            },
        ),
    ];

    for (name, filter) in cases {
        let mut scene = Scene::new(160, 120);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 120.0);
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgb8(246, 243, 238)),
        );
        scene.push_layer(
            Layer {
                bounds: LayerBounds::new(36, 26, 70, 70),
                scene: Scene::new(160, 120),
                filter,
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            |s| {
                let mut pb = PathBuilder::new();
                pb.rect(44.0, 34.0, 96.0, 86.0);
                s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 60, 50)));
            },
        );

        let gpu = renderer
            .render_blocking(&scene, 0.05)
            .unwrap_or_else(|e| panic!("{name}层：GPU 渲染失败（{e}）"));
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

        let mut max_diff = 0u32;
        let mut worst = (0usize, 0usize);
        for i in 0..(160 * 120) {
            for k in 0..4 {
                let d = (gpu[i * 4 + k] as i32 - cpu[i * 4 + k] as i32).unsigned_abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i % 160, i / 160);
                }
            }
        }
        assert!(
            max_diff <= 2,
            "{name}层：GPU 与 CPU 最大通道差 {max_diff}（位于 {worst:?}）超出容差"
        );
        eprintln!("GPU {name}层与 CPU 一致（最大通道差 {max_diff}）");
    }
}

/// M2f-1：通用 GPU 前缀和 vs CPU 对拍（随机长度/数值）。
#[test]
fn gpu_scan_matches_cpu() {
    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let mut s = 0xC0FF_EE12_3456_789Au64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for &n in &[1usize, 2, 63, 64, 65, 255, 256, 257, 1000, 4097] {
        let data: Vec<u32> = (0..n).map(|_| (rnd() * 1000.0) as u32).collect();
        let buf = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scan_test"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let mut enc = renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.run_scan(&mut enc, &buf, n as u32, None);
        renderer.queue.submit(Some(enc.finish()));
        let out_bytes = renderer.read_buffer(&buf, n * 4).expect("读回失败");
        let out: Vec<u32> = bytemuck::cast_slice(&out_bytes).to_vec();
        // CPU exclusive 前缀
        let mut expect = Vec::with_capacity(n);
        let mut acc = 0u32;
        for &v in &data {
            expect.push(acc);
            acc += v;
        }
        assert_eq!(out, expect, "GPU 前缀和与 CPU 不一致 (n={n})");
    }
}

/// M2e：全 16 混合模式 GPU vs CPU 对拍。
#[test]
fn gpu_blend_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    let modes = [
        BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
        BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
        BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference, BlendMode::Exclusion,
        BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity,
    ];
    for (mi, mode) in modes.iter().enumerate() {
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 56.0, 56.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 90, 160)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 22.0, false);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(230, 120, 60, 200)),
            *mode,
        );
        let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
        let mut max_diff = 0u32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        }
        assert!(max_diff <= 2, "混合模式 #{mi}({mode:?}) GPU/CPU 最大差 {max_diff}");
    }
}

/// M2d：裁剪 GPU vs CPU 对拍（单层 + 嵌套两层）。
#[test]
fn gpu_clip_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    for nested in [false, true] {
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 56.0, 56.0);
        let c0 = scene.push_clip(Affine::IDENTITY, pb.build());
        let mut clips = vec![c0];
        if nested {
            let mut pb = PathBuilder::new();
            pb.rect(16.0, 16.0, 64.0, 64.0);
            let c1 = scene.push_clip(Affine::IDENTITY, pb.build());
            clips.push(c1);
        }
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 28.0, false);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 120, 255)));
        scene.draws.last_mut().unwrap().clips = clips;

        let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
        let mut max_diff = 0u32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        }
        assert!(max_diff <= 3, "clip({nested}) GPU/CPU 最大差 {max_diff}");
    }
}

// ===================== M3a：活跃 tile 表（sparse strips 底座） =====================

/// 运行一次管线并转储中间数据（M3a 验证用：固定走稠密参照路径）。
fn dump_pipeline(renderer: &GpuRenderer, scene: &Scene, eps: f32) -> (GpuScene, PipelineDump) {
    let gs = build_gpu_scene(scene, eps);
    let tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("amiki_m3a_out"),
        size: wgpu::Extent3d { width: gs.width, height: gs.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut dump = PipelineDump {
        counters: Some(Vec::new()),
        carries: Some(Vec::new()),
        strip_meta: Some(Vec::new()),
        ..Default::default()
    };
    let mut prof = FrameProfile::default();
    renderer
        .run_pipeline_impl(&gs, &tex, false, &[], None, Some(&mut dump), None, &mut prof)
        .expect("管线运行失败");
    (gs, dump)
}

/// 稠密网格活动单元 ↔ strip 区间一致性：
/// · 覆盖性（恒成立）：任何活动单元（成员计数>0 / 真实 carry≠0）必落在对应 strip 内；
/// · 无幻影：完全无活动的 (draw, tr) 不应有 strip；
/// · `exact = true` 时额外要求区间与活动包围盒逐格相等——画布内轴对齐几何
///   应精确成立；越界内容会按稠密语义保守扩张左/右界（见 strips.wgsl）。
fn check_strips(gs: &GpuScene, dump: &PipelineDump, exact: bool) {
    /// 低于此视为闭合路径求和的浮点残差（约 1e-7），非真实 carry。
    const CARRY_EPS: f32 = 1e-4;
    let counters = dump.counters.as_ref().expect("counters 未转储");
    let carries = dump.carries.as_ref().expect("carries 未转储");
    let meta = dump.strip_meta.as_ref().expect("strip_meta 未转储");
    let n_draws = gs.colors.len() as u32;
    let (mut n_active, mut n_covered) = (0usize, 0usize);
    for d in 0..n_draws {
        for tr in 0..gs.tiles_y {
            let si = (tr * n_draws + d) as usize; // 转置布局：si = tr * n_draws + d
            let (lo_s, hi_s, y0_s, y1_s) =
                (meta[si * 4], meta[si * 4 + 1], meta[si * 4 + 2], meta[si * 4 + 3]);
            let (mut mn_c, mut mx_c, mut mn_r, mut mx_r) = (u32::MAX, 0u32, u32::MAX, 0u32);
            for col in 0..gs.tiles_x {
                for row in 0..16 {
                    let b = (((tr * gs.tiles_x + col) * n_draws + d) * 16 + row) as usize;
                    if counters[b] > 0 || carries[b].abs() > CARRY_EPS {
                        mn_c = mn_c.min(col);
                        mx_c = mx_c.max(col);
                        mn_r = mn_r.min(row);
                        mx_r = mx_r.max(row);
                    }
                }
            }
            if mn_c == u32::MAX {
                assert_eq!(lo_s, gs.tiles_x, "无活动的 (draw={d}, tr={tr}) 不应有 strip");
                continue;
            }
            n_active += ((mx_c - mn_c + 1) as usize) * ((mx_r - mn_r + 1) as usize);
            assert!(lo_s != gs.tiles_x, "有活动但 strip 缺失 (draw={d}, tr={tr})");
            // 覆盖性：strip 必须框住全部活动单元
            assert!(
                lo_s <= mn_c && mx_c <= hi_s && y0_s <= mn_r && mx_r <= y1_s,
                "strip 未覆盖活动单元 (draw={d}, tr={tr}): strip=({lo_s},{hi_s},{y0_s},{y1_s}) 活动=({mn_c},{mx_c},{mn_r},{mx_r})"
            );
            n_covered += ((hi_s - lo_s + 1) as usize) * ((y1_s - y0_s + 1) as usize);
            if exact {
                assert_eq!(
                    (lo_s, hi_s, y0_s, y1_s),
                    (mn_c, mx_c, mn_r, mx_r),
                    "strip 必须与稠密活动包围盒逐格相等 (draw={d}, tr={tr})"
                );
            }
        }
    }
    eprintln!("M3a 对拍：活动单元 {n_active}，strip 覆盖单元 {n_covered}");
}

// ===================== M3b：稀疏桶表 =====================

/// M3b：稀疏路径 vs 稠密路径逐字节一致（多场景，含越界/混合模式/裁剪）。
#[test]
fn sparse_matches_dense() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Ok(mut renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let mut scenes: Vec<(&str, Scene)> = Vec::new();
    {
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 40.0, 40.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 40, 40)));
        scenes.push(("rect", scene));
    }
    {
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 72.0, 72.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(36.0, 36.0), 28.0, false);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40, 90, 160, 200)),
            BlendMode::Multiply,
        );
        let mut pb = PathBuilder::new();
        pb.rect(-16.0, -10.0, 16.0, 10.0);
        let t = Affine::translate(Vec2::new(44.0, 36.0)).then(Affine::rotate(0.5));
        scene.push(t, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 180)));
        scenes.push(("mixed", scene));
    }
    {
        let mut scene = Scene::new(96, 96);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        let t = Affine::translate(Vec2::new(48.0, 48.0))
            .then(Affine::rotate(0.3))
            .then(Affine::translate(Vec2::new(-48.0, -48.0)));
        scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(60.0, 60.0), 34.0);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(230, 120, 60, 200)),
            BlendMode::Multiply,
        );
        scenes.push(("curves", scene));
    }
    {
        // 越出画布右缘 / 左缘 / 四角（carry 右界扩张与桶 0 钳制路径）
        let mut scene = Scene::new(80, 64);
        let mut pb = PathBuilder::new();
        pb.rect(40.0, 8.0, 140.0, 48.0); // 跨右缘
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 160, 90)));
        let mut pb = PathBuilder::new();
        pb.rect(-30.0, 4.0, 30.0, 40.0); // 跨左缘
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(200, 90, 30, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(78.0, 60.0), 22.0, false); // 跨右下角
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(60, 60, 220, 160)),
            BlendMode::Screen,
        );
        scenes.push(("offscreen", scene));
    }
    {
        // 裁剪 + 非可分离混合模式（clip 覆盖率纹理亦走稀疏路径）
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
        let c0 = scene.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.rect(16.0, 16.0, 64.0, 64.0);
        let c1 = scene.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(32.0, 32.0), 26.0);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40, 90, 160, 220)),
            BlendMode::Hue,
        );
        scene.draws.last_mut().unwrap().clips = vec![c0, c1];
        scenes.push(("clips", scene));
    }
    {
        // 全 16 种混合模式（逐个对比）
        let modes = [
            BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
            BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
            BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference, BlendMode::Exclusion,
            BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity,
        ];
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 56.0, 56.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 90, 160)));
        for (i, mode) in modes.iter().enumerate() {
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(16.0 + (i % 4) as f32 * 10.0, 16.0 + (i / 4) as f32 * 10.0), 14.0, false);
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(230, 120, 60, 200)),
                *mode,
            );
        }
        scenes.push(("blend16", scene));
    }

    for (name, scene) in &scenes {
        renderer.path = RenderPath::Dense;
        let dense = renderer.render_blocking(scene, 0.05).expect("稠密渲染失败");
        renderer.path = RenderPath::Sparse;
        let sparse = renderer.render_blocking(scene, 0.05).expect("稀疏渲染失败");
        let mut max_diff = 0u32;
        let mut n_diff = 0usize;
        for (a, b) in dense.iter().zip(sparse.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            max_diff = max_diff.max(d);
            if d > 0 {
                n_diff += 1;
            }
        }
        eprintln!("{name}: 最大差 {max_diff}，不同字节 {n_diff}/{}", dense.len());
        assert_eq!(max_diff, 0, "{name}: 稀疏与稠密渲染不一致（最大差 {max_diff}）");
    }
}

/// M3c：稀疏路径在「稠密不可能」的规模下可运行，且显存与内容成比例。
/// 目标参照 §4.5（1080p × 10k 曲线）；测试用 2k 曲线控制耗时。
#[test]
fn sparse_scales_beyond_dense() {
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, Scene};
    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let (w, h, n) = (1920u32, 1080u32, 2000usize);
    let mut scene = Scene::new(w, h);
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..n {
        let c = Vec2::new(rnd() * w as f32, rnd() * h as f32);
        let mut pb = amiki_scene::PathBuilder::new();
        pb.circle(c, 4.0 + 4.0 * rnd(), false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 120, 60, 200)),
        );
    }

    // 稠密网格：显存 ∝ tiles×draws，早已越过 128MB 存储绑定上限
    let gs = build_gpu_scene(&scene, 0.1);
    let n_grid = (gs.tiles_x as u64) * (gs.tiles_y as u64) * (n as u64) * 16;
    let dense_bytes = n_grid * 12 + (n as u64) * (gs.tiles_y as u64) * 16 * ((gs.tiles_x + 1) as u64) * 4;
    assert!(
        dense_bytes > 128 * 1024 * 1024,
        "本测试应处于稠密不可行区间（实际稠密需求 {:.1} MB）",
        dense_bytes as f64 / 1e6
    );

    // 稀疏路径：可运行，且占用与内容成比例
    let img = renderer.render_blocking(&scene, 0.1).expect("稀疏渲染失败");
    assert_eq!(img.len(), (w * h * 4) as usize);
    let st = renderer.sparse_stats(&scene, 0.1).expect("统计失败").expect("稀疏统计缺失");
    let sparse_bytes = st.n_cells as u64 * 24
        + st.n_buckets as u64 * 4
        + st.n_tile_entries as u64 * 16
        + st.n_items as u64 * 16;
    eprintln!(
        "1080p × {n} 曲线：稠密需求 {:.0} MB → 稀疏 {:.1} MB（条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}）",
        dense_bytes as f64 / 1e6,
        sparse_bytes as f64 / 1e6,
        st.n_strips, st.n_cells, st.n_buckets, st.n_tile_entries, st.n_items
    );
    assert!(
        sparse_bytes * 50 < dense_bytes,
        "稀疏占用应显著小于稠密（稀疏 {sparse_bytes} vs 稠密 {dense_bytes}）"
    );
}

/// M3.6：投机分配的**容量表**必须来自实测用量，而不是写死的猜测，
/// 且必须**按场景特征分表**（共用一个全局值会让大小场景互相踩，
/// 实测 720p×64 曾因此退化 3 倍）。
///
/// 本测试回答"学到了什么"：渲染一帧后，表里应当出现**该场景**的表项，
/// 且其规模恰好等于该帧实测的用量（与 `sparse_stats` 逐项一致）。
/// 容量**够不够用**、不够时能否被检出并重做，见
/// `speculative_overflow_is_redone_correctly`。
#[test]
fn sparse_capacity_is_learned_from_measured_usage() {
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    assert_eq!(renderer.capacity_table_len(), 0, "首帧之前不应有任何容量表项");

    let mut scene = Scene::new(160, 160);
    for i in 0..12 {
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(24.0 + 9.0 * i as f32, 60.0 + 5.0 * i as f32), 9.0, false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(60, 140, 220, 200)),
        );
    }

    renderer.render_blocking(&scene, 0.05).expect("渲染失败");

    let entries = renderer.capacity_entries();
    assert_eq!(entries.len(), 1, "应为该场景建立一个表项：{entries:?}");
    let (key, cap) = entries[0];
    assert_eq!(
        (key.width, key.height, key.draws),
        (160, 160, 12),
        "key 应描述刚渲染的场景（宽/高/draw 数）"
    );
    assert!(
        key.seg_bucket.is_power_of_two(),
        "几何桶应为 2 的幂（量化后才稳定）：{key:?}"
    );

    let st = renderer.sparse_stats(&scene, 0.05).expect("统计失败").expect("稀疏统计缺失");
    assert_eq!(
        (cap.cells, cap.buckets, cap.tile_entries, cap.items),
        (st.n_cells, st.n_buckets, st.n_tile_entries, st.n_items),
        "容量必须等于实测用量（单元/桶/条目/成员）"
    );
}

/// P3：**启动开销**必须保持在目标以内。
///
/// 指标来自发布计划：`GpuRenderer::new()` < 200 ms（原为 0.5–0.7 s）。
///
/// # 实测（RTX 3050 / Windows / release）
///
/// | 配置 | 冷启动 | 热启动 | 达标 |
/// |---|---|---|---|
/// | **默认**（全部后端 → 本机选 Vulkan） | **337 ms** | 202 ms | ✗ |
/// | `AMIKI_BACKENDS=vulkan` | **154 ms** | 53 ms | ✓ |
/// | `AMIKI_BACKENDS=dx12` | 2272 ms | 2093 ms | ✗ |
///
/// **结论：本项"有条件达标"** —— 默认配置下未达标，显式指定后端时达标。
/// 这个条件必须写明，不能含糊成"已达标"。
///
/// 为什么默认不指定后端：默认值要保证"哪台机器都能起来"，而指定单一后端
/// 会削弱项目"一套 WGSL 通吃"的既定价值（我一度改成默认只用 Vulkan，
/// 快一倍，但按这个理由改回了）。
///
/// 需要更快启动的应用有两条正当途径：
/// 1. `AMIKI_BACKENDS` 显式指定后端；
/// 2. `GpuRenderer::new` 本就是 `async` —— 与窗口初始化**重叠**，
///    不让驱动初始化阻塞首帧（这才是引擎该给的答案）。
///
/// 断言取 **1200 ms**：只作为"病态回归"的兜底（例如误改成同步阻塞），
/// 不去卡一个本项尚未达成的指标 —— 那种断言会在慢机器上产生假失败，
/// 也会掩盖"其实没达标"这个事实。
///
/// 定位结论：启动开销的 **86% 在后端枚举**（`Instance` + 请求适配器），
/// 21 条管线只占 12.7 ms。所以发布计划里"用 PipelineCache / 惰性创建"
/// 的假设是**错的**，已按实测更正。
#[test]
fn startup_cost_stays_within_budget() {
    let t0 = std::time::Instant::now();
    let Ok(r1) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t1 = std::time::Instant::now();
    let Ok(r2) = pollster::block_on(GpuRenderer::new()) else { return };
    let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[启动] 适配器 = {}；首次(冷) = {cold_ms:.1} ms；第二次(热) = {warm_ms:.1} ms",
        r1.adapter_name
    );
    assert!(
        r1.adapter_name == r2.adapter_name,
        "同一进程内两次创建应选中同一个适配器"
    );
    // 显式指定后端时（`AMIKI_BACKENDS`）不套用默认预算：那时测的不是默认配置。
    // DX12 走 FXC 运行期编译，冷启动实测约 2.7 s，本就不该套用本预算。
    if std::env::var("AMIKI_BACKENDS").is_ok() {
        eprintln!("（已显式指定后端，跳过预算断言）");
        return;
    }
    assert!(
        cold_ms < 1200.0,
        "冷启动 {cold_ms:.1} ms —— 这是病态回归的兜底阈值；\
         注意**本项未达标**（目标 200 ms，实测默认 ~340 ms），详见测试文档注释"
    );
}

/// P0-3：**帧路径必须能渲染含层场景**，且与参考逐像素一致。
///
/// 覆盖三种滤镜（`None` / `Blur` / `DropShadow`）× 两条入口
/// （`render_into` 无 damage、`render_into_persistent` 带 damage）。
///
/// 对拍基准有两个，缺一不可：
/// - `render_blocking`（GPU 侧重合成参考）—— 验证帧路径的接线；
/// - CPU `amiki_raster`（唯一真值）—— 验证数学本身。
///
/// 本测试在开发中抓到过一个**参数传反**的缺陷（`composite_pass` 的
/// `layer_tex` 与 `shadow_tex` 位置写反），症状是层边缘出现一圈不该有的淡影；
/// 若只跟 `render_blocking` 比而不跟 CPU 比，这个错误会被漏掉若干轮。
#[test]
fn gpu_frame_path_composites_layers() {
    use amiki_math::Affine;
    use amiki_scene::{BlendMode, Color, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    let build = |filter: Filter| -> Scene {
        let mut scene = Scene::new(160, 120);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 120.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(246, 243, 238)));
        // 层外图元，用来确认"层 + 图元"都要在
        let mut pb = PathBuilder::new();
        pb.rect(6.0, 90.0, 40.0, 114.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 140, 90)));
        scene.push_layer(
            Layer {
                bounds: LayerBounds::new(36, 26, 70, 70),
                scene: Scene::new(160, 120),
                filter,
                opacity: 0.95,
                blend: BlendMode::Normal,
            },
            |s| {
                let mut pb = PathBuilder::new();
                pb.rect(44.0, 34.0, 96.0, 86.0);
                s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 60, 50)));
            },
        );
        scene
    };

    let cases: Vec<(&str, Filter)> = vec![
        ("分组", Filter::None),
        ("模糊", Filter::Blur { sigma: 6.0 }),
        (
            "投影",
            Filter::DropShadow {
                dx: 9.0,
                dy: 11.0,
                sigma: 7.0,
                color: Color::rgba8(25, 22, 20, 190),
            },
        ),
        (
            "投影无模糊",
            Filter::DropShadow {
                dx: 9.0,
                dy: 11.0,
                sigma: 0.0,
                color: Color::rgba8(25, 22, 20, 190),
            },
        ),
    ];

    for (name, filter) in cases {
        let scene = build(filter);
        let reference = renderer
            .render_blocking(&scene, 0.05)
            .unwrap_or_else(|e| panic!("{name}层：render_blocking 失败（{e}）"));
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

        // ① 帧路径（无 damage）
        let frame = renderer.prepare(&scene, 0.05).expect("prepare 失败");
        assert!(frame.scene.has_layers, "场景应被识别为含层");
        let tex = renderer.create_output_texture(&frame.scene);
        renderer
            .render_into(&tex, &frame)
            .unwrap_or_else(|e| panic!("{name}层：帧路径不得拒绝含层场景（{e}）"));
        let got = renderer.readback(&tex).expect("读回失败");

        // ② 帧持久路径（全域 damage，等价全量）
        let frame2 = renderer.prepare(&scene, 0.05).expect("prepare 失败");
        let tex2 = renderer.create_output_texture(&frame2.scene);
        let d = amiki_scene::Damage::new(160, 120);
        renderer
            .render_into_persistent(&tex2, &frame2, Some(&d))
            .unwrap_or_else(|e| panic!("{name}层：帧持久路径不得拒绝含层场景（{e}）"));
        let got_persist = renderer.readback(&tex2).expect("读回失败");

        let diff = |a: &[u8], b: &[u8]| -> u32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
                .max()
                .unwrap_or(0)
        };
        // 与 `render_blocking`：两者不是同一条浮点路径（那条是"读回 u8 →
        // 主机 f32 → 写回 u8"的往返），允许 ±1 舍入差。
        let max_ref = diff(&got, &reference);
        let max_cpu = diff(&got, &cpu);
        let max_persist = diff(&got_persist, &cpu);
        assert!(max_ref <= 2, "{name}层：与 render_blocking 差 {max_ref} 超容差");
        assert!(max_cpu <= 2, "{name}层：与 CPU 差 {max_cpu} 超容差");
        assert!(max_persist <= 2, "{name}层：帧持久路径与 CPU 差 {max_persist} 超容差");
        eprintln!(
            "帧路径 {name}层：render_blocking {max_ref} / CPU {max_cpu} / 持久 {max_persist}"
        );
    }
}



/// M3a：轴对齐矩形（多 draw、半透明、跨 tile、部分出画布）——strip 逐格对拍。
#[test]
fn strips_match_dense_rects() {
    use amiki_math::Affine;
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    // 高 72 = 4.5 个 tile 行 → 末行带为部分带（验证高度钳制）
    let mut scene = Scene::new(72, 72);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 72.0, 72.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
    let mut pb = PathBuilder::new();
    pb.rect(8.0, 8.0, 40.0, 56.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
    // 左/上出画布（carry 桶 0 钳制路径）
    let mut pb = PathBuilder::new();
    pb.rect(-24.0, -12.0, 20.0, 30.0);
    scene.push_blend(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(230, 120, 60)), BlendMode::Multiply);
    // 完全出画布（左下方）→ 不得产生 strip
    let mut pb = PathBuilder::new();
    pb.rect(-200.0, 200.0, -160.0, 260.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 0, 0)));

    let (gs, dump) = dump_pipeline(&renderer, &scene, 0.05);
    check_strips(&gs, &dump, true);
}

/// M3a：曲线/旋转/自相交混合场景——strip 覆盖性（含越界保守扩张路径）。
#[test]
fn strips_match_dense_curves() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let mut scene = Scene::new(96, 96);
    // 环形（曲线 + 反绕洞）+ 旋转（斜边跨带）
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
    pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
    let t = Affine::translate(Vec2::new(48.0, 48.0))
        .then(Affine::rotate(0.3))
        .then(Affine::translate(Vec2::new(-48.0, -48.0)));
    scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
    // 自相交五角星（直线，W=2 区域）
    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(60.0, 60.0), 34.0);
    scene.push_blend(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgba8(230, 120, 60, 200)),
        BlendMode::Multiply,
    );
    // 右下角出画布曲线（钳制路径）
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(104.0, 100.0), 24.0, false);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(52, 152, 219)));

    let (gs, dump) = dump_pipeline(&renderer, &scene, 0.05);
    check_strips(&gs, &dump, false);
}

/// M3.5：稠密容量预检——超限必须返回**可读错误**，而不是让 wgpu 校验层 panic。
///
/// 回归背景：720p × 500 曲线走稠密时，`grid_aux` 需 230.4 MB，超过默认
/// `max_storage_buffer_binding_size`（128 MiB），此前表现为
/// `create_bind_group` 内部的 wgpu Validation Error panic（进程终止、无法降级）。
#[test]
fn dense_ceiling_reports_error_not_panic() {
    let Ok(mut renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    // 1) 纯函数判定：默认限制下 720p × 500 draw 必须超限，且错误信息含关键数字
    let limits = wgpu::Limits::default();
    let (tx, ty) = (1280u32.div_ceil(16), 720u32.div_ceil(16)); // 80 × 45
    let err = dense_capacity(&limits, tx, ty, 500).expect_err("720p×500 应判定超限");
    let msg = err.to_string();
    assert!(msg.contains("稠密网格超限"), "错误信息应说明是稠密超限：{msg}");
    assert!(msg.contains("grid_aux"), "错误信息应指明是哪个缓冲：{msg}");
    assert!(msg.contains("RenderPath::Sparse"), "错误信息应给出降级建议：{msg}");
    eprintln!("预检信息（应可读）：{msg}");

    // 2) 正常规模不应被误判
    assert!(dense_capacity(&limits, tx, ty, 8).is_ok(), "小场景不应被预检拦下");

    // 3) 端到端：走稠密路径渲染超限场景必须返回 Err（而非 panic）
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, PathBuilder};
    let mut scene = Scene::new(1280, 720);
    let mut s = 0x5EED_1234_5678_9ABCu64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..500 {
        let c = Vec2::new(rnd() * 1280.0, rnd() * 720.0);
        let mut pb = PathBuilder::new();
        pb.circle(c, 4.0 + 6.0 * rnd(), false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 120, 60, 200)),
        );
    }

    let mut r = pollster::block_on(GpuRenderer::new()).expect("无 GPU");
    r.path = RenderPath::Dense;
    let res = r.render_blocking(&scene, 0.1);
    assert!(
        matches!(res, Err(ref e) if e.to_string().contains("稠密网格超限")),
        "稠密超限应返回 Err(稠密网格超限)，实际：{:?}",
        res.map(|v| format!("Ok({} 字节)", v.len()))
    );

    // 4) 同一场景走稀疏路径必须成功（验证建议的降级方案确实可行）
    renderer.path = RenderPath::Sparse;
    let img = renderer.render_blocking(&scene, 0.1).expect("稀疏路径应可渲染同一场景");
    assert_eq!(img.len(), 1280 * 720 * 4);
    eprintln!("同一场景：稠密返回可读错误，稀疏渲染成功（{} 字节）", img.len());
}

// ===================== M3.5：稀疏自洽检查 =====================

/// 覆盖各类稀疏结构的场景集（矩形 / 曲线+洞 / 自相交 / 越界 / 裁剪+混合）。
fn consistency_scenes() -> Vec<(&'static str, Scene)> {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder};
    let mut out: Vec<(&'static str, Scene)> = Vec::new();
    {
        // 轴对齐 + 跨 tile 边界（tile 区间与局部行区间最严格）
        let mut s = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 72.0, 72.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(16.0, 16.0, 33.0, 49.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
        out.push(("rects", s));
    }
    {
        // 曲线 + 反绕洞 + 旋转（斜边跨带，carry 非零多）
        let mut s = Scene::new(96, 96);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        let t = Affine::translate(Vec2::new(48.0, 48.0))
            .then(Affine::rotate(0.3))
            .then(Affine::translate(Vec2::new(-48.0, -48.0)));
        s.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
        out.push(("donut_rot", s));
    }
    {
        // 自相交（W=2）+ 越出画布四缘（strip 保守扩张路径）
        let mut s = Scene::new(80, 64);
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(40.0, 30.0), 34.0);
        s.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(230, 120, 60, 200)),
            BlendMode::Multiply,
        );
        let mut pb = PathBuilder::new();
        pb.rect(-30.0, 4.0, 30.0, 40.0); // 跨左缘（桶 0 钳制）
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 160, 90)));
        let mut pb = PathBuilder::new();
        pb.rect(40.0, 8.0, 140.0, 48.0); // 跨右缘（桶 tiles_x 排除）
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(200, 90, 30, 200)));
        out.push(("pentagram_offscreen", s));
    }
    {
        // 裁剪 + 非可分离混合
        let mut s = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
        let c0 = s.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(32.0, 32.0), 26.0);
        s.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40, 90, 160, 220)),
            BlendMode::Hue,
        );
        s.draws.last_mut().unwrap().clips = vec![c0];
        out.push(("clip_blend", s));
    }
    out
}

/// 深拷贝转储（破坏性测试用）。
fn clone_dump(d: &PipelineDump) -> PipelineDump {
    PipelineDump {
        seg_ends: d.seg_ends.clone(),
        seg_draws: d.seg_draws.clone(),
        counters: d.counters.clone(),
        scan_base: d.scan_base.clone(),
        carries: d.carries.clone(),
        item_total: d.item_total,
        seg_refs: d.seg_refs.clone(),
        bucket_sums: d.bucket_sums.clone(),
        strip_meta: d.strip_meta.clone(),
        sparse_stats: d.sparse_stats,
        tr_offsets: d.tr_offsets.clone(),
        tr_list: d.tr_list.clone(),
        tile_count: d.tile_count.clone(),
        tile_start: d.tile_start.clone(),
        tile_list: d.tile_list.clone(),
        cell_meta: d.cell_meta.clone(),
        cell_segs: d.cell_segs.clone(),
        strip_cell_off: d.strip_cell_off.clone(),
        strip_bucket_off: d.strip_bucket_off.clone(),
        cell_aux: d.cell_aux.clone(),
        cell_csr_off: d.cell_csr_off.clone(),
        sparse_buckets: d.sparse_buckets.clone(),
    }
}

/// M3.5：稀疏路径自洽——**不依赖稠密路径**的结构正确性验证。
///
/// 这是 M3 之后最重要的安全网：稠密参照在大场景（1080p 约 31 draw 起）跑不起来，
/// 稀疏此前没有独立裁判。
#[test]
fn sparse_self_consistency() {
    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    for (name, scene) in consistency_scenes() {
        let (gs, dump) = renderer.dump_sparse(&scene, 0.05).expect("稀疏管线失败");
        if let Err(e) = check_sparse_consistency(&gs, &dump) {
            panic!("[{name}] 稀疏自洽检查失败：{e}");
        }
        let s = dump.sparse_stats.expect("统计缺失");
        eprintln!(
            "[{name}] 自洽 ✓  条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}",
            s[4], s[0], s[1], s[2], s[3]
        );
    }
}

/// M3.5 反向验证：**故意破坏稀疏结构，检查器必须抓住**。
///
/// 没有这一步，"检查通过"可能只是因为检查器什么都没查。
#[test]
fn sparse_consistency_detects_corruption() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Paint, PathBuilder};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let mut scene = Scene::new(96, 96);
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(40.0, 40.0), 30.0, false);
    pb.circle(Vec2::new(40.0, 40.0), 14.0, true);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 160, 90, 220)));
    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(60.0, 60.0), 28.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 200)));

    let (gs, dump) = renderer.dump_sparse(&scene, 0.05).expect("稀疏管线失败");
    check_sparse_consistency(&gs, &dump).expect("基准场景应自洽");

    // 破坏 1：tile_count 人为 +1 → 逐 tile 桶表条目数应不符
    let mut bad = clone_dump(&dump);
    let tile = bad
        .tile_count
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|c| **c > 0)
        .expect("本场景应有非空 tile");
    *tile += 1;
    let e1 = check_sparse_consistency(&gs, &bad).expect_err("应检出桶表条目数不符");
    assert!(e1.to_string().contains("桶表条目数不一致"), "错误信息不符：{e1}");
    eprintln!("破坏 tile_count → 已检出：{e1}");

    // 破坏 2：cell_meta 的 carry 位模式扰动 → carry 与桶前缀应不符
    let mut bad = clone_dump(&dump);
    let cm = bad
        .cell_meta
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|c| f32::from_bits(c[0]) != 0.0)
        .expect("本场景应有非零 carry 的单元");
    cm[0] = (f32::from_bits(cm[0]) + 0.5).to_bits();
    let e2 = check_sparse_consistency(&gs, &bad).expect_err("应检出 carry 不符");
    assert!(e2.to_string().contains("carry"), "错误信息不符：{e2}");
    eprintln!("破坏 carry 位模式 → 已检出：{e2}");

    // 破坏 3：段数量 +1 → 与 emit 成员计数应不符
    let mut bad = clone_dump(&dump);
    let cm = bad
        .cell_meta
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c[2] > 0)
        .expect("本场景应有非空单元");
    cm[2] += 1;
    let e3 = check_sparse_consistency(&gs, &bad).expect_err("应检出段数量不符");
    assert!(
        e3.to_string().contains("段数量与 emit 成员计数不符")
            || e3.to_string().contains("段范围越界")
            || e3.to_string().contains("CSR 非连续"),
        "错误信息不符：{e3}"
    );
    eprintln!("破坏段数量 → 已检出：{e3}");

    // 破坏 4：tr_list 交换两项 → draw 序错乱应被检出（若该行有 ≥2 个活跃 strip）
    let mut bad = clone_dump(&dump);
    let tr_list = bad.tr_list.as_mut().unwrap();
    if let Some(tr) = bad.tr_offsets.as_ref().unwrap().iter().position(|o| o[1] >= 2) {
        let start = bad.tr_offsets.as_ref().unwrap()[tr][0] as usize;
        tr_list.swap(start, start + 1);
        let e4 = check_sparse_consistency(&gs, &bad).expect_err("交换 tr_list 应被检出");
        eprintln!("交换 tr_list 相邻项 → 已检出：{e4}");
    }
}

/// M3.5：**大场景**（稠密跑不起来）的全链路转储必须可完成且自洽。
///
/// 这是 ③ 的存在理由：稠密参照在 1080p 约 31 draw 起就撞 128 MiB 绑定上限，
/// 稀疏此前在大场景没有裁判。本测试证明：该规模下稀疏**有自己的裁判**，
/// 且全字段转储不会因缓冲过大而失败。
#[test]
fn sparse_consistency_at_scale() {
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, PathBuilder};

    let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let (w, h, n) = (1920u32, 1080u32, 600usize);
    let mut scene = Scene::new(w, h);
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..n {
        let c = Vec2::new(rnd() * w as f32, rnd() * h as f32);
        let mut pb = PathBuilder::new();
        pb.circle(c, 4.0 + 6.0 * rnd(), false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 120, 60, 200)),
        );
    }

    // 该规模下稠密应当已被预检拦下（坐实"稠密跑不起来"这一前提）
    let (tx, ty) = (w.div_ceil(16), h.div_ceil(16));
    if dense_capacity(&wgpu::Limits::default(), tx, ty, n as u32).is_ok() {
        eprintln!("注意：本规模稠密仍可运行，测试未处于预期区间");
    }

    let (gs, dump) = renderer.dump_sparse(&scene, 0.1).expect("大场景稀疏转储失败");
    let st = dump.sparse_stats.expect("统计缺失");
    eprintln!(
        "1080p × {n} 曲线：转储完成（条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}）",
        st[4], st[0], st[1], st[2], st[3]
    );
    match check_sparse_consistency(&gs, &dump) {
        Ok(()) => eprintln!("大场景稀疏自洽 ✓"),
        Err(e) => panic!("1080p × {n} 曲线稀疏自洽检查失败：{e}"),
    }
}

/// M4：**GPU 侧只派发脏 tile** 必须与全量派发**逐字节一致**。
///
/// 这是 GPU 增量渲染的核心不变量：fine 内核由 `global_invocation_id` 推导
/// 像素与 tile 坐标，未派发的 tile 不写入、保留纹理中的旧值。若派发范围算错
/// （例如偏移没被正确处理、或脏区外被误写），本测试会立刻失败。
///
/// 覆盖三类情形：全脏 / 局部脏 / 无脏区（跳过）。
#[test]
fn gpu_damage_dispatch_matches_full() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Damage, Paint, PathBuilder};

    let Ok(mut renderer) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    renderer.path = RenderPath::Sparse;

    let mut scene = Scene::new(128, 96);
    for i in 0..6 {
        let mut pb = PathBuilder::new();
        let x = 10.0 + (i % 3) as f32 * 38.0;
        let y = 14.0 + (i / 3) as f32 * 44.0;
        pb.circle(Vec2::new(x, y), 16.0, false);
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40 + i as u8 * 30, 120, 200 - i as u8 * 20, 210)),
        );
    }

    let frame = renderer.prepare(&scene, 0.1).expect("prepare 失败");
    let (tx, ty) = (frame.scene.tiles_x, frame.scene.tiles_y);
    let tex = renderer.create_output_texture(&frame.scene);

    // 参照：全量派发
    renderer
        .render_into_persistent(&tex, &frame, None)
        .expect("全量渲染失败");
    let full = renderer.readback(&tex).expect("读回失败");

    // 1) 局部脏区（含偏移，确保不是左上角起步）
    let mut d = Damage::new(scene.width, scene.height);
    d.clear();
    d.invalidate_rect(42.0, 20.0, 100.0, 84.0);
    let skipped = renderer
        .render_into_persistent(&tex, &frame, Some(&d))
        .expect("局部脏区渲染失败");
    assert!(!skipped, "有脏区时不得跳过");
    let after = renderer.readback(&tex).expect("读回失败");
    assert_eq!(after, full, "只派发脏 tile 后，画面与全量派发不一致");

    // 2) 无脏区：必须完全跳过且不改变像素
    let mut d2 = Damage::new(scene.width, scene.height);
    d2.clear();
    assert!(d2.is_empty());
    let skipped = renderer
        .render_into_persistent(&tex, &frame, Some(&d2))
        .expect("空脏区渲染失败");
    assert!(skipped, "无脏区时必须跳过");
    let after2 = renderer.readback(&tex).expect("读回失败");
    assert_eq!(after2, full, "跳过的帧不得改变任何像素");

    // 3) 全脏：也必须等于全量
    let d3 = Damage::new(scene.width, scene.height);
    assert!(d3.is_full());
    renderer
        .render_into_persistent(&tex, &frame, Some(&d3))
        .expect("全脏渲染失败");
    let after3 = renderer.readback(&tex).expect("读回失败");
    assert_eq!(after3, full, "全脏渲染与全量派发不一致");

    // 4) 脏区 tile 数校验：尺寸不匹配的 damage 必须被断言拦下（防"全脏被误判为局部"）
    let wrong = Damage::new(scene.width + 64, scene.height);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        renderer.render_into_persistent(&tex, &frame, Some(&wrong))
    }))
    .is_err();
    assert!(caught, "画布尺寸不匹配的 damage 应被断言拦下");

    eprintln!(
        "GPU 脏区派发：全脏 / 局部脏 / 空脏区 三类均与全量逐字节一致（tiles={tx}x{ty}）"
    );
}
