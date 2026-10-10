//! 基础对拍：GPU 与 CPU 逐像素一致（矩形/渐变/位图/mipmap/层/scan/混合/裁剪）。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

/// GPU/CPU 渲染对拍（donut + 旋转）。
/// M2f 已知问题：曲线边缘像素覆盖率偏差（圆极端点行约 -0.28，且不随 ε 变化，
/// 怀疑 flatten 段端点或 seg_ends 数据链路），调查恢复后回归 tolerance 2。
#[test]
fn gpu_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };

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

    let Some((renderer, _gpu_guard)) = gpu() else { return };
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

    let Some((renderer, _gpu_guard)) = gpu() else { return };

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

    let Some((renderer, _gpu_guard)) = gpu() else { return };

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

/// B4：**mip 采样下 GPU 与 CPU 逐像素一致**。
///
/// 用伪随机噪声图（各 mip 层内容真实不同，层级选错立刻暴露），
/// 缩小到 **LOD 带小数部分**（≈1.68）——整数 LOD 只采一层，只有小数部分
/// 才能验证"层间插值 + 层内双线性"的两端语义一致。
/// 同时覆盖旋转（雅可比含切变，LOD 从轴向量算）与平铺（LOD 按单元算）。
#[test]
fn gpu_image_mipmap_matches_cpu() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Image, ImagePlacement, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };

    // 64×64 噪声图，带 mip 链
    let (iw, ih) = (64u32, 64u32);
    let mut data = Vec::with_capacity((iw * ih * 4) as usize);
    for i in 0..(iw * ih) {
        let v = (i * 7 % 251) as u8;
        let g = (v as u32 * 3 % 255) as u8;
        let b = (v as u32 * 11 % 255) as u8;
        data.extend([v, g, b, 255]);
    }
    let img = std::sync::Arc::new(Image::from_rgba8_with_mipmap(iw, ih, data).expect("图像数据合法"));

    // 缩小 3.2×：LOD = log2(64/20) ≈ 1.68（带小数 → 驱动层间插值）
    let cases: Vec<(&str, ImagePlacement)> = vec![
        ("缩小", ImagePlacement::rect(4.0, 4.0, 24.0, 24.0)),
        (
            "缩小+旋转",
            ImagePlacement {
                origin: Vec2::new(30.0, 6.0),
                u_axis: Vec2::new(14.0, 7.0), // |u_axis| = 15.6 → LOD ≈ 2.04 附近
                v_axis: Vec2::new(-7.0, 14.0),
                repeat: false,
            },
        ),
        ("缩小+平铺", ImagePlacement { repeat: true, ..ImagePlacement::rect(0.0, 0.0, 40.0, 40.0) }),
    ];

    for (name, place) in cases {
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
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
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(250, 250, 250)));
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Image { image: img.clone(), placement: place },
        );

        // GL 后端读不出位图，引擎显式报错（同 gpu_image_matches_cpu 的处理）
        let gpu = match renderer.render_blocking_with(&scene, 0.05, amiki_scene::ToneMap::Raw, 0.0) {
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
        assert!(
            max_diff <= 3,
            "{name}放置（mip）：GPU 与 CPU 最大通道差 {max_diff}（字节偏移 {worst}）超出容差"
        );
        eprintln!("GPU 位图 mip（{name}放置）与 CPU 一致（最大通道差 {max_diff}）");
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

    let Some((renderer, _gpu_guard)) = gpu() else { return };

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
    let Some((renderer, _gpu_guard)) = gpu() else { return };
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

    let Some((renderer, _gpu_guard)) = gpu() else { return };

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

    let Some((renderer, _gpu_guard)) = gpu() else { return };

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

