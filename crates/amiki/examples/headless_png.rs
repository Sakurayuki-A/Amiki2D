//! 离屏渲染示例：构建演示场景 → GPU（或 CPU 兜底）渲染 → 输出 PNG。
//!
//! ```bash
//! cargo run --release -p amiki --example headless_png
//! ```

use amiki::color::*;
use amiki::math::Vec2;
use amiki::raster;
use amiki::scene::{BlendMode, Paint, PathBuilder};
use amiki::{Canvas, gpu};
use std::time::Instant;

const W: u32 = 1280;
const H: u32 = 720;

fn build_scene(t: f32) -> amiki::scene::Scene {
    let mut canvas = Canvas::new(W, H);

    // 背景
    canvas.rect(0.0, 0.0, W as f32, H as f32, Paint::Solid(INK));

    // 1) 旋转五角星（自相交，nonzero：中央五边形 W=2 → clamp 1）
    canvas.translate(Vec2::new(320.0, 300.0));
    canvas.rotate(t * 0.4);
    canvas.pentagram(Vec2::ZERO, 210.0, Paint::Solid(RED));

    // 2) 有洞的圆环（外圈 + 反绕内圈），缓慢平移
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(700.0, 220.0), 120.0, false);
    pb.circle(Vec2::new(700.0, 220.0), 58.0, true);
    canvas.reset();
    canvas.fill_path(&pb.build(), Paint::Solid(GREEN));

    // 3) 半透明大圆叠加（src-over alpha 合成）
    canvas.circle(
        Vec2::new(780.0, 300.0 + 24.0 * t.sin()),
        140.0,
        Paint::Solid(Color { r: BLUE.r, g: BLUE.g, b: BLUE.b, a: 0.55 }),
    );

    // 4) 裁剪展示（M2d）：固定圆窗内嵌旋转星
    let clip_c = Vec2::new(1010.0, 180.0);
    let mut pb = PathBuilder::new();
    pb.circle(clip_c, 105.0, false);
    canvas.push_clip(&pb.build());
    canvas.translate(clip_c);
    canvas.rotate(-t * 0.7);
    canvas.pentagram(
        Vec2::ZERO,
        150.0,
        Paint::Solid(Color { r: PURPLE.r, g: PURPLE.g, b: PURPLE.b, a: 0.95 }),
    );
    canvas.reset();
    canvas.pop_clip();
    let mut pb = PathBuilder::new();
    pb.circle(clip_c, 105.0, false);
    pb.circle(clip_c, 101.0, true);
    canvas.fill_path(&pb.build(), Paint::Solid(Color { r: 1.0, g: 1.0, b: 1.0, a: 0.12 }));

    // 5) 混合模式画廊（M2e）：彩色底条 + 8 种模式色块
    let gy = 500.0;
    let gh = 150.0;
    let cols = [RED, ORANGE, GREEN, BLUE, PURPLE];
    let bw = (W as f32 - 80.0) / 5.0;
    for (i, c) in cols.iter().enumerate() {
        let x0 = 40.0 + i as f32 * bw;
        canvas.rect(x0, gy, x0 + bw, gy + gh, Paint::Solid(*c));
    }
    let modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Luminosity,
    ];
    let tw = (W as f32 - 80.0) / modes.len() as f32;
    for (i, m) in modes.iter().enumerate() {
        let x0 = 40.0 + i as f32 * tw;
        canvas.set_blend(*m);
        canvas.rect(
            x0 + tw * 0.12,
            gy + gh * 0.15,
            x0 + tw * 0.88,
            gy + gh * 0.85,
            Paint::Solid(Color { r: 0.92, g: 0.86, b: 0.25, a: 0.85 }),
        );
    }
    canvas.set_blend(BlendMode::Normal);

    // 6) 沿正弦线的一串小圆：海量路径 + AA 质量展示
    for i in 0..50 {
        let x = 50.0 + i as f32 * 24.0;
        let y = 680.0 + (i as f32 * 0.45 + t).sin() * 18.0;
        let r = 5.0 + 4.0 * (i as f32 * 0.3).sin();
        canvas.circle(Vec2::new(x, y), r, Paint::Solid(ORANGE));
    }

    canvas.scene
}

fn main() {
    std::fs::create_dir_all("out").ok();
    let t0 = Instant::now();
    let scene = build_scene(0.7);
    let binned = raster::bin(&scene, 0.1);
    let bin_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let img = match pollster::block_on(gpu::GpuRenderer::new()) {
        Ok(r) => match r.render_blocking(&scene, 0.1) {
            Ok(px) => {
                println!("后端: GPU (wgpu)");
                px
            }
            Err(e) => {
                println!("GPU 渲染失败（{e}），使用 CPU 引擎");
                raster::render(&binned)
            }
        },
        Err(e) => {
            println!("后端: CPU（GPU 不可用: {e}）");
            raster::render(&binned)
        }
    };
    let render_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let path = "out/amiki_headless.png";
    image::save_buffer(path, &img, W, H, image::ColorType::Rgba8).expect("保存 PNG 失败");
    println!("分箱 {bin_ms:.2} ms | 渲染 {render_ms:.2} ms | 输出 {path}");
}
