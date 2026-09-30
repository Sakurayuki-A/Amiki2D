//! 临时诊断：CPU 与 GPU 的层合成逐像素对比（非交付物）。

use amiki_math::Affine;
use amiki_scene::{BlendMode, Color, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene};

fn main() {
    let filter = Filter::Blur { sigma: 6.0 };
    let mut scene = Scene::new(120, 100);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 120.0, 100.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 236, 230)));

    scene.push_layer(
        Layer {
            bounds: LayerBounds::new(26, 20, 68, 58),
            scene: Scene::new(120, 100),
            filter,
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |s| {
            let mut pb = PathBuilder::new();
            pb.rect(40.0, 34.0, 80.0, 66.0);
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 60, 50)));
        },
    );

    let renderer = pollster::block_on(amiki_gpu::GpuRenderer::new()).expect("GPU");
    let gpu = renderer.render_blocking(&scene, 0.05).expect("gpu");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

    println!("横向剖面 y=50（x 从 26 到 94）：");
    print!("  x:   ");
    for x in (26..95).step_by(6) {
        print!("{x:>6}");
    }
    println!();
    print!("  CPU: ");
    for x in (26..95).step_by(6) {
        print!("{:>6}", cpu[(50 * 120 + x) * 4]);
    }
    println!();
    print!("  GPU: ");
    for x in (26..95).step_by(6) {
        print!("{:>6}", gpu[(50 * 120 + x) * 4]);
    }
    println!();

    // 统计
    let mut diffs = 0;
    let mut maxd = 0i32;
    let mut hist = std::collections::BTreeMap::new();
    for y in 0..100 {
        for x in 0..120 {
            let o = (y * 120 + x) * 4;
            let d = (gpu[o] as i32 - cpu[o] as i32).abs();
            if d > 0 {
                diffs += 1;
                *hist.entry(d / 10 * 10).or_insert(0) += 1;
            }
            maxd = maxd.max(d);
        }
    }
    println!("\nR 通道：不同像素 {diffs}/12000，最大差 {maxd}");
    println!("差值分桶（每 10）: {hist:?}");
}
