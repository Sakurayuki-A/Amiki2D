// 这个测试用 GPU 后端做对拍，所以纯 CPU 构建（`--no-default-features`）下
// 整个文件编译为空 —— 没有 `gpu` feature 时 `amiki::gpu` 根本不存在。
#![cfg(feature = "gpu")]

//! 门面描边 API 测试（P0-1）。
//!
//! 这里验证的是**整条链路**：`Canvas::stroke_path` → 用户空间轮廓 →
//! 画布变换**恰好作用一次** → 光栅化。几何的精确面积已在 `amiki-scene`
//! 内验证，渲染链路已在 `amiki-raster` 内验证；本文件专测**变换的施加次数**。
//!
//! 这一点非测不可：`stroke_path` 不接收变换（刻意如此），变换只在画布推入时
//! 发生。若哪天有人把变换加回 `stroke_path`，轮廓会被变换两次 —— 而"画出来
//! 看着还行"会让这种错误活很久。用面积即可精确判定。

use amiki::math::{Affine, Vec2};
use amiki::scene::{Color, LineCap, LineJoin, Paint, PathBuilder, StrokeStyle};

fn style(width: f32) -> StrokeStyle {
    StrokeStyle { width, cap: LineCap::Butt, join: LineJoin::Bevel }
}

/// 渲染一张画布并返回**覆盖率之和**（总面积）。
///
/// 画法：白底 + 不透明黑描边 → 覆盖率 = `1 − R/255`，避免碰 alpha 合成。
fn total_coverage(canvas: &amiki::Canvas) -> f32 {
    let img = amiki::raster::render(&amiki::raster::bin(&canvas.scene, 0.05));
    let (w, h) = (canvas.scene.width as usize, canvas.scene.height as usize);
    (0..w * h).map(|i| 1.0 - img[i * 4] as f32 / 255.0).sum()
}

/// 造一张"白底 + 一条水平粗线"的画布。
///
/// 画布取 260×200，线放在用户空间 y=40、x∈[20,80]：
/// 1 倍时线在 y=40，3 倍时在 y=120（±半线宽都在画布内），
/// 这样"面积随尺度平方变化"才能干净地测出来。
fn canvas_with_line(scale: f32, width: f32) -> amiki::Canvas {
    let mut c = amiki::Canvas::new(260, 200);
    c.rect(0.0, 0.0, 260.0, 200.0, Paint::Solid(Color::rgb8(255, 255, 255)));

    let mut pb = PathBuilder::new();
    pb.move_to(Vec2::new(20.0, 40.0));
    pb.line_to(Vec2::new(80.0, 40.0));
    c.set_transform(Affine::scale(scale));
    c.stroke_path_with(
        &pb.build(),
        style(width),
        Paint::Solid(Color::rgb8(0, 0, 0)),
    );
    c.set_transform(Affine::IDENTITY);
    c
}

/// **变换恰好作用一次**：线宽随画布缩放一起放大。
///
/// 用户空间里是 `长 60 × 宽 4 = 240`。
/// - 1 倍缩放 → 240
/// - 3 倍缩放 → 线长与线宽各 3 倍 → **2160**（9 倍）
///
/// 若变换被施加两次，会得到 3 倍（长 3 倍、宽仍 1 倍于用户空间再乘 3）
/// 之类的错值 —— 面积判定足以区分。
#[test]
fn canvas_transform_applies_exactly_once() {
    let w = 4.0f32;
    let base = total_coverage(&canvas_with_line(1.0, w));
    assert!(
        (base - 240.0).abs() < 6.0,
        "1 倍缩放下面积应约为 60×4=240，实际 {base}"
    );

    let s3 = total_coverage(&canvas_with_line(3.0, w));
    assert!(
        (s3 - 2160.0).abs() < 60.0,
        "3 倍缩放下面积应约为 240×9=2160（线宽随之放大），实际 {s3}"
    );
}

/// 线宽确实是**用户单位**：同一变换下，线宽加倍则面积加倍。
#[test]
fn stroke_width_is_user_units() {
    let a = total_coverage(&canvas_with_line(2.0, 3.0));
    let b = total_coverage(&canvas_with_line(2.0, 6.0));
    assert!(
        (b - a * 2.0).abs() < a * 0.05,
        "线宽加倍应使面积加倍：宽 3 → {a}，宽 6 → {b}"
    );
}

/// 描边产出的是**普通填充路径** —— 场景里不应出现新的绘制种类，
/// 且后端无需任何改动即可渲染。
#[test]
fn stroke_emits_an_ordinary_fill_draw() {
    let mut c = amiki::Canvas::new(100, 100);
    let before = c.scene.draws.len();
    let mut pb = PathBuilder::new();
    pb.move_to(Vec2::new(10.0, 10.0));
    pb.line_to(Vec2::new(90.0, 10.0));
    c.stroke_path(&pb.build(), style(5.0), Color::rgb8(0, 0, 0));
    assert_eq!(c.scene.draws.len() - before, 1, "描边应只产生一条 draw");
    assert_eq!(c.scene.layers.len(), 0, "描边不该引入层");
}

/// **GPU 与 CPU 对拍**：描边轮廓在 GPU 上必须与 CPU 一致。
///
/// 因为描边在提交前已退化成普通的多边形填充，这条对拍实际上是
/// "多边形填充的双端一致性"在描边场景下的复核。
#[test]
fn gpu_stroke_matches_cpu() {
    let Ok(renderer) = pollster::block_on(amiki::gpu::GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };

    let mut c = amiki::Canvas::new(160, 120);
    c.rect(0.0, 0.0, 160.0, 120.0, Paint::Solid(Color::rgb8(246, 243, 238)));

    // 开放折线（三种 join）
    for (i, join) in [
        LineJoin::Bevel,
        LineJoin::Round,
        LineJoin::Miter { limit: 4.0 },
    ]
    .into_iter()
    .enumerate()
    {
        let mut pb = PathBuilder::new();
        let y = 24.0 + i as f32 * 34.0;
        pb.move_to(Vec2::new(16.0, y));
        pb.line_to(Vec2::new(80.0, y + 16.0));
        pb.line_to(Vec2::new(144.0, y));
        c.stroke_path_with(
            &pb.build(),
            StrokeStyle { width: 9.0, cap: LineCap::Round, join },
            Paint::Solid(Color::rgba8(200, 60, 50, 220)),
        );
    }
    // 闭合环
    let mut pb = PathBuilder::new();
    pb.move_to(Vec2::new(40.0, 40.0));
    pb.line_to(Vec2::new(120.0, 40.0));
    pb.line_to(Vec2::new(120.0, 100.0));
    pb.line_to(Vec2::new(40.0, 100.0));
    pb.close();
    c.stroke_path(&pb.build(), style(10.0), Color::rgba8(40, 90, 160, 200));

    let gpu = renderer.render_blocking(&c.scene, 0.05).expect("GPU 渲染失败");
    let cpu = amiki::raster::render(&amiki::raster::bin(&c.scene, 0.05));

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
        "描边场景 GPU 与 CPU 最大通道差 {max_diff}（位于 {worst:?}）超出容差"
    );
    eprintln!("描边 GPU 与 CPU 一致（最大通道差 {max_diff}）");
}
