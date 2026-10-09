//! 虚线（0.10.2）的几何单测 + 演示图。
use amiki_math::Vec2;
use amiki_scene::{dash_polyline, stroke_path_dashed, LineCap, LineJoin, PathBuilder, StrokeStyle};

fn style(w: f32) -> StrokeStyle {
    StrokeStyle { width: w, cap: LineCap::Butt, join: LineJoin::Bevel }
}

/// 一条 100 长的水平线：[10,5] 应切出 7 段，起点间隔 15，每段长 10。
#[test]
fn dash_on_straight_line() {
    let pts = [Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0)];
    let d = dash_polyline(&pts, false, &[10.0, 5.0]).expect("合法模式");
    assert_eq!(d.len(), 7, "100 / 15 = 6.67 ⇒ 7 段实线");
    for (i, seg) in d.iter().enumerate() {
        assert_eq!(seg.len(), 2, "直线上的虚线每段两个点");
        assert!((seg[0].x - i as f32 * 15.0).abs() < 1e-3, "第 {i} 段起点 x");
        assert!((seg[1].x - (i as f32 * 15.0 + 10.0)).abs() < 1e-3, "第 {i} 段终点 x");
    }
    eprintln!("直线 [10,5]：{} 段，首段 {:.1}→{:.1}，末段 {:.1}→{:.1}",
        d.len(), d[0][0].x, d[0][1].x, d[6][0].x, d[6][1].x);
}

/// 奇数长度模式按 SVG/Skia 语义**重复一遍**。
#[test]
fn odd_pattern_is_doubled() {
    let pts = [Vec2::new(0.0, 0.0), Vec2::new(40.0, 0.0)];
    let a = dash_polyline(&pts, false, &[10.0]).expect("合法");
    let b = dash_polyline(&pts, false, &[10.0, 10.0]).expect("合法");
    assert_eq!(a.len(), b.len(), "[10] 应等价 [10,10]");
    assert_eq!(a.len(), 2, "40 / 20 = 2 段");
    eprintln!("奇数模式 [10] 等价 [10,10]：{} 段", a.len());
}

/// 闭合路径沿周长切。
#[test]
fn dash_wraps_around_closed_path() {
    let sq = vec![
        Vec2::new(0.0, 0.0),
        Vec2::new(10.0, 0.0),
        Vec2::new(10.0, 10.0),
        Vec2::new(0.0, 10.0),
    ];
    let d = dash_polyline(&sq, true, &[4.0, 4.0]).expect("合法");
    assert_eq!(d.len(), 5, "周长 40 / 每轮 8 = 5 段实线");
    eprintln!("闭合正方形 [4,4]：{} 段（周长 40）", d.len());
}

/// 非法模式**显式拒绝**，而不是静默按实线画。
#[test]
fn invalid_pattern_is_rejected() {
    let pts = [Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
    for bad in [vec![], vec![0.0, 0.0], vec![5.0, -1.0], vec![f32::NAN, 2.0], vec![f32::INFINITY]] {
        assert!(dash_polyline(&pts, false, &bad).is_none(), "应拒绝 {bad:?}");
    }
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 10.0, 10.0);
    assert!(stroke_path_dashed(&pb.build(), &style(2.0), &[], 0.05).is_none());
    eprintln!("非法模式（空/全 0/负数/NaN/Inf）全部显式返回 None");
}

/// 虚线描边确实"断"了：轮廓数量明显增多。
#[test]
fn dashed_stroke_makes_more_contours() {
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 100.0, 60.0);
    let path = pb.build();
    let solid = amiki_scene::stroke_path(&path, &style(2.0), 0.05);
    let dashed = stroke_path_dashed(&path, &style(2.0), &[8.0, 6.0], 0.05).expect("合法");
    assert!(dashed.els.len() > solid.els.len() * 3, "虚线轮廓应远多于实线");
    eprintln!("描边轮廓元素：实线 {}，虚线 {}（[8,6]）", solid.els.len(), dashed.els.len());
}

/// 出演示图（out/dash_demo.png）：实线 / 虚线两行对照。
#[test]
fn dash_demo_image() {
    use amiki::Canvas;
    use amiki_scene::{Color, Paint};
    let mut c = Canvas::new(300, 120);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 300.0, 120.0);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(252, 252, 254)));

    let mut pb = PathBuilder::new();
    pb.rect(20.0, 20.0, 280.0, 60.0);
    let r = pb.build();
    c.stroke_path(&r, style(3.0), Color::rgb8(30, 60, 140));
    c.stroke_path_dashed(&r, style(3.0), Color::rgb8(200, 70, 40), &[14.0, 8.0]);
    // 再画一条竖直虚线做对照
    let mut pb = PathBuilder::new();
    pb.move_to(Vec2::new(150.0, 70.0));
    pb.line_to(Vec2::new(150.0, 110.0));
    let v = pb.build();
    c.stroke_path_dashed(&v, style(4.0), Color::rgb8(40, 140, 80), &[6.0, 6.0]);

    let img = amiki::render_cpu(&c, 0.05);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/dash_demo.png");
    if let Some(px) = image::RgbaImage::from_raw(300, 120, img) {
        let _ = std::fs::create_dir_all(out.parent().unwrap());
        px.save(&out).expect("写演示图");
        eprintln!("演示图 -> {}", out.display());
    }
}
