//! M6：渐变测试 —— 几何参数、解析采样、编码往返、渲染可见性。
//!
//! 渐变的"正确"有两条独立判据，都要满足：
//! 1. **几何**：`param_at` 必须给出正确的投影/距离/角度参数；
//! 2. **采样**：色标之间的插值必须解析精确（不是查表近似）。

use amiki_math::Vec2;
use amiki_raster::{bin, render};
use amiki_scene::{
    decode_scene, encode_scene, expand_text_run, Color, Gradient, GradientStop, Paint,
    PathBuilder, Scene,
};

fn two_stop() -> Vec<GradientStop> {
    vec![
        GradientStop::new(0.0, Color::rgb8(0, 0, 0)),
        GradientStop::new(1.0, Color::rgb8(255, 255, 255)),
    ]
}

#[test]
fn linear_param_is_projection() {
    let g = Gradient::linear(Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0), two_stop()).unwrap();
    assert!((g.param_at(Vec2::new(0.0, 0.0)) - 0.0).abs() < 1e-5);
    assert!((g.param_at(Vec2::new(50.0, 0.0)) - 0.5).abs() < 1e-5);
    assert!((g.param_at(Vec2::new(100.0, 0.0)) - 1.0).abs() < 1e-5);
    // 垂直于轴向的位置不影响参数（投影语义）
    assert!((g.param_at(Vec2::new(50.0, 999.0)) - 0.5).abs() < 1e-5);
    // 超出端点（默认 clamp 由 sample 处理，param 本身不钳制）
    assert!((g.param_at(Vec2::new(150.0, 0.0)) - 1.5).abs() < 1e-5);
}

#[test]
fn radial_and_sweep_params() {
    let r = Gradient::radial(Vec2::new(10.0, 20.0), 50.0, two_stop()).unwrap();
    assert!((r.param_at(Vec2::new(10.0, 20.0)) - 0.0).abs() < 1e-5);
    assert!((r.param_at(Vec2::new(60.0, 20.0)) - 1.0).abs() < 1e-5);
    assert!((r.param_at(Vec2::new(10.0, 70.0)) - 1.0).abs() < 1e-5);

    let s = Gradient::sweep(Vec2::ZERO, 0.0, two_stop()).unwrap();
    assert!((s.param_at(Vec2::new(1.0, 0.0)) - 0.0).abs() < 1e-4);
    // 正上方（屏幕 y 向下 ⇒ 角度 -PI/2）应归一到 0.75
    assert!((s.param_at(Vec2::new(0.0, -1.0)) - 0.75).abs() < 1e-3);
}

#[test]
fn sample_interpolates_exactly_between_stops() {
    let g = Gradient::linear(
        Vec2::ZERO,
        Vec2::new(1.0, 0.0),
        vec![
            GradientStop::new(0.0, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
            GradientStop::new(0.5, Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }),
            GradientStop::new(1.0, Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
        ],
    )
    .unwrap();

    // 精确中点：0.25 处灰→红的 1/2 = (0.5, 0, 0)
    let c = g.sample(0.25);
    assert!((c.r - 0.5).abs() < 1e-6, "r 应为 0.5，实际 {}", c.r);
    assert!(c.g.abs() < 1e-6);

    // 0.75 处红→白的 1/2 = (1, 0.5, 0.5)
    let c = g.sample(0.75);
    assert!((c.r - 1.0).abs() < 1e-6);
    assert!((c.g - 0.5).abs() < 1e-6, "g 应为 0.5，实际 {}", c.g);

    // 端点外默认 clamp
    assert!(g.sample(-5.0).r.abs() < 1e-6);
    assert!((g.sample(5.0).g - 1.0).abs() < 1e-6);
}

#[test]
fn sample_repeat_wraps() {
    let mut g = Gradient::linear(
        Vec2::ZERO,
        Vec2::new(1.0, 0.0),
        vec![
            GradientStop::new(0.0, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
            GradientStop::new(1.0, Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
        ],
    )
    .unwrap();
    g.repeat = true;
    // 1.25 应等于 0.25
    let a = g.sample(1.25);
    let b = g.sample(0.25);
    assert!((a.r - b.r).abs() < 1e-6, "repeat 模式应取模：{} vs {}", a.r, b.r);
    assert!((a.r - 0.25).abs() < 1e-6);
}

#[test]
fn unsorted_stops_are_sorted_at_construction() {
    let g = Gradient::linear(
        Vec2::ZERO,
        Vec2::new(1.0, 0.0),
        vec![
            GradientStop::new(1.0, Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
            GradientStop::new(0.0, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
        ],
    )
    .unwrap();
    assert!(g.stops[0].t <= g.stops[1].t, "构造时应按 t 排序");
    // 排序后采样仍应正确
    assert!(g.sample(0.0).r.abs() < 1e-6);
    assert!((g.sample(1.0).r - 1.0).abs() < 1e-6);
}

#[test]
fn gradient_rejects_degenerate_input() {
    assert!(Gradient::linear(Vec2::ZERO, Vec2::new(1.0, 0.0), vec![]).is_none());
    assert!(
        Gradient::linear(Vec2::ZERO, Vec2::new(1.0, 0.0), vec![GradientStop::new(
            0.0,
            Color::rgb8(0, 0, 0)
        )])
        .is_none(),
        "少于 2 个色标应被拒绝"
    );
    assert!(
        Gradient::radial(Vec2::ZERO, 0.0, two_stop()).is_none(),
        "半径 0 应被拒绝"
    );
}

#[test]
fn gradient_actually_renders_a_ramp() {
    let g = std::sync::Arc::new(
        Gradient::linear(
            Vec2::new(0.0, 0.0),
            Vec2::new(128.0, 0.0),
            vec![
                GradientStop::new(0.0, Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                GradientStop::new(1.0, Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }),
            ],
        )
        .unwrap(),
    );
    let mut scene = Scene::new(128, 32);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 128.0, 32.0);
    scene.push(amiki_math::Affine::IDENTITY, pb.build(), Paint::Gradient(g));

    let img = render(&bin(&scene, 0.05));
    let px = |x: usize, y: usize| img[(y * 128 + x) * 4];

    // 左端应接近黑、右端接近白、中间居中 —— 这就是"渐变真的生效了"
    let l = px(1, 16);
    let m = px(64, 16);
    let r = px(126, 16);
    assert!(l < 12, "左端应接近黑，实际 {l}");
    assert!(r > 243, "右端应接近白，实际 {r}");
    assert!(
        (m as i32 - 128).abs() < 12,
        "中点应接近中灰（≈128），实际 {m}"
    );
    // 单调递增（无跳变）
    let mut prev = 0u8;
    for x in 0..128 {
        let v = px(x, 16);
        assert!(v >= prev, "采样应单调不减，x={x} 处 {v} < {prev}");
        prev = v;
    }
    eprintln!("渐变渲染：左端 {l} / 中点 {m} / 右端 {r}，且逐列单调不减");
}

#[test]
fn gradient_survives_stream_roundtrip() {
    let g = std::sync::Arc::new(
        Gradient::linear(
            Vec2::new(3.0, 4.0),
            Vec2::new(90.0, 4.0),
            vec![
                GradientStop::new(0.0, Color::rgb8(10, 20, 30)),
                GradientStop::new(0.5, Color::rgb8(200, 100, 50)),
                GradientStop::new(1.0, Color::rgb8(255, 255, 255)),
            ],
        )
        .unwrap(),
    );
    let mut scene = Scene::new(100, 40);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 100.0, 40.0);
    scene.push(amiki_math::Affine::IDENTITY, pb.build(), Paint::Gradient(g));

    let back = decode_scene(&encode_scene(&scene)).expect("解码失败");
    match &back.draws[0].paint {
        Paint::Gradient(g2) => {
            assert_eq!(g2.stops.len(), 3, "色标数量应保持");
            let expect = Gradient::linear(
                Vec2::new(3.0, 4.0),
                Vec2::new(90.0, 4.0),
                vec![
                    GradientStop::new(0.0, Color::rgb8(10, 20, 30)),
                    GradientStop::new(0.5, Color::rgb8(200, 100, 50)),
                    GradientStop::new(1.0, Color::rgb8(255, 255, 255)),
                ],
            )
            .unwrap();
            assert_eq!(g2.kind, expect.kind, "几何应保持");
            for (a, b) in g2.stops.iter().zip(expect.stops.iter()) {
                assert!((a.t - b.t).abs() < 1e-6);
                assert!((a.color.r - b.color.r).abs() < 1e-6);
                assert!((a.color.g - b.color.g).abs() < 1e-6);
                assert!((a.color.b - b.color.b).abs() < 1e-6);
            }
        }
        other => panic!("往返后应为渐变，实际 {other:?}"),
    }
    eprintln!("渐变编码往返：几何与 3 个色标全部保持");
}

#[test]
fn text_can_be_painted_with_gradient() {
    // M5 + M6 交汇：文字用渐变色 —— 证明两个特性正交可组合
    use amiki_scene::{GlyphOutline, PathEl, TextRun};
    // 渐变跨度取 0..50：这样第 1 个字形（x≈0-10，t<0.2）明确偏红，
    // 第 2 个字形（x≈80-90，t>1 已 clamp）明确取到端色蓝。
    // 注意跨度不能取太大——否则第二个字形仍落在红蓝过渡区，
    // 判定"偏蓝"就不再成立。
    let g = std::sync::Arc::new(
        Gradient::linear(
            Vec2::new(0.0, 0.0),
            Vec2::new(50.0, 0.0),
            vec![
                GradientStop::new(0.0, Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }),
                GradientStop::new(1.0, Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }),
            ],
        )
        .unwrap(),
    );
    let bar = GlyphOutline::new(vec![
        PathEl::MoveTo(Vec2::new(0.0, 0.0)),
        PathEl::LineTo(Vec2::new(0.1, 0.0)),
        PathEl::LineTo(Vec2::new(0.1, 0.6)),
        PathEl::LineTo(Vec2::new(0.0, 0.6)),
        PathEl::Close,
    ]);
    let mut run = TextRun::new(100.0);
    run.push_glyph(1, 0.0, 0.0);
    run.push_glyph(1, 0.8, 0.0);

    let mut scene = Scene::new(200, 80);
    for (p, t) in expand_text_run(&run, |_| Some(bar.clone())) {
        scene.push(t, p, Paint::Gradient(g.clone()));
    }
    let img = render(&bin(&scene, 0.05));
    // 第一个字形（x∈[0,10]）应偏红，第二个（x∈[80,90]）应偏蓝
    let sample = |x: usize| {
        let o = (40 * 200 + x) * 4;
        (img[o], img[o + 2])
    };
    let (r1, b1) = sample(4);
    let (r2, b2) = sample(84);
    assert!(r1 > b1, "左字形应偏红，实际 r={r1} b={b1}");
    assert!(b2 > r2, "右字形应偏蓝，实际 r={r2} b={b2}");
    eprintln!("文字 + 渐变可组合：左字形 r={r1}/b={b1}，右字形 r={r2}/b={b2}");
}
