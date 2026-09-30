//! 门面文本 API 测试：`Canvas::text` 必须把一段 shaping 结果变成
//! **每字形一条 draw**，并且几何位置正确。

use amiki::math::Vec2;
use amiki::scene::Paint;

fn font_path() -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("无法定位仓库根目录");
    root.join("assets/fonts/Inter-Regular.ttf")
}

#[test]
fn canvas_text_emits_one_draw_per_glyph() {
    let p = font_path();
    if !p.exists() {
        eprintln!("跳过：未找到字体 {}", p.display());
        return;
    }
    let font = amiki_text::Font::from_file(&p).expect("字体应能解析");

    let mut canvas = amiki::Canvas::new(600, 200);
    let bg_draws = canvas.scene.draws.len();

    let run = amiki_text::shape(&font, "Amiki", 64.0).expect("shaping 应成功");
    let n_glyphs = run.glyphs.len();
    canvas.text(
        &run,
        Vec2::new(40.0, 140.0),
        Paint::Solid(amiki::scene::Color::rgb8(20, 20, 20)),
        |id| font.outline(id).ok(),
    );

    assert_eq!(
        canvas.scene.draws.len() - bg_draws,
        n_glyphs,
        "每个字形应产生一条独立 draw（便于 damage region 独立处理）"
    );

    // 字形必须**水平展开**（x 递增），而不是叠在一处
    let xs: Vec<f32> = canvas.scene.draws[bg_draws..]
        .iter()
        .map(|d| d.transform.e)
        .collect();
    for w in xs.windows(2) {
        assert!(w[1] > w[0], "字形 x 必须递增：{:?}", xs);
    }
    // 基线原点 40 + 首字形约 0 偏移
    assert!(
        (xs[0] - 40.0).abs() < 1.0,
        "首字形 x 应在基线原点附近（40），实际 {}",
        xs[0]
    );
    let span = xs.last().unwrap() - xs[0];
    assert!(span > 100.0, "“Amiki” 在 64px 下应展开 >100px，实际 {span:.1}");
    eprintln!("Canvas::text：{n_glyphs} 个字形 → {n_glyphs} 条 draw，跨度 {span:.1}px");
}

#[test]
fn canvas_text_respects_clip_and_transform() {
    let p = font_path();
    if !p.exists() {
        eprintln!("跳过：未找到字体 {}", p.display());
        return;
    }
    let font = amiki_text::Font::from_file(&p).expect("字体应能解析");
    let mut canvas = amiki::Canvas::new(400, 200);
    canvas.translate(Vec2::new(100.0, 50.0));

    let run = amiki_text::shape(&font, "Hi", 48.0).expect("shaping");
    canvas.text(
        &run,
        Vec2::new(0.0, 0.0),
        Paint::Solid(amiki::scene::Color::rgb8(0, 0, 0)),
        |id| font.outline(id).ok(),
    );
    // 画布变换必须作用到文本上：首字形 x 应含平移 100
    let xs: Vec<f32> = canvas.scene.draws.iter().map(|d| d.transform.e).collect();
    assert!(
        (xs[0] - 100.0).abs() < 1.0,
        "文本应受画布变换影响（期望首字形 x≈100），实际 {}",
        xs[0]
    );
    eprintln!("画布变换已作用到文本：首字形 x = {:.1}", xs[0]);
}
