//! amiki-text 集成测试：用真实字体（Inter）验证解析与 shaping。
//!
//! 字体位于仓库 `assets/fonts/Inter-Regular.ttf`（SIL OFL 1.1）。
//! 若该文件不存在，测试跳过并明确提示——不静默通过。

use amiki_text::{shape, Font};

fn font_path() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/amiki-text
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("无法定位仓库根目录");
    root.join("assets/fonts/Inter-Regular.ttf")
}

fn load() -> Option<Font> {
    let p = font_path();
    if !p.exists() {
        eprintln!("跳过：未找到字体文件 {}", p.display());
        return None;
    }
    Some(Font::from_file(&p).expect("字体应能解析"))
}

#[test]
fn parses_real_font_metadata() {
    let Some(font) = load() else { return };
    // Inter 的 upem 是 2816（这是该字体的真实属性，可作健全性检查）
    assert!(
        (2000.0..=4096.0).contains(&font.units_per_em()),
        "units_per_em 应在合理范围，实际 {}",
        font.units_per_em()
    );
    assert!(font.glyph_count() > 500, "Inter 字形数应可观，实际 {}", font.glyph_count());
    eprintln!(
        "Inter 解析成功：units_per_em = {}, 字形数 = {}",
        font.units_per_em(),
        font.glyph_count()
    );
}

#[test]
fn extracts_outline_in_em_units() {
    let Some(font) = load() else { return };
    let gid = font.glyph_index('H').expect("'H' 应有字形");
    let o = font.outline(gid).expect("应能提取 'H' 的轮廓");
    assert!(!o.els.is_empty(), "'H' 的轮廓不应为空");

    // 轮廓必须归一化到 em：所有坐标都应落在大致 [-1, 1]（含少量越界如上下伸部）
    let mut max_abs = 0.0f32;
    for el in &o.els {
        let pts: Vec<_> = match el {
            amiki_scene::PathEl::MoveTo(p) | amiki_scene::PathEl::LineTo(p) => vec![*p],
            amiki_scene::PathEl::QuadTo(a, b) => vec![*a, *b],
            amiki_scene::PathEl::CubicTo(a, b, c) => vec![*a, *b, *c],
            amiki_scene::PathEl::Close => vec![],
        };
        for p in pts {
            max_abs = max_abs.max(p.x.abs()).max(p.y.abs());
        }
    }
    assert!(
        max_abs < 2.0,
        "轮廓应已归一化为 em 单位（|坐标| < 2），实际最大 {max_abs}"
    );
    eprintln!("'H' 轮廓：{} 个元素，最大 |坐标| = {:.3} em", o.els.len(), max_abs);
}

#[test]
fn outline_is_flipped_to_screen_orientation() {
    let Some(font) = load() else { return };
    let gid = font.glyph_index('H').expect("'H' 应有字形");
    let o = font.outline(gid).expect("轮廓");
    // 字体坐标 y 向上；引擎用屏幕朝向（y 向下）。
    // 'H' 的轮廓在字体坐标里 y ∈ [0, capHeight]，翻转后应变为 y ∈ [-capHeight, 0]。
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for el in &o.els {
        let pts: Vec<_> = match el {
            amiki_scene::PathEl::MoveTo(p) | amiki_scene::PathEl::LineTo(p) => vec![*p],
            amiki_scene::PathEl::QuadTo(a, b) => vec![*a, *b],
            amiki_scene::PathEl::CubicTo(a, b, c) => vec![*a, *b, *c],
            amiki_scene::PathEl::Close => vec![],
        };
        for p in pts {
            lo = lo.min(p.y);
            hi = hi.max(p.y);
        }
    }
    assert!(
        hi <= 0.01 && lo < -0.4,
        "基线以上应映射为负 y（屏幕朝向）：实际 y ∈ [{lo:.3}, {hi:.3}]"
    );
    eprintln!("capHeight 区域：y ∈ [{lo:.3}, {hi:.3}] em（基线以上为负，符合屏幕朝向）");
}

#[test]
fn shaping_advances_do_not_overlap() {
    let Some(font) = load() else { return };
    let run = shape(&font, "Hello", 40.0).expect("shaping 应成功");
    assert_eq!(run.glyphs.len(), 5, "'Hello' 应有 5 个字形");

    // 位置必须**严格递增**——这正是"必须累加 advance"的检验：
    // 若忽略 advance，所有字形会叠在同一 x 上。
    for w in run.glyphs.windows(2) {
        assert!(
            w[1].x > w[0].x,
            "字形 x 必须递增（advance 已累加）：{:?} → {:?}",
            w[0],
            w[1]
        );
    }
    let total = run.glyphs.last().unwrap().x;
    assert!(total > 0.5, "'Hello' 总前进量应 > 0.5 em，实际 {total:.3}");
    eprintln!(
        "shaping 'Hello'：{} 个字形，末字形 x = {:.3} em",
        run.glyphs.len(),
        total
    );
}

#[test]
fn missing_char_reports_error_not_panic() {    let Some(font) = load() else { return };
    // 一个 Inter 几乎肯定没有的码位
    let gid = font.glyph_index('\u{10FFFD}');
    if let Some(g) = gid {
        // 若字体给了 .notdef，轮廓仍应能取到（不应 panic）
        let _ = font.outline(g);
    }
    // 越界字形 id 必须返回 Err 而不是 panic
    let bad = font.glyph_count() + 1000;
    assert!(font.outline(bad).is_err(), "越界字形 id 应返回 Err");
}

#[test]
fn metrics_are_sane_and_consistent() {
    let Some(font) = load() else { return };
    let m = font.metrics().expect("应能读取度量");

    // Inter 的真实量级：ascent ≈ 0.97、descent ≈ 0.24（em 单位）
    assert!(m.ascent > 0.5 && m.ascent < 1.5, "ascent 应在合理范围，实际 {}", m.ascent);
    assert!(m.descent > 0.0 && m.descent < 1.0, "descent 应为正且合理，实际 {}", m.descent);
    // 行高必须大于 ascent（否则行与行会重叠）
    assert!(
        m.line_height() > m.ascent,
        "行高 {} 应大于 ascent {}",
        m.line_height(),
        m.ascent
    );
    // cap height 应大于 x height（这是字体设计的硬约束）
    assert!(
        m.cap_height > m.x_height,
        "cap height {} 应大于 x height {}",
        m.cap_height,
        m.x_height
    );
    // x height 应小于 ascent
    assert!(m.x_height < m.ascent, "x height 应小于 ascent");
    eprintln!(
        "度量：ascent={:.3} descent={:.3} line_gap={:.3} 行高={:.3} x={:.3} cap={:.3}（em）",
        m.ascent,
        m.descent,
        m.line_gap,
        m.line_height(),
        m.x_height,
        m.cap_height
    );
}

#[test]
fn advance_width_is_consistent_with_shaping() {
    let Some(font) = load() else { return };
    let text = "Hello";
    let size = 48.0f32;

    // 快速测量（不做 shaping，无 kerning）
    let approx = font.advance_width(text) * size;
    // 精确宽度：shaping 后末字形位置 + 其前进量
    let run = shape(&font, text, size).expect("shaping");
    let last = run.glyphs.last().unwrap();
    let last_adv = font.advance(last.id).unwrap() * size;
    let exact = last.x * size + last_adv;

    assert!(approx > 0.0 && exact > 0.0);
    // 无 kerning 的近似与精确值应接近（"Hello" 里没有明显负 kerning）
    let rel = (approx - exact).abs() / exact;
    assert!(
        rel < 0.05,
        "快速测量 {approx:.2} 与 shaping 精确宽 {exact:.2} 相差 {:.1}% 过大",
        rel * 100.0
    );
    eprintln!("宽度：快速测量 {approx:.2}px，shaping 精确 {exact:.2}px（相对差 {:.2}%）", rel * 100.0);
}

#[test]
fn advance_of_each_glyph_matches_shaping_positions() {
    // 逐字形校验：若 shaping 的游标累加正确，则相邻字形位置差
    // 应等于前一个字形的 advance（无 kerning 时严格成立）。
    let Some(font) = load() else { return };
    let run = shape(&font, "lllll", 40.0).expect("shaping");
    assert_eq!(run.glyphs.len(), 5);
    // 'l' 重复五次，无 kerning，故位置差应恒等于 advance
    for w in run.glyphs.windows(2) {
        let adv = font.advance(w[0].id).unwrap();
        let diff = w[1].x - w[0].x;
        assert!(
            (diff - adv).abs() < 1e-5,
            "位置差 {diff:.6} 应等于 advance {adv:.6}"
        );
    }
    eprintln!("逐字形 advance 与 shaping 位置严格一致（5 个 'l'）");
}
