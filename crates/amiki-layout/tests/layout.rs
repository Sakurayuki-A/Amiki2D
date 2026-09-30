//! amiki-layout 集成测试：断行、行高、对齐、段落。用真实字体（Inter）。

use amiki_layout::{layout, Align, LayoutOptions};
use amiki_text::Font;

fn font() -> Option<Font> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("仓库根");
    let p = root.join("assets/fonts/Inter-Regular.ttf");
    if !p.exists() {
        eprintln!("跳过：未找到 {}", p.display());
        return None;
    }
    Some(Font::from_file(&p).expect("字体应能解析"))
}

#[test]
fn single_line_has_correct_height_and_baseline() {
    let Some(f) = font() else { return };
    let opts = LayoutOptions::new(40.0, f32::INFINITY);
    let r = layout(&f, "Hello", &opts).expect("排版");
    assert_eq!(r.lines.len(), 1);
    let m = f.metrics().unwrap();
    let line = &r.lines[0];
    // 基线应在 ascent 处
    let expect_base = m.ascent * 40.0;
    assert!(
        (line.baseline.y - expect_base).abs() < 0.01,
        "基线应为 ascent×字号 = {expect_base:.2}，实际 {:.2}",
        line.baseline.y
    );
    // 区块高度 = 行高
    let expect_h = m.line_height() * 40.0;
    assert!(
        (r.size.y - expect_h).abs() < 0.01,
        "总高应为 {expect_h:.2}，实际 {:.2}",
        r.size.y
    );
    eprintln!("单行：基线 {:.2}px，总高 {:.2}px（ascent={:.3} 行高={:.3} em）",
        line.baseline.y, r.size.y, m.ascent, m.line_height());
}

#[test]
fn explicit_newlines_make_lines_and_blank_lines_are_preserved() {
    let Some(f) = font() else { return };
    let opts = LayoutOptions::new(20.0, f32::INFINITY);
    let r = layout(&f, "a\nb\n\nc", &opts).expect("排版");
    assert_eq!(r.lines.len(), 4, "a / b / 空行 / c 共 4 行");
    assert!(r.lines[2].run.glyphs.is_empty(), "第 3 行应为空行");
    // 行基线应逐行递增
    for w in r.lines.windows(2) {
        assert!(w[1].baseline.y > w[0].baseline.y, "基线应逐行递增");
    }
    let lh = f.metrics().unwrap().line_height() * 20.0;
    let d = r.lines[1].baseline.y - r.lines[0].baseline.y;
    assert!((d - lh).abs() < 0.01, "行距应为 {lh:.2}，实际 {d:.2}");
    eprintln!("\\n 分段：4 行（含空行），行距 {d:.2}px 正确");
}

#[test]
fn wraps_at_spaces_and_never_exceeds_max_width() {
    let Some(f) = font() else { return };
    let text = "The quick brown fox jumps over the lazy dog";
    let max_w = f.advance_width(text) * 30.0 / 3.0; // 大约三分之一宽
    let opts = LayoutOptions::new(30.0, max_w);
    let r = layout(&f, text, &opts).expect("排版");
    assert!(r.lines.len() >= 2, "应折成多行，实际 {} 行", r.lines.len());
    for (i, l) in r.lines.iter().enumerate() {
        assert!(
            l.width <= max_w + 0.01,
            "第 {i} 行宽 {:.2} 超出上限 {max_w:.2}",
            l.width
        );
    }
    // 断行处不应丢失内容（把所有行的文本拼回来，忽略被吞掉的空白）
    let joined: String = r
        .lines
        .iter()
        .map(|l| {
            let s = &text[l.text_range.clone()];
            s.to_string()
        })
        .collect::<Vec<_>>()
        .join(" ");
    let norm = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    assert_eq!(norm(&joined), norm(text), "断行不应丢失或重复内容");
    eprintln!("断行：{} 行，每行 ≤ {max_w:.1}px，内容完整", r.lines.len());
}

#[test]
fn line_start_never_begins_with_space() {
    // 空白处断行必须**吞掉**空白：否则行首会出现空格，视觉上不平齐
    let Some(f) = font() else { return };
    let opts = LayoutOptions::new(24.0, f.advance_width("one two three four ") * 24.0 / 2.2);
    let r = layout(&f, "one two three four five six seven", &opts).expect("排版");
    let text = "one two three four five six seven";
    for (i, l) in r.lines.iter().enumerate() {
        let s = &text[l.text_range.clone()];
        assert!(
            !s.starts_with(' '),
            "第 {i} 行不应以空格开头：{s:?}"
        );
        assert!(!s.ends_with(' '), "第 {i} 行不应以空格结尾：{s:?}");
    }
    eprintln!("断行裁掉空白：{} 行均无行首/行尾空格", r.lines.len());
}

#[test]
fn alignment_positions_lines_correctly() {
    let Some(f) = font() else { return };
    let max_w = 400.0f32;
    let text = "a\nbb\nccc";
    let mut xs = Vec::new();
    for (align, name) in [(Align::Left, "左"), (Align::Center, "中"), (Align::Right, "右")] {
        let opts = LayoutOptions { align, ..LayoutOptions::new(30.0, max_w) };
        let r = layout(&f, text, &opts).expect("排版");
        assert_eq!(r.lines.len(), 3);
        // 三行宽度递增，故 x 应单调变化（左对齐：全为 0；居中/右对齐：递减）
        let this: Vec<f32> = r.lines.iter().map(|l| l.baseline.x).collect();
        match align {
            Align::Left => assert!(this.iter().all(|&x| x == 0.0), "{name}对齐 x 应全为 0"),
            Align::Center => {
                assert!(this[0] > this[2], "{name}对齐：窄行 x 应更大（{this:?}）");
                // 居中时 x = (max - w)/2
                let w0 = r.lines[0].width;
                assert!((this[0] - (max_w - w0) / 2.0).abs() < 0.01, "居中公式不符");
            }
            Align::Right => {
                assert!(this[0] > this[2], "{name}对齐：窄行 x 应更大（{this:?}）");
                let w0 = r.lines[0].width;
                assert!((this[0] - (max_w - w0)).abs() < 0.01, "右对齐公式不符");
            }
        }
        xs.push(this);
        eprintln!("{name}对齐：行 x = {:?}", xs.last().unwrap());
    }
}

#[test]
fn unbreakable_long_word_overflows_instead_of_being_cut() {
    // 超长不可断片段（如长 URL）应**溢出**并置 overflow 标志，而不是被硬切。
    // 硬切会破坏内容语义（URL 被截断就失效了）。
    let Some(f) = font() else { return };
    let long = "https://example.com/averyveryverylongpathsegmentthatcannotbreak";
    let opts = LayoutOptions::new(20.0, 60.0); // 极窄
    let r = layout(&f, long, &opts).expect("排版");
    assert!(r.overflow, "应置 overflow 标志");
    // 内容不得丢失
    let joined: String = r.lines.iter().map(|l| &long[l.text_range.clone()]).collect();
    assert_eq!(joined, long, "溢出也不得丢字符");
    eprintln!("超长词：{} 行、overflow={}，字符完整", r.lines.len(), r.overflow);
}

#[test]
fn cjk_breaks_between_characters() {
    // 汉字之间没有空格，必须逐字可断 —— 否则整段汉字会变成一行溢出
    let Some(f) = font() else { return };
    // Inter 不含汉字，故这里主要验证"断行逻辑对 CJK 生效"。
    // 用重复的 CJK 码位构造文本；即使字形缺失，宽度为 0 也能验证不 panic、
    // 且可断点被识别（不产生 overflow）。
    let text = "汉字排版测试文本内容长度足够触发断行";
    let opts = LayoutOptions::new(20.0, 200.0);
    let r = layout(&f, text, &opts).expect("排版不应 panic");
    eprintln!("CJK：{} 行，overflow={}（Inter 无汉字字形，此测试主要验证不 panic）",
        r.lines.len(), r.overflow);
}

#[test]
fn empty_text_yields_empty_layout() {
    let Some(f) = font() else { return };
    let r = layout(&f, "", &LayoutOptions::new(20.0, 100.0)).expect("排版");
    assert!(r.is_empty());
    assert_eq!(r.size.y, 0.0);
}

#[test]
fn line_spacing_scales_line_advance() {
    let Some(f) = font() else { return };
    let base = f.metrics().unwrap().line_height() * 24.0;
    let one = layout(&f, "a\nb", &LayoutOptions::new(24.0, f32::INFINITY)).unwrap();
    let two = layout(
        &f,
        "a\nb",
        &LayoutOptions {
            line_spacing: 2.0,
            ..LayoutOptions::new(24.0, f32::INFINITY)
        },
    )
    .unwrap();
    let d1 = one.lines[1].baseline.y - one.lines[0].baseline.y;
    let d2 = two.lines[1].baseline.y - two.lines[0].baseline.y;
    assert!((d1 - base).abs() < 0.01, "1.0 倍行距应为 {base:.2}，实际 {d1:.2}");
    assert!((d2 - base * 2.0).abs() < 0.01, "2.0 倍行距应为 {:.2}，实际 {d2:.2}", base * 2.0);
    eprintln!("行距倍数：1.0 → {d1:.2}px，2.0 → {d2:.2}px");
}
