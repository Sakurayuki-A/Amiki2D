//! amiki-layout 集成测试：**字体回退布局**（`layout_with_stack`）。
//!
//! 用真实字体：Inter（拉丁）+ Noto Sans SC（CJK），任一缺失则跳过。

use amiki_layout::{is_no_line_end, is_no_line_start, layout, layout_with_stack, layout_with_stack_policy,
                   LayoutOptions};
use amiki_text::{shape, Font, FontStack, MissingGlyphPolicy, TextOptions};

const INTER: &str = "Inter-Regular.ttf";
const NOTO: &str = "NotoSansSC-Regular.ttf";

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("仓库根")
        .to_path_buf()
}

fn load(name: &str) -> Option<Font> {
    let p = root().join("assets/fonts").join(name);
    if !p.exists() {
        eprintln!("跳过：未找到 {}", p.display());
        return None;
    }
    Some(Font::from_file(&p).expect("字体应能解析"))
}

fn stack() -> Option<FontStack> {
    let mut s = FontStack::new();
    s.push(load(INTER)?);
    s.push(load(NOTO)?);
    Some(s)
}

/// 把一行按"解析到的字体"切成连续片段，各自单独 shaping，返回宽度**之和**。
///
/// 这是"排版宽度 = 各 run 之和"的**独立算法**：它不复用 layout 内部的前缀数组，
/// 因此能抓出"pen 跨 run 没接上"或"用了错字体的前进量"这类错误。
fn sum_of_runs(stack: &FontStack, text: &str, size: f32) -> f32 {
    let mut total = 0.0f32;
    let mut cur = String::new();
    let mut cur_font: Option<usize> = None;
    let flush = |fi: usize, s: &str, total: &mut f32| {
        if s.is_empty() {
            return;
        }
        let font = stack.get(fi).expect("字体下标应有效");
        let run = shape(font, s, 1.0).expect("shaping");
        let last = run.glyphs.last().expect("非空片段应有字形");
        *total += (last.x + font.advance(last.id).unwrap_or(0.0)) * size;
    };
    for c in text.chars() {
        let fi = stack.resolve(c).expect("测试文本里的字符都应能解析");
        if Some(fi) != cur_font {
            if let Some(f) = cur_font {
                flush(f, &cur, &mut total);
                cur.clear();
            }
            cur_font = Some(fi);
        }
        cur.push(c);
    }
    if let Some(f) = cur_font {
        flush(f, &cur, &mut total);
    }
    total
}

#[test]
fn mixed_text_lays_out_in_one_line_with_two_fonts() {
    let Some(stack) = stack() else { return };
    let opts = LayoutOptions::new(32.0, f32::INFINITY);
    let r = layout_with_stack(&stack, "Hello 世界", &opts).expect("排版");

    assert_eq!(r.lines.len(), 1, "不折行时应只有一行");
    let fonts: Vec<usize> = r.lines[0].run.glyphs.iter().map(|g| g.font_index).collect();
    assert_eq!(
        fonts,
        vec![0, 0, 0, 0, 0, 0, 1, 1],
        "拉丁与空格走 Inter(0)，汉字走 Noto(1)"
    );
    eprintln!(
        "混合单行：{} 个字形，宽度 {:.2}px，字体 {fonts:?}",
        r.lines[0].run.glyphs.len(),
        r.lines[0].width
    );
}

/// **零破坏**：单字体栈 ⇒ 与老 `layout` 完全一致（行数、宽度、基线、字形）。
#[test]
fn single_font_stack_layout_is_identical_to_legacy() {
    let (Some(a), Some(b)) = (load(INTER), load(INTER)) else { return };
    let opts = LayoutOptions::new(24.0, 200.0);
    let text = "The quick brown fox jumps over the lazy dog";
    let old = layout(&a, text, &opts).expect("老路径");
    let via = layout_with_stack(&FontStack::single(b), text, &opts).expect("栈路径");

    assert_eq!(old.lines.len(), via.lines.len(), "行数必须一致");
    assert_eq!(old.overflow, via.overflow, "溢出标记必须一致");
    for (i, (x, y)) in old.lines.iter().zip(via.lines.iter()).enumerate() {
        assert_eq!(x.width.to_bits(), y.width.to_bits(), "第 {i} 行宽度必须逐位一致");
        assert_eq!(x.baseline.x.to_bits(), y.baseline.x.to_bits());
        assert_eq!(x.baseline.y.to_bits(), y.baseline.y.to_bits());
        assert_eq!(x.text_range, y.text_range, "第 {i} 行字符范围必须一致");
        assert_eq!(x.run.glyphs.len(), y.run.glyphs.len());
        for (g1, g2) in x.run.glyphs.iter().zip(y.run.glyphs.iter()) {
            assert_eq!((g1.id, g1.x.to_bits(), g1.y.to_bits()),
                       (g2.id, g2.x.to_bits(), g2.y.to_bits()));
        }
    }
    eprintln!("单字体栈 ⇄ 老 layout：{} 行逐位一致（宽度 {:.2}）", old.lines.len(), old.size.x);
}

#[test]
fn mixed_text_wraps_without_losing_content() {
    let Some(stack) = stack() else { return };
    let opts = LayoutOptions::new(28.0, 220.0);
    let text = "Hello 世界你好 this is 一段混排文本 for wrapping";
    let r = layout_with_stack(&stack, text, &opts).expect("排版");

    assert!(r.lines.len() > 2, "这个宽度下应折成多行，实际 {}", r.lines.len());
    for (i, l) in r.lines.iter().enumerate() {
        assert!(
            l.width <= opts.max_width + 0.01 || r.overflow,
            "第 {i} 行宽 {:.2} 超过 max_width {:.2}",
            l.width,
            opts.max_width
        );
    }
    // 内容不丢：把各行拼起来，去掉被吞掉的空白后应与原文一致
    let joined: String = r.lines.iter().map(|l| &text[l.text_range.clone()]).collect();
    assert_eq!(
        joined.replace(' ', ""),
        text.replace(' ', ""),
        "折行不得丢字符（只允许吞掉行尾空白）"
    );
    eprintln!(
        "折行：{} 行（{} 字符），最大行宽 {:.2}",
        r.lines.len(),
        text.chars().count(),
        r.size.x
    );
}

/// 验收判据：**排版宽度 = 各 run 之和 ±0.5px**。
#[test]
fn line_width_equals_sum_of_per_font_runs() {
    let Some(stack) = stack() else { return };
    let opts = LayoutOptions::new(30.0, 260.0);
    let text = "Hello 世界你好 mixed 混排 text 排版 wrapping here";
    let r = layout_with_stack(&stack, text, &opts).expect("排版");

    for (i, l) in r.lines.iter().enumerate() {
        let line_text = &text[l.text_range.clone()];
        // 空行跳过（空行没有 run）
        if line_text.is_empty() {
            continue;
        }
        let expect = sum_of_runs(&stack, line_text, opts.size);
        let diff = (l.width - expect).abs();
        assert!(
            diff <= 0.5,
            "第 {i} 行 {line_text:?}：宽度 {:.3} 与各 run 之和 {expect:.3} 相差 {diff:.3}px（> 0.5）",
            l.width
        );
    }
    eprintln!(
        "各 run 之和 ⇄ 行宽：{} 行全部在 0.5px 内",
        r.lines.len()
    );
}

/// CJK 禁则在**真实字体 + 混排**下仍然生效。
#[test]
fn kinsoku_holds_in_mixed_wrapped_text() {
    let Some(stack) = stack() else { return };
    // 句尾标点（。」、）不得出现在行首；开引号不得出现在行尾
    let text = "这是一个测试。「你好」，世界。（完）Hello 世界再见";
    let opts = LayoutOptions::new(30.0, 210.0);
    let r = layout_with_stack(&stack, text, &opts).expect("排版");

    assert!(r.lines.len() > 1, "应折成多行");
    for (i, l) in r.lines.iter().enumerate() {
        let s = &text[l.text_range.clone()];
        if let Some(first) = s.chars().next() {
            assert!(
                !is_no_line_start(first),
                "第 {i} 行以禁则字符 {first:?} 开头：{s:?}"
            );
        }
        if let Some(last) = s.chars().last() {
            assert!(
                !is_no_line_end(last),
                "第 {i} 行以禁则字符 {last:?} 结尾：{s:?}"
            );
        }
    }
    eprintln!("禁则：{} 行全部守规（行首/行尾都没有违规字符）", r.lines.len());
}

/// 行高与基线取**栈里第一个字体**的度量（顺序即优先级，不做隐式合并）。
#[test]
fn line_height_follows_the_first_font() {
    let (Some(inter), Some(noto)) = (load(INTER), load(NOTO)) else { return };
    let m_inter = inter.metrics().expect("Inter 度量");
    let m_noto = noto.metrics().expect("Noto 度量");
    assert!(
        (m_inter.line_height() - m_noto.line_height()).abs() > 0.05,
        "两个字体的行高应明显不同（否则这条测试没有判别力）"
    );

    let opts = LayoutOptions::new(30.0, f32::INFINITY);
    let mut latin_first = FontStack::new();
    latin_first.push(inter);
    latin_first.push(noto);
    let mut cjk_first = FontStack::new();
    cjk_first.push(load(NOTO).expect("Noto"));
    cjk_first.push(load(INTER).expect("Inter"));

    let a = layout_with_stack(&latin_first, "Hello", &opts).expect("排版");
    let b = layout_with_stack(&cjk_first, "Hello", &opts).expect("排版");
    let expect_a = m_inter.line_height() * opts.size;
    let expect_b = m_noto.line_height() * opts.size;
    assert!((a.size.y - expect_a).abs() < 0.01, "应取 Inter 的行高 {expect_a:.2}，实际 {:.2}", a.size.y);
    assert!((b.size.y - expect_b).abs() < 0.01, "应取 Noto 的行高 {expect_b:.2}，实际 {:.2}", b.size.y);
    eprintln!(
        "行高随首字体：Inter {:.2}px、Noto {:.2}px（em 行高 {:.3} / {:.3}）",
        a.size.y,
        b.size.y,
        m_inter.line_height(),
        m_noto.line_height()
    );
}

#[test]
fn missing_glyph_is_an_error_but_notdef_policy_recovers() {
    let Some(stack) = stack() else { return };
    let opts = LayoutOptions::new(30.0, f32::INFINITY);
    // 两个字体都没有的码位
    let missing = ['\u{10FFFD}', '\u{E000}', '\u{F0000}']
        .into_iter()
        .find(|c| stack.resolve(*c).is_none());
    let Some(c) = missing else {
        eprintln!("跳过：找不到两个字体都没有的字符");
        return;
    };
    let text = format!("A{c}B");

    let err = layout_with_stack(&stack, &text, &opts);
    assert!(err.is_err(), "默认策略下缺字必须报错，实际 {err:?}");

    let ok = layout_with_stack_policy(
        &stack,
        &text,
        &opts,
        &TextOptions::default().with_missing_glyph(MissingGlyphPolicy::NotDef),
    )
    .expect("NotDef 策略应能排版");
    assert_eq!(ok.lines.len(), 1);
    assert_eq!(ok.lines[0].run.glyphs.len(), 3, "缺字位置仍应占一个 .notdef 字形");
    eprintln!("缺字：默认报错；NotDef 策略排出 3 个字形（含 1 个方框）");
}
