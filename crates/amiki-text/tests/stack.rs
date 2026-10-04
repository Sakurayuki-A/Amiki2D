//! amiki-text 集成测试：**字体回退**（`FontStack` / `shape_with_stack`）。
//!
//! 用真实字体：`assets/fonts/Inter-Regular.ttf`（拉丁）+ `NotoSansSC-Regular.ttf`（CJK）。
//! 任一缺失则跳过并明确提示——不静默通过。

use amiki_text::{
    shape, shape_with_stack, Font, FontStack, MissingGlyphPolicy, TextError, TextOptions,
};

const INTER: &str = "Inter-Regular.ttf";
const NOTO: &str = "NotoSansSC-Regular.ttf";

fn repo_root() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/amiki-text
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("无法定位仓库根目录")
        .to_path_buf()
}

fn load(name: &str) -> Option<Font> {
    let p = repo_root().join("assets/fonts").join(name);
    if !p.exists() {
        eprintln!("跳过：未找到字体文件 {}", p.display());
        return None;
    }
    Some(Font::from_file(&p).expect("字体应能解析"))
}

/// Inter 在前、Noto 在后（顺序即优先级）。
fn latin_first() -> Option<FontStack> {
    let mut s = FontStack::new();
    s.push(load(INTER)?);
    s.push(load(NOTO)?);
    Some(s)
}

fn missing_char(stack: &FontStack) -> Option<char> {
    // 私有区 / 非字符区：这两个字体都不该有
    ['\u{10FFFD}', '\u{E000}', '\u{F0000}', '\u{2FA1F}']
        .into_iter()
        .find(|c| stack.resolve(*c).is_none())
}

// ---------------------------------------------------------------- 字体身份

#[test]
fn font_id_is_stable_across_loads_and_distinct_between_fonts() {
    let (Some(a), Some(b)) = (load(INTER), load(INTER)) else { return };
    let Some(noto) = load(NOTO) else { return };

    assert_eq!(a.id(), b.id(), "同一份字体两次加载必须得到同一个 FontId");
    assert_ne!(a.id(), noto.id(), "Inter 与 Noto 的 FontId 必须不同");
    assert_ne!(a.id(), amiki_text::FontId::UNKNOWN, "真实字体不得是 UNKNOWN");

    // ID 来自字体自己声明的名字（不是路径、不是字节顺序）
    assert!(
        a.id_name().contains("Inter"),
        "Inter 的 id_name 应含 \"Inter\"，实际 {:?}",
        a.id_name()
    );
    eprintln!("FontId：Inter = {}（{}），Noto = {}", a.id(), a.id_name(), noto.id());
}

#[test]
fn font_id_is_stable_for_the_same_bytes() {
    let Some(font) = load(INTER) else { return };
    let bytes = std::fs::read(repo_root().join("assets/fonts").join(INTER)).expect("读字体");
    let again = Font::from_bytes(bytes).expect("应能解析");
    assert_eq!(font.id(), again.id(), "同一份字节必须得到同一个 FontId");
}

// ---------------------------------------------------------------- 解析

#[test]
fn stack_resolves_latin_to_inter_and_cjk_to_noto() {
    let Some(stack) = latin_first() else { return };
    assert_eq!(stack.len(), 2);
    assert_eq!(stack.resolve('H'), Some(0), "拉丁字母应解析到 Inter");
    assert_eq!(stack.resolve(' '), Some(0), "空格在 Inter 里也有");
    assert_eq!(stack.resolve('世'), Some(1), "汉字应解析到 Noto");
    assert_eq!(stack.resolve('好'), Some(1));
    assert!(missing_char(&stack).is_some(), "应存在两个字体都没有的字符（作为缺字用例）");
    eprintln!("解析：H→0、空格→0、世→1、好→1");
}

/// **顺序显式**：谁在前谁赢 —— 不依赖任何"按脚本优先级猜"的隐式规则。
#[test]
fn stack_order_decides_priority() {
    let (Some(noto), Some(inter)) = (load(NOTO), load(INTER)) else { return };
    let mut reversed = FontStack::new();
    reversed.push(noto);
    reversed.push(inter);
    // Noto Sans SC 也含拉丁字母，因此倒过来后 'H' 由 Noto 胜出
    assert_eq!(reversed.resolve('H'), Some(0), "Noto 在前时 'H' 应解析到 Noto");
    assert_eq!(reversed.resolve('世'), Some(0));
    let run = shape_with_stack(&reversed, "H世", 32.0, &TextOptions::default()).expect("shaping");
    assert!(
        run.glyphs.iter().all(|g| g.font_index == 0),
        "整个串都应来自栈里第 0 个字体，实际 {:?}",
        run.glyphs.iter().map(|g| g.font_index).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------- 混合串

#[test]
fn shape_with_stack_splits_runs_by_font() {
    let Some(stack) = latin_first() else { return };
    let text = "Hello 世界 123 你好";
    let run = shape_with_stack(&stack, text, 32.0, &TextOptions::default())
        .expect("混合串应能 shaping（两个字体合起来覆盖全部字符）");

    // "Hello " = 6、世界 = 2、空格+123+空格 = 5、你好 = 2 → 15
    assert_eq!(run.glyphs.len(), 15, "字形数应与字符数一致（本串无连字）");
    let fonts: Vec<usize> = run.glyphs.iter().map(|g| g.font_index).collect();
    assert_eq!(
        fonts,
        vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        "拉丁与空格应走 Inter(0)，汉字应走 Noto(1)"
    );

    // 位置必须严格递增（pen 跨 run 累加）
    for w in run.glyphs.windows(2) {
        assert!(w[1].x > w[0].x, "字形 x 必须递增：{:?} → {:?}", w[0], w[1]);
    }
    eprintln!(
        "混合串：{} 个字形，字体序列 {:?}，末字形 x = {:.3} em",
        run.glyphs.len(),
        fonts,
        run.glyphs.last().unwrap().x
    );
}

/// **零破坏的可执行证据**：单字体栈 + LEGACY 选项 ⇒ 与老 `shape` 逐字节一致。
#[test]
fn single_font_stack_is_identical_to_legacy_shape() {
    let Some(inter) = load(INTER) else { return };
    let Some(again) = load(INTER) else { return };
    let legacy = shape(&inter, "Hello AV fi", 40.0).expect("老路径");
    let stack = FontStack::single(again);
    let via_stack =
        shape_with_stack(&stack, "Hello AV fi", 40.0, &TextOptions::LEGACY).expect("栈路径");

    assert_eq!(legacy.glyphs.len(), via_stack.glyphs.len(), "字形数应一致");
    for (i, (a, b)) in legacy.glyphs.iter().zip(via_stack.glyphs.iter()).enumerate() {
        assert_eq!(a.id, b.id, "第 {i} 个字形 id 必须一致");
        assert_eq!(a.x.to_bits(), b.x.to_bits(), "第 {i} 个字形 x 必须逐位一致");
        assert_eq!(a.y.to_bits(), b.y.to_bits(), "第 {i} 个字形 y 必须逐位一致");
        assert_eq!(b.font_index, 0, "单字体栈的 font_index 恒为 0");
    }
    eprintln!("单字体栈 ⇄ 老 shape：{} 个字形逐位一致", legacy.glyphs.len());
}

#[test]
fn clusters_point_back_into_the_original_text() {
    let Some(stack) = latin_first() else { return };
    let text = "Hi 世界!";
    let run = shape_with_stack(&stack, text, 32.0, &TextOptions::default()).expect("shaping");
    for g in &run.glyphs {
        let off = g.cluster as usize;
        assert!(off < text.len(), "cluster {off} 必须落在原文内");
        assert!(text.is_char_boundary(off), "cluster {off} 必须是字符边界");
        let c = text[off..].chars().next().expect("该偏移处应有字符");
        assert_eq!(
            stack.resolve(c),
            Some(g.font_index),
            "cluster 指回的字符 {c:?} 应解析到字体 {}（字形 {:?}）",
            g.font_index,
            g
        );
    }
    eprintln!("cluster 全部指回原文且与字体一致（{} 个字形）", run.glyphs.len());
}

#[test]
fn empty_text_shapes_to_empty_run() {
    let Some(stack) = latin_first() else { return };
    let run = shape_with_stack(&stack, "", 32.0, &TextOptions::default()).expect("空串不是错误");
    assert!(run.glyphs.is_empty());
    assert_eq!(run.size, 32.0);
}

// ---------------------------------------------------------------- 缺字三策略

#[test]
fn missing_glyph_error_is_the_default_policy() {
    let Some(stack) = latin_first() else { return };
    let Some(c) = missing_char(&stack) else {
        eprintln!("跳过：找不到两个字体都没有的字符");
        return;
    };
    assert_eq!(TextOptions::default().missing_glyph, MissingGlyphPolicy::Error);

    let text = format!("A{c}B");
    match shape_with_stack(&stack, &text, 32.0, &TextOptions::default()) {
        Err(TextError::MissingGlyph(cp)) => {
            assert_eq!(cp, c as u32, "错误里应带上缺的**码点**");
            eprintln!("缺字默认报错：码点 U+{:04X}", cp);
        }
        other => panic!("默认策略必须返回 Err(MissingGlyph)，实际 {other:?}"),
    }
}

#[test]
fn missing_glyph_notdef_policy_emits_glyph_zero() {
    let Some(stack) = latin_first() else { return };
    let Some(c) = missing_char(&stack) else { return };
    let opts = TextOptions::default().with_missing_glyph(MissingGlyphPolicy::NotDef);
    let run = shape_with_stack(&stack, &format!("A{c}B"), 32.0, &opts).expect("NotDef 策略不应报错");

    assert_eq!(run.glyphs.len(), 3, "缺字位置仍应产生一个字位（.notdef 方框）");
    assert_eq!(run.glyphs[1].id, 0, "缺字应渲染成 .notdef（字形 0）");
    assert!(
        run.glyphs[1].x > run.glyphs[0].x && run.glyphs[2].x > run.glyphs[1].x,
        "方框必须占位（前进量正确）"
    );
    eprintln!(
        "NotDef：缺字 → 字形 {}, 前进量 {:.4} em",
        run.glyphs[1].id,
        run.glyphs[2].x - run.glyphs[1].x
    );
}

#[test]
fn missing_glyph_skip_policy_leaves_no_gap() {
    let Some(stack) = latin_first() else { return };
    let Some(c) = missing_char(&stack) else { return };
    let opts = TextOptions::default().with_missing_glyph(MissingGlyphPolicy::Skip);
    let skipped =
        shape_with_stack(&stack, &format!("A{c}B"), 32.0, &opts).expect("Skip 策略不应报错");
    let plain = shape_with_stack(&stack, "AB", 32.0, &opts).expect("AB");

    assert_eq!(skipped.glyphs.len(), 2, "缺字不应产生字形");
    assert_eq!(
        skipped.glyphs.len(),
        plain.glyphs.len(),
        "Skip 后的字形数应与直接排 \"AB\" 相同"
    );
    for (i, (a, b)) in skipped.glyphs.iter().zip(plain.glyphs.iter()).enumerate() {
        assert_eq!(a.id, b.id, "第 {i} 个字形应与 \"AB\" 相同");
        assert!(
            (a.x - b.x).abs() < 1e-6,
            "第 {i} 个字形位置应与 \"AB\" 相同（缺字不留空隙）：{} vs {}",
            a.x,
            b.x
        );
    }
    // cluster 仍要指回**原文**里的 B（Skip 会挖洞，位置不能错位）
    let b_off = format!("A{c}B").find('B').expect("含 B") as u32;
    assert_eq!(skipped.glyphs[1].cluster, b_off, "cluster 应指回原文中的 B");
    eprintln!("Skip：缺字不留空隙，且 cluster 仍指回原文偏移 {b_off}");
}

// ---------------------------------------------------------------- 选项

#[test]
fn kerning_toggle_changes_advance() {
    let Some(stack) = latin_first() else { return };
    let on = shape_with_stack(&stack, "AV", 100.0, &TextOptions::default()).expect("shaping");
    let off = shape_with_stack(
        &stack,
        "AV",
        100.0,
        &TextOptions::default().without_kerning(),
    )
    .expect("shaping");
    assert_eq!(on.glyphs.len(), 2);
    assert_eq!(off.glyphs.len(), 2);
    assert!(
        off.glyphs[1].x > on.glyphs[1].x,
        "关掉字距后 V 应更靠右（不再收紧）：开={:.4} 关={:.4}",
        on.glyphs[1].x,
        off.glyphs[1].x
    );
    eprintln!(
        "kern：开 {:.4} em、关 {:.4} em（差 {:.4}）",
        on.glyphs[1].x,
        off.glyphs[1].x,
        off.glyphs[1].x - on.glyphs[1].x
    );
}

/// 连字开关。Inter 的 `liga` 是否真的有 `fi` 连字由**实测**决定，
/// 因此这里断言的是不变量（关掉不会让字形变多），并把实测数字打出来。
#[test]
fn ligature_toggle_never_increases_glyphs() {
    let Some(stack) = latin_first() else { return };
    for text in ["fi", "ffi", "ffl", "flight"] {
        let on = shape_with_stack(&stack, text, 64.0, &TextOptions::default()).expect("shaping");
        let off = shape_with_stack(
            &stack,
            text,
            64.0,
            &TextOptions::default().without_ligatures(),
        )
        .expect("shaping");
        assert!(
            off.glyphs.len() >= on.glyphs.len(),
            "{text:?}：关掉连字后字形数不应减少（开 {}、关 {}）",
            on.glyphs.len(),
            off.glyphs.len()
        );
        eprintln!(
            "liga {text:?}：开 {} 个字形、关 {} 个",
            on.glyphs.len(),
            off.glyphs.len()
        );
    }
}
