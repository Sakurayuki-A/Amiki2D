//! 字体与字体栈（含 CJK 回退）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

/// `amiki_text_options_default()` 必须与 header 里写的三个默认值一致。
#[test]
fn text_options_default_matches_the_header() {
    let o = amiki_text_options_default();
    assert_eq!(o.missing_glyph, AMIKI_MISSING_GLYPH_ERROR);
    assert!(o.kerning, "字距默认开");
    assert!(o.ligatures, "连字默认开");
    assert_eq!(o._pad, 0);
}

/// `resolve` 按**顺序**解析：拉丁走 Inter(0)、汉字走 Noto(1)。
#[test]
fn font_stack_resolve_follows_push_order() {
    let (Some(inter), Some(noto)) = (load_font("Inter-Regular.ttf"), load_font("NotoSansSC-Regular.ttf"))
    else {
        return;
    };
    let st = amiki_font_stack_create();
    assert!(!st.is_null(), "创建字体栈不应失败");
    assert_eq!(amiki_font_stack_push(st, inter), AMIKI_OK);
    assert_eq!(amiki_font_stack_push(st, noto), AMIKI_OK);

    let mut idx = 999u32;
    assert_eq!(amiki_font_stack_resolve(st, 'H' as u32, &mut idx), AMIKI_OK);
    assert_eq!(idx, 0, "拉丁应在 Inter（先压的）");
    assert_eq!(amiki_font_stack_resolve(st, '世' as u32, &mut idx), AMIKI_OK);
    assert_eq!(idx, 1, "汉字应在 Noto");

    // 两个字体都没有的码点 ⇒ NOT_FOUND，且**不写出参**
    let mut untouched = 12345u32;
    assert_eq!(
        amiki_font_stack_resolve(st, 0x10FFFD, &mut untouched),
        AMIKI_ERROR_NOT_FOUND
    );
    assert_eq!(untouched, 12345, "失败时不得写出参");
    // 非法标量值 ⇒ INVALID_ARGUMENT（而不是 NOT_FOUND）
    assert_eq!(
        amiki_font_stack_resolve(st, 0xD800, &mut untouched),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    // 出参可为 NULL
    assert_eq!(amiki_font_stack_resolve(st, 'A' as u32, ptr::null_mut()), AMIKI_OK);

    amiki_font_stack_destroy(st);
    amiki_font_destroy(inter);
    amiki_font_destroy(noto);
}

/// **压栈是拷贝**：push 之后立刻销毁 `amiki_font` 句柄，仍然应当画得出来。
///
/// 这条直接验证 header 所有权第 1 条里写的那句话 —— 如果哪天 push 改成
/// 借用指针，这里会立刻崩。
#[test]
fn font_may_be_destroyed_right_after_push() {
    let Some(noto) = load_font("NotoSansSC-Regular.ttf") else { return };
    let st = amiki_font_stack_create();
    assert_eq!(amiki_font_stack_push(st, noto), AMIKI_OK);
    amiki_font_destroy(noto); // **立刻销毁**：栈里已有自己的一份

    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv, st, c"你好".as_ptr(), 0, 4.0, 48.0, 40.0, ink, ptr::null()),
        AMIKI_OK
    );
    let out = render(cv, 64, 64);
    assert!(out.iter().any(|b| *b > 0), "销毁字体句柄后仍应画出内容");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    amiki_font_stack_destroy(st);
}

/// 缺字：默认策略返回 `AMIKI_ERROR_NOT_FOUND` **且画布状态不变**；
/// 换成 `NOTDEF` 策略后能画（画出方框）。
#[test]
fn missing_glyph_policy_is_honoured_end_to_end() {
    let (Some(inter), Some(noto)) = (load_font("Inter-Regular.ttf"), load_font("NotoSansSC-Regular.ttf"))
    else {
        return;
    };
    let st = amiki_font_stack_create();
    assert_eq!(amiki_font_stack_push(st, inter), AMIKI_OK);
    assert_eq!(amiki_font_stack_push(st, noto), AMIKI_OK);

    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 48, 48, &mut cv), AMIKI_OK);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    let before = render(cv, 48, 48);

    // 默认策略（options = NULL ⇒ ERROR）
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv, st, c"A\u{10FFFD}B".as_ptr(), 0, 2.0, 34.0, 28.0, ink, ptr::null()),
        AMIKI_ERROR_NOT_FOUND,
        "缺字默认应报 NOT_FOUND"
    );
    let after = render(cv, 48, 48);
    assert_eq!(before, after, "失败的绘制不得改动画布");

    // NOTDEF 策略：同一个串应当能画出来
    let opts = AmikiTextOptions {
        missing_glyph: AMIKI_MISSING_GLYPH_NOTDEF,
        ..amiki_text_options_default()
    };
    assert_eq!(
        amiki_canvas_draw_text_with_stack(
            cv,
            st,
            c"A\u{10FFFD}B".as_ptr(),
            0,
            2.0,
            34.0,
            28.0,
            ink,
            &opts
        ),
        AMIKI_OK
    );
    let with_box = render(cv, 48, 48);
    assert_ne!(before, with_box, "NOTDEF 策略应真的画出东西（方框）");

    // 越界策略值被拒绝
    let bad = AmikiTextOptions { missing_glyph: 99, ..amiki_text_options_default() };
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv, st, c"A".as_ptr(), 0, 2.0, 34.0, 28.0, ink, &bad),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    amiki_font_stack_destroy(st);
    amiki_font_destroy(inter);
    amiki_font_destroy(noto);
}

/// **字体回退真的画出了汉字**：只用 Noto 的字体栈与单字体路径
/// `amiki_canvas_draw_text(noto, "你好")` 必须给出**逐字节相同**的画面。
///
/// 若栈路径画的是方框（或不含 CJK），这里必然不等 —— 这条比"像素非空"有牙得多。
#[test]
fn stack_path_matches_single_font_path_for_cjk() {
    let (Some(noto_a), Some(noto_b)) = (load_font("NotoSansSC-Regular.ttf"), load_font("NotoSansSC-Regular.ttf"))
    else {
        return;
    };
    let text = c"你好世界";
    let (w, h) = (200u32, 64u32);

    // ① 单字体路径
    let (ctx, _) = cpu_context();
    let mut cv_a: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv_a), AMIKI_OK);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    assert_eq!(
        amiki_canvas_draw_text(cv_a, noto_a, text.as_ptr(), 0, 4.0, 48.0, 40.0, ink),
        AMIKI_OK
    );
    let a = render(cv_a, w, h);

    // ② 只含同一个 Noto 的字体栈
    let st = amiki_font_stack_create();
    assert_eq!(amiki_font_stack_push(st, noto_b), AMIKI_OK);
    let mut cv_b: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv_b), AMIKI_OK);
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv_b, st, text.as_ptr(), 0, 4.0, 48.0, 40.0, ink, ptr::null()),
        AMIKI_OK
    );
    let b = render(cv_b, w, h);

    // 有牙的前提：真的画出了东西（全黑画面会让"相等"变得毫无意义）
    assert!(a.iter().any(|x| *x > 0), "单字体路径应画出汉字");
    assert_eq!(a, b, "字体栈路径（同一字体）必须与单字体路径逐字节一致");

    amiki_canvas_destroy(cv_a);
    amiki_canvas_destroy(cv_b);
    amiki_context_destroy(ctx);
    amiki_font_stack_destroy(st);
    amiki_font_destroy(noto_a);
    amiki_font_destroy(noto_b);
}

// ============================================================================
// 异步输出：begin / poll / try_read（P3 第 ③b 步）
// ============================================================================

