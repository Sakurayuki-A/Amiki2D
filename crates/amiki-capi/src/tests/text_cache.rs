//! 字形缓存（per-font）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

fn cache_stats(f: *mut AmikiFont) -> (u64, u64) {
    // 测试要读字体内部的缓存统计（ABI 不暴露它），裸指针引用无法避免。
    #[allow(dangerous_implicit_autorefs)]
    unsafe {
        let c = (*f).cache.lock().unwrap_or_else(|e| e.into_inner());
        (c.hits(), c.misses())
    }
}

/// 重复绘制同一段文本：第二轮**不再缺失**（缓存真的在复用）。
#[test]
fn text_outline_cache_is_reused_across_draws() {
    let Some(font) = load_font("Inter-Regular.ttf") else {
        return;
    };
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 220, 40, &mut cv), AMIKI_OK);
    let text = c"Cache me twice";
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    assert_eq!(amiki_canvas_draw_text(cv, font, text.as_ptr(), 0, 4.0, 28.0, 16.0, ink), AMIKI_OK);
    let (h1, m1) = cache_stats(font);
    // SAFETY: 句柄有效
    unsafe {
        (*cv).target().scene.draws.clear();
    }
    assert_eq!(amiki_canvas_draw_text(cv, font, text.as_ptr(), 0, 4.0, 28.0, 16.0, ink), AMIKI_OK);
    let (h2, m2) = cache_stats(font);

    assert!(m1 > 0, "第一轮应有缺失（首次取轮廓）");
    assert!(h2 > h1, "第二轮应命中：h2={h2} h1={h1}");
    // 缺失可能仍 +N（实测 +2）：**没有轮廓的字形**（空格/无轮廓标记）本来就存不进
    // 缓存 —— 每次都得再问一次。所以判据是"命中在涨 + 缺失增量很小"，不是"零缺失"。
    assert!(m2 - m1 <= 2, "第二轮缺失增量应极小（仅无轮廓字形），实测 {} ", m2 - m1);
    eprintln!("重复绘制：第一轮 hits={h1} misses={m1}；第二轮 hits={h2} misses={m2}");

    amiki_canvas_destroy(cv);
    amiki_font_destroy(font);
}

/// **每个字体句柄各自一份缓存** —— 这就是"构造上不会串字体"的证据（D8 的核心）。
#[test]
fn cache_is_per_font_handle() {
    let (Some(a), Some(b)) = (load_font("Inter-Regular.ttf"), load_font("Inter-Regular.ttf")) else {
        return;
    };
    let (ctx, _) = cpu_context();
    let mut cva: *mut AmikiCanvas = ptr::null_mut();
    let mut cvb: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 120, 40, &mut cva), AMIKI_OK);
    assert_eq!(amiki_canvas_create(ctx, 120, 40, &mut cvb), AMIKI_OK);
    let text = c"AAA";
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    assert_eq!(amiki_canvas_draw_text(cva, a, text.as_ptr(), 0, 2.0, 28.0, 16.0, ink), AMIKI_OK);
    let (ha, _) = cache_stats(a);
    let (hb, _) = cache_stats(b);
    assert_eq!(hb, 0, "另一个句柄的缓存不应被碰到");

    assert_eq!(amiki_canvas_draw_text(cvb, b, text.as_ptr(), 0, 2.0, 28.0, 16.0, ink), AMIKI_OK);
    let (ha2, _) = cache_stats(a);
    let (hb2, _) = cache_stats(b);
    assert_eq!(ha, ha2, "b 的绘制不影响 a 的缓存");
    assert!(hb2 > 0, "b 自己的缓存开始计数");
    eprintln!("per-font 缓存：a hits={ha2}、b hits={hb2}（互不影响 ⇒ 不会串字体）");

    amiki_canvas_destroy(cva);
    amiki_canvas_destroy(cvb);
    amiki_font_destroy(a);
    amiki_font_destroy(b);
}

/// 缓存不改变错误路径（回归）。
#[test]
fn cache_does_not_change_error_paths() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 32, &mut cv), AMIKI_OK);
    let text = c"x";
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    assert_eq!(
        amiki_canvas_draw_text(cv, ptr::null(), text.as_ptr(), 0, 0.0, 0.0, 12.0, ink),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    amiki_canvas_destroy(cv);
    eprintln!("坏参数路径未受缓存影响（font=NULL ⇒ INVALID_ARGUMENT）");
}

// ============================================================================
// 图像编解码（0.10.0）
// ============================================================================

