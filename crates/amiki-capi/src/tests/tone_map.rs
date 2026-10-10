//! 色调映射（HDR 出口）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

#[test]
fn tone_map_defaults_to_raw_and_round_trips() {
    let (ctx, rc) = cpu_context();
    assert_eq!(rc, AMIKI_OK);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);

    // 默认必须是 Raw —— 也就是 0.5.0 之前的行为
    assert_eq!(amiki_canvas_tone_map(cv), AMIKI_TONEMAP_RAW);
    for v in [AMIKI_TONEMAP_REINHARD, AMIKI_TONEMAP_ACES, AMIKI_TONEMAP_RAW] {
        assert_eq!(amiki_canvas_set_tone_map(cv, v), AMIKI_OK);
        assert_eq!(amiki_canvas_tone_map(cv), v);
    }
    // 空画布不崩，返回默认值
    assert_eq!(amiki_canvas_tone_map(ptr::null()), AMIKI_TONEMAP_RAW);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn out_of_range_tone_map_is_rejected_and_leaves_state_untouched() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_tone_map(cv, AMIKI_TONEMAP_ACES), AMIKI_OK);

    // 越界值必须**显式失败**，且**不改动**已有设置 ——
    // 静默退回默认会让"设错了"看起来像"设上了"
    for bad in [-1i32, 3, 99, i32::MAX] {
        assert_eq!(
            amiki_canvas_set_tone_map(cv, bad),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "越界值 {bad} 应被拒绝"
        );
        assert_eq!(
            amiki_canvas_tone_map(cv),
            AMIKI_TONEMAP_ACES,
            "越界值 {bad} 不该改动已有设置"
        );
    }
    assert_eq!(
        amiki_canvas_set_tone_map(ptr::null_mut(), AMIKI_TONEMAP_ACES),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn tone_map_changes_the_output_of_an_hdr_pixel() {
    // **这条是"真的接进去了"的证明**：若 render_rgba8 忘了用 c.tone_map，
    // 三种设置会给出完全一样的图。
    let raw = render_hot_square(AMIKI_TONEMAP_RAW);
    let rei = render_hot_square(AMIKI_TONEMAP_REINHARD);
    let aces = render_hot_square(AMIKI_TONEMAP_ACES);

    let diff = |a: &[u8], b: &[u8]| -> u32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
            .max()
            .unwrap_or(0)
    };
    let (d_rei, d_ace, d_ra) = (diff(&raw, &rei), diff(&raw, &aces), diff(&rei, &aces));
    eprintln!("C ABI 色调映射：Reinhard 差 {d_rei}/255，ACES 差 {d_ace}，两者互差 {d_ra}");

    assert!(d_rei > 0 && d_ace > 0, "色调映射没有影响输出（{d_rei} / {d_ace}）—— 没接进去");
    assert!(d_ra > 0, "Reinhard 与 ACES 给出相同的输出 —— 那等于只有一个映射");

    // 方块**内部**的像素：alpha 不透明，且不该被色调映射改动
    // （(0,0) 在方块外、alpha 自然是 0，不能用它断言"不透明"）
    let inside = (32 * 64 + 32) * 4;
    assert_eq!(raw[inside + 3], 255, "方块内应当不透明");
    assert_eq!(aces[inside + 3], 255, "色调映射不该改动 alpha");
}

// ============================================================================
// B4：mipmap 模式与 mip bias（0.6.0）
// ============================================================================

