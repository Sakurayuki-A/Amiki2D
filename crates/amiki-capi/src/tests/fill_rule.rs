//! 填充规则（非零 / even-odd）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

// ============================================================================
// 填充规则 / even-odd（0.8.0）
// ============================================================================
/// 往返与默认值：不调用时必须是 `AMIKI_FILL_NONZERO`。
#[test]
fn fill_rule_defaults_to_nonzero_and_roundtrips() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_NONZERO, "默认必须是 NonZero");
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_EVENODD);
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_NONZERO), AMIKI_OK);
    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_NONZERO);
    // NULL 不崩，给默认值
    assert_eq!(amiki_canvas_get_fill_rule(ptr::null()), AMIKI_FILL_NONZERO);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 越界枚举必须被拒绝，且**不改动已有状态**。
#[test]
fn fill_rule_rejects_out_of_range() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);

    for bad in [-1, 2, 99] {
        assert_eq!(
            amiki_canvas_set_fill_rule(cv, bad),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "填充规则 {bad} 必须被拒绝"
        );
    }
    assert_eq!(
        amiki_canvas_get_fill_rule(cv),
        AMIKI_FILL_EVENODD,
        "被拒绝的调用不得改动状态"
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// **端到端**：同向双环在 NonZero 下实心、EvenOdd 下镂空 —— 经过 C ABI 也一样。
#[test]
fn even_odd_changes_the_render_through_the_abi() {
    let (ctx, _) = cpu_context();
    let (w, h) = (32u32, 32u32);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    let draw_rings = |cv: *mut AmikiCanvas| {
        let mut p: *mut AmikiPath = ptr::null_mut();
        assert_eq!(amiki_path_create(&mut p), AMIKI_OK);
        // 外环 + **同向**内环（两圈都是 rect，方向相同）
        assert_eq!(amiki_path_rect(p, 3.3, 3.1, 28.7, 28.9), AMIKI_OK);
        assert_eq!(amiki_path_rect(p, 11.2, 11.3, 21.8, 20.7), AMIKI_OK);
        assert_eq!(amiki_canvas_fill_path(cv, p, ink), AMIKI_OK);
        amiki_path_destroy(p);
    };

    // NonZero
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_rings(cv);
    let nz = render(cv, w, h);

    // EvenOdd（同一个画布上改状态再画一遍，用到 clear）
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_clear(cv), AMIKI_OK);
    // clear 之后规则回到默认 —— 这里重新设一次，顺便验证"粘性"是每次设置后生效
    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_NONZERO);
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);
    draw_rings(cv);
    let eo = render(cv, w, h);

    let center = ((16 * w + 16) * 4) as usize + 3;
    eprintln!(
        "C ABI even-odd：中心 NonZero alpha={} / EvenOdd alpha={}",
        nz[center], eo[center]
    );
    assert_eq!(nz[center], 255, "NonZero 下同向双环中心应不透明");
    assert_eq!(eo[center], 0, "EvenOdd 下同向双环中心应镂空");
    let diff = nz.iter().zip(eo.iter()).filter(|(a, b)| a != b).count();
    assert!(diff > 100, "两条规则必须真的画出不同结果，实际不同字节 {diff}");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 文本轮廓缓存（0.10.6，D8 的另一半）
// ============================================================================

