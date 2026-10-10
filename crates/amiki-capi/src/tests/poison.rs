//! 中毒机制。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

/// 直接测 `Poison` —— 要**故意**让引擎 panic 来测端到端中毒很难，
/// 而机制本身可以穷尽验证。
#[test]
fn poison_blocks_every_later_call() {
    let mut p = Poison::default();
    assert!(!p.is_poisoned());
    assert!(p.check().is_ok());

    p.mark();
    assert!(p.is_poisoned());
    assert_eq!(p.check().unwrap_err(), AMIKI_ERROR_PANIC);
    // 重复标记是幂等的
    p.mark();
    assert_eq!(p.check().unwrap_err(), AMIKI_ERROR_PANIC);
}

#[test]
fn a_poisoned_canvas_refuses_to_work() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 8, 8, &mut cv), AMIKI_OK);

    // SAFETY: 指针来自 amiki_canvas_create
    unsafe { (*cv).poison.mark() };

    let red = amiki_paint_solid(amiki_color_rgb8(255, 0, 0));
    assert_eq!(amiki_canvas_draw_rect(cv, 0.0, 0.0, 8.0, 8.0, red), AMIKI_ERROR_PANIC);
    assert_eq!(amiki_canvas_clear(cv), AMIKI_ERROR_PANIC);
    let mut out = vec![0u8; 8 * 8 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_ERROR_PANIC);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 后端
// ============================================================================

