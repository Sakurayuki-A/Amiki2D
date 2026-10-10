//! 路径构造。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

#[test]
fn path_commands_accumulate_and_clear_resets() {
    let mut p: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut p), AMIKI_OK);

    assert_eq!(amiki_path_move_to(p, 0.0, 0.0), AMIKI_OK);
    assert_eq!(amiki_path_line_to(p, 10.0, 0.0), AMIKI_OK);
    assert_eq!(amiki_path_quad_to(p, 5.0, 5.0, 10.0, 10.0), AMIKI_OK);
    assert_eq!(amiki_path_cubic_to(p, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0), AMIKI_OK);
    assert_eq!(amiki_path_rect(p, 0.0, 0.0, 4.0, 4.0), AMIKI_OK);
    assert_eq!(amiki_path_circle(p, 0.0, 0.0, 2.0), AMIKI_OK);
    assert_eq!(amiki_path_close(p), AMIKI_OK);

    // SAFETY: 指针来自 amiki_path_create
    let n = unsafe { (*p).path.els.len() };
    assert!(n > 8, "路径元素应累积起来，实际 {n}");

    assert_eq!(amiki_path_clear(p), AMIKI_OK);
    // SAFETY: 同上
    assert_eq!(unsafe { (*p).path.els.len() }, 0);

    amiki_path_destroy(p);
}

// ============================================================================
// 中毒机制
// ============================================================================

