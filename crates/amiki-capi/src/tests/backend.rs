//! 后端选择与回落。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

#[test]
fn cpu_backend_is_always_available() {
    let (ctx, r) = cpu_context();
    assert_eq!(r, AMIKI_OK);
    assert_eq!(amiki_context_backend(ctx), AMIKI_BACKEND_CPU);

    // SAFETY: 指针来自 amiki_context_create
    let name = unsafe { std::ffi::CStr::from_ptr(amiki_context_adapter_name(ctx)) };
    assert_eq!(name.to_str().unwrap(), "CPU");

    // 非法后端值
    let mut c2: *mut AmikiContext = ptr::null_mut();
    assert_eq!(amiki_context_create(99, &mut c2), AMIKI_ERROR_INVALID_ARGUMENT);
    assert!(c2.is_null());

    amiki_context_destroy(ctx);
}

/// `AUTO` 在**没有 GPU** 的机器上必须落到 CPU，而不是失败 ——
/// 这是 header 里对 `AMIKI_BACKEND_AUTO` 的承诺。
#[test]
fn auto_backend_falls_back_to_cpu_when_gpu_is_unavailable() {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    let r = amiki_context_create(AMIKI_BACKEND_AUTO, &mut ctx);
    assert_eq!(r, AMIKI_OK, "AUTO 不应失败");
    let b = amiki_context_backend(ctx);
    assert!(
        b == AMIKI_BACKEND_GPU || b == AMIKI_BACKEND_CPU,
        "AUTO 必须落到某个具体后端，实际 {b}"
    );
    amiki_context_destroy(ctx);
}



// ============================================================================
// 色调映射（HDR 的出口）
// ============================================================================

