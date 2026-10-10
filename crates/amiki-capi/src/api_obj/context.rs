//! 上下文：创建 / 销毁 / 查询。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::ffi_free;
use super::make_backend;
use crate::objects::*;
use crate::types::*;


/// 创建上下文。`AUTO` 先试 GPU，失败则落到 CPU；`GPU` 失败返回
/// `AMIKI_ERROR_BACKEND_UNAVAILABLE`；`CPU` 强制 CPU。
///
/// 失败时 `*out` **不被写入**（header 承诺）。
#[no_mangle]
pub extern "C" fn amiki_context_create(backend: i32, out: *mut *mut AmikiContext) -> i32 {
    ffi_free!({
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let (b, name) = match make_backend(backend) {
            Ok(v) => v,
            Err(r) => return r,
        };
        let ctx = Box::new(AmikiContext { backend: b, adapter_name: name, poison: Poison::default() });
        // SAFETY: `out` 已判空且调用方保证可写
        unsafe { *out = Box::into_raw(ctx) };
        AMIKI_OK
    })
}

/// 销毁上下文。传 NULL 是空操作。
///
/// 注意：**其下所有 canvas 必须先销毁** —— 本函数不追踪它们。
#[no_mangle]
pub extern "C" fn amiki_context_destroy(ctx: *mut AmikiContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_context_create` 且未被销毁。
    drop(unsafe { Box::from_raw(ctx) });
}

/// 实际生效的后端。传 NULL 返回 `AMIKI_BACKEND_AUTO`。
#[no_mangle]
pub extern "C" fn amiki_context_backend(ctx: *const AmikiContext) -> i32 {
    if ctx.is_null() {
        return AMIKI_BACKEND_AUTO;
    }
    // SAFETY: 已判空
    match unsafe { &*ctx }.backend {
        #[cfg(feature = "gpu")]
        Backend::Gpu(_) => AMIKI_BACKEND_GPU,
        Backend::Cpu => AMIKI_BACKEND_CPU,
    }
}

/// 适配器名（静态字符串，**不可释放**）。传 NULL 返回 `"(null)"`。
#[no_mangle]
pub extern "C" fn amiki_context_adapter_name(ctx: *const AmikiContext) -> *const std::ffi::c_char {
    if ctx.is_null() {
        return c"(null)".as_ptr();
    }
    // SAFETY: 已判空；返回的指针指向 context 自己持有的 `CString`，
    // 因此只要 context 活着就一直有效（header 承诺"永远有效"，
    // 实际语义是"至少和 context 一样久"）。
    unsafe { (*ctx).adapter_name.as_ptr() }
}

// ============================================================================
// 画布
// ============================================================================

