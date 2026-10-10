//! 画布生命周期与尺寸查询。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::{ffi, ffi_free, write_out};
use crate::objects::*;
use crate::types::*;


#[no_mangle]
pub extern "C" fn amiki_canvas_create(
    ctx: *mut AmikiContext,
    width: u32,
    height: u32,
    out: *mut *mut AmikiCanvas,
) -> i32 {
    ffi_free!({
        if ctx.is_null() || out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        if width == 0 || height == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let cv = Box::new(AmikiCanvas {
            tone_map: amiki_scene::ToneMap::Raw,
            mip_bias: 0.0,
            fill_rule: amiki_scene::FillRule::NonZero,
            ctx,
            width,
            height,
            eps: 0.1,
            root: amiki::Canvas::new(width, height),
            layers: Vec::new(),
            clip_depth: 0,
            pending: None,
            poison: Poison::default(),
        });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(cv) };
        AMIKI_OK
    })
}

/// 销毁画布。传 NULL 是空操作。
#[no_mangle]
pub extern "C" fn amiki_canvas_destroy(canvas: *mut AmikiCanvas) {
    if canvas.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_canvas_create` 且未被销毁
    drop(unsafe { Box::from_raw(canvas) });
}

/// 清空内容与状态（变换、混合、裁剪栈、层栈），复用内部缓冲。
#[no_mangle]
pub extern "C" fn amiki_canvas_clear(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        c.reset_state();
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_size(
    canvas: *const AmikiCanvas,
    out_w: *mut u32,
    out_h: *mut u32,
) -> i32 {
    if canvas.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let c = unsafe { &*canvas };
    if let Err(r) = c.poison.check() {
        return r;
    }
    // SAFETY: 出参可为 NULL（header 写明）
    unsafe {
        write_out(out_w, c.width);
        write_out(out_h, c.height);
    }
    AMIKI_OK
}

// ---- 状态 ----

