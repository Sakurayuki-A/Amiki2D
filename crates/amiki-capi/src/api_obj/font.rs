//! 字体与字体栈（含 shaping 选项默认值）。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::{ffi, ffi_free, write_out};
use crate::objects::*;
use crate::types::*;

use std::ffi::CStr;

#[no_mangle]
pub extern "C" fn amiki_font_create_from_file(
    utf8_path: *const std::ffi::c_char,
    out: *mut *mut AmikiFont,
) -> i32 {
    ffi_free!({
        if utf8_path.is_null() || out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 调用方保证以 \0 结尾
        let Ok(s) = unsafe { CStr::from_ptr(utf8_path) }.to_str() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // `Font::from_file` 会把文件读进内存，所以文件句柄不需要调用方保留。
        let Ok(font) = amiki_text::Font::from_file(s) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiFont {
            font,
            poison: Poison::default(),
            cache: std::sync::Mutex::new(amiki_scene::TextCache::new()),
        });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_font_create_from_memory(
    data: *const u8,
    len: usize,
    out: *mut *mut AmikiFont,
) -> i32 {
    ffi_free!({
        if data.is_null() || out.is_null() || len == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 调用方保证 [data, data+len) 可读
        let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        let Ok(font) = amiki_text::Font::from_bytes(bytes) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiFont {
            font,
            poison: Poison::default(),
            cache: std::sync::Mutex::new(amiki_scene::TextCache::new()),
        });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_font_destroy(font: *mut AmikiFont) {
    if font.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_font_create_*` 且未被销毁
    drop(unsafe { Box::from_raw(font) });
}

#[no_mangle]
pub extern "C" fn amiki_font_metrics(
    font: *const AmikiFont,
    out_ascent: *mut f32,
    out_descent: *mut f32,
    out_line_gap: *mut f32,
) -> i32 {
    if font.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let h = unsafe { &*font };
    if let Err(r) = h.poison.check() {
        return r;
    }
    let Ok(m) = h.font.metrics() else {
        return AMIKI_ERROR_UNKNOWN;
    };
    // SAFETY: 三个出参都可为 NULL
    unsafe {
        write_out(out_ascent, m.ascent);
        write_out(out_descent, m.descent);
        write_out(out_line_gap, m.line_gap);
    }
    AMIKI_OK
}

// ---- 字体栈（字体回退，0.7.0）----

/// 创建一个**空**字体栈。失败（分配失败 / panic）返回 NULL。
#[no_mangle]
pub extern "C" fn amiki_font_stack_create() -> *mut AmikiFontStack {
    // 这里不用 `ffi_free!`：它在 panic 时返回 `AMIKI_ERROR_PANIC`（i32），
    // 而本函数返回指针 —— 用 `unwrap_or(null)` 表达"创建失败"。
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Box::into_raw(Box::new(AmikiFontStack::new()))
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// 销毁字体栈。传 NULL 是空操作。
///
/// **不影响**压栈用过的 `amiki_font`：压栈是克隆（字节用 Arc 共享），
/// 句柄仍归调用方，该 `amiki_font_destroy` 还得调。
#[no_mangle]
pub extern "C" fn amiki_font_stack_destroy(stack: *mut AmikiFontStack) {
    if stack.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_font_stack_create` 且未被销毁
    drop(unsafe { Box::from_raw(stack) });
}

/// 把一个字体压入栈（**追加**：顺序即优先级，先压的先赢）。
#[no_mangle]
pub extern "C" fn amiki_font_stack_push(
    stack: *mut AmikiFontStack,
    font: *const AmikiFont,
) -> i32 {
    if font.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(stack, |s| {
        // SAFETY: 已判空
        let f = unsafe { &*font };
        if let Err(r) = f.poison.check() {
            return r;
        }
        // 克隆 `Font` 很便宜：字体字节在 `Arc` 里共享，这里不复制那几 MB。
        s.stack.push(f.font.clone());
        AMIKI_OK
    })
}

/// 查询**第一个**含该码点的字体下标（0 起）。
///
/// 都不含时返回 `AMIKI_ERROR_NOT_FOUND`（并且不写出参）。
/// `codepoint` 不是合法标量值（代理区 / > U+10FFFF）⇒ `AMIKI_ERROR_INVALID_ARGUMENT`。
#[no_mangle]
pub extern "C" fn amiki_font_stack_resolve(
    stack: *const AmikiFontStack,
    codepoint: u32,
    out_index: *mut u32,
) -> i32 {
    if stack.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let s = unsafe { &*stack };
    if let Err(r) = s.poison.check() {
        return r;
    }
    let Some(c) = char::from_u32(codepoint) else {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    };
    match s.stack.resolve(c) {
        Some(i) => {
            // SAFETY: out 可为 NULL（header 契约）
            unsafe { write_out(out_index, i as u32) };
            AMIKI_OK
        }
        None => AMIKI_ERROR_NOT_FOUND,
    }
}

/// `amiki_text_options` 的默认值（与 header 里写的三个默认值一致）。
#[no_mangle]
pub extern "C" fn amiki_text_options_default() -> AmikiTextOptions {
    AmikiTextOptions::default_options()
}

// ---- 渐变 ----

