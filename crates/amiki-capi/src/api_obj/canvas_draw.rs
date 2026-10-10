//! 画布绘制：填充 / 描边 / 矩形 / 圆 / 五角星 / 位图 / 文本。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::ffi;
use crate::objects::*;
use crate::types::*;

use std::ffi::CStr;
use std::sync::Arc;

#[no_mangle]
pub extern "C" fn amiki_canvas_fill_path(
    canvas: *mut AmikiCanvas,
    path: *const AmikiPath,
    paint: AmikiPaint,
) -> i32 {
    if path.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        // SAFETY: 已判空
        let p = unsafe { &*path };
        if let Err(r) = p.poison.check() {
            return r;
        }
        with_paint!(c, paint, |paint| {
            c.target().fill_path(&p.path, paint);
            AMIKI_OK
        })
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_stroke_path(
    canvas: *mut AmikiCanvas,
    path: *const AmikiPath,
    style: AmikiStrokeStyle,
    paint: AmikiPaint,
) -> i32 {
    if path.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        // SAFETY: 已判空
        let p = unsafe { &*path };
        if let Err(r) = p.poison.check() {
            return r;
        }
        let Some(style) = style.to_style() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        with_paint!(c, paint, |paint| {
            c.target().stroke_path_with(&p.path, style, paint);
            AMIKI_OK
        })
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_draw_rect(
    canvas: *mut AmikiCanvas,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    paint: AmikiPaint,
) -> i32 {
    ffi!(canvas, |c| {
        for v in [x0, y0, x1, y1] {
            if !v.is_finite() {
                return AMIKI_ERROR_INVALID_ARGUMENT;
            }
        }
        with_paint!(c, paint, |paint| {
            c.target().rect(x0, y0, x1, y1, paint);
            AMIKI_OK
        })
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_draw_circle(
    canvas: *mut AmikiCanvas,
    cx: f32,
    cy: f32,
    r: f32,
    paint: AmikiPaint,
) -> i32 {
    ffi!(canvas, |c| {
        if !cx.is_finite() || !cy.is_finite() || !r.is_finite() || r < 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        with_paint!(c, paint, |paint| {
            c.target().circle(amiki_math::Vec2::new(cx, cy), r, paint);
            AMIKI_OK
        })
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_draw_pentagram(
    canvas: *mut AmikiCanvas,
    cx: f32,
    cy: f32,
    r: f32,
    paint: AmikiPaint,
) -> i32 {
    ffi!(canvas, |c| {
        if !cx.is_finite() || !cy.is_finite() || !r.is_finite() || r < 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        with_paint!(c, paint, |paint| {
            c.target().pentagram(amiki_math::Vec2::new(cx, cy), r, paint);
            AMIKI_OK
        })
    })
}

/// 从 C 句柄取 `Arc<Image>`（`Arc::clone`，于是 C 侧可立刻销毁自己的句柄）。
///
/// # Safety
/// `img` 若非空必须有效。
unsafe fn image_arc(img: *const AmikiImage) -> Option<Arc<amiki_scene::Image>> {
    if img.is_null() {
        return None;
    }
    // SAFETY: 调用方保证有效
    let h = unsafe { &*img };
    if h.poison.is_poisoned() {
        return None;
    }
    Some(Arc::clone(&h.image))
}

#[no_mangle]
pub extern "C" fn amiki_canvas_draw_image(
    canvas: *mut AmikiCanvas,
    image: *const AmikiImage,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> i32 {
    ffi!(canvas, |c| {
        for v in [x0, y0, x1, y1] {
            if !v.is_finite() {
                return AMIKI_ERROR_INVALID_ARGUMENT;
            }
        }
        // SAFETY: 契约要求 image 有效
        let Some(a) = (unsafe { image_arc(image) }) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        c.target().image(a, x0, y0, x1, y1);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_draw_image_placed(
    canvas: *mut AmikiCanvas,
    image: *const AmikiImage,
    placement: AmikiPlacement,
) -> i32 {
    ffi!(canvas, |c| {
        // SAFETY: 契约要求 image 有效
        let Some(a) = (unsafe { image_arc(image) }) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        c.target().image_with_placement(a, placement.to_placement());
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_fill_path_image(
    canvas: *mut AmikiCanvas,
    path: *const AmikiPath,
    image: *const AmikiImage,
    placement: AmikiPlacement,
) -> i32 {
    if path.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        // SAFETY: 已判空
        let p = unsafe { &*path };
        if let Err(r) = p.poison.check() {
            return r;
        }
        // SAFETY: 契约要求 image 有效
        let Some(a) = (unsafe { image_arc(image) }) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let paint = image_paint(a, placement.to_placement());
        c.target().fill_path_with(&p.path, paint);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_draw_text(
    canvas: *mut AmikiCanvas,
    font: *const AmikiFont,
    utf8: *const std::ffi::c_char,
    utf8_len: usize,
    x: f32,
    y: f32,
    size: f32,
    paint: AmikiPaint,
) -> i32 {
    if font.is_null() || utf8.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        if !x.is_finite() || !y.is_finite() || !size.is_finite() || size <= 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 已判空
        let f = unsafe { &*font };
        if let Err(r) = f.poison.check() {
            return r;
        }
        // 显式长度：0 表示"按 \0 求长"（header 承诺的语义）
        let bytes = if utf8_len == 0 {
            // SAFETY: 调用方保证 utf8 以 \0 结尾
            unsafe { CStr::from_ptr(utf8) }.to_bytes()
        } else {
            // SAFETY: 调用方保证 [utf8, utf8+utf8_len) 可读
            unsafe { std::slice::from_raw_parts(utf8.cast::<u8>(), utf8_len) }
        };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let Ok(run) = amiki_text::shape(&f.font, text, size) else {
            // 字体里缺字形不是错误，只是没有内容可画 —— 但 shaping 失败
            // 意味着字体数据有问题，明确报出来而不是静默画出空画面。
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        with_paint!(c, paint, |paint| {
            let font_ref = &f.font;
            let mut cache = f.cache.lock().unwrap_or_else(|p| p.into_inner());
                c.target().text_cached(&mut cache, &run, amiki_math::Vec2::new(x, y), paint, |id| {
                font_ref.outline(id).ok()
            });
            AMIKI_OK
        })
    })
}

/// 绘制一段 UTF-8 文本，字形按**字体栈**做回退（0.7.0）。
///
/// 与 [`amiki_canvas_draw_text`] 的区别只有一处：字体来源从单个 `amiki_font`
/// 换成有序的 `amiki_font_stack` —— 因此一段文本里拉丁与汉字可以来自不同字体。
/// 其余（基线原点、显式长度、字形走同一管线）完全相同。
#[no_mangle]
pub extern "C" fn amiki_canvas_draw_text_with_stack(
    canvas: *mut AmikiCanvas,
    stack: *const AmikiFontStack,
    utf8: *const std::ffi::c_char,
    utf8_len: usize,
    x: f32,
    y: f32,
    size: f32,
    paint: AmikiPaint,
    options: *const AmikiTextOptions,
) -> i32 {
    if stack.is_null() || utf8.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        if !x.is_finite() || !y.is_finite() || !size.is_finite() || size <= 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 已判空
        let s = unsafe { &*stack };
        if let Err(r) = s.poison.check() {
            return r;
        }
        // 选项可为 NULL ⇒ 用 header 承诺的默认值（缺字报错、字距与连字都开）
        let opts = if options.is_null() {
            AmikiTextOptions::default_options()
        } else {
            // SAFETY: 已判空
            unsafe { *options }
        };
        let Ok(text_opts) = opts.to_options() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 显式长度：0 表示"按 \0 求长"（与 amiki_canvas_draw_text 同语义）
        let bytes = if utf8_len == 0 {
            // SAFETY: 调用方保证 utf8 以 \0 结尾
            unsafe { CStr::from_ptr(utf8) }.to_bytes()
        } else {
            // SAFETY: 调用方保证 [utf8, utf8+utf8_len) 可读
            unsafe { std::slice::from_raw_parts(utf8.cast::<u8>(), utf8_len) }
        };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let run = match amiki_text::shape_with_stack(&s.stack, text, size, &text_opts) {
            Ok(r) => r,
            // 缺字（策略为 Error 时）是**内容**问题而不是参数问题：明确告诉调用方
            // "栈里没有这个字"，它据此可以换策略或补字体 —— 比含糊的
            // INVALID_ARGUMENT 有用得多。其余失败仍按字体数据问题处理。
            Err(amiki_text::TextError::MissingGlyph(_)) => return AMIKI_ERROR_NOT_FOUND,
            Err(_) => return AMIKI_ERROR_INVALID_ARGUMENT,
        };
        with_paint!(c, paint, |paint| {
            let stack_ref = &s.stack;
            c.target()
                .text_with_stack(&run, amiki_math::Vec2::new(x, y), paint, |fi, id| {
                    stack_ref.get(fi)?.outline(id).ok()
                });
            AMIKI_OK
        })
    })
}

// ---- 离屏层 ----

