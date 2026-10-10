//! 渐变创建与销毁。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::objects::*;
use crate::types::*;

use amiki_scene::GradientKind;

/// 三个渐变构造函数共用的外壳。
fn gradient_create(
    kind: GradientKind,
    stops: *const AmikiGradientStop,
    count: u32,
    out: *mut *mut AmikiGradient,
) -> i32 {
    crate::api::ffi_free!({
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 契约要求 stops 指向 count 个元素
        let Some(stops) = (unsafe { stops_from_raw(stops, count) }) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiGradient {
            gradient: make_gradient(kind, stops),
            poison: Poison::default(),
        });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_gradient_create_linear(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    stops: *const AmikiGradientStop,
    stop_count: u32,
    out: *mut *mut AmikiGradient,
) -> i32 {
    if !x0.is_finite() || !y0.is_finite() || !x1.is_finite() || !y1.is_finite() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    gradient_create(
        GradientKind::Linear {
            from: amiki_math::Vec2::new(x0, y0),
            to: amiki_math::Vec2::new(x1, y1),
        },
        stops,
        stop_count,
        out,
    )
}

#[no_mangle]
pub extern "C" fn amiki_gradient_create_radial(
    cx: f32,
    cy: f32,
    radius: f32,
    stops: *const AmikiGradientStop,
    stop_count: u32,
    out: *mut *mut AmikiGradient,
) -> i32 {
    // 半径必须为正 —— 引擎的 `Gradient::radial` 也这么要求，
    // 在这里先拒绝能给出更准确的错误位置。
    if !cx.is_finite() || !cy.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    gradient_create(
        GradientKind::Radial { center: amiki_math::Vec2::new(cx, cy), radius },
        stops,
        stop_count,
        out,
    )
}

#[no_mangle]
pub extern "C" fn amiki_gradient_create_sweep(
    cx: f32,
    cy: f32,
    start_radians: f32,
    stops: *const AmikiGradientStop,
    stop_count: u32,
    out: *mut *mut AmikiGradient,
) -> i32 {
    if !cx.is_finite() || !cy.is_finite() || !start_radians.is_finite() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    gradient_create(
        GradientKind::Sweep { center: amiki_math::Vec2::new(cx, cy), start: start_radians },
        stops,
        stop_count,
        out,
    )
}

#[no_mangle]
pub extern "C" fn amiki_gradient_destroy(gradient: *mut AmikiGradient) {
    if gradient.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_gradient_create_*` 且未被销毁
    drop(unsafe { Box::from_raw(gradient) });
}

