//! 渐变参数校验。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

#[test]
fn gradient_needs_at_least_two_stops() {
    let mut g: *mut AmikiGradient = ptr::null_mut();
    let stop = AmikiGradientStop { t: 0.0, color: amiki_color_rgb8(1, 2, 3) };

    // 0 个、1 个色标都不行
    assert_eq!(
        amiki_gradient_create_linear(0.0, 0.0, 1.0, 1.0, ptr::null(), 0, &mut g),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_gradient_create_linear(0.0, 0.0, 1.0, 1.0, &stop, 1, &mut g),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    // 2 个可以
    let stops = [stop, AmikiGradientStop { t: 1.0, color: amiki_color_rgb8(9, 9, 9) }];
    assert_eq!(
        amiki_gradient_create_linear(0.0, 0.0, 1.0, 1.0, stops.as_ptr(), 2, &mut g),
        AMIKI_OK
    );
    assert!(!g.is_null());
    amiki_gradient_destroy(g);
}

/// `amiki_paint_gradient` 的 kind 必须**由渐变几何决定**，不是调用方指定的。
#[test]
fn paint_gradient_kind_follows_the_gradient_geometry() {
    let stops = [
        AmikiGradientStop { t: 0.0, color: amiki_color_rgb8(0, 0, 0) },
        AmikiGradientStop { t: 1.0, color: amiki_color_rgb8(255, 255, 255) },
    ];
    let mut g: *mut AmikiGradient = ptr::null_mut();

    assert_eq!(amiki_gradient_create_radial(0.0, 0.0, 4.0, stops.as_ptr(), 2, &mut g), AMIKI_OK);
    assert_eq!(amiki_paint_gradient(g).kind, AMIKI_PAINT_RADIAL);

    amiki_gradient_destroy(g);
    assert_eq!(amiki_gradient_create_sweep(0.0, 0.0, 0.0, stops.as_ptr(), 2, &mut g), AMIKI_OK);
    assert_eq!(amiki_paint_gradient(g).kind, AMIKI_PAINT_SWEEP);

    amiki_gradient_destroy(g);
    assert_eq!(
        amiki_gradient_create_linear(0.0, 0.0, 1.0, 0.0, stops.as_ptr(), 2, &mut g),
        AMIKI_OK
    );
    assert_eq!(amiki_paint_gradient(g).kind, AMIKI_PAINT_LINEAR);
    amiki_gradient_destroy(g);

    // 空句柄不崩，返回一个"用到时必然失败"的 paint
    let p = amiki_paint_gradient(ptr::null());
    assert!(p.gradient.is_null());
}

#[test]
fn radial_gradient_rejects_non_positive_radius() {
    let stops = [
        AmikiGradientStop { t: 0.0, color: amiki_color_rgb8(0, 0, 0) },
        AmikiGradientStop { t: 1.0, color: amiki_color_rgb8(255, 255, 255) },
    ];
    let mut g: *mut AmikiGradient = ptr::null_mut();
    for r in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            amiki_gradient_create_radial(0.0, 0.0, r, stops.as_ptr(), 2, &mut g),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "半径 {r} 应被拒绝"
        );
    }
}

// ============================================================================
// 栈配对：压了没弹必须被挡住
// ============================================================================

