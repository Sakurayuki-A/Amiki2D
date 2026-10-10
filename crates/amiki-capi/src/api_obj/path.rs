//! 路径构造。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::{ffi, ffi_free};
use crate::objects::*;
use crate::types::*;


#[no_mangle]
pub extern "C" fn amiki_path_create(out: *mut *mut AmikiPath) -> i32 {
    ffi_free!({
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let p = Box::new(AmikiPath {
            path: amiki_scene::Path::default(),
            poison: Poison::default(),
        });
        // SAFETY: 已判空
        unsafe { *out = Box::into_raw(p) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_destroy(path: *mut AmikiPath) {
    if path.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_path_create` 且未被销毁
    drop(unsafe { Box::from_raw(path) });
}

#[no_mangle]
pub extern "C" fn amiki_path_clear(path: *mut AmikiPath) -> i32 {
    ffi!(path, |p| {
        p.path.els.clear();
        AMIKI_OK
    })
}

/// 三条路径构造函数的公共外壳。
macro_rules! path_op {
    ($path:expr, |$p:ident| $body:block) => {
        ffi!($path, |$p| {
            $body
            AMIKI_OK
        })
    };
}

#[no_mangle]
pub extern "C" fn amiki_path_move_to(path: *mut AmikiPath, x: f32, y: f32) -> i32 {
    path_op!(path, |p| {
        if !x.is_finite() || !y.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        p.path.els.push(amiki_scene::PathEl::MoveTo(amiki_math::Vec2::new(x, y)));
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_line_to(path: *mut AmikiPath, x: f32, y: f32) -> i32 {
    path_op!(path, |p| {
        if !x.is_finite() || !y.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        p.path.els.push(amiki_scene::PathEl::LineTo(amiki_math::Vec2::new(x, y)));
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_quad_to(
    path: *mut AmikiPath,
    cx: f32,
    cy: f32,
    x: f32,
    y: f32,
) -> i32 {
    path_op!(path, |p| {
        for v in [cx, cy, x, y] {
            if !v.is_finite() {
                return AMIKI_ERROR_INVALID_ARGUMENT;
            }
        }
        p.path.els.push(amiki_scene::PathEl::QuadTo(
            amiki_math::Vec2::new(cx, cy),
            amiki_math::Vec2::new(x, y),
        ));
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_cubic_to(
    path: *mut AmikiPath,
    c1x: f32,
    c1y: f32,
    c2x: f32,
    c2y: f32,
    x: f32,
    y: f32,
) -> i32 {
    path_op!(path, |p| {
        for v in [c1x, c1y, c2x, c2y, x, y] {
            if !v.is_finite() {
                return AMIKI_ERROR_INVALID_ARGUMENT;
            }
        }
        p.path.els.push(amiki_scene::PathEl::CubicTo(
            amiki_math::Vec2::new(c1x, c1y),
            amiki_math::Vec2::new(c2x, c2y),
            amiki_math::Vec2::new(x, y),
        ));
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_close(path: *mut AmikiPath) -> i32 {
    path_op!(path, |p| {
        p.path.els.push(amiki_scene::PathEl::Close);
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_rect(
    path: *mut AmikiPath,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
) -> i32 {
    path_op!(path, |p| {
        for v in [x0, y0, x1, y1] {
            if !v.is_finite() {
                return AMIKI_ERROR_INVALID_ARGUMENT;
            }
        }
        let mut pb = amiki_scene::PathBuilder::new();
        pb.rect(x0, y0, x1, y1);
        p.path.els.extend(pb.build().els);
    })
}

#[no_mangle]
pub extern "C" fn amiki_path_circle(path: *mut AmikiPath, cx: f32, cy: f32, r: f32) -> i32 {
    path_op!(path, |p| {
        if !cx.is_finite() || !cy.is_finite() || !r.is_finite() || r <= 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let mut pb = amiki_scene::PathBuilder::new();
        pb.circle(amiki_math::Vec2::new(cx, cy), r, false);
        p.path.els.extend(pb.build().els);
    })
}

// ============================================================================
// 资源：位图 / 字体 / 渐变
// ============================================================================

