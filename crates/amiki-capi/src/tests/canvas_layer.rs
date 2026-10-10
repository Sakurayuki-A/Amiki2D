//! 裁剪与离屏层。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

#[test]
fn clip_pop_without_push_is_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    // 引擎的 `Canvas::pop_clip` 在栈空时是**静默空操作** ——
    // 那会让"弹多了"这类错误无声通过。C ABI 这一层显式拒绝。
    assert_eq!(amiki_canvas_pop_clip(cv), AMIKI_ERROR_INVALID_ARGUMENT);

    let mut path: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut path), AMIKI_OK);
    assert_eq!(amiki_path_rect(path, 0.0, 0.0, 8.0, 8.0), AMIKI_OK);

    assert_eq!(amiki_canvas_push_clip(cv, path), AMIKI_OK);
    assert_eq!(amiki_canvas_pop_clip(cv), AMIKI_OK);
    // 配平之后再弹仍然要拒绝
    assert_eq!(amiki_canvas_pop_clip(cv), AMIKI_ERROR_INVALID_ARGUMENT);

    amiki_path_destroy(path);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn layer_end_without_begin_is_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_ERROR_INVALID_ARGUMENT);

    let b = AmikiLayerBounds { x: 0, y: 0, width: 8, height: 8 };
    assert_eq!(
        amiki_canvas_begin_layer(cv, b, amiki_filter_blur(2.0), 1.0, AMIKI_BLEND_NORMAL),
        AMIKI_OK
    );
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_ERROR_INVALID_ARGUMENT);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// **打开中的层没配平就渲染** —— 直接渲染会静默丢掉那些绘制。
/// 这是本项目最不能接受的一类错误，必须显式拒绝。
#[test]
fn rendering_with_an_open_layer_is_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let b = AmikiLayerBounds { x: 0, y: 0, width: 8, height: 8 };
    assert_eq!(
        amiki_canvas_begin_layer(cv, b, amiki_filter_blur(2.0), 1.0, AMIKI_BLEND_NORMAL),
        AMIKI_OK
    );
    let mut out = vec![0u8; 16 * 16 * 4];
    assert_eq!(
        amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "层没配平就渲染会静默丢内容，必须拒绝"
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn layer_opacity_must_be_in_unit_range() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    let b = AmikiLayerBounds { x: 0, y: 0, width: 8, height: 8 };

    for bad in [-0.1f32, 1.1, f32::NAN] {
        assert_eq!(
            amiki_canvas_begin_layer(cv, b, amiki_filter_none(), bad, AMIKI_BLEND_NORMAL),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "opacity {bad} 应被拒绝"
        );
    }
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 输出缓冲
// ============================================================================

