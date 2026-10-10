//! 画布：缓冲、绘制、清空、尺寸。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

#[test]
fn render_rejects_short_output_buffer_without_writing() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 8, 8, &mut cv), AMIKI_OK);

    let mut out = vec![0xAAu8; 8 * 8 * 4];
    // 少一个字节
    assert_eq!(
        amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len() - 1),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(out.iter().all(|&b| b == 0xAA), "失败时不应写入任何字节");

    // 够长就行（多给也无妨）
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 端到端：真的画出东西
// ============================================================================

#[test]
fn cpu_render_draws_the_requested_geometry() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let red = amiki_paint_solid(amiki_color_rgb8(255, 0, 0));
    assert_eq!(amiki_canvas_draw_rect(cv, 0.0, 0.0, 16.0, 16.0, red), AMIKI_OK);

    let mut out = vec![0u8; 16 * 16 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);

    // 中心像素应为红且不透明
    let i = (8 * 16 + 8) * 4;
    assert!(out[i] > 200, "R 应为红：{}", out[i]);
    assert!(out[i + 1] < 40, "G 应为 0：{}", out[i + 1]);
    assert!(out[i + 2] < 40, "B 应为 0：{}", out[i + 2]);
    assert!(out[i + 3] > 200, "A 应不透明：{}", out[i + 3]);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// **header 所有权第 4 条的实测**：位图与渐变在绘制后即可销毁，
/// 因为引擎内部持有计数引用。这条如果错了，用户的代码会偶发崩溃。
#[test]
fn images_and_gradients_may_be_destroyed_right_after_drawing() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    // 4×4 的绿图
    let px = [0u8, 255, 0, 255].repeat(16);
    let mut img: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_rgba8(px.as_ptr(), 4, 4, 0, &mut img), AMIKI_OK);
    assert_eq!(amiki_canvas_draw_image(cv, img, 0.0, 0.0, 16.0, 16.0), AMIKI_OK);
    // **立刻销毁句柄** —— 渲染时引擎应仍持有它
    amiki_image_destroy(img);

    let stops = [
        AmikiGradientStop { t: 0.0, color: amiki_color_rgb8(0, 0, 255) },
        AmikiGradientStop { t: 1.0, color: amiki_color_rgb8(255, 255, 0) },
    ];
    let mut g: *mut AmikiGradient = ptr::null_mut();
    assert_eq!(
        amiki_gradient_create_linear(0.0, 0.0, 16.0, 0.0, stops.as_ptr(), 2, &mut g),
        AMIKI_OK
    );
    let mut path: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut path), AMIKI_OK);
    assert_eq!(amiki_path_circle(path, 8.0, 8.0, 6.0), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(cv, path, amiki_paint_gradient(g)), AMIKI_OK);
    amiki_gradient_destroy(g);
    amiki_path_destroy(path);

    // 两个句柄都已销毁，这里必须仍然渲染成功
    let mut out = vec![0u8; 16 * 16 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert!(out.iter().any(|&b| b != 0), "销毁句柄后画面变成了空的");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn clear_resets_state_so_the_next_frame_is_independent() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 8, 8, &mut cv), AMIKI_OK);
    let red = amiki_paint_solid(amiki_color_rgb8(255, 0, 0));

    assert_eq!(amiki_canvas_draw_rect(cv, 0.0, 0.0, 8.0, 8.0, red), AMIKI_OK);
    assert_eq!(amiki_canvas_set_blend(cv, AMIKI_BLEND_MULTIPLY), AMIKI_OK);
    assert_eq!(amiki_canvas_scale(cv, 3.0), AMIKI_OK);
    let mut path: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut path), AMIKI_OK);
    assert_eq!(amiki_path_rect(path, 0.0, 0.0, 4.0, 4.0), AMIKI_OK);
    assert_eq!(amiki_canvas_push_clip(cv, path), AMIKI_OK);

    assert_eq!(amiki_canvas_clear(cv), AMIKI_OK);

    // clear 之后：裁剪栈、变换、混合都应复位 → 弹裁剪会失败
    assert_eq!(amiki_canvas_pop_clip(cv), AMIKI_ERROR_INVALID_ARGUMENT);
    let mut out = vec![0xAAu8; 8 * 8 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert!(out.iter().all(|&b| b == 0), "clear 后应是空画面（全透明）");

    amiki_path_destroy(path);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn canvas_size_reports_what_was_requested() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 37, 19, &mut cv), AMIKI_OK);
    let (mut w, mut h) = (0u32, 0u32);
    assert_eq!(amiki_canvas_size(cv, &mut w, &mut h), AMIKI_OK);
    assert_eq!((w, h), (37, 19));
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 路径构造
// ============================================================================

