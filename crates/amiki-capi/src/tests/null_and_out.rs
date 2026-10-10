//! 空指针 / 出参 / 越界值。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

// ============================================================================
// 空指针：**必须被拒绝，且不能崩**
// ============================================================================
#[test]
fn every_pointer_entry_point_rejects_null() {
    // 这个测试的价值在于"跑完不崩" —— 空指针解引用会直接段错误。
    assert_eq!(amiki_context_backend(ptr::null()), AMIKI_BACKEND_AUTO);
    amiki_context_destroy(ptr::null_mut()); // 空操作
    assert!(!amiki_context_adapter_name(ptr::null()).is_null()); // 返回 "(null)"，不崩

    assert_eq!(amiki_canvas_clear(ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    amiki_canvas_destroy(ptr::null_mut());
    assert_eq!(
        amiki_canvas_size(ptr::null(), ptr::null_mut(), ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_set_transform(ptr::null_mut(), amiki_affine_identity()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_render_rgba8(ptr::null_mut(), ptr::null_mut(), 0),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(amiki_canvas_begin_render_rgba8(ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_poll_render_rgba8(ptr::null_mut(), ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(ptr::null_mut(), ptr::null_mut(), 0),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    assert_eq!(amiki_path_clear(ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    amiki_path_destroy(ptr::null_mut());
    assert_eq!(amiki_path_move_to(ptr::null_mut(), 0.0, 0.0), AMIKI_ERROR_INVALID_ARGUMENT);

    amiki_image_destroy(ptr::null_mut());
    assert_eq!(
        amiki_image_size(ptr::null(), ptr::null_mut(), ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_font_destroy(ptr::null_mut());
    assert_eq!(
        amiki_font_metrics(ptr::null(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_gradient_destroy(ptr::null_mut());

    // 字体栈（0.7.0）：destroy(NULL) 是空操作；其余入口都要拒绝空指针
    amiki_font_stack_destroy(ptr::null_mut());
    assert_eq!(
        amiki_font_stack_push(ptr::null_mut(), ptr::null()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_font_stack_resolve(ptr::null(), 'A' as u32, ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_draw_text_with_stack(
            ptr::null_mut(),
            ptr::null(),
            ptr::null(),
            0,
            0.0,
            0.0,
            12.0,
            amiki_paint_solid(amiki_color_rgb8(0, 0, 0)),
            ptr::null()
        ),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    // create 类：out 为空也要拒绝
    assert_eq!(
        amiki_context_create(AMIKI_BACKEND_CPU, ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_create(ptr::null_mut(), 4, 4, ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(amiki_path_create(ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_image_create_rgba8(ptr::null(), 1, 1, 0, ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_font_create_from_file(ptr::null(), ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
}

#[test]
fn context_adapter_name_is_never_null() {
    // 传 NULL 也要返回一个可打印的字符串，而不是 NULL —— 方便调用方直接 printf。
    let p = amiki_context_adapter_name(ptr::null());
    assert!(!p.is_null());
    // SAFETY: 上一条断言保证非空，且实现返回静态字符串
    let s = unsafe { std::ffi::CStr::from_ptr(p) };
    assert_eq!(s.to_str().unwrap(), "(null)");
}

// ============================================================================
// 失败时不写出参（header 的明确承诺）
// ============================================================================

#[test]
fn out_params_are_untouched_on_failure() {
    const SENTINEL: usize = 0xDEAD_BEEF;
    let mut canvas_ptr: *mut AmikiCanvas = SENTINEL as *mut AmikiCanvas;

    let (ctx, r) = cpu_context();
    assert_eq!(r, AMIKI_OK);

    // 宽或高为 0 → 必须失败，且**不写** out
    assert_eq!(
        amiki_canvas_create(ctx, 0, 8, &mut canvas_ptr),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(canvas_ptr as usize, SENTINEL, "失败时不应写出参");

    assert_eq!(
        amiki_canvas_create(ctx, 8, 0, &mut canvas_ptr),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(canvas_ptr as usize, SENTINEL, "失败时不应写出参");

    // 现在成功一次，确认这条路径确实会写
    assert_eq!(amiki_canvas_create(ctx, 8, 8, &mut canvas_ptr), AMIKI_OK);
    assert_ne!(canvas_ptr as usize, SENTINEL);
    amiki_canvas_destroy(canvas_ptr);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 非法输入
// ============================================================================

#[test]
fn invalid_values_are_rejected_not_silently_accepted() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    let red = amiki_paint_solid(amiki_color_rgb8(255, 0, 0));

    // 越界枚举
    assert_eq!(amiki_canvas_set_blend(cv, 16), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(amiki_canvas_set_blend(cv, -1), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(amiki_canvas_set_blend(cv, AMIKI_BLEND_NORMAL), AMIKI_OK);

    // NaN / Inf：不能让它们流进光栅化
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    assert_eq!(amiki_canvas_translate(cv, nan, 0.0), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(amiki_canvas_scale(cv, inf), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(amiki_canvas_rotate(cv, nan), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_draw_rect(cv, 0.0, 0.0, nan, 1.0, red),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_draw_circle(cv, 0.0, 0.0, -1.0, red),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    // 压平容差必须为正
    assert_eq!(amiki_canvas_set_flatten_tolerance(cv, 0.0), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(amiki_canvas_set_flatten_tolerance(cv, -1.0), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(amiki_canvas_set_flatten_tolerance(cv, nan), AMIKI_ERROR_INVALID_ARGUMENT);

    // paint 的 kind 越界
    let bad = AmikiPaint { kind: 99, ..Default::default() };
    let mut path: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut path), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(cv, path, bad), AMIKI_ERROR_INVALID_ARGUMENT);

    // 渐变 paint 但句柄为空
    let no_grad =
        AmikiPaint { kind: AMIKI_PAINT_LINEAR, color: AmikiColor::default(), gradient: ptr::null() };
    assert_eq!(amiki_canvas_fill_path(cv, path, no_grad), AMIKI_ERROR_INVALID_ARGUMENT);

    amiki_path_destroy(path);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

