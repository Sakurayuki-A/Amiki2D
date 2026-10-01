//! ABI 层自测。
//!
//! 这些测试**从 C 侧的角度**调用导出函数（裸指针、out 参数、错误码），
//! 因此它们验证的是 `include/amiki.h` 承诺的那些行为：
//!
//! - 版本号两处一致
//! - 空指针 / 越界枚举 / 非法数值**都被拒绝**，且不崩溃
//! - **失败时不写出参**（header 的明确承诺）
//! - 中毒机制真的会挡住后续调用
//! - 位图 / 渐变可以**在绘制后立刻销毁**（header 所有权第 4 条）
//! - 端到端的 CPU 渲染真的画出东西
//!
//! 这些测试不替代 C / C# / Node 三个示例 —— 那三个验证的是"别的语言真能调进来"，
//! 而这里验证的是"契约本身成立"。

use crate::api::*;
use crate::api_obj::*;
use crate::objects::*;
use crate::types::*;
use std::ptr;

/// 建一个 CPU 后端的上下文（测试要确定、不能依赖机器有没有 GPU）。
fn cpu_context() -> (*mut AmikiContext, i32) {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    let r = amiki_context_create(AMIKI_BACKEND_CPU, &mut ctx);
    (ctx, r)
}

// ============================================================================
// 版本
// ============================================================================

#[test]
fn version_string_is_taken_from_cargo_toml() {
    // SAFETY: `amiki_version_string` 返回静态 C 字符串
    let s = unsafe { std::ffi::CStr::from_ptr(amiki_version_string()) };
    // 直接与包版本比 —— 这条断言**不可能因为"忘了改某处"而漂移**，
    // 因为实现就是编译期从 Cargo.toml 取的。
    assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn numeric_version_matches_package_version() {
    // 数值版本没法在 const 上下文里从字符串解析，所以只能各写一份 ——
    // 这条测试就是保证两份同步的东西。
    let pkg = env!("CARGO_PKG_VERSION");
    let core = pkg.split('-').next().unwrap(); // 去掉 -alpha 之类
    let parts: Vec<u32> = core.split('.').map(|x| x.parse().expect("版本段不是数字")).collect();
    assert_eq!(parts.len(), 3, "包版本应为 major.minor.patch：{pkg}");

    let v = amiki_version();
    assert_eq!(v >> 16, parts[0], "major 与 Cargo.toml 不一致（包版本 {pkg}）");
    assert_eq!((v >> 8) & 0xFF, parts[1], "minor 与 Cargo.toml 不一致（包版本 {pkg}）");
    assert_eq!(v & 0xFF, parts[2], "patch 与 Cargo.toml 不一致（包版本 {pkg}）");
}

#[test]
fn result_string_covers_every_value_and_rejects_out_of_range() {
    for r in [
        AMIKI_OK,
        AMIKI_ERROR_INVALID_ARGUMENT,
        AMIKI_ERROR_OUT_OF_MEMORY,
        AMIKI_ERROR_BACKEND_UNAVAILABLE,
        AMIKI_ERROR_NOT_SUPPORTED,
        AMIKI_ERROR_PANIC,
        AMIKI_ERROR_UNKNOWN,
    ] {
        // SAFETY: 返回静态 C 字符串
        let s = unsafe { std::ffi::CStr::from_ptr(amiki_result_string(r)) };
        assert!(!s.to_bytes().is_empty(), "错误码 {r} 没有说明文字");
        // `"unknown"` 只留给**越界的错误码**（那是调用方自己的 bug）；
        // 每个已定义的错误码都要有自己的说明。
        assert_ne!(s.to_str().unwrap(), "unknown", "错误码 {r} 落到了 unknown");
    }
    // 越界值不能崩，且明确返回 "unknown"
    // SAFETY: 同上
    let s = unsafe { std::ffi::CStr::from_ptr(amiki_result_string(9999)) };
    assert_eq!(s.to_str().unwrap(), "unknown");
}

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

#[test]
fn path_commands_accumulate_and_clear_resets() {
    let mut p: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut p), AMIKI_OK);

    assert_eq!(amiki_path_move_to(p, 0.0, 0.0), AMIKI_OK);
    assert_eq!(amiki_path_line_to(p, 10.0, 0.0), AMIKI_OK);
    assert_eq!(amiki_path_quad_to(p, 5.0, 5.0, 10.0, 10.0), AMIKI_OK);
    assert_eq!(amiki_path_cubic_to(p, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0), AMIKI_OK);
    assert_eq!(amiki_path_rect(p, 0.0, 0.0, 4.0, 4.0), AMIKI_OK);
    assert_eq!(amiki_path_circle(p, 0.0, 0.0, 2.0), AMIKI_OK);
    assert_eq!(amiki_path_close(p), AMIKI_OK);

    // SAFETY: 指针来自 amiki_path_create
    let n = unsafe { (*p).path.els.len() };
    assert!(n > 8, "路径元素应累积起来，实际 {n}");

    assert_eq!(amiki_path_clear(p), AMIKI_OK);
    // SAFETY: 同上
    assert_eq!(unsafe { (*p).path.els.len() }, 0);

    amiki_path_destroy(p);
}

// ============================================================================
// 中毒机制
// ============================================================================

/// 直接测 `Poison` —— 要**故意**让引擎 panic 来测端到端中毒很难，
/// 而机制本身可以穷尽验证。
#[test]
fn poison_blocks_every_later_call() {
    let mut p = Poison::default();
    assert!(!p.is_poisoned());
    assert!(p.check().is_ok());

    p.mark();
    assert!(p.is_poisoned());
    assert_eq!(p.check().unwrap_err(), AMIKI_ERROR_PANIC);
    // 重复标记是幂等的
    p.mark();
    assert_eq!(p.check().unwrap_err(), AMIKI_ERROR_PANIC);
}

#[test]
fn a_poisoned_canvas_refuses_to_work() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 8, 8, &mut cv), AMIKI_OK);

    // SAFETY: 指针来自 amiki_canvas_create
    unsafe { (*cv).poison.mark() };

    let red = amiki_paint_solid(amiki_color_rgb8(255, 0, 0));
    assert_eq!(amiki_canvas_draw_rect(cv, 0.0, 0.0, 8.0, 8.0, red), AMIKI_ERROR_PANIC);
    assert_eq!(amiki_canvas_clear(cv), AMIKI_ERROR_PANIC);
    let mut out = vec![0u8; 8 * 8 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_ERROR_PANIC);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 后端
// ============================================================================

#[test]
fn cpu_backend_is_always_available() {
    let (ctx, r) = cpu_context();
    assert_eq!(r, AMIKI_OK);
    assert_eq!(amiki_context_backend(ctx), AMIKI_BACKEND_CPU);

    // SAFETY: 指针来自 amiki_context_create
    let name = unsafe { std::ffi::CStr::from_ptr(amiki_context_adapter_name(ctx)) };
    assert_eq!(name.to_str().unwrap(), "CPU");

    // 非法后端值
    let mut c2: *mut AmikiContext = ptr::null_mut();
    assert_eq!(amiki_context_create(99, &mut c2), AMIKI_ERROR_INVALID_ARGUMENT);
    assert!(c2.is_null());

    amiki_context_destroy(ctx);
}

/// `AUTO` 在**没有 GPU** 的机器上必须落到 CPU，而不是失败 ——
/// 这是 header 里对 `AMIKI_BACKEND_AUTO` 的承诺。
#[test]
fn auto_backend_falls_back_to_cpu_when_gpu_is_unavailable() {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    let r = amiki_context_create(AMIKI_BACKEND_AUTO, &mut ctx);
    assert_eq!(r, AMIKI_OK, "AUTO 不应失败");
    let b = amiki_context_backend(ctx);
    assert!(
        b == AMIKI_BACKEND_GPU || b == AMIKI_BACKEND_CPU,
        "AUTO 必须落到某个具体后端，实际 {b}"
    );
    amiki_context_destroy(ctx);
}
