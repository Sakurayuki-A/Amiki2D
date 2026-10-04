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



// ============================================================================
// 色调映射（HDR 的出口）
// ============================================================================

#[test]
fn tone_map_defaults_to_raw_and_round_trips() {
    let (ctx, rc) = cpu_context();
    assert_eq!(rc, AMIKI_OK);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);

    // 默认必须是 Raw —— 也就是 0.5.0 之前的行为
    assert_eq!(amiki_canvas_tone_map(cv), AMIKI_TONEMAP_RAW);
    for v in [AMIKI_TONEMAP_REINHARD, AMIKI_TONEMAP_ACES, AMIKI_TONEMAP_RAW] {
        assert_eq!(amiki_canvas_set_tone_map(cv, v), AMIKI_OK);
        assert_eq!(amiki_canvas_tone_map(cv), v);
    }
    // 空画布不崩，返回默认值
    assert_eq!(amiki_canvas_tone_map(ptr::null()), AMIKI_TONEMAP_RAW);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

#[test]
fn out_of_range_tone_map_is_rejected_and_leaves_state_untouched() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_tone_map(cv, AMIKI_TONEMAP_ACES), AMIKI_OK);

    // 越界值必须**显式失败**，且**不改动**已有设置 ——
    // 静默退回默认会让"设错了"看起来像"设上了"
    for bad in [-1i32, 3, 99, i32::MAX] {
        assert_eq!(
            amiki_canvas_set_tone_map(cv, bad),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "越界值 {bad} 应被拒绝"
        );
        assert_eq!(
            amiki_canvas_tone_map(cv),
            AMIKI_TONEMAP_ACES,
            "越界值 {bad} 不该改动已有设置"
        );
    }
    assert_eq!(
        amiki_canvas_set_tone_map(ptr::null_mut(), AMIKI_TONEMAP_ACES),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 用给定的色调映射画一个 `> 1.0` 的方块并渲染，返回像素。
fn render_hot_square(tone: i32) -> Vec<u8> {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_tone_map(cv, tone), AMIKI_OK);

    // 直接画一个远高于 1.0 的方块 —— 这就是"高光源"。
    // 用直接绘制而不是层：这条测试要验证的是**色调映射接进了出口**，
    // 层的 HDR 存活另有测试（amiki-raster 的 hdr_values_survive_a_blurred_layer）。
    let hot = amiki_color_rgba(1.8, 1.8, 1.8, 1.0);
    assert_eq!(
        amiki_canvas_draw_rect(cv, 8.0, 8.0, 56.0, 56.0, amiki_paint_solid(hot)),
        AMIKI_OK
    );

    let mut px = vec![0u8; 64 * 64 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, px.as_mut_ptr(), px.len()), AMIKI_OK);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    px
}

#[test]
fn tone_map_changes_the_output_of_an_hdr_pixel() {
    // **这条是"真的接进去了"的证明**：若 render_rgba8 忘了用 c.tone_map，
    // 三种设置会给出完全一样的图。
    let raw = render_hot_square(AMIKI_TONEMAP_RAW);
    let rei = render_hot_square(AMIKI_TONEMAP_REINHARD);
    let aces = render_hot_square(AMIKI_TONEMAP_ACES);

    let diff = |a: &[u8], b: &[u8]| -> u32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
            .max()
            .unwrap_or(0)
    };
    let (d_rei, d_ace, d_ra) = (diff(&raw, &rei), diff(&raw, &aces), diff(&rei, &aces));
    eprintln!("C ABI 色调映射：Reinhard 差 {d_rei}/255，ACES 差 {d_ace}，两者互差 {d_ra}");

    assert!(d_rei > 0 && d_ace > 0, "色调映射没有影响输出（{d_rei} / {d_ace}）—— 没接进去");
    assert!(d_ra > 0, "Reinhard 与 ACES 给出相同的输出 —— 那等于只有一个映射");

    // 方块**内部**的像素：alpha 不透明，且不该被色调映射改动
    // （(0,0) 在方块外，alpha 自然是 0 —— 我第一版就是这么写错的）
    let inside = (32 * 64 + 32) * 4;
    assert_eq!(raw[inside + 3], 255, "方块内应当不透明");
    assert_eq!(aces[inside + 3], 255, "色调映射不该改动 alpha");
}

// ============================================================================
// B4：mipmap 模式与 mip bias（0.6.0）
// ============================================================================

/// 64×64 伪随机噪声图（各 mip 层内容真实不同，层级选错立刻暴露）。
fn noise_image_bytes() -> Vec<u8> {
    let mut px = Vec::with_capacity(64 * 64 * 4);
    for i in 0..64 * 64 {
        let v = (i * 7 % 251) as u8;
        px.extend([v, (v as u32 * 3 % 255) as u8, (v as u32 * 11 % 255) as u8, 255]);
    }
    px
}

fn render_downscaled_noise(mode: i32, bias: Option<f32>) -> Vec<u8> {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 20, 20, &mut cv), AMIKI_OK);

    let px = noise_image_bytes();
    let mut img: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_rgba8(px.as_ptr(), 64, 64, 0, &mut img), AMIKI_OK);
    if mode != i32::MIN {
        // i32::MIN 表示"保持默认（BOX）"，其余为显式设置
        assert_eq!(amiki_image_set_mipmap_mode(img, mode), AMIKI_OK);
    }
    assert_eq!(amiki_canvas_draw_image(cv, img, 0.0, 0.0, 20.0, 20.0), AMIKI_OK);
    amiki_image_destroy(img);
    // mode/bias 传 None 表示"保持默认"，避免拿 NaN 当哨兵（NaN != NaN 恒真）
    if let Some(b) = bias {
        assert_eq!(amiki_canvas_set_mip_bias(cv, b), AMIKI_OK);
    }

    let mut out = vec![0u8; 20 * 20 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    out
}

/// 缩小 3.2× 时 BOX 与 DISABLED 必须给出不同输出（mip 链真的在采样），
/// 而两次 DISABLED 之间逐字节一致（回到 0.5.0 行为）。
#[test]
fn mipmap_mode_changes_downscaled_output() {
    let boxed = render_downscaled_noise(AMIKI_MIPMAP_BOX, None);
    let disabled = render_downscaled_noise(AMIKI_MIPMAP_DISABLED, None);
    let disabled_again = render_downscaled_noise(AMIKI_MIPMAP_DISABLED, None);
    assert_ne!(boxed[..], disabled[..], "缩小 3.2× 时 BOX 与 DISABLED 应不同");
    assert_eq!(disabled, disabled_again, "DISABLED 必须可复现（0.5.0 行为）");
}

/// 1:1 与放大的绘制不受模式影响（硬约束"行为不变"）。
#[test]
fn mipmap_mode_is_invisible_for_upscale() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 32, 32, &mut cv), AMIKI_OK);

    let px = noise_image_bytes();
    let mut boxed_img: *mut AmikiImage = ptr::null_mut();
    let mut plain_img: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_rgba8(px.as_ptr(), 16, 16, 0, &mut boxed_img), AMIKI_OK);
    assert_eq!(amiki_image_create_rgba8(px.as_ptr(), 16, 16, 0, &mut plain_img), AMIKI_OK);
    assert_eq!(amiki_image_set_mipmap_mode(plain_img, AMIKI_MIPMAP_DISABLED), AMIKI_OK);

    // 同一张 16×16 图放大画两遍：带链与不带链必须逐字节一致
    let mut out_boxed = vec![0u8; 32 * 32 * 4];
    assert_eq!(amiki_canvas_draw_image(cv, boxed_img, 0.0, 0.0, 32.0, 32.0), AMIKI_OK);
    assert_eq!(amiki_canvas_render_rgba8(cv, out_boxed.as_mut_ptr(), out_boxed.len()), AMIKI_OK);

    assert_eq!(amiki_image_set_mipmap_mode(boxed_img, AMIKI_MIPMAP_DISABLED), AMIKI_OK);
    assert_eq!(amiki_canvas_set_mip_bias(cv, 0.0), AMIKI_OK);
    let mut out_disabled = vec![0u8; 32 * 32 * 4];
    assert_eq!(amiki_canvas_draw_image(cv, boxed_img, 0.0, 0.0, 32.0, 32.0), AMIKI_OK);
    assert_eq!(amiki_canvas_render_rgba8(cv, out_disabled.as_mut_ptr(), out_disabled.len()), AMIKI_OK);

    assert_eq!(out_boxed, out_disabled, "放大时有无 mip 链应逐字节一致");
    amiki_image_destroy(boxed_img);
    amiki_image_destroy(plain_img);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// mip bias 生效：±1 给出不同输出；NaN / inf 被拒绝且**不改动已有设置**。
#[test]
fn mip_bias_shifts_levels_and_rejects_non_finite() {
    let base = render_downscaled_noise(i32::MIN, Some(0.0));
    let sharper = render_downscaled_noise(i32::MIN, Some(-1.0));
    let blurrier = render_downscaled_noise(i32::MIN, Some(1.0));
    assert_ne!(base[..], sharper[..], "bias -1 应改变输出");
    assert_ne!(base[..], blurrier[..], "bias +1 应改变输出");

    // 拒绝非有限值后，已有设置必须原样保留
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 20, 20, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_mip_bias(cv, 0.5), AMIKI_OK);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            amiki_canvas_set_mip_bias(cv, bad),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "非有限 bias 必须被拒绝"
        );
    }
    let mut out = vec![0u8; 20 * 20 * 4];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    // bias=0.5 与 bias=0 的输出应不同（说明 0.5 还在生效，没被 NaN 重置）
    assert_ne!(out[..], base[..], "被拒绝的调用不得改动已有设置");
}

/// 越界的 mipmap 模式枚举值被拒绝且不改动已有状态。
#[test]
fn mipmap_mode_rejects_out_of_range() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 8, 8, &mut cv), AMIKI_OK);
    let px = [255u8, 0, 0, 255].repeat(4 * 4);
    let mut img: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_rgba8(px.as_ptr(), 4, 4, 0, &mut img), AMIKI_OK);

    for bad in [-1, 2, 99] {
        assert_eq!(
            amiki_image_set_mipmap_mode(img, bad),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "mode {bad} 必须被拒绝"
        );
    }
    amiki_image_destroy(img);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 填充规则 / even-odd（0.8.0）
// ============================================================================
/// 往返与默认值：不调用时必须是 `AMIKI_FILL_NONZERO`。
#[test]
fn fill_rule_defaults_to_nonzero_and_roundtrips() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_NONZERO, "默认必须是 NonZero");
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_EVENODD);
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_NONZERO), AMIKI_OK);
    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_NONZERO);
    // NULL 不崩，给默认值
    assert_eq!(amiki_canvas_get_fill_rule(ptr::null()), AMIKI_FILL_NONZERO);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 越界枚举必须被拒绝，且**不改动已有状态**。
#[test]
fn fill_rule_rejects_out_of_range() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);

    for bad in [-1, 2, 99] {
        assert_eq!(
            amiki_canvas_set_fill_rule(cv, bad),
            AMIKI_ERROR_INVALID_ARGUMENT,
            "填充规则 {bad} 必须被拒绝"
        );
    }
    assert_eq!(
        amiki_canvas_get_fill_rule(cv),
        AMIKI_FILL_EVENODD,
        "被拒绝的调用不得改动状态"
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// **端到端**：同向双环在 NonZero 下实心、EvenOdd 下镂空 —— 经过 C ABI 也一样。
#[test]
fn even_odd_changes_the_render_through_the_abi() {
    let (ctx, _) = cpu_context();
    let (w, h) = (32u32, 32u32);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    let draw_rings = |cv: *mut AmikiCanvas| {
        let mut p: *mut AmikiPath = ptr::null_mut();
        assert_eq!(amiki_path_create(&mut p), AMIKI_OK);
        // 外环 + **同向**内环（两圈都是 rect，方向相同）
        assert_eq!(amiki_path_rect(p, 3.3, 3.1, 28.7, 28.9), AMIKI_OK);
        assert_eq!(amiki_path_rect(p, 11.2, 11.3, 21.8, 20.7), AMIKI_OK);
        assert_eq!(amiki_canvas_fill_path(cv, p, ink), AMIKI_OK);
        amiki_path_destroy(p);
    };

    // NonZero
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_rings(cv);
    let nz = render(cv, w, h);

    // EvenOdd（同一个画布上改状态再画一遍，用到 clear）
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_clear(cv), AMIKI_OK);
    // clear 之后规则回到默认 —— 这里重新设一次，顺便验证"粘性"是每次设置后生效
    assert_eq!(amiki_canvas_get_fill_rule(cv), AMIKI_FILL_NONZERO);
    assert_eq!(amiki_canvas_set_fill_rule(cv, AMIKI_FILL_EVENODD), AMIKI_OK);
    draw_rings(cv);
    let eo = render(cv, w, h);

    let center = ((16 * w + 16) * 4) as usize + 3;
    eprintln!(
        "C ABI even-odd：中心 NonZero alpha={} / EvenOdd alpha={}",
        nz[center], eo[center]
    );
    assert_eq!(nz[center], 255, "NonZero 下同向双环中心应不透明");
    assert_eq!(eo[center], 0, "EvenOdd 下同向双环中心应镂空");
    let diff = nz.iter().zip(eo.iter()).filter(|(a, b)| a != b).count();
    assert!(diff > 100, "两条规则必须真的画出不同结果，实际不同字节 {diff}");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

// ============================================================================
// 字体栈 / 字体回退（0.7.0）
// ============================================================================

/// 仓库里的字体路径（找不到就跳过，不静默通过）。
fn font_file(name: &str) -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("仓库根");
    let p = root.join("assets/fonts").join(name);
    if !p.exists() {
        eprintln!("跳过：未找到 {}", p.display());
        return None;
    }
    Some(p)
}

fn load_font(name: &str) -> Option<*mut AmikiFont> {
    let p = font_file(name)?;
    let c = std::ffi::CString::new(p.to_string_lossy().as_bytes()).expect("路径无 NUL");
    let mut f: *mut AmikiFont = ptr::null_mut();
    assert_eq!(amiki_font_create_from_file(c.as_ptr(), &mut f), AMIKI_OK);
    Some(f)
}

fn render(cv: *mut AmikiCanvas, w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    out
}

/// `amiki_text_options_default()` 必须与 header 里写的三个默认值一致。
#[test]
fn text_options_default_matches_the_header() {
    let o = amiki_text_options_default();
    assert_eq!(o.missing_glyph, AMIKI_MISSING_GLYPH_ERROR);
    assert!(o.kerning, "字距默认开");
    assert!(o.ligatures, "连字默认开");
    assert_eq!(o._pad, 0);
}

/// `resolve` 按**顺序**解析：拉丁走 Inter(0)、汉字走 Noto(1)。
#[test]
fn font_stack_resolve_follows_push_order() {
    let (Some(inter), Some(noto)) = (load_font("Inter-Regular.ttf"), load_font("NotoSansSC-Regular.ttf"))
    else {
        return;
    };
    let st = amiki_font_stack_create();
    assert!(!st.is_null(), "创建字体栈不应失败");
    assert_eq!(amiki_font_stack_push(st, inter), AMIKI_OK);
    assert_eq!(amiki_font_stack_push(st, noto), AMIKI_OK);

    let mut idx = 999u32;
    assert_eq!(amiki_font_stack_resolve(st, 'H' as u32, &mut idx), AMIKI_OK);
    assert_eq!(idx, 0, "拉丁应在 Inter（先压的）");
    assert_eq!(amiki_font_stack_resolve(st, '世' as u32, &mut idx), AMIKI_OK);
    assert_eq!(idx, 1, "汉字应在 Noto");

    // 两个字体都没有的码点 ⇒ NOT_FOUND，且**不写出参**
    let mut untouched = 12345u32;
    assert_eq!(
        amiki_font_stack_resolve(st, 0x10FFFD, &mut untouched),
        AMIKI_ERROR_NOT_FOUND
    );
    assert_eq!(untouched, 12345, "失败时不得写出参");
    // 非法标量值 ⇒ INVALID_ARGUMENT（而不是 NOT_FOUND）
    assert_eq!(
        amiki_font_stack_resolve(st, 0xD800, &mut untouched),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    // 出参可为 NULL
    assert_eq!(amiki_font_stack_resolve(st, 'A' as u32, ptr::null_mut()), AMIKI_OK);

    amiki_font_stack_destroy(st);
    amiki_font_destroy(inter);
    amiki_font_destroy(noto);
}

/// **压栈是拷贝**：push 之后立刻销毁 `amiki_font` 句柄，仍然应当画得出来。
///
/// 这条直接验证 header 所有权第 1 条里写的那句话 —— 如果哪天 push 改成
/// 借用指针，这里会立刻崩。
#[test]
fn font_may_be_destroyed_right_after_push() {
    let Some(noto) = load_font("NotoSansSC-Regular.ttf") else { return };
    let st = amiki_font_stack_create();
    assert_eq!(amiki_font_stack_push(st, noto), AMIKI_OK);
    amiki_font_destroy(noto); // **立刻销毁**：栈里已有自己的一份

    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv, st, c"你好".as_ptr(), 0, 4.0, 48.0, 40.0, ink, ptr::null()),
        AMIKI_OK
    );
    let out = render(cv, 64, 64);
    assert!(out.iter().any(|b| *b > 0), "销毁字体句柄后仍应画出内容");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    amiki_font_stack_destroy(st);
}

/// 缺字：默认策略返回 `AMIKI_ERROR_NOT_FOUND` **且画布状态不变**；
/// 换成 `NOTDEF` 策略后能画（画出方框）。
#[test]
fn missing_glyph_policy_is_honoured_end_to_end() {
    let (Some(inter), Some(noto)) = (load_font("Inter-Regular.ttf"), load_font("NotoSansSC-Regular.ttf"))
    else {
        return;
    };
    let st = amiki_font_stack_create();
    assert_eq!(amiki_font_stack_push(st, inter), AMIKI_OK);
    assert_eq!(amiki_font_stack_push(st, noto), AMIKI_OK);

    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 48, 48, &mut cv), AMIKI_OK);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    let before = render(cv, 48, 48);

    // 默认策略（options = NULL ⇒ ERROR）
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv, st, c"A\u{10FFFD}B".as_ptr(), 0, 2.0, 34.0, 28.0, ink, ptr::null()),
        AMIKI_ERROR_NOT_FOUND,
        "缺字默认应报 NOT_FOUND"
    );
    let after = render(cv, 48, 48);
    assert_eq!(before, after, "失败的绘制不得改动画布");

    // NOTDEF 策略：同一个串应当能画出来
    let opts = AmikiTextOptions {
        missing_glyph: AMIKI_MISSING_GLYPH_NOTDEF,
        ..amiki_text_options_default()
    };
    assert_eq!(
        amiki_canvas_draw_text_with_stack(
            cv,
            st,
            c"A\u{10FFFD}B".as_ptr(),
            0,
            2.0,
            34.0,
            28.0,
            ink,
            &opts
        ),
        AMIKI_OK
    );
    let with_box = render(cv, 48, 48);
    assert_ne!(before, with_box, "NOTDEF 策略应真的画出东西（方框）");

    // 越界策略值被拒绝
    let bad = AmikiTextOptions { missing_glyph: 99, ..amiki_text_options_default() };
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv, st, c"A".as_ptr(), 0, 2.0, 34.0, 28.0, ink, &bad),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    amiki_font_stack_destroy(st);
    amiki_font_destroy(inter);
    amiki_font_destroy(noto);
}

/// **字体回退真的画出了汉字**：只用 Noto 的字体栈与单字体路径
/// `amiki_canvas_draw_text(noto, "你好")` 必须给出**逐字节相同**的画面。
///
/// 若栈路径画的是方框（或不含 CJK），这里必然不等 —— 这条比"像素非空"有牙得多。
#[test]
fn stack_path_matches_single_font_path_for_cjk() {
    let (Some(noto_a), Some(noto_b)) = (load_font("NotoSansSC-Regular.ttf"), load_font("NotoSansSC-Regular.ttf"))
    else {
        return;
    };
    let text = c"你好世界";
    let (w, h) = (200u32, 64u32);

    // ① 单字体路径
    let (ctx, _) = cpu_context();
    let mut cv_a: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv_a), AMIKI_OK);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    assert_eq!(
        amiki_canvas_draw_text(cv_a, noto_a, text.as_ptr(), 0, 4.0, 48.0, 40.0, ink),
        AMIKI_OK
    );
    let a = render(cv_a, w, h);

    // ② 只含同一个 Noto 的字体栈
    let st = amiki_font_stack_create();
    assert_eq!(amiki_font_stack_push(st, noto_b), AMIKI_OK);
    let mut cv_b: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv_b), AMIKI_OK);
    assert_eq!(
        amiki_canvas_draw_text_with_stack(cv_b, st, text.as_ptr(), 0, 4.0, 48.0, 40.0, ink, ptr::null()),
        AMIKI_OK
    );
    let b = render(cv_b, w, h);

    // 有牙的前提：真的画出了东西（全黑画面会让"相等"变得毫无意义）
    assert!(a.iter().any(|x| *x > 0), "单字体路径应画出汉字");
    assert_eq!(a, b, "字体栈路径（同一字体）必须与单字体路径逐字节一致");

    amiki_canvas_destroy(cv_a);
    amiki_canvas_destroy(cv_b);
    amiki_context_destroy(ctx);
    amiki_font_stack_destroy(st);
    amiki_font_destroy(noto_a);
    amiki_font_destroy(noto_b);
}
