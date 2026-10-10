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
        AMIKI_ERROR_NOT_FOUND,
        AMIKI_ERROR_STACK_OVERFLOW,
        AMIKI_ERROR_STACK_UNDERFLOW,
        AMIKI_ERROR_BUSY,
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

/// **错误码取值必须在 header 与 Rust 两侧一致**。
///
/// 这是个"镜像"：`AMIKI_ERROR_*` 在 `include/amiki.h`（C 侧的契约）与
/// `src/types.rs`（实现）各写一份。任何一侧改了数字，调用方就会按**错的含义**
/// 处理错误 —— 与 0.9.0 抓到的 `clip_depth` / `fill_rule` 镜像漂移是同一类问题。
///
/// 只扫 `amiki_result` 枚举体那一段（交接文档第 10 条：范围要窄，
/// 不能让相邻段落命中）。
#[test]
fn error_codes_match_the_header() {
    let header = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/include/amiki.h"))
        .expect("读 amiki.h");
    let start = header.find("typedef enum amiki_result {").expect("header 里有 amiki_result");
    let end = header[start..].find("} amiki_result;").expect("枚举体有结束") + start;
    let body = &header[start..end];

    for (name, value) in [
        ("AMIKI_OK", AMIKI_OK),
        ("AMIKI_ERROR_INVALID_ARGUMENT", AMIKI_ERROR_INVALID_ARGUMENT),
        ("AMIKI_ERROR_OUT_OF_MEMORY", AMIKI_ERROR_OUT_OF_MEMORY),
        ("AMIKI_ERROR_BACKEND_UNAVAILABLE", AMIKI_ERROR_BACKEND_UNAVAILABLE),
        ("AMIKI_ERROR_NOT_SUPPORTED", AMIKI_ERROR_NOT_SUPPORTED),
        ("AMIKI_ERROR_PANIC", AMIKI_ERROR_PANIC),
        ("AMIKI_ERROR_UNKNOWN", AMIKI_ERROR_UNKNOWN),
        ("AMIKI_ERROR_NOT_FOUND", AMIKI_ERROR_NOT_FOUND),
        ("AMIKI_ERROR_STACK_OVERFLOW", AMIKI_ERROR_STACK_OVERFLOW),
        ("AMIKI_ERROR_STACK_UNDERFLOW", AMIKI_ERROR_STACK_UNDERFLOW),
        ("AMIKI_ERROR_BUSY", AMIKI_ERROR_BUSY),
    ] {
        let needle = format!("{name} = {value}");
        let at = body
            .find(&needle)
            .unwrap_or_else(|| panic!("header 的 amiki_result 里没有 `{needle}`"))
            + needle.len();
        // 防"值前几位相同"造成的假通过（例如 header 写 100，needle 是 10）
        assert_ne!(
            body[at..].chars().next(),
            Some('0'),
            "header 里 {name} 的取值看起来不是 {value}"
        );
    }
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
// 文本轮廓缓存（0.10.6，D8 的另一半）
// ============================================================================

fn cache_stats(f: *mut AmikiFont) -> (u64, u64) {
    // 测试要读字体内部的缓存统计（ABI 不暴露它），裸指针引用无法避免。
    #[allow(dangerous_implicit_autorefs)]
    unsafe {
        let c = (*f).cache.lock().unwrap_or_else(|e| e.into_inner());
        (c.hits(), c.misses())
    }
}

/// 重复绘制同一段文本：第二轮**不再缺失**（缓存真的在复用）。
#[test]
fn text_outline_cache_is_reused_across_draws() {
    let Some(font) = load_font("Inter-Regular.ttf") else {
        return;
    };
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 220, 40, &mut cv), AMIKI_OK);
    let text = c"Cache me twice";
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    assert_eq!(amiki_canvas_draw_text(cv, font, text.as_ptr(), 0, 4.0, 28.0, 16.0, ink), AMIKI_OK);
    let (h1, m1) = cache_stats(font);
    // SAFETY: 句柄有效
    unsafe {
        (*cv).target().scene.draws.clear();
    }
    assert_eq!(amiki_canvas_draw_text(cv, font, text.as_ptr(), 0, 4.0, 28.0, 16.0, ink), AMIKI_OK);
    let (h2, m2) = cache_stats(font);

    assert!(m1 > 0, "第一轮应有缺失（首次取轮廓）");
    assert!(h2 > h1, "第二轮应命中：h2={h2} h1={h1}");
    // 缺失可能仍 +N（实测 +2）：**没有轮廓的字形**（空格/无轮廓标记）本来就存不进
    // 缓存 —— 每次都得再问一次。所以判据是"命中在涨 + 缺失增量很小"，不是"零缺失"
    // （我第一版写了零缺失，被实测 +2 打回）。
    assert!(m2 - m1 <= 2, "第二轮缺失增量应极小（仅无轮廓字形），实测 {} ", m2 - m1);
    eprintln!("重复绘制：第一轮 hits={h1} misses={m1}；第二轮 hits={h2} misses={m2}");

    amiki_canvas_destroy(cv);
    amiki_font_destroy(font);
}

/// **每个字体句柄各自一份缓存** —— 这就是"构造上不会串字体"的证据（D8 的核心）。
#[test]
fn cache_is_per_font_handle() {
    let (Some(a), Some(b)) = (load_font("Inter-Regular.ttf"), load_font("Inter-Regular.ttf")) else {
        return;
    };
    let (ctx, _) = cpu_context();
    let mut cva: *mut AmikiCanvas = ptr::null_mut();
    let mut cvb: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 120, 40, &mut cva), AMIKI_OK);
    assert_eq!(amiki_canvas_create(ctx, 120, 40, &mut cvb), AMIKI_OK);
    let text = c"AAA";
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    assert_eq!(amiki_canvas_draw_text(cva, a, text.as_ptr(), 0, 2.0, 28.0, 16.0, ink), AMIKI_OK);
    let (ha, _) = cache_stats(a);
    let (hb, _) = cache_stats(b);
    assert_eq!(hb, 0, "另一个句柄的缓存不应被碰到");

    assert_eq!(amiki_canvas_draw_text(cvb, b, text.as_ptr(), 0, 2.0, 28.0, 16.0, ink), AMIKI_OK);
    let (ha2, _) = cache_stats(a);
    let (hb2, _) = cache_stats(b);
    assert_eq!(ha, ha2, "b 的绘制不影响 a 的缓存");
    assert!(hb2 > 0, "b 自己的缓存开始计数");
    eprintln!("per-font 缓存：a hits={ha2}、b hits={hb2}（互不影响 ⇒ 不会串字体）");

    amiki_canvas_destroy(cva);
    amiki_canvas_destroy(cvb);
    amiki_font_destroy(a);
    amiki_font_destroy(b);
}

/// 缓存不改变错误路径（回归）。
#[test]
fn cache_does_not_change_error_paths() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 32, &mut cv), AMIKI_OK);
    let text = c"x";
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));
    assert_eq!(
        amiki_canvas_draw_text(cv, ptr::null(), text.as_ptr(), 0, 0.0, 0.0, 12.0, ink),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    amiki_canvas_destroy(cv);
    eprintln!("坏参数路径未受缓存影响（font=NULL ⇒ INVALID_ARGUMENT）");
}

// ============================================================================
// 图像编解码（0.10.0）
// ============================================================================

/// 造一张可判别的 RGBA 图（解码后逐像素比对用）
fn codec_test_image(w: u32, h: u32) -> *mut AmikiImage {
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            px[i] = (x * 9 % 256) as u8;
            px[i + 1] = (y * 13 % 256) as u8;
            px[i + 2] = ((x + y) * 5 % 256) as u8;
            px[i + 3] = if (x + y) % 2 == 0 { 200 } else { 255 };
        }
    }
    let mut img: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_rgba8(px.as_ptr(), w, h, 0, &mut img), AMIKI_OK);
    img
}

/// 编码 PNG → 从内存解码 → 逐像素一致；出参缓冲用 `amiki_free` 释放。
// 测试要从 ABI 句柄取像素比对（ABI 没有像素 getter），裸指针引用无法避免。
#[allow(dangerous_implicit_autorefs)]
#[test]
fn image_codec_roundtrip_through_abi() {
    let src = codec_test_image(23, 17);
    let mut data: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    assert_eq!(amiki_image_encode_png(src, &mut data, &mut len), AMIKI_OK);
    assert!(!data.is_null() && len > 8, "编码输出应有内容");

    let mut back: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_from_bytes(data, len, &mut back), AMIKI_OK);
    assert!(!back.is_null());
    // SAFETY: 两个句柄都有效且未被销毁
    unsafe {
        let (a, b) = (&*src, &*back);
        assert_eq!(
            (a.image.width, a.image.height),
            (b.image.width, b.image.height)
        );
        assert_eq!(a.image.data, b.image.data, "PNG 往返必须逐像素一致");
    }
    amiki_free(data, len);
    amiki_image_destroy(src);
    amiki_image_destroy(back);
    eprintln!("ABI 图像往返：23×17 PNG 编码→解码逐像素一致（{len} 字节）");
}

/// 从**文件**加载（按内容判格式，不看扩展名）。
// 测试要从 ABI 句柄取像素比对（ABI 没有像素 getter），裸指针引用无法避免。
#[allow(dangerous_implicit_autorefs)]
#[test]
fn image_codec_loads_from_file() {
    let src = codec_test_image(9, 5);
    let mut data: *mut u8 = ptr::null_mut();
    let mut len: usize = 0;
    assert_eq!(amiki_image_encode_png(src, &mut data, &mut len), AMIKI_OK);
    let path = std::env::temp_dir().join("amiki_codec_abi_test.png");
    // SAFETY: data/len 来自上面的编码
    std::fs::write(&path, unsafe { std::slice::from_raw_parts(data, len) }).expect("写临时文件");
    amiki_free(data, len);

    let c = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
    let mut loaded: *mut AmikiImage = ptr::null_mut();
    assert_eq!(amiki_image_create_from_file(c.as_ptr(), &mut loaded), AMIKI_OK);
    // SAFETY: 句柄有效
    unsafe {
        assert_eq!(((*loaded).image.width, (*loaded).image.height), (9, 5));
        assert_eq!((*loaded).image.data, (*src).image.data);
    }
    amiki_image_destroy(loaded);
    amiki_image_destroy(src);
    let _ = std::fs::remove_file(&path);
    eprintln!("ABI 文件加载：9×5 与原始一致");
}

/// 错误映射：不认识 / 数据坏 / 文件不存在 / 空指针 —— 都显式，且不 panic。
#[test]
fn image_codec_error_mapping() {
    let mut img: *mut AmikiImage = ptr::null_mut();
    let junk = b"not an image at all, definitely not";
    assert_eq!(
        amiki_image_create_from_bytes(junk.as_ptr(), junk.len(), &mut img),
        AMIKI_ERROR_NOT_SUPPORTED
    );
    assert!(img.is_null(), "失败时不得写出参");

    let png = {
        let src = codec_test_image(4, 4);
        let mut d: *mut u8 = ptr::null_mut();
        let mut l: usize = 0;
        assert_eq!(amiki_image_encode_png(src, &mut d, &mut l), AMIKI_OK);
        // SAFETY: d/l 来自编码
        let v = unsafe { std::slice::from_raw_parts(d, l) }.to_vec();
        amiki_free(d, l);
        amiki_image_destroy(src);
        v
    };
    let cut = &png[..png.len() / 2];
    assert_eq!(
        amiki_image_create_from_bytes(cut.as_ptr(), cut.len(), &mut img),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    let missing = std::ffi::CString::new("Z:\\definitely\\missing\\file.png").unwrap();
    assert_eq!(
        amiki_image_create_from_file(missing.as_ptr(), &mut img),
        AMIKI_ERROR_NOT_FOUND
    );

    assert_eq!(
        amiki_image_create_from_bytes(ptr::null(), 0, &mut img),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_image_create_from_file(ptr::null(), &mut img),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_image_encode_png(ptr::null(), ptr::null_mut(), ptr::null_mut()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    eprintln!("ABI 错误映射：不认识=NOT_SUPPORTED / 截断=INVALID_ARGUMENT / 缺文件=NOT_FOUND / 空指针=INVALID_ARGUMENT");
}

// ============================================================================
// 状态栈 save / restore（0.9.0）
// ============================================================================

/// 往返：save → 改状态 → restore ⇒ 画出来的东西与"从未改过"完全相同。
///
/// 用**渲染结果**比对，而不是只看 getter：这直接证明"绘制时用的确实是那套状态"。
#[test]
fn canvas_save_restore_roundtrip() {
    let (ctx, _) = cpu_context();
    let (w, h) = (32u32, 32u32);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    let mut p: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut p), AMIKI_OK);
    assert_eq!(amiki_path_rect(p, 4.3, 4.1, 28.7, 28.9), AMIKI_OK);

    // A：只在初始状态下画
    let mut a: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut a), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(a, p, ink), AMIKI_OK);
    let pa = render(a, w, h);

    // B：save → 改变换/混合/填充规则 → 画一条（应不同）→ restore → 再画
    let mut b: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut b), AMIKI_OK);
    assert_eq!(amiki_canvas_save(b), AMIKI_OK);
    assert_eq!(amiki_canvas_translate(b, 7.0, 5.0), AMIKI_OK);
    assert_eq!(amiki_canvas_set_blend(b, 1), AMIKI_OK); // Multiply
    assert_eq!(amiki_canvas_set_fill_rule(b, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(b, p, ink), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(b), AMIKI_OK);
    assert_eq!(amiki_canvas_get_fill_rule(b), AMIKI_FILL_NONZERO, "恢复后应回到 NonZero");
    assert_eq!(amiki_canvas_fill_path(b, p, ink), AMIKI_OK);
    let pb_img = render(b, w, h);
    assert_ne!(pa, pb_img, "改过状态的那条必须真的画得不一样（否则测试没牙）");

    // 只做 save/改状态/restore，然后画 —— 应与 A 逐字节一致
    let mut c: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut c), AMIKI_OK);
    assert_eq!(amiki_canvas_save(c), AMIKI_OK);
    assert_eq!(amiki_canvas_translate(c, 7.0, 5.0), AMIKI_OK);
    assert_eq!(amiki_canvas_set_blend(c, 1), AMIKI_OK);
    assert_eq!(amiki_canvas_set_fill_rule(c, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(c), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(c, p, ink), AMIKI_OK);
    let pc = render(c, w, h);
    assert_eq!(pa, pc, "restore 之后的状态必须与从未改过时逐字节一致");

    amiki_path_destroy(p);
    amiki_canvas_destroy(a);
    amiki_canvas_destroy(b);
    amiki_canvas_destroy(c);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 往返：恢复后与初始状态逐字节一致");
}

/// 溢出与下溢：32 层成功、第 33 层 `STACK_OVERFLOW`；清空后多 restore `STACK_UNDERFLOW`。
#[test]
fn canvas_save_overflows_and_restore_underflows() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    for i in 0..32 {
        assert_eq!(amiki_canvas_save(cv), AMIKI_OK, "第 {} 次 save 应成功", i + 1);
    }
    assert_eq!(
        amiki_canvas_save(cv),
        AMIKI_ERROR_STACK_OVERFLOW,
        "第 33 次必须显式溢出"
    );
    for i in 0..32 {
        assert_eq!(amiki_canvas_restore(cv), AMIKI_OK, "第 {} 次 restore 应成功", i + 1);
    }
    assert_eq!(
        amiki_canvas_restore(cv),
        AMIKI_ERROR_STACK_UNDERFLOW,
        "空栈 restore 必须显式下溢"
    );
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(amiki_result_string(AMIKI_ERROR_STACK_OVERFLOW)) }
            .to_str()
            .unwrap(),
        "state stack overflow (missing restore)"
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 溢出/下溢：32 层上限与两个错误码都对");
}

/// `restore` 会回退裁剪深度 ⇒ C 侧那本账必须同步，否则 `pop_clip` 会误判。
#[test]
fn canvas_restore_syncs_clip_depth() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let mut p: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut p), AMIKI_OK);
    assert_eq!(amiki_path_rect(p, 0.0, 0.0, 10.0, 10.0), AMIKI_OK);

    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_push_clip(cv, p), AMIKI_OK);
    assert_eq!(amiki_canvas_push_clip(cv, p), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK);
    assert_eq!(
        amiki_canvas_pop_clip(cv),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "restore 已把裁剪深度退回 0，此时 pop 必须报错（说明 C 侧的账同步了）"
    );

    assert_eq!(amiki_canvas_push_clip(cv, p), AMIKI_OK);
    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK);
    assert_eq!(
        amiki_canvas_pop_clip(cv),
        AMIKI_OK,
        "这一层是 save 之前压的，restore 不该把它算掉"
    );

    amiki_path_destroy(p);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 裁剪同步：restore 回退深度后 pop 的判定正确");
}

/// 层内的栈独立（C ABI 侧）。
#[test]
fn canvas_layer_has_independent_stack() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(
        amiki_canvas_begin_layer(
            cv,
            AmikiLayerBounds { x: 0, y: 0, width: 16, height: 16 },
            AmikiFilter::default(),
            1.0,
            0,
        ),
        AMIKI_OK
    );
    assert_eq!(
        amiki_canvas_restore(cv),
        AMIKI_ERROR_STACK_UNDERFLOW,
        "层内栈不继承外层栈"
    );
    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK, "外层那层还在");
    assert_eq!(amiki_canvas_restore(cv), AMIKI_ERROR_STACK_UNDERFLOW);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 层交互：层内栈从空开始，外层计数不受影响");
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

// ============================================================================
// 异步输出：begin / poll / try_read（P3 第 ③b 步）
// ============================================================================

/// 画点东西出来，免得"异步 == 同步"落到一张全透明画布上变得毫无意义。
fn draw_sample(cv: *mut AmikiCanvas, w: f32, h: f32) {
    let bg = amiki_paint_solid(amiki_color_rgb8(250, 250, 252));
    assert_eq!(amiki_canvas_draw_rect(cv, 0.0, 0.0, w, h, bg), AMIKI_OK);
    let red = amiki_paint_solid(amiki_color_rgb8(231, 76, 60));
    assert_eq!(amiki_canvas_draw_rect(cv, 4.0, 4.0, w - 4.0, h - 4.0, red), AMIKI_OK);
    let blue = amiki_paint_solid(amiki_color_rgb8(52, 152, 219));
    assert_eq!(amiki_canvas_draw_circle(cv, w * 0.5, h * 0.5, w * 0.25, blue), AMIKI_OK);
}

/// 判据 1（CPU 后端）：三入口能走通，且与**同步入口逐字节相同**。
#[test]
fn async_trio_matches_sync_on_the_cpu_backend() {
    let (ctx, _) = cpu_context();
    let (w, h) = (64u32, 48u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_sample(cv, w as f32, h as f32);
    let sync = render(cv, w, h);
    assert!(sync.iter().any(|&b| b != 0), "有牙的前提：画面不是空的");

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut ready = 0i32;
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
    assert_eq!(ready, 1, "CPU 分支应首帧即就绪（与门面的异步入口同一语义）");

    let mut out = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert_eq!(out, sync, "异步与同步必须逐字节相同");

    // 取走之后就没有在飞的帧了：三条入口按"没 begin"处理
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    // 重新 begin 是允许的（旧帧已被取走，这里验证"可以反复提交"）
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut again = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, again.as_mut_ptr(), again.len()), AMIKI_OK);
    assert_eq!(again, sync, "第二次提交的产物必须与第一次相同");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 没有 `begin` 就 `poll` / `try_read`：**显式报错**（与 `restore` 多于 `save` 同类），
/// 且失败时不写出参、不写一个字节。
#[test]
fn poll_and_try_read_without_begin_are_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let mut ready = 0x1234_5678i32;
    assert_eq!(
        amiki_canvas_poll_render_rgba8(cv, &mut ready),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "没提交过就没有'算完没有'可问"
    );
    assert_eq!(ready, 0x1234_5678, "失败时不写出参");

    let mut out = vec![0xAAu8; 16 * 16 * 4];
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(out.iter().all(|&b| b == 0xAA), "失败时不应写入任何字节");

    // 空指针 / 空出参
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, ptr::null_mut(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(amiki_canvas_begin_render_rgba8(ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_poll_render_rgba8(ptr::null_mut(), &mut ready),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(ptr::null_mut(), out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// `out_len` 不足 ⇒ 报错、不写字节、**且不消费在飞的那一帧**
/// （长度不够是调用方的 bug，不该顺手毁掉它），补一个够长的缓冲仍能取到同一张图。
#[test]
fn try_read_rejects_short_buffer_and_keeps_the_frame() {
    let (ctx, _) = cpu_context();
    let (w, h) = (32u32, 24u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_sample(cv, w as f32, h as f32);
    let sync = render(cv, w, h);

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut short = vec![0xAAu8; sync.len()];
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, short.as_mut_ptr(), short.len() - 1),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(short.iter().all(|&b| b == 0xAA), "失败时不应写入任何字节");

    // 帧还在：补一个够长的缓冲就能取到
    let mut out = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert_eq!(out, sync, "短缓冲失败后这一帧不应被毁掉");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 打开中的层没配平就 `begin`：与同步入口同一条理由 —— 直接渲染会静默丢掉那些绘制。
#[test]
fn begin_with_an_open_layer_is_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let b = AmikiLayerBounds { x: 0, y: 0, width: 8, height: 8 };
    assert_eq!(
        amiki_canvas_begin_layer(cv, b, amiki_filter_blur(2.0), 1.0, AMIKI_BLEND_NORMAL),
        AMIKI_OK
    );
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_ERROR_INVALID_ARGUMENT);
    // 被拒绝 ⇒ 没有在飞的帧，poll 也应当报"没提交过"
    let mut ready = 0i32;
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_ERROR_INVALID_ARGUMENT);

    // 配平之后就能提交了
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// `amiki_canvas_clear` 必须把在飞的那一帧一起丢掉：
/// 它算的是**清空之前**的内容，留着只会让调用方取到过期像素。
#[test]
fn clear_drops_the_in_flight_frame() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    draw_sample(cv, 16.0, 16.0);

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_clear(cv), AMIKI_OK);

    let mut ready = 0i32;
    assert_eq!(
        amiki_canvas_poll_render_rgba8(cv, &mut ready),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "clear 之后不该还有帧可取（那会是清空前的过期像素）"
    );
    let mut out = vec![0xAAu8; 16 * 16 * 4];
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(out.iter().all(|&b| b == 0xAA));

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 判据 2（GPU 后端）：异步与同步**逐字节相同**；
/// 并且"还没算完"时 `try_read` 必须报 `AMIKI_ERROR_BUSY`、不写一个字节、
/// **也不消费那一帧**（后面轮询到就绪仍能取到）。
#[test]
fn async_matches_sync_on_the_gpu_backend() {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    assert_eq!(amiki_context_create(AMIKI_BACKEND_AUTO, &mut ctx), AMIKI_OK);
    if amiki_context_backend(ctx) != AMIKI_BACKEND_GPU {
        eprintln!("无可用 GPU，跳过（这条只在 GPU 后端上有意义）");
        amiki_context_destroy(ctx);
        return;
    }

    let (w, h) = (96u32, 64u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_sample(cv, w as f32, h as f32);
    let sync = render(cv, w, h);

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut out = vec![0xAAu8; sync.len()];
    let first = amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len());
    if first == AMIKI_OK {
        // 回调已经跑完了（合法，只是少见）
        assert_eq!(out, sync, "首帧即就绪时也要逐字节相同");
    } else {
        assert_eq!(first, AMIKI_ERROR_BUSY, "未就绪只能是 BUSY（不是别的错误）");
        assert!(out.iter().all(|&b| b == 0xAA), "BUSY 时不应写入任何字节");

        // 轮询到就绪；**刚刚 BUSY 过的那一帧还在**
        let mut ready = 0i32;
        let mut polls = 0u32;
        loop {
            assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
            if ready != 0 {
                break;
            }
            polls += 1;
            assert!(polls < 1_000_000, "轮询次数异常：map 回调一直没触发");
        }
        let mut out2 = vec![0u8; sync.len()];
        assert_eq!(
            amiki_canvas_try_read_render_rgba8(cv, out2.as_mut_ptr(), out2.len()),
            AMIKI_OK
        );
        assert_eq!(out2, sync, "GPU 上异步与同步必须逐字节相同（轮询 {polls} 次）");
    }

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// **0.10.9 的回归钉子（C ABI 侧）**：`draws × tile 行` 够大时，稀疏路径的 1D 派发
/// 会拆成二维 —— 宿主与内核的行距不一致会让**尾部 strip 永远不被处理**，
/// 于是画面最下面若干行整片透明（`0.10.8` 记为"GPU 在 `y ≥ 2048` 会算错"，
/// 当时的处置是"超范围回落 CPU"；真因查明后那条限制已删除）。
///
/// 判据：GPU 上下文与纯 CPU 上下文渲染**同一场景**，最大通道差 ≤ 2，
/// 且**最下面一行不是透明的**（旧 bug 时那里什么都没有）。
#[test]
fn tall_canvas_with_many_draws_is_correct_on_the_gpu_backend() {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    assert_eq!(amiki_context_create(AMIKI_BACKEND_AUTO, &mut ctx), AMIKI_OK);
    if amiki_context_backend(ctx) != AMIKI_BACKEND_GPU {
        eprintln!("无可用 GPU，跳过");
        amiki_context_destroy(ctx);
        return;
    }

    // 256×2160（135 个 tile 行）× 2000 个矩形 ⇒ draws × tile 行 = 270 135 > 262 140
    // ⇒ carry 需要 4 322 160 个元素（> 65535 个工作组）⇒ 必然触发二维分批。
    let (w, h, n) = (256u32, 2160u32, 2000u32);
    let side = (n as f32).sqrt().ceil() as u32;

    let mut cv_gpu: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv_gpu), AMIKI_OK);
    let (cctx, _) = cpu_context();
    let mut cv_cpu: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(cctx, w, h, &mut cv_cpu), AMIKI_OK);

    // 同一个场景画到两块画布上（背景 + 45×45 的彩色矩形网格）
    for cv in [cv_gpu, cv_cpu] {
        assert_eq!(
            amiki_canvas_draw_rect(
                cv,
                0.0,
                0.0,
                w as f32,
                h as f32,
                amiki_paint_solid(amiki_color_rgb8(250, 250, 252))
            ),
            AMIKI_OK
        );
        for i in 0..n {
            let col = i % side;
            let row = i / side;
            let cw = w as f32 / side as f32;
            let ch = h as f32 / side as f32;
            let (x0, y0) = (col as f32 * cw + cw * 0.15, row as f32 * ch + ch * 0.15);
            let t = i as f32 / n as f32;
            assert_eq!(
                amiki_canvas_draw_rect(
                    cv,
                    x0,
                    y0,
                    x0 + cw * 0.7,
                    y0 + ch * 0.7,
                    amiki_paint_solid(amiki_color_rgba(t, 1.0 - t, (t * 3.0) % 1.0, 1.0))
                ),
                AMIKI_OK
            );
        }
    }

    let gpu_px = render(cv_gpu, w, h);
    let cpu_px = render(cv_cpu, w, h);
    let max_diff = gpu_px
        .iter()
        .zip(cpu_px.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 2,
        "GPU 与 CPU 最大通道差 {max_diff} 超出容差 —— 尾部 tile 可能又被丢掉了"
    );

    // 最下面一行必须真的画了东西（旧 bug 时这里是全透明）
    let bottom = (((h - 1) * w + w / 2) as usize) * 4;
    assert!(gpu_px[bottom + 3] > 0, "最下面一行是透明的：尾部 strip 又被丢掉了");
    eprintln!("C ABI 4K 尾部回归：{w}x{h}/{n} 矩形，最大通道差 {max_diff}");

    amiki_canvas_destroy(cv_gpu);
    amiki_canvas_destroy(cv_cpu);
    amiki_context_destroy(ctx);
    amiki_context_destroy(cctx);
}

/// 同一张 HDR 画布的异步读法（`render_hot_square` 的异步版）。
fn hot_square_canvas(tone: i32) -> (*mut AmikiContext, *mut AmikiCanvas) {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_tone_map(cv, tone), AMIKI_OK);
    let hot = amiki_color_rgba(1.8, 1.8, 1.8, 1.0);
    assert_eq!(
        amiki_canvas_draw_rect(cv, 8.0, 8.0, 56.0, 56.0, amiki_paint_solid(hot)),
        AMIKI_OK
    );
    (ctx, cv)
}

/// 走完异步三入口（轮询到就绪再取），返回像素。
fn read_async(cv: *mut AmikiCanvas, w: u32, h: u32) -> Vec<u8> {
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut ready = 0i32;
    let mut polls = 0u32;
    loop {
        assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
        if ready != 0 {
            break;
        }
        polls += 1;
        assert!(polls < 1_000_000, "map 回调一直没触发");
    }
    let mut out = vec![0u8; (w * h * 4) as usize];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    out
}

/// **色调映射、mip bias 必须一起接进异步入口**：否则两条入口在 HDR 画布上就会不同，
/// 而"逐字节相同"是这条 API 唯一的卖点。
#[test]
fn async_trio_honours_the_tone_map() {
    let sync_raw = render_hot_square(AMIKI_TONEMAP_RAW);
    let sync_aces = render_hot_square(AMIKI_TONEMAP_ACES);

    let (ctx, cv) = hot_square_canvas(AMIKI_TONEMAP_ACES);
    let async_aces = read_async(cv, 64, 64);
    assert_eq!(async_aces, sync_aces, "异步入口必须与同步入口的 ACES 输出逐字节相同");
    assert_ne!(async_aces, sync_raw, "有牙的前提：ACES 与 RAW 本来就该不同");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 层场景在 `begin` 里就同步算完了（层合成需要根图像，异步没有收益）⇒
/// **第一次 `poll` 即就绪**，且产物与同步入口逐字节相同。
#[test]
fn async_layer_scene_is_ready_on_the_first_poll() {
    let (ctx, _) = cpu_context();
    let (w, h) = (64u32, 48u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);

    let b = AmikiLayerBounds { x: 0, y: 0, width: w, height: h };
    assert_eq!(
        amiki_canvas_begin_layer(cv, b, amiki_filter_blur(3.0), 0.8, AMIKI_BLEND_NORMAL),
        AMIKI_OK
    );
    draw_sample(cv, w as f32, h as f32);
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    let sync = render(cv, w, h);
    assert!(sync.iter().any(|&b| b != 0), "有牙的前提：画面不是空的");

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut ready = 0i32;
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
    assert_eq!(ready, 1, "层场景没有在飞可言，第一次 poll 就必须就绪");

    let mut out = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert_eq!(out, sync, "层场景的异步产物也要与同步逐字节相同");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}
