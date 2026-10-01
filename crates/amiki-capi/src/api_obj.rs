//! 对象句柄的导出函数：上下文、画布、路径、资源。
//!
//! 每个函数都以 `crate::api::ffi` / `crate::api::ffi_free` 宏包裹 ——
//! 空指针、中毒、panic 三件事在那里统一处理，这里只写正常路径。

use crate::api::{ffi, ffi_free, write_out};
use crate::objects::*;
use crate::types::*;

use amiki_scene::{GradientKind, ImagePlacement, Layer};
use std::ffi::{CStr, CString};
use std::sync::Arc;

// ============================================================================
// 后端选择（`gpu` feature 分派）
// ============================================================================

/// 把 C 侧的 `amiki_backend` 解析成一个真正可用的后端。
///
/// # `gpu` feature 关掉时（WASM 构建）
///
/// `AMIKI_BACKEND_GPU` **显式失败**（`AMIKI_ERROR_BACKEND_UNAVAILABLE`），
/// 而不是静默给一个 CPU 后端 —— 调用方明确点名要 GPU，就得让它知道拿不到。
/// `AUTO` 则正常落到 CPU（那正是 `AUTO` 的语义）。
#[allow(clippy::type_complexity)]
fn make_backend(backend: i32) -> Result<(Backend, CString), i32> {
    match backend {
        AMIKI_BACKEND_CPU => Ok((Backend::Cpu, c"CPU".to_owned())),

        #[cfg(feature = "gpu")]
        AMIKI_BACKEND_GPU => match pollster::block_on(amiki_gpu::GpuRenderer::new()) {
            // 先取名字再 move —— `Box::new(r)` 会把 `r` 拿走，
            // 同一个表达式里再用 `&r.adapter_name` 是借用已移动的值。
            Ok(r) => {
                let name = adapter_cstring(&r.adapter_name);
                Ok((Backend::Gpu(Box::new(r)), name))
            }
            Err(_) => Err(AMIKI_ERROR_BACKEND_UNAVAILABLE),
        },
        #[cfg(not(feature = "gpu"))]
        AMIKI_BACKEND_GPU => Err(AMIKI_ERROR_BACKEND_UNAVAILABLE),

        #[cfg(feature = "gpu")]
        AMIKI_BACKEND_AUTO => match pollster::block_on(amiki_gpu::GpuRenderer::new()) {
            Ok(r) => {
                let name = adapter_cstring(&r.adapter_name);
                Ok((Backend::Gpu(Box::new(r)), name))
            }
            // 拿不到 GPU 就落 CPU —— 这是 AUTO 的承诺
            Err(_) => Ok((Backend::Cpu, c"CPU".to_owned())),
        },
        #[cfg(not(feature = "gpu"))]
        AMIKI_BACKEND_AUTO => Ok((Backend::Cpu, c"CPU".to_owned())),

        _ => Err(AMIKI_ERROR_INVALID_ARGUMENT),
    }
}

/// `CString::new` 只在适配器名里有内嵌 NUL 时失败 —— 那种名字不可信，
/// 退回 `"GPU"` 而不是让整个创建失败。
#[cfg(feature = "gpu")]
fn adapter_cstring(name: &str) -> CString {
    CString::new(name).unwrap_or_else(|_| c"GPU".to_owned())
}

// ============================================================================
// 上下文
// ============================================================================

/// 创建上下文。`AUTO` 先试 GPU，失败则落到 CPU；`GPU` 失败返回
/// `AMIKI_ERROR_BACKEND_UNAVAILABLE`；`CPU` 强制 CPU。
///
/// 失败时 `*out` **不被写入**（header 承诺）。
#[no_mangle]
pub extern "C" fn amiki_context_create(backend: i32, out: *mut *mut AmikiContext) -> i32 {
    ffi_free!({
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let (b, name) = match make_backend(backend) {
            Ok(v) => v,
            Err(r) => return r,
        };
        let ctx = Box::new(AmikiContext { backend: b, adapter_name: name, poison: Poison::default() });
        // SAFETY: `out` 已判空且调用方保证可写
        unsafe { *out = Box::into_raw(ctx) };
        AMIKI_OK
    })
}

/// 销毁上下文。传 NULL 是空操作。
///
/// 注意：**其下所有 canvas 必须先销毁** —— 本函数不追踪它们。
#[no_mangle]
pub extern "C" fn amiki_context_destroy(ctx: *mut AmikiContext) {
    if ctx.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_context_create` 且未被销毁。
    drop(unsafe { Box::from_raw(ctx) });
}

/// 实际生效的后端。传 NULL 返回 `AMIKI_BACKEND_AUTO`。
#[no_mangle]
pub extern "C" fn amiki_context_backend(ctx: *const AmikiContext) -> i32 {
    if ctx.is_null() {
        return AMIKI_BACKEND_AUTO;
    }
    // SAFETY: 已判空
    match unsafe { &*ctx }.backend {
        #[cfg(feature = "gpu")]
        Backend::Gpu(_) => AMIKI_BACKEND_GPU,
        Backend::Cpu => AMIKI_BACKEND_CPU,
    }
}

/// 适配器名（静态字符串，**不可释放**）。传 NULL 返回 `"(null)"`。
#[no_mangle]
pub extern "C" fn amiki_context_adapter_name(ctx: *const AmikiContext) -> *const std::ffi::c_char {
    if ctx.is_null() {
        return c"(null)".as_ptr();
    }
    // SAFETY: 已判空；返回的指针指向 context 自己持有的 `CString`，
    // 因此只要 context 活着就一直有效（header 承诺"永远有效"，
    // 实际语义是"至少和 context 一样久"）。
    unsafe { (*ctx).adapter_name.as_ptr() }
}

// ============================================================================
// 画布
// ============================================================================

#[no_mangle]
pub extern "C" fn amiki_canvas_create(
    ctx: *mut AmikiContext,
    width: u32,
    height: u32,
    out: *mut *mut AmikiCanvas,
) -> i32 {
    ffi_free!({
        if ctx.is_null() || out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        if width == 0 || height == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let cv = Box::new(AmikiCanvas {
            ctx,
            width,
            height,
            eps: 0.1,
            root: amiki::Canvas::new(width, height),
            layers: Vec::new(),
            clip_depth: 0,
            poison: Poison::default(),
        });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(cv) };
        AMIKI_OK
    })
}

/// 销毁画布。传 NULL 是空操作。
#[no_mangle]
pub extern "C" fn amiki_canvas_destroy(canvas: *mut AmikiCanvas) {
    if canvas.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_canvas_create` 且未被销毁
    drop(unsafe { Box::from_raw(canvas) });
}

/// 清空内容与状态（变换、混合、裁剪栈、层栈），复用内部缓冲。
#[no_mangle]
pub extern "C" fn amiki_canvas_clear(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        c.reset_state();
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_size(
    canvas: *const AmikiCanvas,
    out_w: *mut u32,
    out_h: *mut u32,
) -> i32 {
    if canvas.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let c = unsafe { &*canvas };
    if let Err(r) = c.poison.check() {
        return r;
    }
    // SAFETY: 出参可为 NULL（header 写明）
    unsafe {
        write_out(out_w, c.width);
        write_out(out_h, c.height);
    }
    AMIKI_OK
}

// ---- 状态 ----

#[no_mangle]
pub extern "C" fn amiki_canvas_set_transform(canvas: *mut AmikiCanvas, t: AmikiAffine) -> i32 {
    ffi!(canvas, |c| {
        if !t.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().set_transform(t.to_affine());
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_reset_transform(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        c.target().reset();
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_translate(canvas: *mut AmikiCanvas, dx: f32, dy: f32) -> i32 {
    ffi!(canvas, |c| {
        if !dx.is_finite() || !dy.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().translate(amiki_math::Vec2::new(dx, dy));
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_rotate(canvas: *mut AmikiCanvas, radians: f32) -> i32 {
    ffi!(canvas, |c| {
        if !radians.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().rotate(radians);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_scale(canvas: *mut AmikiCanvas, s: f32) -> i32 {
    ffi!(canvas, |c| {
        if !s.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().scale(s);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_set_blend(canvas: *mut AmikiCanvas, blend: i32) -> i32 {
    ffi!(canvas, |c| {
        let Some(b) = blend_from_i32(blend) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        c.target().set_blend(b);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_push_clip(
    canvas: *mut AmikiCanvas,
    path: *const AmikiPath,
) -> i32 {
    if path.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        // SAFETY: 已判空；契约要求指针有效
        let p = unsafe { &*path };
        if let Err(r) = p.poison.check() {
            return r;
        }
        c.target().push_clip(&p.path);
        c.clip_depth += 1;
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_pop_clip(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        // `Canvas::pop_clip` 在栈空时是静默空操作 —— 那会让"压了没弹/弹多了"
        // 这类错误无声通过。C ABI 这一层显式拒绝。
        if c.clip_depth == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().pop_clip();
        c.clip_depth -= 1;
        AMIKI_OK
    })
}

// ---- 绘制 ----

/// 公共绘制外壳：把 paint 解析成引擎类型，然后交给 `f`。
macro_rules! with_paint {
    ($c:expr, $paint:expr, $f:expr) => {{
        // SAFETY: 契约要求 paint 里的句柄有效
        let Some(p) = (unsafe { paint_from_raw($paint) }) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        $f(p)
    }};
}

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
            c.target().text(&run, amiki_math::Vec2::new(x, y), paint, |id| {
                font_ref.outline(id).ok()
            });
            AMIKI_OK
        })
    })
}

// ---- 离屏层 ----

#[no_mangle]
pub extern "C" fn amiki_canvas_begin_layer(
    canvas: *mut AmikiCanvas,
    bounds: AmikiLayerBounds,
    filter: AmikiFilter,
    opacity: f32,
    blend: i32,
) -> i32 {
    ffi!(canvas, |c| {
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let Some(blend) = blend_from_i32(blend) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let Some(filter) = filter.to_filter() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 层里再开层会让"谁在谁里面"变得难以推理，而 C 侧没有作用域语法兜底。
        // 这里放开嵌套（引擎支持），但深度设上限以免调用方写错时无限增长。
        const MAX_LAYER_DEPTH: usize = 32;
        if c.layers.len() >= MAX_LAYER_DEPTH {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let (w, h) = (c.width, c.height);
        let mut sub = amiki::Canvas::new(w, h);
        sub.set_blend(blend);
        c.layers.push(PendingLayer {
            canvas: sub,
            bounds: layer_bounds_from_raw(bounds),
            filter,
            opacity,
            blend,
        });
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_end_layer(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        let Some(l) = c.layers.pop() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 把子场景作为一个 layer 挂到**当前目标**（可能是更外层）上。
        // 这里用 `Scene::push_layer` 而不是门面的 `push_layer_filtered`：
        // 后者接收闭包，而 C 侧没有闭包 —— 绘制是在 begin/end 之间发生的。
        let target = c.target();
        target.scene.push_layer(
            Layer {
                bounds: l.bounds,
                scene: l.canvas.scene,
                filter: l.filter,
                opacity: l.opacity,
                blend: l.blend,
            },
            |_| {},
        );
        AMIKI_OK
    })
}

// ---- 输出 ----

#[no_mangle]
pub extern "C" fn amiki_canvas_render_rgba8(
    canvas: *mut AmikiCanvas,
    out: *mut u8,
    out_len: usize,
) -> i32 {
    ffi!(canvas, |c| {
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let need = (c.width as usize) * (c.height as usize) * 4;
        if out_len < need {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // 打开中的层没配对的 end —— 直接渲染会**静默丢掉**那些绘制。
        // 这正是本项目最不能接受的一类错误，所以显式拒绝。
        if !c.layers.is_empty() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let eps = c.eps;
        let scene = &c.root.scene;
        let ctx = c.ctx;
        if ctx.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: context 必须比 canvas 活得久（header 契约）
        let ctx = unsafe { &*ctx };
        if let Err(r) = ctx.poison.check() {
            return r;
        }
        let rendered: Result<Vec<u8>, String> = match &ctx.backend {
            #[cfg(feature = "gpu")]
            Backend::Gpu(r) => r.render_blocking(scene, eps).map_err(|e| e.to_string()),
            // CPU 路径与 GPU 无关 —— 这也是 WASM 构建下唯一的路径
            Backend::Cpu => Ok(amiki_raster::render(&amiki_raster::bin(scene, eps))),
        };
        match rendered {
            Ok(px) if px.len() >= need => {
                // SAFETY: need <= out_len，且 out 非空
                unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), out, need) };
                AMIKI_OK
            }
            Ok(_) => AMIKI_ERROR_UNKNOWN,
            // GPU 后端的"不支持"要单独报出来（例如 GL 遇到位图），
            // 让调用方能区分"引擎坏了"与"这个后端做不了"。
            Err(msg) => {
                if msg.contains("不支持") || msg.contains("无法") {
                    AMIKI_ERROR_NOT_SUPPORTED
                } else {
                    AMIKI_ERROR_UNKNOWN
                }
            }
        }
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_set_flatten_tolerance(
    canvas: *mut AmikiCanvas,
    eps: f32,
) -> i32 {
    ffi!(canvas, |c| {
        if !eps.is_finite() || eps <= 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.eps = eps;
        AMIKI_OK
    })
}

// ============================================================================
// 路径
// ============================================================================

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

#[no_mangle]
pub extern "C" fn amiki_image_create_rgba8(
    pixels: *const u8,
    width: u32,
    height: u32,
    stride: usize,
    out: *mut *mut AmikiImage,
) -> i32 {
    ffi_free!({
        if pixels.is_null() || out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        if width == 0 || height == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let row = (width as usize) * 4;
        let stride = if stride == 0 { row } else { stride };
        if stride < row {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let need = stride * (height as usize - 1) + row;
        // SAFETY: 调用方保证 [pixels, pixels+need) 可读
        let src = unsafe { std::slice::from_raw_parts(pixels, need) };
        // 逐行拷贝：`stride` 可能大于 `row`，不能整块 memcpy。
        let mut data = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            let off = y * stride;
            data.extend_from_slice(&src[off..off + row]);
        }
        let Some(img) = amiki_scene::Image::from_rgba8(width, height, data) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiImage { image: Arc::new(img), poison: Poison::default() });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_image_destroy(image: *mut AmikiImage) {
    if image.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_image_create_rgba8` 且未被销毁
    drop(unsafe { Box::from_raw(image) });
}

#[no_mangle]
pub extern "C" fn amiki_image_size(
    image: *const AmikiImage,
    out_w: *mut u32,
    out_h: *mut u32,
) -> i32 {
    if image.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let h = unsafe { &*image };
    if let Err(r) = h.poison.check() {
        return r;
    }
    // SAFETY: 出参可为 NULL
    unsafe {
        write_out(out_w, h.image.width);
        write_out(out_h, h.image.height);
    }
    AMIKI_OK
}

#[no_mangle]
pub extern "C" fn amiki_font_create_from_file(
    utf8_path: *const std::ffi::c_char,
    out: *mut *mut AmikiFont,
) -> i32 {
    ffi_free!({
        if utf8_path.is_null() || out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 调用方保证以 \0 结尾
        let Ok(s) = unsafe { CStr::from_ptr(utf8_path) }.to_str() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // `Font::from_file` 会把文件读进内存，所以文件句柄不需要调用方保留。
        let Ok(font) = amiki_text::Font::from_file(s) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiFont { font, poison: Poison::default() });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_font_create_from_memory(
    data: *const u8,
    len: usize,
    out: *mut *mut AmikiFont,
) -> i32 {
    ffi_free!({
        if data.is_null() || out.is_null() || len == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 调用方保证 [data, data+len) 可读
        let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        let Ok(font) = amiki_text::Font::from_bytes(bytes) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiFont { font, poison: Poison::default() });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_font_destroy(font: *mut AmikiFont) {
    if font.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_font_create_*` 且未被销毁
    drop(unsafe { Box::from_raw(font) });
}

#[no_mangle]
pub extern "C" fn amiki_font_metrics(
    font: *const AmikiFont,
    out_ascent: *mut f32,
    out_descent: *mut f32,
    out_line_gap: *mut f32,
) -> i32 {
    if font.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let h = unsafe { &*font };
    if let Err(r) = h.poison.check() {
        return r;
    }
    let Ok(m) = h.font.metrics() else {
        return AMIKI_ERROR_UNKNOWN;
    };
    // SAFETY: 三个出参都可为 NULL
    unsafe {
        write_out(out_ascent, m.ascent);
        write_out(out_descent, m.descent);
        write_out(out_line_gap, m.line_gap);
    }
    AMIKI_OK
}

// ---- 渐变 ----

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

/// 供测试与内部使用：把 `ImagePlacement::rect` 暴露成同一个构造路径。
#[allow(dead_code)]
pub(crate) fn rect_placement(x0: f32, y0: f32, x1: f32, y1: f32) -> ImagePlacement {
    ImagePlacement::rect(x0, y0, x1, y1)
}
