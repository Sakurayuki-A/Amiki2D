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
        AMIKI_BACKEND_GPU => match amiki_gpu::GpuRenderer::new_blocking() {
            // 先取名字再 move —— `Arc::new(r)` 会把 `r` 拿走，
            // 同一个表达式里再用 `&r.adapter_name` 是借用已移动的值。
            Ok(r) => {
                let name = adapter_cstring(&r.adapter_name);
                Ok((Backend::Gpu(Arc::new(r)), name))
            }
            Err(_) => Err(AMIKI_ERROR_BACKEND_UNAVAILABLE),
        },
        #[cfg(not(feature = "gpu"))]
        AMIKI_BACKEND_GPU => Err(AMIKI_ERROR_BACKEND_UNAVAILABLE),

        #[cfg(feature = "gpu")]
        AMIKI_BACKEND_AUTO => match amiki_gpu::GpuRenderer::new_blocking() {
            Ok(r) => {
                let name = adapter_cstring(&r.adapter_name);
                Ok((Backend::Gpu(Arc::new(r)), name))
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
            tone_map: amiki_scene::ToneMap::Raw,
            mip_bias: 0.0,
            fill_rule: amiki_scene::FillRule::NonZero,
            ctx,
            width,
            height,
            eps: 0.1,
            root: amiki::Canvas::new(width, height),
            layers: Vec::new(),
            clip_depth: 0,
            pending: None,
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
            let mut cache = f.cache.lock().unwrap_or_else(|p| p.into_inner());
                c.target().text_cached(&mut cache, &run, amiki_math::Vec2::new(x, y), paint, |id| {
                font_ref.outline(id).ok()
            });
            AMIKI_OK
        })
    })
}

/// 绘制一段 UTF-8 文本，字形按**字体栈**做回退（0.7.0）。
///
/// 与 [`amiki_canvas_draw_text`] 的区别只有一处：字体来源从单个 `amiki_font`
/// 换成有序的 `amiki_font_stack` —— 因此一段文本里拉丁与汉字可以来自不同字体。
/// 其余（基线原点、显式长度、字形走同一管线）完全相同。
#[no_mangle]
pub extern "C" fn amiki_canvas_draw_text_with_stack(
    canvas: *mut AmikiCanvas,
    stack: *const AmikiFontStack,
    utf8: *const std::ffi::c_char,
    utf8_len: usize,
    x: f32,
    y: f32,
    size: f32,
    paint: AmikiPaint,
    options: *const AmikiTextOptions,
) -> i32 {
    if stack.is_null() || utf8.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        if !x.is_finite() || !y.is_finite() || !size.is_finite() || size <= 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 已判空
        let s = unsafe { &*stack };
        if let Err(r) = s.poison.check() {
            return r;
        }
        // 选项可为 NULL ⇒ 用 header 承诺的默认值（缺字报错、字距与连字都开）
        let opts = if options.is_null() {
            AmikiTextOptions::default_options()
        } else {
            // SAFETY: 已判空
            unsafe { *options }
        };
        let Ok(text_opts) = opts.to_options() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 显式长度：0 表示"按 \0 求长"（与 amiki_canvas_draw_text 同语义）
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
        let run = match amiki_text::shape_with_stack(&s.stack, text, size, &text_opts) {
            Ok(r) => r,
            // 缺字（策略为 Error 时）是**内容**问题而不是参数问题：明确告诉调用方
            // "栈里没有这个字"，它据此可以换策略或补字体 —— 比含糊的
            // INVALID_ARGUMENT 有用得多。其余失败仍按字体数据问题处理。
            Err(amiki_text::TextError::MissingGlyph(_)) => return AMIKI_ERROR_NOT_FOUND,
            Err(_) => return AMIKI_ERROR_INVALID_ARGUMENT,
        };
        with_paint!(c, paint, |paint| {
            let stack_ref = &s.stack;
            c.target()
                .text_with_stack(&run, amiki_math::Vec2::new(x, y), paint, |fi, id| {
                    stack_ref.get(fi)?.outline(id).ok()
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
        // 填充规则是画布级状态：新开的层要继承当前规则，
        // 否则"设了 even-odd 再开层"会在这层里静默退回 NonZero。
        sub.set_fill_rule(c.fill_rule);
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

/// CPU 后端渲染（带画布自己的色调映射与 mip bias）。
///
/// 同步入口与异步入口**共用这一份实现** —— 否则"两条入口逐字节相同"没有根据。
fn render_cpu_with_tone(
    scene: &amiki_scene::Scene,
    eps: f32,
    tone: amiki_scene::ToneMap,
    bias: f32,
) -> Vec<u8> {
    let mut binned = amiki_raster::bin(scene, eps);
    binned.mip_bias = bias;
    amiki_raster::render_with_tone(&binned, tone)
}

/// GPU 错误 → C 错误码。
///
/// 后端的"不支持"要单独报出来（B1：GL 遇到位图），让调用方能区分
/// "引擎坏了"与"这个后端做不了"。
fn render_error_to_i32(msg: &str) -> i32 {
    if msg.contains("不支持") || msg.contains("无法") {
        AMIKI_ERROR_NOT_SUPPORTED
    } else {
        AMIKI_ERROR_UNKNOWN
    }
}

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
        // 色调映射作用在**出口量化**这一步，两条后端路径都要用上它；
        // mip bias（B4）同理，两条路径共用同一设置
        let tone = c.tone_map;
        let bias = c.mip_bias;
        let rendered: Result<Vec<u8>, String> = match &ctx.backend {
            #[cfg(feature = "gpu")]
            Backend::Gpu(r) => {
                r.render_blocking_with(scene, eps, tone, bias).map_err(|e| e.to_string())
            }
            // CPU 路径与 GPU 无关 —— 这也是 WASM 构建下唯一的路径
            Backend::Cpu => Ok(render_cpu_with_tone(scene, eps, tone, bias)),
        };
        match rendered {
            Ok(px) if px.len() >= need => {
                // SAFETY: need <= out_len，且 out 非空
                unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), out, need) };
                AMIKI_OK
            }
            Ok(_) => AMIKI_ERROR_UNKNOWN,
            Err(msg) => render_error_to_i32(&msg),
        }
    })
}

// ---- 输出（异步三入口，P3 第 ③b 步）----
//
// 同步入口 `amiki_canvas_render_rgba8` **一行都没变**（除了上面那条超范围回落）：
// 异步是**另加**的能力，不是替换。两条入口共用同一套前置判断与同一份 CPU 实现，
// 所以产物逐字节相同（`tests.rs` 里按字节比对）。

/// **提交一帧、立即返回**（P3 第 ③b 步）。
///
/// 之后用 `amiki_canvas_poll_render_rgba8` 问就绪、`amiki_canvas_try_read_render_rgba8` 取值。
/// 在飞帧是**提交那一刻场景的快照**：提交后再改画布不影响它。
///
/// - **重复 begin 是允许的**：上一个还没取走的帧被丢弃（它只是"一次还没算完的计算"）。
///   这是**有意**的 —— 否则调用方一旦不想等了就再无合法出路，只能再加一个 cancel 符号。
/// - 打开中的层没配对的 `end` ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`（与同步入口同一条理由：
///   直接渲染会静默丢掉那些绘制）。
/// - 提交阶段就失败的（准备 / 派发）**立刻**报错，不装作"在飞"再让 `try_read` 报。
#[no_mangle]
pub extern "C" fn amiki_canvas_begin_render_rgba8(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        if !c.layers.is_empty() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let ctx = c.ctx;
        if ctx.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: context 必须比 canvas 活得久（header 契约）
        let ctx = unsafe { &*ctx };
        if let Err(r) = ctx.poison.check() {
            return r;
        }
        let (eps, tone, bias) = (c.eps, c.tone_map, c.mip_bias);
        let scene = &c.root.scene;
        let pending = match &ctx.backend {
            #[cfg(feature = "gpu")]
            Backend::Gpu(r) => match r.begin_with(scene, eps, tone, bias) {
                Ok(p) => PendingCanvasRender::InFlight(p),
                // 提交阶段的失败（准备 / 派发）是同步可见的 —— 立刻报出去，
                // 不让它装作"在飞"然后再由 try_read 报
                Err(e) => return render_error_to_i32(&e.to_string()),
            },
            Backend::Cpu => PendingCanvasRender::Ready(render_cpu_with_tone(scene, eps, tone, bias)),
        };
        c.pending = Some(pending);
        AMIKI_OK
    })
}

/// **非阻塞地问一次：在飞的那一帧算完了吗**（P3 第 ③b 步）。
///
/// 写入 `*out_ready`：非 0 = 可以 `try_read` 了；0 = 还没好。**0 不是错误** ——
/// 它是"过一会儿再问"，这正是异步渲染存在的理由（提交与读回分到不同帧）。
///
/// 失败时**不写** `*out_ready`（含"没 begin 就 poll" ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`：
/// 与 `restore` 多于 `save` 同类，是调用方的 bug，**不假装已就绪**）。
#[no_mangle]
pub extern "C" fn amiki_canvas_poll_render_rgba8(
    canvas: *mut AmikiCanvas,
    out_ready: *mut i32,
) -> i32 {
    ffi!(canvas, |c| {
        if out_ready.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let Some(p) = c.pending.as_mut() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let ready = match p {
            PendingCanvasRender::Ready(_) => true,
            #[cfg(feature = "gpu")]
            PendingCanvasRender::InFlight(f) => f.poll() == amiki_gpu::PollStatus::Ready,
        };
        // SAFETY: 已判空
        unsafe { *out_ready = if ready { 1 } else { 0 } };
        AMIKI_OK
    })
}

/// **就绪则取值**：写入 `out`（长度至少 `width * height * 4`）并消费掉这一帧
/// （P3 第 ③b 步）。行序与 alpha 约定与同步入口完全相同。
///
/// - 还没算完 ⇒ `AMIKI_ERROR_BUSY`，**不写入任何字节**，且这一帧**留在原地**
///   （可以继续 `poll` / 再 `try_read`）；
/// - `out_len` 不足 ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`，且这一帧**不被消费**
///   （长度不够是调用方的 bug，不该顺手毁掉在飞的那一帧，也不写任何字节）；
/// - 没 begin 就调用 ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`；
/// - 取值阶段的 GPU 错误 ⇒ 帧被消费，并按与同步入口**同一套规则**报出来
///   （"不支持 / 无法" ⇒ `AMIKI_ERROR_NOT_SUPPORTED`，其余 ⇒ `AMIKI_ERROR_UNKNOWN`）——
///   错误是终局的，要重来就重新 `begin`。
#[no_mangle]
pub extern "C" fn amiki_canvas_try_read_render_rgba8(
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
        let Some(p) = c.pending.as_mut() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 未就绪时**提前返回**：这一帧一个字节都没动过，调用方可以继续问。
        let got: Result<Vec<u8>, i32> = match p {
            // `mem::take` 而不是克隆：CPU 分支的结果可能有好几 MB
            PendingCanvasRender::Ready(v) => Ok(std::mem::take(v)),
            #[cfg(feature = "gpu")]
            PendingCanvasRender::InFlight(f) => match f.take_ready() {
                Some(Ok(v)) => Ok(v),
                Some(Err(e)) => Err(render_error_to_i32(&e.to_string())),
                // `take_ready` 的承诺：未就绪时**什么都不动**（还能继续 poll）
                None => return AMIKI_ERROR_BUSY,
            },
        };
        c.pending = None;
        match got {
            Ok(px) if px.len() >= need => {
                // SAFETY: need <= out_len，且 out 非空
                unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), out, need) };
                AMIKI_OK
            }
            // CPU 分支的像素数由画布尺寸决定，长度不符只可能是引擎侧算错；
            // 报 UNKNOWN 而不是写出半张图
            Ok(_) => AMIKI_ERROR_UNKNOWN,
            Err(code) => code,
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
// 色调映射
// ============================================================================

/// 设置输出量化时用的色调映射。默认 `AMIKI_TONEMAP_RAW`（只裁剪）。
#[no_mangle]
pub extern "C" fn amiki_canvas_set_tone_map(cv: *mut AmikiCanvas, tone: i32) -> i32 {
    ffi!(cv, |c| {
        // 越界值**显式失败**，不静默退回默认 ——
        // 静默退回会让"设错了"看起来像"设上了"
        let Some(t) = crate::types::tone_map_from_raw(tone) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        c.tone_map = t;
        AMIKI_OK
    })
}

/// 读回当前的色调映射设置。空画布返回 `AMIKI_TONEMAP_RAW`。
#[no_mangle]
pub extern "C" fn amiki_canvas_tone_map(cv: *const AmikiCanvas) -> i32 {
    if cv.is_null() {
        return crate::types::AMIKI_TONEMAP_RAW;
    }
    crate::types::tone_map_to_raw(unsafe { &*cv }.tone_map)
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
        // B4：**默认生成 mip 链**（AMIKI_MIPMAP_BOX）。要省那 1/3 内存
        // 或锁死"只双线性"的行为，加载后调
        // `amiki_image_set_mipmap_mode(img, AMIKI_MIPMAP_DISABLED)`。
        // 1:1 与放大的绘制不受影响（LOD < 0 钳回 level 0）。
        let Some(img) = amiki_scene::Image::from_rgba8_with_mipmap(width, height, data) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let h = Box::new(AmikiImage { image: Arc::new(img), poison: Poison::default() });
        // SAFETY: `out` 已判空
        unsafe { *out = Box::into_raw(h) };
        AMIKI_OK
    })
}

// ---- 图像编解码（0.10.0，feature = "codec"）----

#[cfg(feature = "codec")]
fn codec_error_to_i32(e: amiki_codec::CodecError) -> i32 {
    use amiki_codec::CodecError as E;
    match e {
        // "不认识这个格式" / "这个格式的这项特性本版不做" —— 都是能力问题
        E::UnknownFormat | E::Unsupported(_) => AMIKI_ERROR_NOT_SUPPORTED,
        // 数据坏、尺寸非法：是参数/内容问题
        E::Decode(_) | E::Encode(_) | E::BadSize(_, _) => AMIKI_ERROR_INVALID_ARGUMENT,
    }
}

#[cfg(feature = "codec")]
fn image_from_decoded(img: amiki_scene::Image) -> *mut AmikiImage {
    Box::into_raw(Box::new(AmikiImage {
        image: Arc::new(img),
        poison: Poison::default(),
    }))
}

/// 从文件加载图像（PNG / JPEG）。**按内容判格式，不看扩展名。**
///
/// 找不到文件返回 `AMIKI_ERROR_NOT_FOUND`；格式不认识或数据坏返回
/// `AMIKI_ERROR_NOT_SUPPORTED` / `AMIKI_ERROR_INVALID_ARGUMENT`（见 header 的说明）。
/// 与 `amiki_image_create_rgba8` 一样，**默认生成 mip 链**。
#[cfg(feature = "codec")]
#[no_mangle]
pub extern "C" fn amiki_image_create_from_file(
    utf8_path: *const std::ffi::c_char,
    out: *mut *mut AmikiImage,
) -> i32 {
    ffi_free!({
        if utf8_path.is_null() || out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let Ok(path) = (unsafe { CStr::from_ptr(utf8_path) }).to_str() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let Ok(bytes) = std::fs::read(path) else {
            return AMIKI_ERROR_NOT_FOUND;
        };
        match amiki_codec::decode(&bytes) {
            // SAFETY: `out` 已判空
            Ok(img) => {
                unsafe { *out = image_from_decoded(img) };
                AMIKI_OK
            }
            Err(e) => codec_error_to_i32(e),
        }
    })
}

/// 从内存加载图像（PNG / JPEG）。**不复制输入缓冲**，调用返回后即可释放 `data`。
#[cfg(feature = "codec")]
#[no_mangle]
pub extern "C" fn amiki_image_create_from_bytes(
    data: *const u8,
    len: usize,
    out: *mut *mut AmikiImage,
) -> i32 {
    ffi_free!({
        if data.is_null() || out.is_null() || len == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 调用方保证 [data, data+len) 可读
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        match amiki_codec::decode(bytes) {
            // SAFETY: `out` 已判空
            Ok(img) => {
                unsafe { *out = image_from_decoded(img) };
                AMIKI_OK
            }
            Err(e) => codec_error_to_i32(e),
        }
    })
}

/// 把图像编码成 PNG（RGBA8 无损）。
///
/// 输出缓冲由引擎用 `amiki_alloc` 分配，**调用方必须用 `amiki_free(ptr, *out_len)`
/// 释放**（与其它"引擎分配"的 API 同一套契约）。失败时不写任何出参。
#[cfg(feature = "codec")]
#[no_mangle]
pub extern "C" fn amiki_image_encode_png(
    image: *const AmikiImage,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    ffi_free!({
        if image.is_null() || out_data.is_null() || out_len.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: 已判空
        let img = unsafe { &*image };
        match amiki_codec::encode_png(&img.image) {
            Ok(bytes) => {
                let n = bytes.len();
                let p = crate::api::amiki_alloc(n);
                if p.is_null() {
                    return AMIKI_ERROR_OUT_OF_MEMORY;
                }
                // SAFETY: p 是刚分配的长度 n 的缓冲；bytes 长度 n；两者不重叠
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, n);
                    *out_data = p;
                    *out_len = n;
                }
                AMIKI_OK
            }
            Err(e) => codec_error_to_i32(e),
        }
    })
}

#[no_mangle]
pub extern "C" fn amiki_image_destroy(image: *mut AmikiImage) {    if image.is_null() {
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

/// 设置位图的 mipmap 模式（B4）。
///
/// - `AMIKI_MIPMAP_BOX`（**默认**，创建时即生效）：按 2×2 盒式滤波生成
///   mip 链（线性光 + 预乘平均，见 `amiki_scene::mipmap`），缩小时三线性
///   采样，消除摩尔纹。代价是约 +1/3 的内存。
/// - `AMIKI_MIPMAP_DISABLED`：丢弃 mip 链，永远双线性采样（0.5.0 的行为）。
///
/// 本函数重建图像的内部数据：**对后续的绘制生效**，已画进画布的引用不变。
/// 越界的枚举值被拒绝（返回 `AMIKI_ERROR_INVALID_ARGUMENT`），状态不变。
#[no_mangle]
pub extern "C" fn amiki_image_set_mipmap_mode(image: *mut AmikiImage, mode: i32) -> i32 {
    ffi!(image, |h| {
        let Some(gen) = crate::types::mipmap_mode_from_raw(mode) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // Image 是 Clone 的纯数据（RGBA8 + 可选链），克隆后换一个 Arc：
        // 已被画布引用的旧数据保持不动，这是"对后续绘制生效"的语义
        let mut img = (*h.image).clone();
        img.mip = if gen {
            Some(amiki_scene::generate_mip_chain(img.width, img.height, &img.data))
        } else {
            None
        };
        h.image = std::sync::Arc::new(img);
        AMIKI_OK
    })
}

/// 设置画布的 mip 采样偏移（B4）。默认 0。
///
/// 正值取更糊的层（适合先放大后缩小的场合），负值取更锐的层；
/// 采样层级会被钳到 mip 链范围内，过大/过小的值只是失效、不会出错。
/// 非有限值（NaN / inf）被拒绝。只影响**缩小**的位图采样，不影响绘制本身。
#[no_mangle]
pub extern "C" fn amiki_canvas_set_mip_bias(cv: *mut AmikiCanvas, bias: f32) -> i32 {
    ffi!(cv, |c| {
        if !bias.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.mip_bias = bias;
        AMIKI_OK
    })
}

// ---- 填充规则（even-odd，0.8.0）----

/// 设置**后续填充**的填充规则。状态式、粘性；默认 `AMIKI_FILL_NONZERO`。
///
/// 越界枚举值被拒绝（返回 `AMIKI_ERROR_INVALID_ARGUMENT`），状态不变。
/// 只作用于填充，不影响描边（描边是轮廓化，与填充规则无关）。
#[no_mangle]
pub extern "C" fn amiki_canvas_set_fill_rule(cv: *mut AmikiCanvas, rule: i32) -> i32 {
    let Some(rule) = fill_rule_from_i32(rule) else {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    };
    ffi!(cv, |c| {
        c.fill_rule = rule;
        // 立刻写进当前绘制目标（根画布或最内层打开中的层）。
        // 层是**之后**才打开的，所以在 `begin_layer` 里也要把规则传下去。
        c.target().set_fill_rule(rule);
        AMIKI_OK
    })
}

/// 当前填充规则。传 NULL 返回 `AMIKI_FILL_NONZERO`（与 `amiki_context_backend`
/// 对 NULL 的处理一致：不崩、给默认值）。
#[no_mangle]
pub extern "C" fn amiki_canvas_get_fill_rule(cv: *const AmikiCanvas) -> i32 {
    if cv.is_null() {
        return AMIKI_FILL_NONZERO;
    }
    // SAFETY: 已判空
    let c = unsafe { &*cv };
    fill_rule_to_i32(c.fill_rule)
}

// ---- 状态栈（0.9.0）----

/// 保存当前状态（变换 / 混合模式 / 裁剪栈 / 填充规则）到状态栈。
///
/// 已存 32 层时返回 `AMIKI_ERROR_STACK_OVERFLOW`，栈不变。
#[no_mangle]
pub extern "C" fn amiki_canvas_save(cv: *mut AmikiCanvas) -> i32 {
    ffi!(cv, |c| {
        match c.target().save() {
            Ok(()) => AMIKI_OK,
            Err(e) => canvas_error_to_i32(e),
        }
    })
}

/// 从状态栈恢复：四项状态**整体覆盖**为最近一次 save 时的值。
///
/// 栈空返回 `AMIKI_ERROR_STACK_UNDERFLOW`（状态不变）。
///
/// ⚠️ **两侧镜像都要重新同步**。C 侧独立维护着两份门面状态的镜像：
/// `clip_depth`（`push_clip` 加、`pop_clip` 减）与 `fill_rule`
/// （`amiki_canvas_set_fill_rule` 写）。恢复会把门面的这两项一起回退，
/// 但**不会**动 C 侧的镜像 —— 不同步的话：
/// - `clip_depth` 停在旧值 ⇒ 下一次 `amiki_canvas_pop_clip` 误判；
/// - `fill_rule` 停在旧值 ⇒ `amiki_canvas_get_fill_rule` 报出一个
///   与实际绘制不一致的规则（本测试抓到的就是这个：恢复后仍报 EvenOdd）。
#[no_mangle]
pub extern "C" fn amiki_canvas_restore(cv: *mut AmikiCanvas) -> i32 {
    ffi!(cv, |c| {
        match c.target().restore() {
            Ok(()) => {
                c.clip_depth = c.target().clip_depth();
                c.fill_rule = c.target().fill_rule();
                AMIKI_OK
            }
            Err(e) => canvas_error_to_i32(e),
        }
    })
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
        let h = Box::new(AmikiFont {
            font,
            poison: Poison::default(),
            cache: std::sync::Mutex::new(amiki_scene::TextCache::new()),
        });
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
        let h = Box::new(AmikiFont {
            font,
            poison: Poison::default(),
            cache: std::sync::Mutex::new(amiki_scene::TextCache::new()),
        });
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

// ---- 字体栈（字体回退，0.7.0）----

/// 创建一个**空**字体栈。失败（分配失败 / panic）返回 NULL。
#[no_mangle]
pub extern "C" fn amiki_font_stack_create() -> *mut AmikiFontStack {
    // 这里不用 `ffi_free!`：它在 panic 时返回 `AMIKI_ERROR_PANIC`（i32），
    // 而本函数返回指针 —— 用 `unwrap_or(null)` 表达"创建失败"。
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Box::into_raw(Box::new(AmikiFontStack::new()))
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// 销毁字体栈。传 NULL 是空操作。
///
/// **不影响**压栈用过的 `amiki_font`：压栈是克隆（字节用 Arc 共享），
/// 句柄仍归调用方，该 `amiki_font_destroy` 还得调。
#[no_mangle]
pub extern "C" fn amiki_font_stack_destroy(stack: *mut AmikiFontStack) {
    if stack.is_null() {
        return;
    }
    // SAFETY: 调用方保证指针来自 `amiki_font_stack_create` 且未被销毁
    drop(unsafe { Box::from_raw(stack) });
}

/// 把一个字体压入栈（**追加**：顺序即优先级，先压的先赢）。
#[no_mangle]
pub extern "C" fn amiki_font_stack_push(
    stack: *mut AmikiFontStack,
    font: *const AmikiFont,
) -> i32 {
    if font.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(stack, |s| {
        // SAFETY: 已判空
        let f = unsafe { &*font };
        if let Err(r) = f.poison.check() {
            return r;
        }
        // 克隆 `Font` 很便宜：字体字节在 `Arc` 里共享，这里不复制那几 MB。
        s.stack.push(f.font.clone());
        AMIKI_OK
    })
}

/// 查询**第一个**含该码点的字体下标（0 起）。
///
/// 都不含时返回 `AMIKI_ERROR_NOT_FOUND`（并且不写出参）。
/// `codepoint` 不是合法标量值（代理区 / > U+10FFFF）⇒ `AMIKI_ERROR_INVALID_ARGUMENT`。
#[no_mangle]
pub extern "C" fn amiki_font_stack_resolve(
    stack: *const AmikiFontStack,
    codepoint: u32,
    out_index: *mut u32,
) -> i32 {
    if stack.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    // SAFETY: 已判空
    let s = unsafe { &*stack };
    if let Err(r) = s.poison.check() {
        return r;
    }
    let Some(c) = char::from_u32(codepoint) else {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    };
    match s.stack.resolve(c) {
        Some(i) => {
            // SAFETY: out 可为 NULL（header 契约）
            unsafe { write_out(out_index, i as u32) };
            AMIKI_OK
        }
        None => AMIKI_ERROR_NOT_FOUND,
    }
}

/// `amiki_text_options` 的默认值（与 header 里写的三个默认值一致）。
#[no_mangle]
pub extern "C" fn amiki_text_options_default() -> AmikiTextOptions {
    AmikiTextOptions::default_options()
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
