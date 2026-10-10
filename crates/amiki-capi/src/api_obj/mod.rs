//! 对象句柄的导出函数：上下文、画布、路径、资源。
//!
//! 每个函数都以 `crate::api::ffi` / `crate::api::ffi_free` 宏包裹 ——
//! 空指针、中毒、panic 三件事在那里统一处理，这里只写正常路径。

use crate::objects::*;
use crate::types::*;

use amiki_scene::ImagePlacement;
use std::ffi::CString;
// `Arc` 只被 `make_backend` 的 GPU 分支用到：拆文件后它不再与"总是存在的"用法同处一室，
// 所以 `--no-default-features` 下必须连导入一起门控（否则 unused_imports）。
#[cfg(feature = "gpu")]
use std::sync::Arc;

// ============================================================================
// 公共外壳（跨文件使用的宏，0.10.9 从 `canvas_state.rs` 提到模块根）
// ============================================================================

/// 公共绘制外壳：把 paint 解析成引擎类型，然后交给 `f`。
///
/// 原本是 `api_obj.rs` 里"状态"一节内的局部宏（`macro_rules!` 按文本顺序生效，
/// 所以后面画布绘制那几个函数能用上它）。拆成多文件后按文本顺序跨不过去，
/// 于是提到模块根、并且放在 `pub mod` 声明**之前**：`macro_rules!` 的文本作用域
/// 覆盖其后声明的子模块，各子模块因此直接可用（不需要再 `use`）。
macro_rules! with_paint {
    ($c:expr, $paint:expr, $f:expr) => {{
        // SAFETY: 契约要求 paint 里的句柄有效
        let Some(p) = (unsafe { paint_from_raw($paint) }) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        $f(p)
    }};
}

// ============================================================================
// 后端选择（`gpu` feature 分派）—— 见文件末尾的 `make_backend`
// ============================================================================

// ---- 按对象分文件（0.10.9：原 api_obj.rs 1691 行 / 70 个导出）----
pub mod context;
pub use context::{amiki_context_adapter_name, amiki_context_backend, amiki_context_create, amiki_context_destroy};
pub mod canvas;
pub use canvas::{amiki_canvas_clear, amiki_canvas_create, amiki_canvas_destroy, amiki_canvas_size};
pub mod canvas_state;
pub use canvas_state::{amiki_canvas_get_fill_rule, amiki_canvas_pop_clip, amiki_canvas_push_clip, amiki_canvas_reset_transform, amiki_canvas_restore, amiki_canvas_rotate, amiki_canvas_save, amiki_canvas_scale, amiki_canvas_set_blend, amiki_canvas_set_fill_rule, amiki_canvas_set_flatten_tolerance, amiki_canvas_set_mip_bias, amiki_canvas_set_tone_map, amiki_canvas_set_transform, amiki_canvas_tone_map, amiki_canvas_translate};
pub mod canvas_draw;
pub use canvas_draw::{amiki_canvas_draw_circle, amiki_canvas_draw_image, amiki_canvas_draw_image_placed, amiki_canvas_draw_pentagram, amiki_canvas_draw_rect, amiki_canvas_draw_text, amiki_canvas_draw_text_with_stack, amiki_canvas_fill_path, amiki_canvas_fill_path_image, amiki_canvas_stroke_path};
pub mod canvas_layer;
pub use canvas_layer::{amiki_canvas_begin_layer, amiki_canvas_end_layer};
pub mod canvas_output;
pub use canvas_output::{amiki_canvas_begin_render_rgba8, amiki_canvas_poll_render_rgba8, amiki_canvas_render_rgba8, amiki_canvas_try_read_render_rgba8};
pub mod path;
pub use path::{amiki_path_circle, amiki_path_clear, amiki_path_close, amiki_path_create, amiki_path_cubic_to, amiki_path_destroy, amiki_path_line_to, amiki_path_move_to, amiki_path_quad_to, amiki_path_rect};
pub mod image;
pub use image::{amiki_image_create_rgba8, amiki_image_destroy, amiki_image_set_mipmap_mode, amiki_image_size};
// 编解码三个入口只在 `codec` feature 下存在（`image.rs` 里它们整体带 cfg），
// 所以这里也必须带上 —— 少了它 `--no-default-features` 会 E0432。
#[cfg(feature = "codec")]
pub use image::{amiki_image_create_from_bytes, amiki_image_create_from_file, amiki_image_encode_png};
pub mod font;
pub use font::{amiki_font_create_from_file, amiki_font_create_from_memory, amiki_font_destroy, amiki_font_metrics, amiki_font_stack_create, amiki_font_stack_destroy, amiki_font_stack_push, amiki_font_stack_resolve, amiki_text_options_default};
pub mod gradient;
pub use gradient::{amiki_gradient_create_linear, amiki_gradient_create_radial, amiki_gradient_create_sweep, amiki_gradient_destroy};

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

/// 供测试与内部使用：把 `ImagePlacement::rect` 暴露成同一个构造路径。
#[allow(dead_code)]
pub(crate) fn rect_placement(x0: f32, y0: f32, x1: f32, y1: f32) -> ImagePlacement {
    ImagePlacement::rect(x0, y0, x1, y1)
}
