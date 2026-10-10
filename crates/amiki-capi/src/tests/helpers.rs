//! 跨模块共用的测试装置（CPU 上下文 / 渲染 / 取样 / 字体 / HDR 方块）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

/// 建一个 CPU 后端的上下文（测试要确定、不能依赖机器有没有 GPU）。
pub(super) fn cpu_context() -> (*mut AmikiContext, i32) {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    let r = amiki_context_create(AMIKI_BACKEND_CPU, &mut ctx);
    (ctx, r)
}

// ============================================================================
// 版本
// ============================================================================

/// 用给定的色调映射画一个 `> 1.0` 的方块并渲染，返回像素。
pub(super) fn render_hot_square(tone: i32) -> Vec<u8> {
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

/// 仓库里的字体路径（找不到就跳过，不静默通过）。
pub(super) fn font_file(name: &str) -> Option<std::path::PathBuf> {
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

pub(super) fn load_font(name: &str) -> Option<*mut AmikiFont> {
    let p = font_file(name)?;
    let c = std::ffi::CString::new(p.to_string_lossy().as_bytes()).expect("路径无 NUL");
    let mut f: *mut AmikiFont = ptr::null_mut();
    assert_eq!(amiki_font_create_from_file(c.as_ptr(), &mut f), AMIKI_OK);
    Some(f)
}

pub(super) fn render(cv: *mut AmikiCanvas, w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; (w * h * 4) as usize];
    assert_eq!(amiki_canvas_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    out
}

/// 画点东西出来，免得"异步 == 同步"落到一张全透明画布上变得毫无意义。
pub(super) fn draw_sample(cv: *mut AmikiCanvas, w: f32, h: f32) {
    let bg = amiki_paint_solid(amiki_color_rgb8(250, 250, 252));
    assert_eq!(amiki_canvas_draw_rect(cv, 0.0, 0.0, w, h, bg), AMIKI_OK);
    let red = amiki_paint_solid(amiki_color_rgb8(231, 76, 60));
    assert_eq!(amiki_canvas_draw_rect(cv, 4.0, 4.0, w - 4.0, h - 4.0, red), AMIKI_OK);
    let blue = amiki_paint_solid(amiki_color_rgb8(52, 152, 219));
    assert_eq!(amiki_canvas_draw_circle(cv, w * 0.5, h * 0.5, w * 0.25, blue), AMIKI_OK);
}

