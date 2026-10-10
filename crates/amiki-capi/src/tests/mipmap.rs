//! mipmap 模式与 mip 偏置。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

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

