//! 位图：RGBA8 构造、解码 / 编码、尺寸与 mip 模式。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::{ffi, ffi_free, write_out};
use crate::objects::*;
use crate::types::*;

// `CStr` 只被下面那三个 `codec` 门控的入口用到（拆文件后不再有"总是存在"的用法兜底），
// 所以连导入一起门控。
#[cfg(feature = "codec")]
use std::ffi::CStr;
use std::sync::Arc;

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

