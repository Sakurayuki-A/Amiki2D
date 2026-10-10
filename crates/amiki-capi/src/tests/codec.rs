//! 图像编解码（0.10.0）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

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

