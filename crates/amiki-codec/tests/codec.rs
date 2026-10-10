//! amiki-codec 的测试。
//!
//! 判据分两类（写死在这里，避免以后有人把 JPEG 也按"逐字节"要求）：
//! - **PNG：逐字节一致**（无损格式，不同实现必须给出同样的像素）
//! - **JPEG：容差**（IDCT/上采样实现不同，实测最大通道差 1–3；这里放到 8 留余量）
//!
//! 参照物是工作区里已有的 `image` crate —— **独立实现**，不是我们自己的代码。

use amiki_codec::{decode, decode_jpeg, decode_png, encode_png, sniff, CodecError, Format};
use amiki_scene::Image;

fn asset(name: &str) -> Vec<u8> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/assets")
        .join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("读不到素材 {}: {e}", p.display()))
}

/// 造一张有梯度 + 半透明 + 不透明像素的图（编码往返用）
fn synth(w: u32, h: u32) -> Image {
    let mut d = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            d[i] = (x * 7 % 256) as u8;
            d[i + 1] = (y * 11 % 256) as u8;
            d[i + 2] = ((x + y) * 3 % 256) as u8;
            d[i + 3] = if (x + y) % 3 == 0 { 128 } else { 255 };
        }
    }
    Image::from_rgba8(w, h, d).expect("尺寸合法")
}

/// PNG 往返必须**逐字节一致**。
#[test]
fn png_roundtrip_is_byte_exact() {
    let src = synth(37, 23);
    let bytes = encode_png(&src).expect("编码");
    assert_eq!(sniff(&bytes), Some(Format::Png), "编码结果应能被嗅探为 PNG");
    let back = decode_png(&bytes).expect("解码");
    assert_eq!((back.width, back.height), (37, 23));
    assert_eq!(back.data, src.data, "PNG 是无损格式，往返必须逐字节一致");
    eprintln!("PNG 往返：{}×{}，{} 字节，逐字节一致", back.width, back.height, bytes.len());
}

/// **索引色（调色板）PNG 必须能解** —— 实验里最先暴露的缺口。
/// 与 `image` crate 对照，要求逐字节一致。
#[test]
fn indexed_png_expands_palette() {
    let bytes = asset("indexed.png");
    let ours = decode_png(&bytes).expect("索引色应能解");
    let theirs = image::load_from_memory(&bytes).expect("参照解码").to_rgba8();
    assert_eq!((ours.width, ours.height), (theirs.width(), theirs.height()));
    assert_eq!(
        ours.data,
        theirs.into_raw(),
        "索引色展开必须与独立实现逐字节一致"
    );
    eprintln!("索引色 PNG：{}×{} 与参照逐字节一致", ours.width, ours.height);
}

/// 灰度、灰度+α：与参照逐字节一致。
#[test]
fn gray_and_gray_alpha_match_reference() {
    for name in ["gray8.png", "gray_alpha8.png"] {
        let bytes = asset(name);
        let ours = decode_png(&bytes).expect("应能解");
        let theirs = image::load_from_memory(&bytes).expect("参照").to_rgba8();
        assert_eq!(ours.data, theirs.into_raw(), "{name} 应与参照一致");
        eprintln!("{name}：{}×{} 逐字节一致", ours.width, ours.height);
    }
}

/// **真 16 位 PNG**：我们取高字节（见模块文档的取舍）。
///
/// 判据是**精确期望值**：素材是照 `v = (y + x*9000) % 65536` 生成的，
/// 所以每个像素应当是 `v >> 8` 的灰、且不透明。硬编码属性（"灰阶数 > N"）
/// 在这里会失手 —— 12 个 16 位样本的高字节本来就只有 4 个不同值
/// （断言"灰阶数 > 16"必然失败）。
#[test]
fn gray16_takes_high_byte() {
    let ours = decode_png(&asset("gray16.png")).expect("16 位灰度应能解");
    let (w, h) = (ours.width as usize, ours.height as usize);
    assert_eq!((w, h), (4, 3), "素材尺寸");
    for y in 0..h {
        for x in 0..w {
            // 生成公式：v = (y + x*9000) % 65536；取高字节
            let expect = (((y + x * 9000) % 65536) as u16 >> 8) as u8;
            let i = (y * w + x) * 4;
            let px = &ours.data[i..i + 4];
            assert_eq!(
                px,
                [expect, expect, expect, 255],
                "像素 ({x},{y}) 应当是 16 位样本的高字节 {expect}"
            );
        }
    }
    eprintln!("16 位灰度：{w}×{h} 全部等于 (v%65536)>>8，alpha 不透明");
}

/// JPEG 与参照比对：**容差**判据，不是逐字节。
#[test]
fn jpeg_matches_reference_within_tolerance() {
    let bytes = asset("photo.jpg");
    let ours = decode_jpeg(&bytes).expect("JPEG 应能解");
    let theirs = image::load_from_memory(&bytes).expect("参照").to_rgba8();
    assert_eq!((ours.width, ours.height), (theirs.width(), theirs.height()));
    let t = theirs.into_raw();
    let max = ours
        .data
        .iter()
        .zip(t.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert!(max <= 8, "JPEG 与独立实现最大通道差 {max} 超出容差");
    eprintln!("JPEG：{}×{}，与参照最大通道差 {max}", ours.width, ours.height);
}

/// `decode` 按**内容**选解码器（不看扩展名）。
#[test]
fn decode_sniffs_by_magic() {
    assert_eq!(decode(&asset("indexed.png")).map(|i| (i.width, i.height)).ok(), Some((4, 3)));
    assert!(decode(&asset("photo.jpg")).is_ok());
    let e = decode(b"not an image at all").expect_err("随机数据必须报错");
    assert_eq!(e, CodecError::UnknownFormat);
}

/// 坏数据一律**返回 Err，不 panic、不静默**。
#[test]
fn bad_data_never_panics() {
    let png = asset("indexed.png");
    let jpg = asset("photo.jpg");
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("空", Vec::new()),
        ("随机", vec![0x13, 0x37, 0x00, 0xFF, 0xAA]),
        ("只有 magic", vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]),
        ("PNG 截断一半", png[..png.len() / 2].to_vec()),
        ("PNG 去掉尾 8 字节", png[..png.len() - 8].to_vec()),
        ("JPEG 截断一半", jpg[..jpg.len() / 2].to_vec()),
        ("JPEG 只有头", jpg[..3].to_vec()),
    ];
    for (name, data) in cases {
        let r = decode(&data);
        assert!(r.is_err(), "{name}：应当报错而不是成功");
        // 真值断言：到这一步没 panic 就已经是主要目的
        let _ = format!("{name}: {r:?}");
    }
    eprintln!("坏数据：7 种情形全部显式报错、无 panic");
}

/// 超大尺寸要早退（护住内存），不能进解码器再说。
#[test]
fn absurd_size_is_rejected() {
    let img = Image::from_rgba8(2, 2, vec![0; 16]).unwrap();
    let mut png = encode_png(&img).unwrap();
    // 把 IHDR 里的宽高改成 60000×60000（CRC 会不匹配 → 解码器应报错）
    if png.len() > 24 {
        png[16..20].copy_from_slice(&60000u32.to_be_bytes());
        png[20..24].copy_from_slice(&60000u32.to_be_bytes());
    }
    let r = decode_png(&png);
    assert!(r.is_err(), "改坏 IHDR 后必须报错（CRC 或尺寸校验）");
    eprintln!("超大尺寸：显式报错 {r:?}");
}
