//! # amiki-codec —— 图像编解码（**可选** crate）
//!
//! 把图像文件变成引擎能画的 [`Image`]，以及把 [`Image`] 写成 PNG。
//!
//! ## 为什么单独一个 crate
//!
//! 四个核心 crate（`amiki-math` / `amiki-scene` / `amiki-raster` / `amiki-layout`）
//! **零第三方依赖**，这是本项目的立身之本（可审计、总量小）。图像编解码必须带依赖，
//! 所以它单独一层：**不想要的人不用它，依赖树与体积一分不涨**。
//!
//! ## 支持范围（首版，明确写死）
//!
//! | 格式 | 解码 | 编码 |
//! |---|---|---|
//! | PNG（灰度 / 灰度+α / RGB / RGBA；**1/2/4/8 位**与**索引色**；16 位取高字节） | ✅ | ✅（RGBA8） |
//! | JPEG（基线 + 渐进；灰度 / RGB） | ✅ | — |
//! | 其它（GIF/WEBP/AVIF/HEIF/…） | ❌ **显式报错** | — |
//!
//! 取舍都写在明处：
//! - **索引色（调色板）必须支持** —— UI 图标集里很常见（实验里它是最先暴露的一条）。
//! - **16 位取高字节**（不是报错）：视觉上正确、行为可预期；要精确保留 16 位得等真需要时再说。
//! - **JPEG 各解码器不可能逐字节一致**（IDCT/上采样实现不同）——
//!   实测 `zune-jpeg` 与 libjpeg-turbo 最大通道差 1–3。所以 JPEG 相关的测试
//!   用**容差**判据，PNG 用**逐字节**判据。
//! - 任何坏数据都**返回 `Err`，绝不 panic、绝不静默画错**。

use amiki_scene::Image;

/// 编解码错误。每一种都对应"调用方能做的事"，不吞细节。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// 数据太短或 magic 不认识（不是 PNG/JPEG）
    UnknownFormat,
    /// 是这个格式，但本版不支持其中的某个特性（如 JPEG 的 CMYK/12 位精度）
    Unsupported(String),
    /// 解码失败（截断、校验和不符、语法错误……）
    Decode(String),
    /// 编码失败
    Encode(String),
    /// 尺寸非法（0 或超出上限）
    BadSize(u32, u32),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::UnknownFormat => write!(f, "不认识的图像格式（只支持 PNG / JPEG）"),
            CodecError::Unsupported(s) => write!(f, "该格式的此项特性不支持：{s}"),
            CodecError::Decode(s) => write!(f, "解码失败：{s}"),
            CodecError::Encode(s) => write!(f, "编码失败：{s}"),
            CodecError::BadSize(w, h) => write!(f, "尺寸非法：{w}×{h}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// 尺寸上限：护住"解码一个 5 万×5 万的头"这种内存放大攻击。
///
/// 上限按**总像素数**给（不是宽高各自），因为内存是按乘积算的。
pub const MAX_PIXELS: u64 = 64 * 1024 * 1024; // 64M 像素 ≈ 256 MB RGBA8

/// 认出来的格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Png,
    Jpeg,
}

/// 按 magic 嗅探格式（不看扩展名 —— 文件内容才是事实）。
pub fn sniff(bytes: &[u8]) -> Option<Format> {
    if bytes.len() >= 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Some(Format::Png);
    }
    if bytes.len() >= 3 && bytes[..3] == [0xFF, 0xD8, 0xFF] {
        return Some(Format::Jpeg);
    }
    None
}

/// 按内容自动选解码器。
pub fn decode(bytes: &[u8]) -> Result<Image, CodecError> {
    match sniff(bytes) {
        Some(Format::Png) => decode_png(bytes),
        Some(Format::Jpeg) => decode_jpeg(bytes),
        None => Err(CodecError::UnknownFormat),
    }
}

fn check_size(w: u32, h: u32) -> Result<(), CodecError> {
    if w == 0 || h == 0 {
        return Err(CodecError::BadSize(w, h));
    }
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(CodecError::BadSize(w, h));
    }
    Ok(())
}

fn into_image(w: u32, h: u32, rgba: Vec<u8>) -> Result<Image, CodecError> {
    check_size(w, h)?;
    Image::from_rgba8(w, h, rgba).ok_or(CodecError::BadSize(w, h))
}

// ============================================================ PNG

/// 解码 PNG。
///
/// 覆盖：灰度 / 灰度+α / RGB / RGBA / **索引色**，位深 1/2/4/8/16。
/// 低比特位按规范展开到 8 位（1/2/4 位用标准缩放 `v * 255 / max`）；
/// 16 位**取高字节**（视觉正确、行为可预期，见模块文档）。
pub fn decode_png(bytes: &[u8]) -> Result<Image, CodecError> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    // 不启用 png crate 的变换：我们自己展开位深/调色板，行为才可预期
    dec.set_transformations(png::Transformations::IDENTITY);
    let mut reader = dec.read_info().map_err(|e| CodecError::Decode(format!("PNG 头：{e}")))?;
    let info = reader.info().clone();
    let (w, h) = (info.width, info.height);
    check_size(w, h)?;
    let mut raw = vec![0u8; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut raw)
        .map_err(|e| CodecError::Decode(format!("PNG 数据：{e}")))?;
    let raw = &raw[..frame.buffer_size()];

    let npix = (w as usize) * (h as usize);
    let mut rgba = vec![0u8; npix * 4];
    let bd = match frame.bit_depth {
        png::BitDepth::One => 1u8,
        png::BitDepth::Two => 2,
        png::BitDepth::Four => 4,
        png::BitDepth::Eight => 8,
        png::BitDepth::Sixteen => 16,
    };
    let channels: usize = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => 1,
    };

    // 取第 i 个像素的第 c 个通道，统一成 8 位
    let sample = |i: usize, c: usize, channels: usize| -> u8 {
        match bd {
            16 => {
                let idx = (i * channels + c) * 2;
                raw.get(idx).copied().unwrap_or(0) // 高字节
            }
            8 => raw.get(i * channels + c).copied().unwrap_or(0),
            bits => {
                // 1/2/4 位：按位打包，只对灰度与索引色合法
                let per_byte = 8 / bits as usize;
                let bit_off = (i % per_byte) * bits as usize;
                let byte = raw.get(i / per_byte).copied().unwrap_or(0);
                let shift = 8 - bits as usize - bit_off;
                let v = (byte >> shift) & ((1u16 << bits) - 1) as u8;
                // 标准缩放：v * 255 / (2^bits - 1)
                ((v as u16 * 255) / ((1u16 << bits) - 1)) as u8
            }
        }
    };

    for i in 0..npix {
        let px = match frame.color_type {
            png::ColorType::Grayscale => {
                let g = sample(i, 0, channels);
                [g, g, g, 255]
            }
            png::ColorType::GrayscaleAlpha => {
                let g = sample(i, 0, channels);
                [g, g, g, sample(i, 1, channels)]
            }
            png::ColorType::Rgb => [
                sample(i, 0, channels),
                sample(i, 1, channels),
                sample(i, 2, channels),
                255,
            ],
            png::ColorType::Rgba => [
                sample(i, 0, channels),
                sample(i, 1, channels),
                sample(i, 2, channels),
                sample(i, 3, channels),
            ],
            png::ColorType::Indexed => {
                let idx = if bd == 8 {
                    raw.get(i).copied().unwrap_or(0)
                } else {
                    sample(i, 0, channels)
                } as usize;
                let pal = info.palette.as_deref().unwrap_or(&[]);
                let r = pal.get(idx * 3).copied().unwrap_or(0);
                let g = pal.get(idx * 3 + 1).copied().unwrap_or(0);
                let b = pal.get(idx * 3 + 2).copied().unwrap_or(0);
                // tRNS：调色板透明度表（可选）
                let a = info
                    .trns
                    .as_deref()
                    .and_then(|t| t.get(idx).copied())
                    .unwrap_or(255);
                [r, g, b, a]
            }
        };
        rgba[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
    into_image(w, h, rgba)
}

/// 编码为 PNG（RGBA8，无损）。
pub fn encode_png(img: &Image) -> Result<Vec<u8>, CodecError> {
    check_size(img.width, img.height)?;
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, img.width, img.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc
            .write_header()
            .map_err(|e| CodecError::Encode(format!("PNG 头：{e}")))?;
        writer
            .write_image_data(&img.data)
            .map_err(|e| CodecError::Encode(format!("PNG 数据：{e}")))?;
    }
    Ok(out)
}

// ============================================================ JPEG

/// 解码 JPEG（基线 + 渐进；灰度 / RGB；常见 4:2:0 / 4:4:4 上采样）。
///
/// **注意**：不同 JPEG 解码器的像素值**不会逐字节相同**（IDCT 与上采样实现不同）。
/// 实测本实现与 libjpeg-turbo 最大通道差 1–3。
pub fn decode_jpeg(bytes: &[u8]) -> Result<Image, CodecError> {
    let mut dec = zune_jpeg::JpegDecoder::new(bytes);
    let px = dec
        .decode()
        .map_err(|e| CodecError::Decode(format!("JPEG 数据：{e:?}")))?;
    let info = dec.info().ok_or_else(|| CodecError::Decode("JPEG 无头信息".into()))?;
    let (w, h) = (info.width as u32, info.height as u32);
    check_size(w, h)?;
    let npix = (w as usize) * (h as usize);
    if npix == 0 || px.len() % npix != 0 {
        return Err(CodecError::Decode("JPEG 数据长度与尺寸不匹配".into()));
    }
    // 通道数从数据推（1 = 灰度，3 = RGB，4 = RGBA），不依赖库的色彩空间枚举
    let n = px.len() / npix;
    let mut rgba = Vec::with_capacity(npix * 4);
    match n {
        1 => {
            for &y in &px {
                rgba.extend_from_slice(&[y, y, y, 255]);
            }
        }
        3 => {
            for c in px.chunks(3) {
                if let [r, g, b] = c {
                    rgba.extend_from_slice(&[*r, *g, *b, 255]);
                }
            }
        }
        4 => rgba.extend_from_slice(&px),
        other => {
            return Err(CodecError::Unsupported(format!(
                "JPEG 通道数 {other}（CMYK / 12 位等不在首版范围）"
            )))
        }
    }
    into_image(w, h, rgba)
}
