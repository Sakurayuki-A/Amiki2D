//! mipmap —— mip 链生成、LOD 计算与三线性采样（B4）。
//!
//! # 为什么这个模块在 `amiki-scene` 而不是 `amiki-raster`
//!
//! 与 [`crate::ToneMap`] 同一个理由：mip 链挂在 [`crate::Image`] 上，而
//! `Image` 住在 `amiki-scene`；GPU 侧（`amiki-gpu`）上传纹理时也要用同一份
//! 生成代码。`amiki-raster` 依赖 `amiki-scene`，反方向的依赖会成环。
//!
//! # 一致性契约（这是本模块存在的理由）
//!
//! - **mip 链只在 CPU 生成**（本模块），GPU 上传时消费生成的字节。
//!   不做 GPU 生成 —— 两套滤波实现必然漂移。
//! - **LOD 只在 CPU 计算**（[`compute_lod`]），由 GPU 侧打包进放置缓冲，
//!   WGSL 只读不算。`log2` 在 CPU libm 与 GPU 内建函数之间可能差 1 ULP，
//!   不在着色器里算就从根上消掉了这个漂移源。
//! - **确定性**：生成过程只用 f32 加乘与查表式分支，同一输入在任何时刻
//!   生成、生成多少次，字节都相同。
//!
//! # 滤波设计
//!
//! 盒式滤波（2×2 等权平均），**在线性光空间、预乘 alpha 上做**：
//!
//! ```text
//! sRGB u8 → 解码到线性 → 预乘 → 平均 → 反预乘 → 编码回 sRGB → 量化 u8
//! ```
//!
//! - 不用 Lanczos / Kaiser：负瓣振铃会让"逐像素一致"更脆弱，且 2D 位图的
//!   缩小质量瓶颈在走样（摩尔纹），不在锐度。
//! - 预乘是**为平均而临时做的**：平均完立刻反预乘，存储仍是直线色。
//!   不预乘直接平均会让透明边缘的 RGB 漏进结果（黑边）。
//! - 奇数尺寸：2×2 块与图像相交的部分等权平均（1/2/4 个纹素），
//!   不复制边缘纹素（复制等于给边缘加权，会整体偏色）。
//! - alpha 按普通标量平均（它是覆盖率，不参与 sRGB 编解码）。

use crate::{srgb_to_linear, Vec2};

// ============================================================================
// mip 链
// ============================================================================

/// 一级 mip：与 `Image` 相同的布局（sRGB 编码直线 RGBA8）。
#[derive(Clone, Debug, PartialEq)]
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// 一张图像的 mip 链。
///
/// **不含 level 0** —— 那就是原图自己（`Image::data`），存两份是纯浪费。
/// `levels[0]` 是 level 1（原图的一半），以此类推。
#[derive(Clone, Debug, PartialEq)]
pub struct MipChain {
    pub levels: Vec<MipLevel>,
}

impl MipChain {
    /// 完整的层级数（含 level 0）。GPU 纹理的 `mip_level_count` 用这个。
    pub fn total_levels(&self) -> u32 {
        self.levels.len() as u32 + 1
    }
}

/// 含 level 0 在内的 mip 层级数：`floor(log2(max(w,h))) + 1`。
pub fn mip_level_count(width: u32, height: u32) -> u32 {
    max_dim(width, height).ilog2() + 1
}

fn max_dim(width: u32, height: u32) -> u32 {
    width.max(height).max(1)
}

/// 从 RGBA8 原图生成 mip 链（level 1 到最深层，level 0 不在其中）。
///
/// 非二次幂支持：每层 `max(1, w>>1) × max(1, h>>1)`，直到 1×1。
pub fn generate_mip_chain(width: u32, height: u32, data: &[u8]) -> MipChain {
    let total = mip_level_count(width, height) as usize;
    let mut levels = Vec::with_capacity(total.saturating_sub(1));
    let mut prev = MipLevel {
        width,
        height,
        data: data.to_vec(),
    };
    // 不透明性与 LUT 全链只算一次：255 的平均还是 255（编码表边界对齐），
    // level 0 不透明 ⇒ 所有层级不透明
    let opaque = data.as_chunks::<4>().0.iter().all(|p| p[3] == 255);
    let dec = srgb_decode_lut();
    let enc = srgb_encode_lut();
    for _ in 1..total {
        let w = (prev.width >> 1).max(1);
        let h = (prev.height >> 1).max(1);
        prev = box_downsample(&prev, w, h, opaque, &dec, &enc);
        levels.push(prev.clone());
    }
    MipChain { levels }
}

/// 为 GPU 纹理数组生成**补齐到数组尺寸**的 mip 层。
///
/// 数组里所有层共享同一尺寸（取全场景最大值），小于数组的图像左上对齐、
/// 其余透明。mip 链必须在**补齐后的布局**上生成，否则透明 padding 会在
/// 高层级污染图像区域之外的采样 —— 这与 level 0 的现有行为一致。
/// 与 [`generate_mip_chain`] 用同一套滤波，同样的输入字节产生同样的输出。
pub fn generate_padded_mip_levels(
    img_w: u32,
    img_h: u32,
    data: &[u8],
    dst_w: u32,
    dst_h: u32,
) -> Vec<MipLevel> {
    debug_assert_eq!(data.len(), (img_w * img_h * 4) as usize);
    // 补齐到数组尺寸（左上对齐，越界透明）
    let mut padded = vec![0u8; (dst_w * dst_h * 4) as usize];
    let cw = img_w.min(dst_w) as usize;
    let ch = img_h.min(dst_h) as usize;
    let stride = dst_w as usize * 4;
    for y in 0..ch {
        let src = y * img_w as usize * 4;
        padded[y * stride..y * stride + cw * 4].copy_from_slice(&data[src..src + cw * 4]);
    }
    let total = mip_level_count(dst_w, dst_h) as usize;
    let mut levels = Vec::with_capacity(total.saturating_sub(1));
    let mut prev = MipLevel {
        width: dst_w,
        height: dst_h,
        data: padded,
    };
    let opaque = prev.data.as_chunks::<4>().0.iter().all(|p| p[3] == 255);
    let dec = srgb_decode_lut();
    let enc = srgb_encode_lut();
    for _ in 1..total {
        let w = (prev.width >> 1).max(1);
        let h = (prev.height >> 1).max(1);
        prev = box_downsample(&prev, w, h, opaque, &dec, &enc);
        levels.push(prev.clone());
    }
    levels
}

/// 一级盒式降采样：2×2 等权平均，线性光 + 预乘。
///
/// sRGB 编解码走**查表**（[`srgb_decode_lut`] / [`srgb_encode_lut`]）：
/// 朴素实现对每个纹素调 7 次 `powf`，1024² 图实测 56 ms —— 查表后同一张图
/// 降到毫秒级。表由 `tone.rs` 的同名函数构造，语义一致；且表 + f32 加乘
/// **不含超越函数**，生成过程天然跨平台字节一致（powf 在 x86 与 wasm 上
/// 可能差 1 ULP，会传导到量化字节）。
///
/// 行级并行（原生目标）：输出行彼此独立，按行分块跑在 scoped 线程上；
/// 每个输出行只读同样的输入、只写自己的行，**分块方式不影响任何字节**，
/// 确定性不受影响。wasm 目标保持串行（`std::thread` 在
/// wasm32-unknown-unknown 上不可用）。说明：KNOWN_ISSUES 的 C5
/// "CPU 并行不做"针对的是**光栅器采样**负载（实测无效，瓶颈在分配）；
/// 链生成是一次性的上传期工作，负载形态不同，实测有效（见
/// `examples/mipmap_perf`）。
fn box_downsample(
    src: &MipLevel,
    dst_w: u32,
    dst_h: u32,
    opaque: bool,
    dec: &[f32; 256],
    enc: &[u8; 16384],
) -> MipLevel {
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
    let row_len = dst_w as usize * 4;

    #[cfg(not(target_arch = "wasm32"))]
    let threads = {
        // spawn ~30µs/个：层级太小（<128 行）或每线程摊不到 ~64 行时不值得开，
        // 否则线程创建的开销比它省下的计算还贵
        let cpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        if dst_h as usize >= 128 {
            cpu.min((dst_h as usize) / 32).max(1)
        } else {
            1
        }
    };
    #[cfg(target_arch = "wasm32")]
    let threads = 1;

    if threads > 1 {
        let chunk = (dst_h as usize).div_ceil(threads);
        std::thread::scope(|s| {
            // 行带切分：每带是互斥的 `&mut [u8]`（无需 unsafe），带内行连续
            for (t, band) in out.chunks_mut(row_len * chunk).enumerate() {
                let y0 = (t * chunk) as u32;
                s.spawn(move || {
                    downsample_rows(src, band, y0, dst_w, opaque, dec, enc)
                });
            }
        });
    } else {
        downsample_rows(src, &mut out, 0, dst_w, opaque, dec, enc);
    }

    MipLevel {
        width: dst_w,
        height: dst_h,
        data: out,
    }
}

/// 计算输出行 `y0` 起的连续 `band.len()/row_len` 行。宽高 ≥ 2 时所有输出
/// 纹素都是**完整的 2×2 块**（floor 减半意味着奇数尺寸的尾行/尾列被丢弃
/// —— `dst = src >> 1`，不存在需要钳制的边缘块）；宽或高为 1 的退化图
/// 走通用慢路径。
fn downsample_rows(
    src: &MipLevel,
    band: &mut [u8],
    y0: u32,
    dst_w: u32,
    opaque: bool,
    dec: &[f32; 256],
    enc: &[u8; 16384],
) {
    let (sw, sh) = (src.width as usize, src.height as usize);
    let dst_w = dst_w as usize;
    let band_rows = band.len() / (dst_w * 4);
    if sw >= 2 && sh >= 2 {
        // 内区快路径：2×2 完整块，无分支。
        // 反预乘保持两步（先 ×0.25 再除），均匀色的往返才能精确到字节：
        // 4l×0.25 = l（2 的幂精确），l/l = 1.0 —— 合并成乘逆会引入 1 ULP，
        // 白色的 1−ulp 会掉进 254 的量化桶。
        let full_w = dst_w.min(sw >> 1);
        let full_h = (band_rows + y0 as usize).min(sh >> 1);
        if opaque {
            // alpha 恒 255：预乘/反预乘是恒等运算，整段跳过（含除法）
            for y in (y0 as usize)..full_h {
                let r0 = &src.data[y * 2 * sw * 4..(y * 2 + 1) * sw * 4];
                let r1 = &src.data[(y * 2 + 1) * sw * 4..(y * 2 + 2) * sw * 4];
                let orow = &mut band[(y - y0 as usize) * dst_w * 4..][..dst_w * 4];
                for x in 0..full_w {
                    let (i0, i1) = (x * 8, x * 8 + 4);
                    let o = x * 4;
                    for ch in 0..3 {
                        // 配对求和：(a+b)+(c+d) 每步都是 2 的幂倍，均匀色精确
                        orow[o + ch] = enc[encode_index(
                            ((dec[r0[i0 + ch] as usize] + dec[r0[i1 + ch] as usize])
                                + (dec[r1[i0 + ch] as usize] + dec[r1[i1 + ch] as usize]))
                                * 0.25,
                        )];
                    }
                    orow[o + 3] = 255;
                }
            }
            return;
        }
        for y in (y0 as usize)..full_h {
            let r0 = &src.data[y * 2 * sw * 4..(y * 2 + 1) * sw * 4];
            let r1 = &src.data[(y * 2 + 1) * sw * 4..(y * 2 + 2) * sw * 4];
            let orow = &mut band[(y - y0 as usize) * dst_w * 4..][..dst_w * 4];
            for x in 0..full_w {
                let (i0, i1) = (x * 8, x * 8 + 4);
                let o = x * 4;
                let a_avg = ((dec[r0[i0 + 3] as usize] + dec[r0[i1 + 3] as usize])
                    + (dec[r1[i0 + 3] as usize] + dec[r1[i1 + 3] as usize]))
                    * 0.25;
                if a_avg > 0.0 {
                    orow[o] = enc[encode_index(
                        ((dec[r0[i0] as usize] + dec[r0[i1] as usize])
                            + (dec[r1[i0] as usize] + dec[r1[i1] as usize]))
                            * 0.25
                            / a_avg,
                    )];
                    orow[o + 1] = enc[encode_index(
                        ((dec[r0[i0 + 1] as usize] + dec[r0[i1 + 1] as usize])
                            + (dec[r1[i0 + 1] as usize] + dec[r1[i1 + 1] as usize]))
                            * 0.25
                            / a_avg,
                    )];
                    orow[o + 2] = enc[encode_index(
                        ((dec[r0[i0 + 2] as usize] + dec[r0[i1 + 2] as usize])
                            + (dec[r1[i0 + 2] as usize] + dec[r1[i1 + 2] as usize]))
                            * 0.25
                            / a_avg,
                    )];
                }
                orow[o + 3] = (a_avg * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
        return;
    }
    // 退化路径：宽或高为 1（1×N / N×1 / 1×1）。2×2 块在长度为 1 的轴上
    // 由同一纹素补位 —— 补位纹素与原纹素**值相同**，等权与否不影响结果。
    for ry in 0..band_rows {
        let y = y0 as usize + ry;
        for x in 0..dst_w {
            let nx = if x * 2 + 1 < sw { 2 } else { 1 };
            let ny = if y * 2 + 1 < sh { 2 } else { 1 };
            let d = &src.data;
            let tap = |xx: usize, yy: usize| {
                let o = (yy * sw + xx) * 4;
                let a = dec[d[o + 3] as usize];
                [
                    dec[d[o] as usize] * a,
                    dec[d[o + 1] as usize] * a,
                    dec[d[o + 2] as usize] * a,
                    a,
                ]
            };
            // 配对结构：每排先配对，再跨排（每步都是 2 的幂倍）；
            // 缺席的纹素（退化边）贡献 0，等权数由 nx*ny 修正
            let t00 = tap(x * 2, y * 2);
            let t01 = if nx == 2 { tap(x * 2 + 1, y * 2) } else { [0.0; 4] };
            let t10 = if ny == 2 { tap(x * 2, y * 2 + 1) } else { [0.0; 4] };
            let t11 = if nx == 2 && ny == 2 { tap(x * 2 + 1, y * 2 + 1) } else { [0.0; 4] };
            let mut acc = [0.0f32; 4];
            for ch in 0..4 {
                acc[ch] = (t00[ch] + t01[ch]) + (t10[ch] + t11[ch]);
            }
            let n = (nx * ny) as f32;
            let a_avg = acc[3] / n;
            let o = x * 4;
            if a_avg > 0.0 {
                band[ry * dst_w * 4 + o] = enc[encode_index(acc[0] / n / a_avg)];
                band[ry * dst_w * 4 + o + 1] = enc[encode_index(acc[1] / n / a_avg)];
                band[ry * dst_w * 4 + o + 2] = enc[encode_index(acc[2] / n / a_avg)];
            }
            band[ry * dst_w * 4 + o + 3] = (a_avg * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
}

/// sRGB 解码表：u8 值 → 线性光 f32。由 [`srgb_to_linear`] 构造，语义一致。
fn srgb_decode_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = srgb_to_linear(i as f32 / 255.0);
    }
    lut
}

/// 线性光 → sRGB u8 的**边界对齐**编码表。
///
/// `encode_index(l)` 把线性值映射到 65536 个桶，表内容按量化边界填充：
/// 值 `l` 得到 `v` 当且仅当 `l` 落在"四舍五入到 v"的线性区间内 ——
/// 与 `(linear_to_srgb(l) * 255 + 0.5) as u8` 逐字节一致（含边界）。
fn srgb_encode_lut() -> [u8; 16384] {
    let mut lut = [0u8; 16384];
    let mut bucket = 0usize;
    for v in 0..=255u8 {
        // v 的量化区间上界（**编码空间**）：enc 超过 (v+0.5)/255 就舍入到 v+1。
        // 换算到线性空间用 srgb_to_linear（单调递增，桶序不反转）
        let hi_enc = ((v as f32 + 0.5) / 255.0).min(1.0);
        let hi_lin = srgb_to_linear(hi_enc);
        // bucket 索引 = (l * 65535 + 0.5)（与 encode_index 同一映射）
        let b = encode_index(hi_lin).min(16383);
        while bucket <= b {
            lut[bucket] = v;
            bucket += 1;
        }
    }
    while bucket < 16384 {
        lut[bucket] = 255;
        bucket += 1;
    }
    lut
}

#[inline]
fn encode_index(l: f32) -> usize {
    // floor 分割（不用 +0.5 偏移）：桶边界与填充时的量化边界严格同序，
    // 均匀色往返（decode→平均→encode）才能精确回到原字节。
    // 16384 个桶是下限：暗部线性段斜率 12.92，每个 u8 量化步长占
    // ~0.0003 线性值，桶宽必须明显小于它（4096 桶只有 1.24 个/步，会丢字节）
    ((l.clamp(0.0, 1.0) * 16384.0) as usize).min(16383)
}

// ============================================================================
// LOD
// ============================================================================

/// 从图像放置的轴向量计算 LOD（含 bias）。
///
/// 放置是 `p = origin + u·u_axis + v·v_axis`（`uv ∈ [0,1]²`），故屏幕像素步长
/// 对应的 uv 步长由逆雅可比给出（与 [`crate::ImagePlacement::uv_at`] 同一套推导）：
///
/// ```text
/// duv/dx = ( v_axis.y, -u_axis.y) / det      duv/dy = ( -v_axis.x, u_axis.x) / det
/// ```
///
/// 每个方向取"该步长在纹素空间覆盖多少纹素"（uv 步长 × 纹理尺寸）的欧氏范数，
/// 取两个方向的最大值，`LOD = log2(ρ)`。纹理宽高不同时**逐分量缩放**，
/// 不在 max 之后乘单一尺寸（那只对正方形纹理成立）。
///
/// 退化（`det ≈ 0`）返回 0：采 level 0，与"宁可画得糊也不产 NaN"的既有约定一致。
pub fn compute_lod(u_axis: Vec2, v_axis: Vec2, tex_w: u32, tex_h: u32, bias: f32) -> f32 {
    let w = tex_w as f32;
    let h = tex_h as f32;
    let det = u_axis.x * v_axis.y - v_axis.x * u_axis.y;
    if !det.is_finite() || det.abs() < 1e-12 {
        return 0.0;
    }
    let inv = 1.0 / det.abs();
    let rho_x = ((v_axis.y * w).powi(2) + (u_axis.y * h).powi(2)).sqrt() * inv;
    let rho_y = ((v_axis.x * w).powi(2) + (u_axis.x * h).powi(2)).sqrt() * inv;
    let rho = rho_x.max(rho_y);
    if rho <= 0.0 || !rho.is_finite() {
        return 0.0;
    }
    // bias 非有限时按 0 处理。C ABI（`amiki_canvas_set_mip_bias`）已经显式拒绝
    // NaN/inf，但 Rust 侧 `BinnedScene::mip_bias` 与 `PersistentRenderer::mip_bias`
    // 是 pub 字段，直接写 NaN 会让 LOD 传染成 NaN → 整幅 NaN 像素（静默画错，
    // 而不是报错）。宁可退回无偏移 —— 与 `det` / `rho` 两处退化处理同一约定。
    if !bias.is_finite() {
        return rho.log2();
    }
    rho.log2() + bias
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linear_to_srgb, Image};

    fn rgba(w: u32, h: u32, px: &[[u8; 4]]) -> Vec<u8> {
        assert_eq!(px.len(), (w * h) as usize);
        px.iter().flat_map(|c| c.iter().copied()).collect()
    }

    #[test]
    fn chain_structure_matches_level_count_formula() {
        // 8×8：level 1..3 = 4×4, 2×2, 1×1（level 0 是原图，不在链里）
        let c = generate_mip_chain(8, 8, &[0u8; 256]);
        assert_eq!(c.levels.len(), 3);
        assert_eq!((c.levels[0].width, c.levels[0].height), (4, 4));
        assert_eq!((c.levels[1].width, c.levels[1].height), (2, 2));
        assert_eq!((c.levels[2].width, c.levels[2].height), (1, 1));
        assert_eq!(c.total_levels(), 4);
        assert_eq!(mip_level_count(8, 8), 4);

        // 非二次幂：3×5 → 1×2 → 1×1
        let c = generate_mip_chain(3, 5, &[0u8; 60]);
        assert_eq!((c.levels[0].width, c.levels[0].height), (1, 2));
        assert_eq!((c.levels[1].width, c.levels[1].height), (1, 1));

        // 1×1：链为空，层级数 1
        let c = generate_mip_chain(1, 1, &[0u8; 4]);
        assert!(c.levels.is_empty());
        assert_eq!(mip_level_count(1, 1), 1);
    }

    #[test]
    fn uniform_color_survives_the_roundtrip_byte_identically() {
        // 均匀色经过 解码→预乘→平均→反预乘→编码 必须回到原字节
        // （若这一步都做不到，整条链就在逐层漂移）
        for v in [0u8, 1, 7, 64, 128, 200, 255] {
            let data = rgba(2, 2, &[[v, v, v, 255]; 4]);
            let c = generate_mip_chain(2, 2, &data);
            let l = &c.levels[0];
            assert_eq!(&l.data[..3], &[v, v, v], "v={v}");
            assert_eq!(l.data[3], 255);
        }
    }

    #[test]
    fn averaging_happens_in_linear_light_not_in_encoded_space() {
        // 黑白棋盘 2×2 → 1×1：线性平均得线性 0.5，编码回 sRGB ≈ 188。
        // 在编码空间直接平均会得到 ~128 —— 这条测试就是钉住两者的分界。
        let data = rgba(
            2,
            2,
            &[[0, 0, 0, 255], [255, 255, 255, 255], [255, 255, 255, 255], [0, 0, 0, 255]],
        );
        let c = generate_mip_chain(2, 2, &data);
        let l = &c.levels[0];
        let expected = (linear_to_srgb(0.5) * 255.0 + 0.5) as u8; // ≈ 188
        assert_eq!(l.data[0], expected);
    }

    #[test]
    fn premultiplied_averaging_leaves_no_black_fringe() {
        // 左半不透明红、右半全透明：平均后 alpha=0.5，反预乘后 RGB 仍是纯红。
        // 不预乘直接平均会把 (0,0,0) 的透明像素混进来 → 暗红（黑边）。
        let data = rgba(2, 1, &[[255, 0, 0, 255], [0, 0, 0, 0]]);
        let c = generate_mip_chain(2, 1, &data);
        let l = &c.levels[0];
        assert_eq!((l.data[0], l.data[1], l.data[2]), (255, 0, 0), "透明边缘不得发黑");
        assert_eq!(l.data[3], 128, "alpha 是普通标量平均：255/2 → 128");
    }

    #[test]
    fn odd_dims_average_the_intersected_block_with_equal_weights() {
        // 3×3 → 1×1（floor 减半：3>>1 = 1）。全图 9 个纹素等权平均。
        let v = 200u8;
        let data = rgba(3, 3, &[[v, 0, 0, 255]; 9]);
        let c = generate_mip_chain(3, 3, &data);
        assert_eq!(c.levels.len(), 1);
        assert_eq!((c.levels[0].width, c.levels[0].height), (1, 1));
        assert_eq!(c.levels[0].data[0], v, "等权平均下均匀色逐层不变");
    }

    #[test]
    fn generation_is_deterministic() {
        let mut data = vec![0u8; 64 * 64 * 4];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 7 % 251) as u8;
        }
        let a = generate_mip_chain(64, 64, &data);
        let b = generate_mip_chain(64, 64, &data);
        assert_eq!(a, b, "同一输入两次生成的链必须字节相同");
    }

    #[test]
    fn padded_chain_matches_direct_chain_for_full_size_images() {
        // 补齐路径（GPU 数组用）在图像恰好等于数组尺寸时必须与直接生成一致 ——
        // 这是"双端一致"的关键等价。
        let mut data = vec![0u8; 16 * 16 * 4];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 13 % 253) as u8;
        }
        let direct = generate_mip_chain(16, 16, &data);
        let padded = generate_padded_mip_levels(16, 16, &data, 16, 16);
        assert_eq!(direct.levels, padded);
    }

    #[test]
    fn padded_chain_keeps_image_region_top_left() {
        // 2×2 图补齐到 4×4：level 1 是 2×2，图像占左上 1×1，其余透明
        let data = rgba(2, 2, &[[255, 0, 0, 255]; 4]);
        let levels = generate_padded_mip_levels(2, 2, &data, 4, 4);
        let l1 = &levels[0];
        assert_eq!((l1.width, l1.height), (2, 2));
        assert_eq!(&l1.data[0..4], &[255, 0, 0, 255], "左上角仍是纯红");
        assert_eq!(&l1.data[4..], &[0; 12], "其余是透明 padding");
    }

    // ---- LOD ---------------------------------------------------------------

    #[test]
    fn lod_is_zero_for_one_to_one_and_log2_of_scale_otherwise() {
        let (w, h) = (64u32, 64u32);
        // 1:1 放置：u_axis=(w,0), v_axis=(0,h) → LOD 0
        let lod = compute_lod(Vec2::new(w as f32, 0.0), Vec2::new(0.0, h as f32), w, h, 0.0);
        assert!(lod.abs() < 1e-5, "1:1 应为 0，得到 {lod}");

        // 缩小 2×：LOD 1；缩小 8×：LOD 3
        let lod = compute_lod(
            Vec2::new(w as f32 / 2.0, 0.0),
            Vec2::new(0.0, h as f32 / 2.0),
            w,
            h,
            0.0,
        );
        assert!((lod - 1.0).abs() < 1e-5);
        let lod = compute_lod(
            Vec2::new(w as f32 / 8.0, 0.0),
            Vec2::new(0.0, h as f32 / 8.0),
            w,
            h,
            0.0,
        );
        assert!((lod - 3.0).abs() < 1e-5);

        // 放大 2×：LOD -1（采样侧钳到 level 0）
        let lod = compute_lod(
            Vec2::new(w as f32 * 2.0, 0.0),
            Vec2::new(0.0, h as f32 * 2.0),
            w,
            h,
            0.0,
        );
        assert!((lod + 1.0).abs() < 1e-5);

        // 平铺的 LOD 按单个平铺单元算：u_axis 是一个单元的跨度，
        // 即使平铺 100 次，轴向量不变 → LOD 不变（任务书 2.4）
        let lod_tiled = compute_lod(
            Vec2::new(w as f32 / 4.0, 0.0),
            Vec2::new(0.0, h as f32 / 4.0),
            w,
            h,
            0.0,
        );
        assert!((lod_tiled - 2.0).abs() < 1e-5);
    }

    #[test]
    fn lod_uses_per_axis_texel_scaling_for_non_square_textures() {
        // 100×10 纹理，放置 u_axis=(50,0), v_axis=(0,10)：
        // x 方向缩小 2×（100 纹素摊到 50 像素），y 方向 1:1 → LOD = 1
        let lod = compute_lod(Vec2::new(50.0, 0.0), Vec2::new(0.0, 10.0), 100, 10, 0.0);
        assert!((lod - 1.0).abs() < 1e-5, "非正方形纹理应逐轴缩放，得到 {lod}");
    }

    #[test]
    fn degenerate_axes_and_bias() {
        // 退化（det≈0）→ 0，不产 NaN
        let lod = compute_lod(Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0), 8, 8, 0.0);
        assert_eq!(lod, 0.0);
        assert!(lod.is_finite());

        // bias 直接相加
        let lod = compute_lod(Vec2::new(64.0, 0.0), Vec2::new(0.0, 64.0), 64, 64, -1.5);
        assert!((lod + 1.5).abs() < 1e-5);

        // 非有限 bias（NaN / ±inf）按 0 处理，不得把 LOD 传染成 NaN。
        // C ABI（`amiki_canvas_set_mip_bias`）会显式拒绝，但 Rust 侧
        // `BinnedScene::mip_bias` / `PersistentRenderer::mip_bias` 是 pub 字段 ——
        // 直接写 NaN 若一路传到像素，就是"静默画出 NaN 像素"，比报错更糟。
        let base = compute_lod(Vec2::new(64.0, 0.0), Vec2::new(0.0, 64.0), 64, 64, 0.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let lod = compute_lod(Vec2::new(64.0, 0.0), Vec2::new(0.0, 64.0), 64, 64, bad);
            assert!(lod.is_finite(), "bias={bad} 时 LOD 必须有限，得到 {lod}");
            assert_eq!(lod, base, "bias={bad} 应按 0 处理");
        }
    }

    // ---- 三线性采样 ---------------------------------------------------------

    #[test]
    fn trilinear_at_lod_zero_equals_bilinear_exactly() {
        // 放大与 1:1（lod ≤ 0）必须走同一条双线性路径 —— 行为不变是硬约束
        let mut data = vec![0u8; 8 * 8 * 4];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 31 % 249) as u8;
        }
        let img = Image::from_rgba8_with_mipmap(8, 8, data).unwrap();
        for u in [0.1f32, 0.37, 0.5, 0.9] {
            for v in [0.2f32, 0.5, 0.83] {
                assert_eq!(
                    img.sample_mipmap(Vec2::new(u, v), false, 0.0),
                    img.sample_bilinear(Vec2::new(u, v), false),
                    "lod=0 时三线性必须与双线性逐字节一致 (u={u}, v={v})"
                );
            }
        }
    }

    #[test]
    fn trilinear_lerps_between_levels_on_the_encoded_values() {
        // level 0 = 黑白 2×2 棋盘，level 1 = 1×1 灰（sRGB(线性 0.5) ≈ 188）。
        // lod=0.5 的三线性 = 两次双线性结果在**编码值上**取中点 ——
        // 与 GPU 的 mipmap_filter: Linear 完全同语义。
        let data = rgba(
            2,
            2,
            &[[0, 0, 0, 255], [255, 255, 255, 255], [255, 255, 255, 255], [0, 0, 0, 255]],
        );
        let img = Image::from_rgba8_with_mipmap(2, 2, data).unwrap();
        let uv = Vec2::new(0.5, 0.5);
        let c0 = img.sample_bilinear(uv, false); // 棋盘双线性 → 编码空间中点 0.5
        // level 1 存的是**量化到 u8 后**的灰 —— GPU 采的也是这些字节，
        // 所以 lerp 的两端都必须是量化值。alpha 是覆盖率：平均后仍是 1.0
        let q = (linear_to_srgb(0.5) * 255.0 + 0.5) as u8;
        let c1 = [q as f32 / 255.0, q as f32 / 255.0, q as f32 / 255.0, 1.0];
        let got = img.sample_mipmap(uv, false, 0.5);
        for i in 0..4 {
            let want = c0[i] + (c1[i] - c0[i]) * 0.5;
            assert!((got[i] - want).abs() < 1e-5, "ch{i}: got {} want {}", got[i], want);
        }
        // lod ≥ 1：完全落在最深层
        let got = img.sample_mipmap(uv, false, 3.0);
        for i in 0..4 {
            assert!((got[i] - c1[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn images_without_chain_fall_back_to_bilinear() {
        let data = vec![128u8; 4 * 4 * 4];
        let img = Image::from_rgba8(4, 4, data).unwrap();
        let uv = Vec2::new(0.5, 0.5);
        assert_eq!(img.sample_mipmap(uv, false, 5.0), img.sample_bilinear(uv, false));
    }
}
