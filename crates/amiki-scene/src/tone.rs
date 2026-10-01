//! 色调映射与 sRGB 编解码 —— **色彩模型里"降范围"那一半**。
//!
//! # 为什么这个模块在 `amiki-scene` 而不是 `amiki-gpu`
//!
//! 它需要被 **CPU 与 GPU 两条路径**同时看到：GPU 侧的色调映射在主机回读时做，
//! 而 CPU 侧（也是 WASM 唯一能用的那条）在光栅器出口做。WASM 是**纯 CPU 构建**，
//! 根本看不到 `amiki-gpu` —— 所以这个类型不能住在那里。
//!
//! `amiki-scene` 是三层共享的最底层（CPU 与 GPU 都依赖它），也是色彩模型的
//! 自然归属。完整的色彩模型说明见 `docs/COLOR.md`。

// ============================================================================
// 色调映射与输出编码
// ============================================================================

/// 把 HDR 值压到显示范围的映射方式。
///
/// # 为什么需要它
///
/// 引擎的层与滤镜路径是 HDR 的：>1.0 的值穿过层边界存活，模糊也在未裁剪的
/// 数据上做。但**显示设备只有 `[0,1]`** —— 必须有一个明确的、可选的步骤把
/// 它压下来。不写这一步，>1.0 就只是被 `clamp` 掉，高光源会糊成一片白。
///
/// # 取值
///
/// - [`ToneMap::Raw`]：只裁剪（`clamp`）。**默认**，与 0.5.0 之前的行为一致，
///   也是与其它 2D 库（Skia / Cairo / canvas）一致的做法。
/// - [`ToneMap::Reinhard`]：`x / (1 + x)`。便宜、单调、保序，
///   高光源会保留可分辨的层次。
/// - [`ToneMap::Aces`]：Narkowicz 的 ACES 近似。对比度与饱和度的观感最好，
///   是电影/游戏里的常用选择。
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ToneMap {
    /// 只做裁剪（`[0,1]` 之外直接截断）。默认。
    #[default]
    Raw,
    /// Reinhard：`x / (1 + x)`。
    Reinhard,
    /// ACES 近似（Narkowicz 2015）。
    Aces,
}

impl ToneMap {
    /// 作用在**单个通道**上。
    ///
    /// 注意：色调映射是**非线性**的，严格来说应在亮度上做、再按比例还原颜色，
    /// 否则会偏色。但逐通道是三家的通行简化（包括 Narkowicz 的原文），
    /// 而本项目对"与其它 2D 库一致"的重视高于色彩科学上的严格 ——
    /// 需要严格做法时应当是一个独立的、明确的选择。
    #[inline]
    pub fn apply(self, x: f32) -> f32 {
        match self {
            ToneMap::Raw => x,
            ToneMap::Reinhard => x / (1.0 + x),
            ToneMap::Aces => {
                // Narkowicz 2015 的拟合：a=2.51 b=0.03 c=2.43 d=0.59 e=0.14
                const A: f32 = 2.51;
                const B: f32 = 0.03;
                const C: f32 = 2.43;
                const D: f32 = 0.59;
                const E: f32 = 0.14;
                ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
            }
        }
    }

    /// 是否需要在映射后再做一次 sRGB 编码。
    ///
    /// 引擎的颜色值是 **sRGB 编码**的（见 `docs/COLOR.md`），不是线性的。
    /// 所以 `Raw` 下**不需要**编码 —— 值本来就是显示空间。
    /// 但 `Reinhard` / `ACES` 是**在线性光上定义**的数学，直接作用在 sRGB 编码
    /// 的值上会偏亮；规范做法是"解码到线性 → 映射 → 再编码回 sRGB"。
    pub fn is_linear_operator(self) -> bool {
        !matches!(self, ToneMap::Raw)
    }
}

/// sRGB 编码的值 → 线性光。
#[inline]
pub fn srgb_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// 线性光 → sRGB 编码。
#[inline]
pub fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// 按指定的色调映射把 f32 直线 RGBA 降到 8 位。
///
/// 这是 **HDR 降到显示范围的唯一位置**。顺序很重要：
/// 1. 若映射是线性算符（Reinhard / ACES），先把 sRGB 编码的值解码到线性；
/// 2. 施加映射；
/// 3. 再编码回 sRGB（显示空间）；
/// 4. 量化到 8 位。
///
/// alpha **不参与映射** —— 它是覆盖率，不是光。
pub fn rgba8_from_f32_with(px: &[[f32; 4]], tone: ToneMap) -> Vec<u8> {
    let linear = tone.is_linear_operator();
    let mut out = Vec::with_capacity(px.len() * 4);
    for p in px {
        // 前三个通道参与映射；alpha 不参与（它是覆盖率，不是光）
        for &c in &p[..3] {
            let v = if linear { srgb_to_linear(c.max(0.0)) } else { c };
            let mapped = tone.apply(v);
            let enc = if linear { linear_to_srgb(mapped) } else { mapped };
            out.push((enc * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
        out.push((p[3] * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
    }
    out
}
