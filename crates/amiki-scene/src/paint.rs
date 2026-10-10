//! 画法：填充规则、混合模式、`Paint`、`Clip` / `Insets`。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

/// 填充规则。
///
/// # even-odd 是怎么实现的（0.8.0）
///
/// 引擎的覆盖率是 `min(|Σ seg_area|, 1)`，几何上等于该像素的**绕数加权面积**
/// `∬ W dA`。even-odd 要的是**奇偶区域**，那是绕数的非线性函数（`W mod 2`）——
/// 从 `∬ W dA` 无论如何改符号都取不到（实测偏差 0.72~0.85，见
/// `crates/amiki-raster/tests/fill_rule_semantics.rs`）。
///
/// 所以实现方式是**几何**上的：把 even-odd 的路径用扫描线跨度分解成一组
/// **互不重叠的梯形**（`amiki_raster::even_odd_spans`，CPU 侧、两端共用），
/// 再交给原有的 NonZero 路径。覆盖率数学、WGSL、绑定号一个都没动。
///
/// 代价：even-odd 的 draw 多一趟 CPU 侧预处理（每次绘制都要跑，不缓存）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillRule {
    /// 非零绕数（默认）。
    #[default]
    NonZero,
    /// 奇偶（even-odd）。
    EvenOdd,
}

impl FillRule {
    /// 是否 even-odd。分箱/提取用它决定要不要做跨度分解。
    pub fn is_even_odd(self) -> bool {
        matches!(self, FillRule::EvenOdd)
    }

    /// AMK1 `flags` bit 12 的取值：`0 = NonZero`、`1 = EvenOdd`。
    pub fn bit(self) -> u32 {
        match self {
            FillRule::NonZero => 0,
            FillRule::EvenOdd => 1,
        }
    }

    /// 从 AMK1 `flags` bit 12 还原（**只解不编的 v1 场景恒为 NonZero**）。
    pub fn from_bit(b: u32) -> FillRule {
        if b & 1 == 0 {
            FillRule::NonZero
        } else {
            FillRule::EvenOdd
        }
    }
}

/// 混合模式（W3C compositing 全集）。id 与 WGSL/CPU 镜像逐一对齐。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal = 0,
    Multiply = 1,
    Screen = 2,
    Overlay = 3,
    Darken = 4,
    Lighten = 5,
    ColorDodge = 6,
    ColorBurn = 7,
    HardLight = 8,
    SoftLight = 9,
    Difference = 10,
    Exclusion = 11,
    Hue = 12,
    Saturation = 13,
    Color = 14,
    Luminosity = 15,
}

impl BlendMode {
    pub const fn id(self) -> u32 {
        self as u32
    }
}

/// 绘制样式。
///
/// **不是 `Copy`**：渐变经 `Arc` 共享，而 `Arc` 只能 `Clone`。
/// 代价是每字形绘制需要 `clone()` 一次（仅原子加，可忽略）；
/// 收益是渐变定义在多条 draw 间共享同一份数据。
#[derive(Clone, Debug, PartialEq)]
pub enum Paint {
    Solid(Color),
    /// 渐变着色。用 `Arc` 让渐变定义在多条 draw 间共享（避免逐字形深拷贝色标表）。
    Gradient(std::sync::Arc<Gradient>),
    /// 位图着色（M6）。放置与采样方式见 [`ImagePlacement`]。
    Image {
        image: std::sync::Arc<Image>,
        placement: ImagePlacement,
    },
}

/// 每 draw 最多引用的裁剪层数（WGSL 侧展开采样）。
pub const MAX_DRAW_CLIPS: usize = 4;

// ============================================================================
// M5：文本 —— 字形轮廓即路径
// ============================================================================
//
// 设计决策（重要）：**字形轮廓直接作为普通 path 进入既有管线**。
//
// 由此免费获得：
//   · 复用压平 / 分箱 / 解析式覆盖率 / 混合 / 裁剪 / damage region，零新增着色代码；
//   · **任意缩放都不糊** —— 抗锯齿来自真实的面积积分，不依赖预渲染的字形位图；
//   · 与图形完全同源的合成路径，文本与形状可以任意叠加、混合、裁剪。
//
// 引擎**不包含字体解析**：调用方（或未来的 `amiki-text`）负责 shaping 与轮廓提取，
// 交给引擎的是设备空间无关的轮廓点。这样核心保持零依赖，字体栈可替换。

/// 裁剪层：变换 + 路径。覆盖率在独立 pass 中渲染，绘制时逐像素乘入。
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub transform: Affine,
    pub path: Path,
}

/// 九宫格的**边距**（源图像素）：左 / 上 / 右 / 下。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Insets {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl Insets {
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Insets {
        Insets { left, top, right, bottom }
    }
    /// 四边同宽。
    pub fn all(v: u32) -> Insets {
        Insets { left: v, top: v, right: v, bottom: v }
    }
}

