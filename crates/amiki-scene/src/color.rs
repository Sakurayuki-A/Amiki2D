//! 颜色与颜色矩阵（`Color` / `ColorMatrix`）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。


/// 颜色：0..1 的**直线（非预乘）RGBA**。预乘纪律在 M4 全链路化。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const TRANSPARENT: Color = Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

    pub const fn rgba8(r: u8, g: u8, b: u8, a: u8) -> Color {
        Color {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }

    pub const fn rgb8(r: u8, g: u8, b: u8) -> Color {
        Color::rgba8(r, g, b, 255)
    }
}

/// **颜色矩阵**（0.10.4）：4×5 仿射，作用在**非预乘 RGBA** 上。
///
/// ```text
///   r' = m00*r + m01*g + m02*b + m03*a + m04
///   g' = m10*r + m11*g + m12*b + m13*a + m14
///   b' = m20*r + m21*g + m22*b + m23*a + m24
///   a' = m30*r + m31*g + m32*b + m33*a + m34
/// ```
///
/// 第 5 列是**常量项**（加性偏移）—— 这是与 SVG `feColorMatrix`、Skia
/// `SkColorMatrix` 同一套约定，也是"灰化禁用态 / 整体调色 / 主题变体"最省事的表达。
///
/// # 为什么先只做类型、不接画布
///
/// 接画布要动**两条**管线（CPU 光栅器与 GPU 着色器）—— 那属于"约定未确认、
/// 风险高一档"的改动。所以本版先落地**纯函数**这一层：
/// 它是可独立验证的完整增量（无渲染改动、无行为变化），接线时只需要决定
/// "在哪一层把滤镜应用到已解析的颜色上"（纯色与渐变停靠点可以在提交时应用；
/// 图像要逐像素，需要另说）。
///
/// # 约定
///
/// - 输入输出都是**非预乘**、分量在 `[0,1]`（`Color` 的既有约定）；
/// - 结果**钳制**到 `[0,1]`（不产生 NaN/∞；非有限输入按钳制处理）；
/// - **不做色彩管理**（无 ICC / 线性空间转换）—— 这是本项目 `C3` 明确不做的范围。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix {
    /// 行主序的 4×5
    pub rows: [[f32; 5]; 4],
}

impl Default for ColorMatrix {
    fn default() -> Self {
        ColorMatrix::IDENTITY
    }
}

impl ColorMatrix {
    /// 恒等（输出 = 输入）。
    pub const IDENTITY: ColorMatrix = ColorMatrix {
        rows: [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
    };

    /// 直接给 4×5。
    pub const fn new(rows: [[f32; 5]; 4]) -> ColorMatrix {
        ColorMatrix { rows }
    }

    /// 对**非预乘**颜色应用，结果钳到 `[0,1]`。
    pub fn apply(&self, c: Color) -> Color {
        let v = [c.r, c.g, c.b, c.a];
        let mut out = [0.0f32; 4];
        for (i, row) in self.rows.iter().enumerate() {
            let mut acc = row[4];
            for (j, x) in v.iter().enumerate() {
                acc += row[j] * x;
            }
            out[i] = if acc.is_finite() { acc.clamp(0.0, 1.0) } else { 0.0 };
        }
        Color { r: out[0], g: out[1], b: out[2], a: out[3] }
    }

    /// **组合**：先 `inner` 再 `self`（与矩阵乘法同序，便于链式叠加）。
    ///
    /// 注意常量列（第 5 列）也要参与：`self` 的偏移会被 `inner` 的线性部分作用一次，
    /// 这正是"先做 A 再做 B"的语义。
    // 4x4 index arithmetic reads clearer than iterator gymnastics here; this is the
    // same kind of local, commented allow as D3 (see docs/KNOWN_ISSUES.md).
    #[allow(clippy::needless_range_loop)]
    pub fn then(&self, inner: &ColorMatrix) -> ColorMatrix {
        let mut rows = [[0.0f32; 5]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for (k, r) in self.rows[i].iter().take(4).enumerate() {
                    acc += r * inner.rows[k][j];
                }
                rows[i][j] = acc;
            }
            // 常量项：self 的常量 + self 线性部分作用于 inner 的常量
            let mut acc = self.rows[i][4];
            for (k, r) in self.rows[i].iter().take(4).enumerate() {
                acc += r * inner.rows[k][4];
            }
            rows[i][4] = acc;
        }
        ColorMatrix { rows }
    }

    /// 去色（Rec.709 亮度权重，alpha 不变）。
    pub fn greyscale() -> ColorMatrix {
        const R: f32 = 0.2126;
        const G: f32 = 0.7152;
        const B: f32 = 0.0722;
        ColorMatrix::new([
            [R, G, B, 0.0, 0.0],
            [R, G, B, 0.0, 0.0],
            [R, G, B, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// 饱和度：`0` = 去色，`1` = 不变，`>1` 更艳（标准 Rec.709 亮度权重）。
    pub fn saturate(s: f32) -> ColorMatrix {
        const R: f32 = 0.2126;
        const G: f32 = 0.7152;
        const B: f32 = 0.0722;
        let inv = 1.0 - s;
        let (rr, rg, rb) = (inv * R, inv * G, inv * B);
        ColorMatrix::new([
            [rr + s, rg, rb, 0.0, 0.0],
            [rr, rg + s, rb, 0.0, 0.0],
            [rr, rg, rb + s, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// 棕褐色（经典 sepia 系数）。
    pub fn sepia() -> ColorMatrix {
        ColorMatrix::new([
            [0.393, 0.769, 0.189, 0.0, 0.0],
            [0.349, 0.686, 0.168, 0.0, 0.0],
            [0.272, 0.534, 0.131, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    }

    /// 乘一个**通道缩放**（例如变暗 `0.5`）。
    pub fn scale(r: f32, g: f32, b: f32, a: f32) -> ColorMatrix {
        ColorMatrix::new([
            [r, 0.0, 0.0, 0.0, 0.0],
            [0.0, g, 0.0, 0.0, 0.0],
            [0.0, 0.0, b, 0.0, 0.0],
            [0.0, 0.0, 0.0, a, 0.0],
        ])
    }

    /// 加一个**常量偏移**（例如整体提亮，末列语义）。
    pub fn offset(r: f32, g: f32, b: f32, a: f32) -> ColorMatrix {
        ColorMatrix::new([
            [1.0, 0.0, 0.0, 0.0, r],
            [0.0, 1.0, 0.0, 0.0, g],
            [0.0, 0.0, 1.0, 0.0, b],
            [0.0, 0.0, 0.0, 1.0, a],
        ])
    }
}

