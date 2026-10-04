//! amiki-scene —— 路径、绘制与场景模型。
//!
//! Scene 是引擎的中间表示：上层 API 产出 Scene，后端（GPU / CPU）消费 Scene。
//! M1 采用直接结构化表示；M2 演进为 Vello 式 tag+data 紧凑编码流（接口保持稳定）。

#![forbid(unsafe_code)]

use amiki_math::{Affine, CubicBez, QuadBez, Vec2};

mod damage;
pub use damage::{changed_bounds, draw_device_bounds, edit_damage};
mod mipmap;
pub use mipmap::{
    compute_lod, generate_mip_chain, generate_padded_mip_levels, mip_level_count, MipChain,
    MipLevel,
};
mod stroke;
pub use stroke::{flatten_path, stroke_path, stroke_polyline, LineCap, LineJoin, StrokeStyle};
mod tone;
pub use tone::{linear_to_srgb, rgba8_from_f32_with, srgb_to_linear, ToneMap};

/// 路径元素。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathEl {
    MoveTo(Vec2),
    LineTo(Vec2),
    QuadTo(Vec2, Vec2),
    CubicTo(Vec2, Vec2, Vec2),
    Close,
}

/// 路径（子路径序列；填充时未闭合子路径自动闭合）。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Path {
    pub els: Vec<PathEl>,
}

impl Path {
    pub fn from_els(els: Vec<PathEl>) -> Path {
        Path { els }
    }
}

/// 路径构建器，附常用形状。
#[derive(Clone, Debug, Default)]
pub struct PathBuilder {
    els: Vec<PathEl>,
}

impl PathBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn move_to(&mut self, p: Vec2) -> &mut Self {
        self.els.push(PathEl::MoveTo(p));
        self
    }

    pub fn line_to(&mut self, p: Vec2) -> &mut Self {
        self.els.push(PathEl::LineTo(p));
        self
    }

    pub fn quad_to(&mut self, c: Vec2, p: Vec2) -> &mut Self {
        self.els.push(PathEl::QuadTo(c, p));
        self
    }

    pub fn cubic_to(&mut self, c1: Vec2, c2: Vec2, p: Vec2) -> &mut Self {
        self.els.push(PathEl::CubicTo(c1, c2, p));
        self
    }

    pub fn close(&mut self) -> &mut Self {
        self.els.push(PathEl::Close);
        self
    }

    /// 轴对齐矩形（屏幕坐标顺时针）。
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) -> &mut Self {
        self.move_to(Vec2::new(x0, y0))
            .line_to(Vec2::new(x1, y0))
            .line_to(Vec2::new(x1, y1))
            .line_to(Vec2::new(x0, y1))
            .close()
    }

    /// 圆（4 段三次贝塞尔，kappa 近似）。`reversed` 反绕，用于挖洞。
    pub fn circle(&mut self, center: Vec2, r: f32, reversed: bool) -> &mut Self {
        const KAPPA: f32 = 0.552_284_7;
        let k = r * KAPPA;
        let (x, y) = (center.x, center.y);
        // 四个锚点：右、下、左、上（屏幕坐标）
        let anchors = [
            Vec2::new(x + r, y),
            Vec2::new(x, y + r),
            Vec2::new(x - r, y),
            Vec2::new(x, y - r),
        ];
        // 正向（右→下→左→上）每段的控制点
        let fwd: [(Vec2, Vec2); 4] = [
            (Vec2::new(x + r, y + k), Vec2::new(x + k, y + r)), // 右→下
            (Vec2::new(x - k, y + r), Vec2::new(x - r, y + k)), // 下→左
            (Vec2::new(x - r, y - k), Vec2::new(x - k, y - r)), // 左→上
            (Vec2::new(x + k, y - r), Vec2::new(x + r, y - k)), // 上→右
        ];
        if !reversed {
            self.move_to(anchors[0]);
            for i in 0..4 {
                let (c1, c2) = fwd[i];
                self.cubic_to(c1, c2, anchors[(i + 1) % 4]);
            }
        } else {
            // 反绕 = 反向遍历锚点（A0→A3→A2→A1→A0）+ 每段控制点互换
            self.move_to(anchors[0]);
            for i in 0..4 {
                let seg = 3 - i; // 反向第 i 段 = 正向第 seg 段的逆
                let (c2f, c1f) = fwd[seg];
                self.cubic_to(c1f, c2f, anchors[3 - i]);
            }
        }
        self.close()
    }

    /// 五角星（pentagram：隔点连接，自相交，nonzero 填充中央五边形为 W=2）。
    pub fn pentagram(&mut self, center: Vec2, r: f32) -> &mut Self {
        let pts: Vec<Vec2> = (0..5)
            .map(|i| {
                let a = -std::f32::consts::FRAC_PI_2
                    + i as f32 * (2.0 * std::f32::consts::PI / 5.0);
                Vec2::new(center.x + r * a.cos(), center.y + r * a.sin())
            })
            .collect();
        self.move_to(pts[0]);
        self.line_to(pts[2]);
        self.line_to(pts[4]);
        self.line_to(pts[1]);
        self.line_to(pts[3]);
        self.close()
    }

    /// 多边形（自动闭合）。
    pub fn polygon(&mut self, pts: &[Vec2]) -> &mut Self {
        if pts.is_empty() {
            return self;
        }
        self.move_to(pts[0]);
        for p in &pts[1..] {
            self.line_to(*p);
        }
        self.close()
    }

    pub fn build(&self) -> Path {
        Path { els: self.els.clone() }
    }
}

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

/// 渐变的一个色标：位置 `t ∈ [0,1]` 处的颜色。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    pub t: f32,
    pub color: Color,
}

impl GradientStop {
    pub fn new(t: f32, color: Color) -> GradientStop {
        GradientStop { t, color }
    }
}

/// 渐变几何（**设备空间**，即已含画布变换）。
///
/// 之所以存设备空间：着色发生在逐像素阶段，那时画布变换已经作用完毕；
/// 若在着色里再套一次变换，成本与出错面都更大。调用方在构造时报设备坐标即可。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientKind {
    /// 从 `from` 到 `to` 的线性渐变；投影参数 `t = (p − from)·(to − from) / |to − from|²`
    Linear { from: Vec2, to: Vec2 },
    /// 以 `center` 为心、`radius` 为半径的径向渐变；`t = |p − center| / radius`
    Radial { center: Vec2, radius: f32 },
    /// 绕 `center` 的角度扫描渐变；`t = (atan2(p.y − c.y, p.x − c.x) − start) / TAU`
    Sweep { center: Vec2, start: f32 },
}

/// 渐变：几何 + 色标表 + 范围外行为。
///
/// **色标按 `t` 升序**（构造时保证）。采样是解析求值而非查表——
/// 查表会引入 1/255 量化的误差，而本项目对"解析覆盖率"的质量承诺
/// 不允许在着色阶段引入额外近似。
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub kind: GradientKind,
    pub stops: Vec<GradientStop>,
    /// 范围外行为：`t` 钳到 `[0,1]`（Clamp）或按 1 取模重复（Repeat）
    pub repeat: bool,
}

impl Gradient {
    /// 构造线性渐变（设备坐标）。色标会被排序；少于 2 个色标时返回 `None`。
    pub fn linear(from: Vec2, to: Vec2, mut stops: Vec<GradientStop>) -> Option<Gradient> {
        if stops.len() < 2 {
            return None;
        }
        stops.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        Some(Gradient { kind: GradientKind::Linear { from, to }, stops, repeat: false })
    }

    /// 构造径向渐变（设备坐标）。
    pub fn radial(center: Vec2, radius: f32, mut stops: Vec<GradientStop>) -> Option<Gradient> {
        if stops.len() < 2 || radius <= 0.0 || !radius.is_finite() {
            return None;
        }
        stops.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        Some(Gradient { kind: GradientKind::Radial { center, radius }, stops, repeat: false })
    }

    /// 构造扫描渐变（设备坐标）。
    pub fn sweep(center: Vec2, start: f32, mut stops: Vec<GradientStop>) -> Option<Gradient> {
        if stops.len() < 2 {
            return None;
        }
        stops.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        Some(Gradient { kind: GradientKind::Sweep { center, start }, stops, repeat: false })
    }

    /// 求某设备坐标处的 `t`（未做范围处理）。
    pub fn param_at(&self, p: Vec2) -> f32 {
        match self.kind {
            GradientKind::Linear { from, to } => {
                let d = to - from;
                let len2 = d.dot(d);
                if len2 <= 1e-12 {
                    return 0.0;
                }
                (p - from).dot(d) / len2
            }
            GradientKind::Radial { center, radius } => (p - center).hypot() / radius,
            GradientKind::Sweep { center, start } => {
                let v = p - center;
                let a = v.y.atan2(v.x);
                // 归一到以 start 为起点的 [0,1)
                let t = (a - start) / std::f32::consts::TAU;
                t - t.floor()
            }
        }
    }

    /// 在参数 `t` 处解析求值（色标之间线性插值，端点外按首尾色延展或按重复取模）。
    pub fn sample(&self, t: f32) -> Color {
        let t = if self.repeat {
            t - t.floor()
        } else {
            t.clamp(0.0, 1.0)
        };
        let s = &self.stops;
        // 排序保证：s[0].t 最小、s[n-1].t 最大
        if t <= s[0].t {
            return s[0].color;
        }
        if t >= s[s.len() - 1].t {
            return s[s.len() - 1].color;
        }
        // 线性查找区间：色标数量很小（通常 2–8），且此处是逐像素热路径，
        // 二分反而因分支预测不稳更慢。
        let mut i = 0;
        while i + 1 < s.len() && s[i + 1].t < t {
            i += 1;
        }
        let (a, b) = (s[i], s[i + 1]);
        let span = b.t - a.t;
        let k = if span.abs() <= 1e-9 { 0.0 } else { (t - a.t) / span };
        Color {
            r: a.color.r + (b.color.r - a.color.r) * k,
            g: a.color.g + (b.color.g - a.color.g) * k,
            b: a.color.b + (b.color.b - a.color.b) * k,
            a: a.color.a + (b.color.a - a.color.a) * k,
        }
    }
}

/// 位图：**线性（非预乘）RGBA8**，行优先，`data.len() == width * height * 4`。
///
/// 引擎只认这一种内存布局：调用方负责解码（PNG/JPEG/…）与色彩空间转换。
/// 这样 amiki-scene / amiki-raster / amiki-gpu 都不需要引入任何图像解码库。
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    /// mip 链（level 1 起，level 0 是 [`Image::data`] 自身）。
    ///
    /// `None` = 不用 mipmap：两侧都只走双线性，行为与 0.5.0 完全一致。
    /// 由 [`Image::from_rgba8_with_mipmap`] 生成；mipmap 是**运行时派生物**，
    /// 不进 AMK1 编码流（编码/解码都只碰原图）。
    ///
    /// # 注意：Rust 与 C ABI 的默认值**不同**（有意的，不是疏漏）
    ///
    /// [`Image::from_rgba8`] **不**生成链 —— 那是对 0.5.0 的硬兼容承诺
    /// （1:1 与放大的绘制必须逐字节不变）；而 C ABI 的
    /// `amiki_image_create_rgba8` **默认**生成 BOX 链，代价约 **+1/3 内存**，
    /// 可用 `amiki_image_set_mipmap_mode` 关掉。两边的取舍都记在
    /// `CHANGELOG.md` 的 0.6.0 条目里。
    pub mip: Option<MipChain>,
}

impl Image {
    /// 从 RGBA8 数据构造；长度不符时返回 `None`。不生成 mip 链。
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Option<Image> {
        if width == 0 || height == 0 {
            return None;
        }
        if data.len() != (width as usize) * (height as usize) * 4 {
            return None;
        }
        Some(Image { width, height, data, mip: None })
    }

    /// 从 RGBA8 数据构造并**立即生成 mip 链**（B4）。
    ///
    /// 生成时机是构造时而非首次采样时：链是确定性的（见 `mipmap` 模块说明），
    /// 早生成只多一次分配，晚生成会在采样路径里埋分支。
    pub fn from_rgba8_with_mipmap(width: u32, height: u32, data: Vec<u8>) -> Option<Image> {
        let mut img = Image::from_rgba8(width, height, data)?;
        img.mip = Some(generate_mip_chain(width, height, &img.data));
        Some(img)
    }

    /// 取某像素（越界返回透明）。
    #[inline]
    pub fn texel(&self, x: u32, y: u32) -> [f32; 4] {
        if x >= self.width || y >= self.height {
            return [0.0; 4];
        }
        let o = ((y * self.width + x) * 4) as usize;
        let d = &self.data;
        [
            d[o] as f32 / 255.0,
            d[o + 1] as f32 / 255.0,
            d[o + 2] as f32 / 255.0,
            d[o + 3] as f32 / 255.0,
        ]
    }

    /// 双线性采样，`uv ∈ [0,1]²` 为图像空间坐标。
    ///
    /// **用双线性而非最近邻**：图像常被缩放，最近邻会产生明显的锯齿与抖动；
    /// 而本引擎的输出质量基准是解析覆盖率，采样阶段不该是质量瓶颈。
    ///
    /// `repeat` 为真时按 1 取模平铺，否则钳到边缘（clamp）。
    pub fn sample_bilinear(&self, uv: Vec2, repeat: bool) -> [f32; 4] {
        sample_bilinear_buf(&self.data, self.width, self.height, uv, repeat)
    }

    /// 三线性采样（B4）：层内双线性 + 层间按 LOD 小数部分插值。
    ///
    /// `lod ≤ 0`（1:1 或放大）时与 [`Image::sample_bilinear`] **完全同一路径**，
    /// 行为不变。没有 mip 链时同样退回双线性 —— 调用方无需先行判断。
    ///
    /// 层的选取与 GPU 的 `textureSampleLevel`（`mipmap_filter: Linear`）同语义：
    /// `floor(lod)` 层与 `floor(lod)+1` 层各做一次双线性，再按小数部分 lerp。
    pub fn sample_mipmap(&self, uv: Vec2, repeat: bool, lod: f32) -> [f32; 4] {
        let Some(chain) = &self.mip else {
            return self.sample_bilinear(uv, repeat);
        };
        let top = chain.levels.len() as u32; // 最深层级号（level 1..=top）
        let lod = lod.clamp(0.0, top as f32);
        let base = lod.floor() as u32;
        let c0 = self.sample_level(chain, base, uv, repeat);
        let f = lod - base as f32;
        if base >= top || f <= 0.0 {
            return c0;
        }
        let c1 = self.sample_level(chain, base + 1, uv, repeat);
        let mut out = [0.0f32; 4];
        for i in 0..4 {
            out[i] = c0[i] + (c1[i] - c0[i]) * f;
        }
        out
    }

    /// 采某一 mip 层（level 0 = 原图）。层级超出链深时钳到最深层。
    fn sample_level(&self, chain: &MipChain, level: u32, uv: Vec2, repeat: bool) -> [f32; 4] {
        let idx = (level as usize).min(chain.levels.len());
        if level == 0 || idx == 0 {
            return sample_bilinear_buf(&self.data, self.width, self.height, uv, repeat);
        }
        let l = &chain.levels[idx - 1];
        sample_bilinear_buf(&l.data, l.width, l.height, uv, repeat)
    }
}

/// 对任意 RGBA8 缓冲做双线性采样（[`Image::sample_bilinear`] 与 mip 层共用）。
///
/// 数学与 0.5.0 的 `sample_bilinear` 完全一致 —— 只是把"缓冲+尺寸"参数化，
/// 让 mip 层走同一条路径，避免两份实现漂移。
fn sample_bilinear_buf(
    data: &[u8],
    width: u32,
    height: u32,
    uv: Vec2,
    repeat: bool,
) -> [f32; 4] {
    let (mut u, mut v) = (uv.x, uv.y);
    if repeat {
        u -= u.floor();
        v -= v.floor();
    } else {
        u = u.clamp(0.0, 1.0);
        v = v.clamp(0.0, 1.0);
    }
    // 映射到纹素坐标（像素中心对齐：u=0.5/w 应对应第 0 个纹素中心）
    let fx = u * width as f32 - 0.5;
    let fy = v * height as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;

    let fetch = |x: i32, y: i32| -> [f32; 4] {
        let (xx, yy) = if repeat {
            // 平铺时按尺寸取模，保证负坐标也正确回绕
            (
                x.rem_euclid(width as i32) as u32,
                y.rem_euclid(height as i32) as u32,
            )
        } else {
            (
                x.clamp(0, width as i32 - 1) as u32,
                y.clamp(0, height as i32 - 1) as u32,
            )
        };
        if xx >= width || yy >= height {
            return [0.0; 4];
        }
        let o = ((yy * width + xx) * 4) as usize;
        [
            data[o] as f32 / 255.0,
            data[o + 1] as f32 / 255.0,
            data[o + 2] as f32 / 255.0,
            data[o + 3] as f32 / 255.0,
        ]
    };
    let c00 = fetch(x0 as i32, y0 as i32);
    let c10 = fetch(x0 as i32 + 1, y0 as i32);
    let c01 = fetch(x0 as i32, y0 as i32 + 1);
    let c11 = fetch(x0 as i32 + 1, y0 as i32 + 1);

    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let a = c00[i] + (c10[i] - c00[i]) * tx;
        let b = c01[i] + (c11[i] - c01[i]) * tx;
        out[i] = a + (b - a) * ty;
    }
    out
}

/// 图像的放置方式：把源图 `[0,1]²` 线性映射到设备空间的一个平行四边形。
///
/// 用**三个点**（原点 + 两个轴向量）而非四个角：图像放置本质是仿射映射，
/// 三点即完整确定它，且天然表达旋转/切变；四点还会引入"四边形是否合法"的问题。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImagePlacement {
    /// `uv = (0,0)` 对应的设备坐标
    pub origin: Vec2,
    /// `uv = (1,0)` 相对 `origin` 的向量
    pub u_axis: Vec2,
    /// `uv = (0,1)` 相对 `origin` 的向量
    pub v_axis: Vec2,
    /// 是否平铺（否则钳制到边缘）
    pub repeat: bool,
}

impl ImagePlacement {
    /// 轴对齐放置：把图像放进设备空间的矩形。`flip_y` 用于图像 y 向下与
    /// 引擎设备坐标一致的情况（本项目设备坐标 y 向下，故通常传 `false`）。
    pub fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> ImagePlacement {
        ImagePlacement {
            origin: Vec2::new(x0, y0),
            u_axis: Vec2::new(x1 - x0, 0.0),
            v_axis: Vec2::new(0.0, y1 - y0),
            repeat: false,
        }
    }

    /// 设备坐标 → 图像空间 `uv`。
    ///
    /// 解 `p = origin + u*u_axis + v*v_axis`。行列式接近 0（退化）时返回 `None`，
    /// 由调用方决定降级行为 —— 而不是悄悄除零产出 NaN 像素。
    #[inline]
    pub fn uv_at(&self, p: Vec2) -> Option<Vec2> {
        let (a, b) = (self.u_axis.x, self.v_axis.x);
        let (c, d) = (self.u_axis.y, self.v_axis.y);
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return None;
        }
        let (px, py) = (p.x - self.origin.x, p.y - self.origin.y);
        // 2×2 逆矩阵
        Some(Vec2::new((d * px - b * py) / det, (-c * px + a * py) / det))
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

/// 一个字形的轮廓（**em 单位**：1.0 = 字号，原点在字形基线左端）。
///
/// 用曲线元素序列表达，与 [`Path`] 同构——因此可零成本喂给既有管线。
/// 引擎不关心它来自 TrueType/OpenType 的哪种表，只要求坐标是 em 单位的
/// 轮廓坐标（后续按 `font_size` 与变换缩放）。
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphOutline {
    /// 轮廓元素；坐标单位为 em（相对字号归一化）
    pub els: Vec<PathEl>,
}

impl GlyphOutline {
    pub fn new(els: Vec<PathEl>) -> GlyphOutline {
        GlyphOutline { els }
    }
}

/// 字体的**内部稳定标识** —— 不是路径，也不是"栈里的下标"。
///
/// # 为什么必须是内部 ID
///
/// 它有两个用途：字形缓存的键（见 [`GlyphKey`]），以及"是不是同一个字体"的判定。
/// 路径会变（换机器、换部署目录）、下标会变（栈顺序调整），拿它们做键会让缓存
/// **悄悄命中错误的字形**（画出来是另一个字体的字）。所以这里用字体**自己声明的**
/// 标识，按优先级取第一个可用的：
///
/// 1. PostScript 名（`name` 表 ID 6）—— 字体厂商给的标准唯一名
/// 2. Unique ID（ID 3）
/// 3. Full name（ID 4）
/// 4. Family + Subfamily（ID 1 + 2）
/// 5. 都没有时退到**字体字节的哈希**（同一份字节仍稳定，但两份"内容相同的不同文件"会撞成一个）
///
/// 计算是 FNV-1a 64，并带一个**来源标签前缀**（`ps:` / `uid:` / …），
/// 因此不同来源不会互相碰撞。
///
/// 定义在 `amiki-scene` 而不是 `amiki-text`：`GlyphKey` 在这里，
/// 而依赖方向是 `text → scene`（不能反过来）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FontId(u64);

impl FontId {
    /// 未知/未指定字体：**旧单字体路径**的缓存键用它（那时调用方只给字形 id，
    /// 拿不到字体身份）。真实字体 ID 永远不是 0。
    pub const UNKNOWN: FontId = FontId(0);

    pub const fn from_bits(bits: u64) -> FontId {
        FontId(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for FontId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            write!(f, "FontId(unknown)")
        } else {
            write!(f, "FontId({:016x})", self.0)
        }
    }
}

/// 一个已就位的字形：id 用于缓存，位置以**em 单位**给出（基线原点向左为 0）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionedGlyph {
    /// 字形标识（调用方定义：字体内 glyph id 即可），供缓存分键
    pub id: u32,
    /// 该字形来自**字体栈里的第几个字体**。
    ///
    /// 单字体路径恒为 0（老代码仍用 [`TextRun::push_glyph`]，行为不变）。
    /// 字体回退场景下，`expand` 必须靠它找到正确的字体取轮廓 ——
    /// 同一个 `id` 在不同字体里是完全不同的字形。
    pub font_index: usize,
    /// 该字形在原文中的**字符字节偏移**（rustybuzz 的 cluster）。
    ///
    /// 连字会把多个字符映射到同一个 cluster —— 这正是断行需要的语义（连字不可拆）。
    /// 旧单字体路径不填（0）。
    pub cluster: u32,
    /// 基线原点（em 单位）
    pub x: f32,
    pub y: f32,
}

/// 一段文本：字号 + 已 shaping 的字形序列。
///
/// `size` 是**字号**（em 到设备空间的缩放基准）。若调用方已把字号折进变换，
/// 传 `1.0` 即可——两条路等价，引擎不做假设。
#[derive(Clone, Debug, PartialEq)]
pub struct TextRun {
    pub size: f32,
    pub glyphs: Vec<PositionedGlyph>,
}

impl TextRun {
    pub fn new(size: f32) -> TextRun {
        TextRun { size, glyphs: Vec::new() }
    }

    /// 追加一个字形（**旧入口**：字体下标 0、不带 cluster，行为与 0.6.1 一致）。
    pub fn push_glyph(&mut self, id: u32, x: f32, y: f32) -> &mut Self {
        self.push_glyph_in(0, id, x, y, 0)
    }

    /// 追加一个字形，并指明它来自哪个字体、对应原文的哪个字节偏移。
    pub fn push_glyph_in(
        &mut self,
        font_index: usize,
        id: u32,
        x: f32,
        y: f32,
        cluster: u32,
    ) -> &mut Self {
        self.glyphs.push(PositionedGlyph { id, font_index, cluster, x, y });
        self
    }
}

/// 字形轮廓缓存键：`(字体, glyph id, 量化后的字号)`。
///
/// 为什么必须缓存：一个字形会被成百上千次复用（每个字符位置一次），
/// 每次都重新构造 `Path` 会让 CPU 侧成为瓶颈。
/// 为什么按**量化**字号分键：同一段文本常以若干离散字号出现，
/// 量化后命中率远高于精确浮点相等，且误差只是一次查表偏移。
/// 为什么带**字体 ID**：同一个 glyph id 在不同字体里是不同字形，
/// 不带字体维度就会跨字体误命中（字体回退场景下必然发生）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlyphKey {
    /// 字体内部 ID（旧路径为 [`FontId::UNKNOWN`]）
    pub font_id: FontId,
    pub id: u32,
    /// 字号 × 16 后取整（1/16 px 粒度）
    pub size_q: u32,
}

impl GlyphKey {
    pub const SIZE_QUANT: f32 = 16.0;

    /// 旧入口：不带字体身份（等价于 0.6.1 的键）。
    pub fn new(id: u32, size: f32) -> GlyphKey {
        GlyphKey::with_font(FontId::UNKNOWN, id, size)
    }

    /// 带字体身份的键（字体回退路径用它）。
    pub fn with_font(font_id: FontId, id: u32, size: f32) -> GlyphKey {
        GlyphKey { font_id, id, size_q: (size.max(0.0) * Self::SIZE_QUANT).round() as u32 }
    }
}

/// 把 [`TextRun`] 展开为若干 `(Path, 变换)`——每个字形一个。
///
/// `lookup` 提供字形轮廓。返回的变换把 em 单位轮廓映射到设备空间：
/// 缩放 `size`、平移字形位置。调用方再乘上文本自身的外部变换。
///
/// 展开成**多个独立 path** 而不是合并成一条大路径，有两个理由：
/// 1. 每个字形各自成 draw，可被分箱与 damage region 独立处理（改一个字只脏一小块）；
/// 2. 避免合并后产生大量子路径带来的绕数判定开销。
pub fn expand_text_run(
    run: &TextRun,
    mut lookup: impl FnMut(u32) -> Option<GlyphOutline>,
) -> Vec<(Path, Affine)> {
    expand_text_run_with(run, |_font_index, id| lookup(id))
}

/// 字体回退版 [`expand_text_run`]：`lookup` 同时收到**字体下标**与字形 id。
///
/// 多字体的一段文本里，同一个 `glyph id` 在不同字体中是**不同字形**，
/// 因此取轮廓必须带上 [`PositionedGlyph::font_index`]；`lookup` 通常实现为
/// `|fi, id| stack.get(fi)?.outline(id).ok()`。
pub fn expand_text_run_with(
    run: &TextRun,
    mut lookup: impl FnMut(usize, u32) -> Option<GlyphOutline>,
) -> Vec<(Path, Affine)> {
    let mut out = Vec::with_capacity(run.glyphs.len());
    for g in &run.glyphs {
        let Some(o) = lookup(g.font_index, g.id) else { continue };
        if o.els.is_empty() {
            continue;
        }
        // em → 设备：先按字号缩放，再平移到字形基线原点
        let scale = Affine::scale(run.size);
        let shift = Affine::translate(Vec2::new(g.x, g.y));
        out.push((Path::from_els(o.els.clone()), scale.then(shift)));
    }
    out
}

/// 带缓存的文本展开器：由调用方持有，跨帧复用。
///
/// 缓存的是**已按字号缩放的路径**（`Affine::scale(size)` 折入坐标），
/// 因此同一 `(id, size)` 第二次出现时不再重复缩放整条轮廓。
#[derive(Default)]
pub struct TextCache {
    /// (键 → 已缩放轮廓)。用 Vec 而非 HashMap：字形数量通常很小（几十到几百），
    /// 线性查找比哈希更快，也避免引入哈希依赖。
    entries: Vec<(GlyphKey, Path)>,
    hits: u64,
    misses: u64,
}

impl TextCache {
    pub fn new() -> TextCache {
        TextCache::default()
    }

    /// 命中率（诊断用）。
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 展开一段文本，命中缓存则复用已缩放轮廓。
    ///
    /// `outline` 只在未命中时调用（避免无谓的轮廓提取开销）。
    pub fn expand<F>(&mut self, run: &TextRun, mut outline: F) -> Vec<(Path, Affine)>
    where
        F: FnMut(u32) -> Option<GlyphOutline>,
    {
        self.expand_with(run, |_| FontId::UNKNOWN, |_fi, id| outline(id))
    }

    /// 字体回退版 [`TextCache::expand`]：缓存键带**字体内部 ID**。
    ///
    /// 两个闭包分开是有意的：`font_of` 很便宜（`stack.get(i).map(Font::id)`），
    /// 而键必须在查缓存**之前**算出来；`outline` 才是贵的那个，只在未命中时调用。
    /// 若只给一个闭包，就得为了拿字体 ID 而先把轮廓提出来 —— 缓存也就白做了。
    pub fn expand_with<FF, FO>(
        &mut self,
        run: &TextRun,
        mut font_of: FF,
        mut outline: FO,
    ) -> Vec<(Path, Affine)>
    where
        FF: FnMut(usize) -> FontId,
        FO: FnMut(usize, u32) -> Option<GlyphOutline>,
    {
        let mut out = Vec::with_capacity(run.glyphs.len());
        for g in &run.glyphs {
            let key = GlyphKey::with_font(font_of(g.font_index), g.id, run.size);
            // 命中：线性查已缩放轮廓
            let idx = self.entries.iter().position(|(k, _)| *k == key);
            let path = match idx {
                Some(i) => {
                    self.hits += 1;
                    self.entries[i].1.clone()
                }
                None => {
                    self.misses += 1;
                    let Some(o) = outline(g.font_index, g.id) else { continue };
                    if o.els.is_empty() {
                        continue;
                    }
                    // 把字号折进坐标，之后只需平移 → 同一字号的复用不再缩放
                    let s = run.size;
                    let scaled: Vec<PathEl> = o
                        .els
                        .iter()
                        .map(|el| match *el {
                            PathEl::MoveTo(p) => PathEl::MoveTo(p * s),
                            PathEl::LineTo(p) => PathEl::LineTo(p * s),
                            PathEl::QuadTo(c, p) => PathEl::QuadTo(c * s, p * s),
                            PathEl::CubicTo(a, b, p) => PathEl::CubicTo(a * s, b * s, p * s),
                            PathEl::Close => PathEl::Close,
                        })
                        .collect();
                    let p = Path::from_els(scaled);
                    self.entries.push((key, p.clone()));
                    p
                }
            };
            // 位置也要乘以字号：`expand_text_run` 用 `scale.then(translate)`，
            // 平移量同样被缩放。缓存把字号折进了**坐标**，因此这里必须显式
            // 缩放位置，否则 size≠1 时字形会被摆到错误位置（曾用测试抓出此 bug）。
            out.push((
                path,
                Affine::translate(Vec2::new(g.x * run.size, g.y * run.size)),
            ));
        }
        out
    }
}

/// 裁剪层：变换 + 路径。覆盖率在独立 pass 中渲染，绘制时逐像素乘入。
#[derive(Clone, Debug, PartialEq)]
pub struct Clip {
    pub transform: Affine,
    pub path: Path,
}

/// 一次绘制：变换 + 路径 + 样式 + 混合模式 + 裁剪引用。
///
/// 派生 `PartialEq` 供 M4 的 [`edit_damage`] 判定"这条 draw 是否被改动"：
/// 任一字段不同即视为改动（含浮点按位比较——对 damage 判定而言这是**正确**语义，
/// 只要有一个比特变了就必须重算，宁可多标不可漏标）。
#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    pub transform: Affine,
    pub path: Path,
    pub paint: Paint,
    pub fill_rule: FillRule,
    pub blend: BlendMode,
    /// 引用 `Scene::clips` 的索引（≤ [`MAX_DRAW_CLIPS`]，超出截断）。
    pub clips: Vec<u32>,
}

/// 一层离屏内容：先渲到独立缓冲，套滤镜，再按变换/混合合成回父场景。
///
/// # 为什么需要它
///
/// 模糊、阴影这类效果**不能逐图元计算** —— 它们要读取"周围像素"，
/// 而周围像素只有在整层画完之后才存在。因此必须把一批绘制放到独立缓冲里
/// 先合成完，再对整层做一次滤波。这就是"层树"存在的原因。
///
/// # 边界由调用方显式给出
///
/// 一个被模糊的层，其影响范围会**超出内容本身的包围盒**（模糊半径向外扩散）。
/// 若引擎自行按内容推导边界，就必须把每个滤镜的外扩量算准，任何一处保守不足
/// 都会导致边缘被切掉（且很难察觉）。因此这里要求调用方显式给出 `bounds`：
/// 语义清楚，且"图层多大"本来就是调用方最清楚的事。
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// 离屏缓冲在**设备空间**的范围（像素），空则跳过该层
    pub bounds: LayerBounds,
    /// 层内绘制（设备空间语义，与父场景一致）
    pub scene: Scene,
    /// 合成前应用的滤镜
    pub filter: Filter,
    /// 层整体不透明度（0..1），乘到每个像素的 alpha 上
    pub opacity: f32,
    /// 合回父场景时的混合模式
    pub blend: BlendMode,
}

/// 离屏层范围（设备像素，整数以避免半像素采样歧义）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerBounds {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl LayerBounds {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> LayerBounds {
        LayerBounds { x, y, width, height }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// 按半径向外扩张（用于模糊/阴影需要的外扩）。
    pub fn inflate(&self, r: f32) -> LayerBounds {
        let d = r.ceil().max(0.0) as i32;
        LayerBounds {
            x: self.x - d,
            y: self.y - d,
            width: self.width + (d as u32) * 2,
            height: self.height + (d as u32) * 2,
        }
    }

    /// 滤镜的**作用范围**：层内容经该滤镜后可能影响到的设备区域。
    ///
    /// 这是层与 damage region 的交汇点，因此它属于**场景语义**而不是光栅化细节 ——
    /// 放在这里（而不是 `amiki-raster`）是为了让 `edit_damage` 也能用它算脏区。
    /// `amiki-raster` 转调本方法，两边不会漂移。
    ///
    /// - `Blur`：三倍标准差之外的高斯尾部可忽略（覆盖 99.7%）
    /// - `DropShadow`：先按 sigma 外扩，再按偏移平移
    pub fn inflate_by_filter(&self, filter: Filter) -> LayerBounds {
        match filter {
            Filter::None => *self,
            Filter::Blur { sigma } => self.inflate(sigma * 3.0),
            Filter::DropShadow { dx, dy, sigma, .. } => {
                let b = self.inflate(sigma * 3.0);
                // 阴影是 `b` 平移 `(dx, dy)` 后的副本，故作用范围是**两者的并集**
                // 的包围盒：`[b.x + min(0,dx), b.x + b.width + max(0,dx)]`。
                //
                // 这里曾经把原点也加上 `dx`（`x: b.x + dx`）—— 那是**平移**而不是
                // **并集**，于是正偏移时左边/上边被裁掉一块。sigma 大时模糊外扩
                // 恰好盖住了这个错误，`sigma = 0` 才暴露（实测与参考差 156/255）。
                let sx = dx.floor() as i32;
                let sy = dy.floor() as i32;
                LayerBounds {
                    x: b.x + sx.min(0),
                    y: b.y + sy.min(0),
                    width: b.width + dx.abs().ceil() as u32,
                    height: b.height + dy.abs().ceil() as u32,
                }
            }
        }
    }
}

impl Layer {
    /// 本层经滤镜后可能影响的设备区域。
    pub fn inflated_bounds(&self) -> LayerBounds {
        self.bounds.inflate_by_filter(self.filter)
    }

    /// 本层"内容"的判等键：决定**滤波后的层缓冲**是否需要重算。
    ///
    /// 只含真正影响缓冲内容的字段：范围、层内场景、滤镜。
    /// `opacity` 与 `blend` 在**合成时**才用，改了它们不需要重新滤波
    /// （但需要重新合成 —— 那由脏区负责）。
    pub fn buffer_key(&self) -> (&LayerBounds, &Scene, &Filter) {
        (&self.bounds, &self.scene, &self.filter)
    }
}

/// 层的滤镜。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    /// 不做滤波（仅用于离屏合成 / 分组不透明度）
    None,
    /// 高斯模糊（`sigma` 为标准差，单位设备像素）
    Blur { sigma: f32 },
    /// 投影：先把层内容按 `dx,dy` 偏移并模糊后着色，再与本源叠加
    DropShadow {
        dx: f32,
        dy: f32,
        sigma: f32,
        color: Color,
    },
}

/// 场景：画布尺寸 + 有序绘制列表（画家算法，后绘制在上）+ 裁剪层 + 离屏层。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub draws: Vec<Draw>,
    pub clips: Vec<Clip>,
    /// M6：离屏层（滤镜）。在 `draws` **之后**按顺序合成 ——
    /// 保证"层内绘制先完成，再整层滤波"的语义，也避免层内绘制被父场景
    /// 后续绘制意外穿插。
    pub layers: Vec<Layer>,
}

impl Scene {
    pub fn new(width: u32, height: u32) -> Scene {
        Scene { width, height, draws: Vec::new(), clips: Vec::new(), layers: Vec::new() }
    }

    /// 推入一个离屏层，并把子场景交给闭包填充。
    ///
    /// 子场景尺寸与父场景相同（层内坐标即设备坐标，`bounds` 只决定
    /// 实际分配的离屏范围与合成位置）——这样层内绘制可以用与父场景
    /// 完全相同的坐标思考，不必做原点平移，减少一类坐标错误。
    pub fn push_layer<F>(&mut self, mut layer: Layer, build: F) -> &mut Self
    where
        F: FnOnce(&mut Scene),
    {
        layer.scene.width = self.width;
        layer.scene.height = self.height;
        let mut sub = std::mem::take(&mut layer.scene);
        build(&mut sub);
        layer.scene = sub;
        self.layers.push(layer);
        self
    }

    /// 场景是否**没有任何可绘制内容**（含层内内容）。
    pub fn is_empty_content(&self) -> bool {
        self.draws.is_empty()
            && self.layers.iter().all(|l| l.scene.is_empty_content() || l.bounds.is_empty())
    }

    pub fn push(&mut self, transform: Affine, path: Path, paint: Paint) -> &mut Self {
        self.draws.push(Draw {
            transform,
            path,
            paint,
            fill_rule: FillRule::NonZero,
            blend: BlendMode::Normal,
            clips: Vec::new(),
        });
        self
    }

    pub fn push_blend(
        &mut self,
        transform: Affine,
        path: Path,
        paint: Paint,
        blend: BlendMode,
    ) -> &mut Self {
        self.draws.push(Draw {
            transform,
            path,
            paint,
            fill_rule: FillRule::NonZero,
            blend,
            clips: Vec::new(),
        });
        self
    }

    /// 登记一个裁剪层，返回其索引。
    pub fn push_clip(&mut self, transform: Affine, path: Path) -> u32 {
        self.clips.push(Clip { transform, path });
        (self.clips.len() - 1) as u32
    }
}

/// 便利函数：把一条曲线元素展开为（控制点序列）供后端消费。
/// 后端实际压平在 amiki-raster / GPU 侧完成；此函数仅用于测试与工具。
pub fn curve_points(el: &PathEl) -> Option<(Vec2, Vec<Vec2>)> {
    match *el {
        PathEl::QuadTo(c, p) => Some((c, vec![p])),
        PathEl::CubicTo(c1, c2, p) => Some((c1, vec![c2, p])),
        _ => None,
    }
}

/// 从元素流构造曲线对象（测试用）。
pub fn quad_of(p0: Vec2, el: &PathEl) -> Option<QuadBez> {
    if let PathEl::QuadTo(c, p) = *el {
        Some(QuadBez::new(p0, c, p))
    } else {
        None
    }
}

pub fn cubic_of(p0: Vec2, el: &PathEl) -> Option<CubicBez> {
    if let PathEl::CubicTo(c1, c2, p) = *el {
        Some(CubicBez::new(p0, c1, c2, p))
    } else {
        None
    }
}

// ============================================================================
// M2c：紧凑编码流 v1（AMK1）—— 稳定二进制交换格式
// ============================================================================
//
// 用途：场景的单次紧凑上传 / 磁盘缓存 / 网络传输 / 跨语言绑定。
// v1 为 CPU 解码的稳定格式；GPU 端直接解码消费在 M3 规划中。
//
// 布局（小端）：
//   "AMK1" | version u32 | width u32 | height u32 | n_clips u32 | n_draws u32
//   每 clip：transform 6×f32 | path
//   每 draw：transform 6×f32 | color 4×f32 | blend/flags u32 | n_refs u32 | refs | path
//   path = count u32 | count × (tag u8 + 坐标 f32)
//   tag：0 MoveTo, 1 LineTo, 2 QuadTo, 3 CubicTo, 4 Close
//
// v2（0.8.0）：draw 的那个 u32 从"纯 blend id"升级为**标志字** ——
// bit 0..3 = blend id，bit 12 = fill_rule（0 = NonZero，1 = EvenOdd）。
// 布局**没有变化**（不加字段、不扩宽）；v1 流里 bit 12 恒为 0，解码即 NonZero。
// v1 只解不编：编码器统一写 v2。

pub const STREAM_MAGIC: &[u8; 4] = b"AMK1";
/// 当前编码版本。
///
/// # v1 → v2（0.8.0）
///
/// v2 与 v1 的**布局完全相同**，唯一差别是每个 draw 的 blend 字里
/// **bit 12 被定义为 `fill_rule`**（v1 该位恒为 0 ⇒ 解码为 `NonZero`）。
///
/// - **只解不编**：编码器统一写 v2（v1 编码器已不存在，避免死代码漂移）；
///   解码器同时接受 v1 与 v2。
/// - 未知版本（≥3）**显式报错**，不静默错解。
pub const STREAM_VERSION: u32 = 2;
/// v1 的版本号（只解不编）。
pub const STREAM_VERSION_V1: u32 = 1;
/// blend 字里 `fill_rule` 占的位（与计划的 bit 12 一致）。
pub const FILL_RULE_BIT: u32 = 12;

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// 写入一个颜色（4×f32：r, g, b, a）。
fn put_color(buf: &mut Vec<u8>, c: &Color) {
    put_f32(buf, c.r);
    put_f32(buf, c.g);
    put_f32(buf, c.b);
    put_f32(buf, c.a);
}

fn put_path(buf: &mut Vec<u8>, path: &Path) {
    put_u32(buf, path.els.len() as u32);
    for el in &path.els {
        match *el {
            PathEl::MoveTo(p) => {
                buf.push(0);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::LineTo(p) => {
                buf.push(1);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::QuadTo(c, p) => {
                buf.push(2);
                put_f32(buf, c.x);
                put_f32(buf, c.y);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::CubicTo(c1, c2, p) => {
                buf.push(3);
                put_f32(buf, c1.x);
                put_f32(buf, c1.y);
                put_f32(buf, c2.x);
                put_f32(buf, c2.y);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::Close => buf.push(4),
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("编码流越界".into());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32x(&mut self, n: usize) -> Result<Vec<f32>, String> {
        (0..n).map(|_| self.f32()).collect()
    }
    fn point(&mut self) -> Result<Vec2, String> {
        let v = self.f32x(2)?;
        Ok(Vec2::new(v[0], v[1]))
    }
    fn path(&mut self) -> Result<Path, String> {
        let count = self.u32()? as usize;
        let mut els = Vec::with_capacity(count);
        for _ in 0..count {
            let tag = self.take(1)?[0];
            els.push(match tag {
                0 => PathEl::MoveTo(self.point()?),
                1 => PathEl::LineTo(self.point()?),
                2 => {
                    let c = self.point()?;
                    PathEl::QuadTo(c, self.point()?)
                }
                3 => {
                    let c1 = self.point()?;
                    let c2 = self.point()?;
                    PathEl::CubicTo(c1, c2, self.point()?)
                }
                4 => PathEl::Close,
                _ => return Err(format!("未知路径元素 tag {tag}")),
            });
        }
        Ok(Path { els })
    }
}

fn put_transform(buf: &mut Vec<u8>, t: &Affine) {
    for v in [t.a, t.b, t.c, t.d, t.e, t.f] {
        put_f32(buf, v);
    }
}

fn read_transform(r: &mut Reader) -> Result<Affine, String> {
    let v = r.f32x(6)?;
    Ok(Affine { a: v[0], b: v[1], c: v[2], d: v[3], e: v[4], f: v[5] })
}

/// 场景编码为紧凑字节流。
pub fn encode_scene(scene: &Scene) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256 + scene.draws.len() * 64);
    buf.extend_from_slice(STREAM_MAGIC);
    put_u32(&mut buf, STREAM_VERSION);
    put_u32(&mut buf, scene.width);
    put_u32(&mut buf, scene.height);
    put_u32(&mut buf, scene.clips.len() as u32);
    put_u32(&mut buf, scene.draws.len() as u32);
    for clip in &scene.clips {
        put_transform(&mut buf, &clip.transform);
        put_path(&mut buf, &clip.path);
    }
    for draw in &scene.draws {
        put_transform(&mut buf, &draw.transform);
        match &draw.paint {
            Paint::Solid(c) => {
                put_u32(&mut buf, 0); // paint 类型标签：0 = 实心
                put_color(&mut buf, c);
            }
            Paint::Gradient(g) => {
                put_u32(&mut buf, 1); // 1 = 渐变
                // 几何：kind 标签 + 参数
                match g.kind {
                    GradientKind::Linear { from, to } => {
                        put_u32(&mut buf, 0);
                        put_f32(&mut buf, from.x);
                        put_f32(&mut buf, from.y);
                        put_f32(&mut buf, to.x);
                        put_f32(&mut buf, to.y);
                    }
                    GradientKind::Radial { center, radius } => {
                        put_u32(&mut buf, 1);
                        put_f32(&mut buf, center.x);
                        put_f32(&mut buf, center.y);
                        put_f32(&mut buf, radius);
                        put_f32(&mut buf, 0.0); // 占位，保持定长便于解码
                    }
                    GradientKind::Sweep { center, start } => {
                        put_u32(&mut buf, 2);
                        put_f32(&mut buf, center.x);
                        put_f32(&mut buf, center.y);
                        put_f32(&mut buf, start);
                        put_f32(&mut buf, 0.0);
                    }
                }
                put_u32(&mut buf, u32::from(g.repeat));
                put_u32(&mut buf, g.stops.len() as u32);
                for s in &g.stops {
                    put_f32(&mut buf, s.t);
                    put_color(&mut buf, &s.color);
                }
            }
            Paint::Image { image, placement } => {
                // M6 现状：AMK1 流**不内嵌位图**（一张 717×692 的图就有约 2 MB，
                // 内嵌会让"场景流"退化成图像容器，失去轻量优势）。
                // 因此这里写入一个**显式的"含图像"标记**：解码方会明确报错，
                // 而不是静默丢掉图像导致画面悄悄缺一块。
                put_u32(&mut buf, 2); // 2 = 图像（当前不可序列化）
                put_u32(&mut buf, image.width);
                put_u32(&mut buf, image.height);
                put_f32(&mut buf, placement.origin.x);
                put_f32(&mut buf, placement.origin.y);
                put_f32(&mut buf, placement.u_axis.x);
                put_f32(&mut buf, placement.u_axis.y);
                put_f32(&mut buf, placement.v_axis.x);
                put_f32(&mut buf, placement.v_axis.y);
                put_u32(&mut buf, u32::from(placement.repeat));
            }
        }
        // v2：这个 u32 从"纯 blend id"升级为**标志字** —— bit 0..3 仍是 blend，
        // bit 12 是 fill_rule（0 = NonZero 默认，1 = EvenOdd）。
        // 布局**一字未改**（没有加字段、没有扩宽），v1 流里这一位恒为 0。
        put_u32(&mut buf, draw.blend.id() | (draw.fill_rule.bit() << FILL_RULE_BIT));
        put_u32(&mut buf, draw.clips.len() as u32);
        for &c in &draw.clips {
            put_u32(&mut buf, c);
        }
        put_path(&mut buf, &draw.path);
    }
    buf
}

/// 从字节流解码场景（格式校验失败返回 Err）。
pub fn decode_scene(bytes: &[u8]) -> Result<Scene, String> {
    let mut r = Reader { data: bytes, pos: 0 };
    if r.take(4)? != STREAM_MAGIC {
        return Err("编码流 magic 不符（需要 AMK1）".into());
    }
    let version = r.u32()?;
    // v1 与 v2 都接受；差别只在 blend 字的 bit 12 是否**定义**为 fill_rule。
    if version != STREAM_VERSION && version != STREAM_VERSION_V1 {
        return Err(format!("编码流版本不支持: {version}"));
    }
    let v2 = version >= 2;
    let width = r.u32()?;
    let height = r.u32()?;
    let n_clips = r.u32()?;
    let n_draws = r.u32()?;
    let mut scene = Scene::new(width, height);
    for _ in 0..n_clips {
        let t = read_transform(&mut r)?;
        let path = r.path()?;
        scene.push_clip(t, path);
    }
    for _ in 0..n_draws {
        let t = read_transform(&mut r)?;
        let paint = match r.u32()? {
            0 => {
                let c = r.f32x(4)?;
                Paint::Solid(Color { r: c[0], g: c[1], b: c[2], a: c[3] })
            }
            1 => {
                let kind_id = r.u32()?;
                let a = r.f32x(4)?;
                let kind = match kind_id {
                    0 => GradientKind::Linear {
                        from: Vec2::new(a[0], a[1]),
                        to: Vec2::new(a[2], a[3]),
                    },
                    1 => GradientKind::Radial { center: Vec2::new(a[0], a[1]), radius: a[2] },
                    2 => GradientKind::Sweep { center: Vec2::new(a[0], a[1]), start: a[2] },
                    other => return Err(format!("渐变类型不支持: {other}")),
                };
                let repeat = r.u32()? != 0;
                let n = r.u32()? as usize;
                let mut stops = Vec::with_capacity(n);
                for _ in 0..n {
                    let st = r.f32()?;
                    let c = r.f32x(4)?;
                    stops.push(GradientStop::new(
                        st,
                        Color { r: c[0], g: c[1], b: c[2], a: c[3] },
                    ));
                }
                if stops.len() < 2 {
                    return Err("渐变色标少于 2 个".into());
                }
                Paint::Gradient(std::sync::Arc::new(Gradient { kind, stops, repeat }))
            }
            2 => {
                // 图像：AMK1 不内嵌位图（见编码侧说明）。
                // 明确报错，而不是返回一个缺了图像的场景。
                let (w, h) = (r.u32()?, r.u32()?);
                return Err(format!(
                    "AMK1 流不内嵌位图，无法解码 {w}×{h} 的图像绘制；\
                     请在解码后重新插入图像"
                ));
            }
            other => return Err(format!("paint 类型不支持: {other}")),
        };        let flags = r.u32()?;
        // v2：bit 12 是 fill_rule；v1：该位保留，恒为 NonZero。
        let fill_rule = if v2 {
            FillRule::from_bit(flags >> FILL_RULE_BIT)
        } else {
            FillRule::NonZero
        };
        // blend id 只占低 4 位（0..15），掩掉标志位再查表。
        let blend_id = flags & 0x0FFF;
        let n_refs = r.u32()? as usize;
        let mut clips = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            clips.push(r.u32()?);
        }
        let path = r.path()?;
        let blend = match blend_id {
            0 => BlendMode::Normal,
            1 => BlendMode::Multiply,
            2 => BlendMode::Screen,
            3 => BlendMode::Overlay,
            4 => BlendMode::Darken,
            5 => BlendMode::Lighten,
            6 => BlendMode::ColorDodge,
            7 => BlendMode::ColorBurn,
            8 => BlendMode::HardLight,
            9 => BlendMode::SoftLight,
            10 => BlendMode::Difference,
            11 => BlendMode::Exclusion,
            12 => BlendMode::Hue,
            13 => BlendMode::Saturation,
            14 => BlendMode::Color,
            15 => BlendMode::Luminosity,
            _ => return Err(format!("未知混合模式 id {blend_id}")),
        };
        scene.draws.push(Draw {
            transform: t,
            path,
            paint,
            fill_rule,
            blend,
            clips,
        });
    }
    Ok(scene)
}

// ============================================================================
// M4：damage region —— tile 级脏区模型
// ============================================================================
//
// 目的：让"与画面内容成比例"从**显存**扩展到 **CPU 时间与提交量**。
// 保留模式下，只重算被改动区域覆盖的 tile；未改动区域复用上一帧结果。
//
// 与分箱的一致性：tile 粒度必须等于分箱粒度，否则增量结果会与全量结果
// 边界不一致。`amiki-raster::TILE` / `amiki-gpu` 的分箱均取 16，此处同值。
// 注意这是**跨 crate 的约定**，若将来调整分箱 tile 尺寸，这里必须同步。

/// tile 边长（像素）—— 必须与分箱层的 tile 尺寸一致。
pub const DAMAGE_TILE: u32 = 16;

/// 一个 tile 级脏区：像素矩形 `[x0, x1) × [y0, y1)`，`f32` 以容纳半个像素的边界。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamageRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl DamageRect {
    /// 从任意两点构造（自动规范化，允许传入逆序坐标）。
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> DamageRect {
        DamageRect {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        }
    }

    /// 两矩形求并（外接包围盒）。
    pub fn union(&self, o: &DamageRect) -> DamageRect {
        DamageRect {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }

    /// 是否为空（零面积或退化）。
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
}

/// 脏区的三种状态。用显式枚举而非 `Option`：本模型有**三**种状态
/// （全脏 / 部分脏 / 无脏），用 `Option` 会把"全脏"和"无脏"混为一谈。
#[derive(Clone, Copy, Debug, PartialEq)]
enum DamageState {
    /// 整幅画布都脏 —— 必须走全量渲染。
    All,
    /// 无脏区 —— 静态画面，可完全跳过重绘。
    Clean,
    /// 部分脏区 —— 走增量渲染。
    Rect(DamageRect),
}

/// 一帧的脏区集合（tile 级）。
///
/// 语义要点：
/// - **只增不减**：同一帧内多次 `invalidate` 取并集，不做精确的矩形差集
///   （差集需要凸分解，收益低于复杂度）。
/// - **`All` 是安全默认**：新建的 `Damage` 表示"整幅画布都脏"，
///   因为新出现的帧没有可复用内容。要进入增量模式必须显式 `clear()`。
/// - **tile 量化**：`tiles()` 把像素矩形向外扩到 tile 边界，
///   保证"覆盖到脏像素的 tile 一定会被重算"。
#[derive(Clone, Debug)]
pub struct Damage {
    width: u32,
    height: u32,
    state: DamageState,
}

impl Damage {
    /// 新建 = **整幅全脏**（安全默认：没有可复用内容）。
    pub fn new(width: u32, height: u32) -> Damage {
        Damage { width, height, state: DamageState::All }
    }

    /// 清空为"无脏区"——进入增量模式。之后只有被 `invalidate` 的区域会重算。
    pub fn clear(&mut self) {
        self.state = DamageState::Clean;
    }

    /// 标记整幅画布为脏。
    pub fn invalidate_all(&mut self) {
        self.state = DamageState::All;
    }

    /// 标记一块像素矩形为脏（自动与已有脏区取并集）。
    pub fn invalidate_rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32) {
        let r = DamageRect::new(x0, y0, x1, y1);
        if r.is_empty() {
            return;
        }
        self.state = match self.state {
            DamageState::All => DamageState::All,
            DamageState::Clean => DamageState::Rect(r),
            DamageState::Rect(d) => DamageState::Rect(d.union(&r)),
        };
    }

    /// 标记某条路径的设备空间包围盒为脏。
    ///
    /// M4 的调用点是"场景被改动时"：调用方给出被改 draw 的设备空间 bbox 即可，
    /// 无需自己算 tile。
    pub fn invalidate_bounds(&mut self, bbox: [f32; 4]) {
        self.invalidate_rect(bbox[0], bbox[1], bbox[2], bbox[3]);
    }

    /// 是否整幅全脏（调用方据此决定走全量路径）。
    pub fn is_full(&self) -> bool {
        matches!(self.state, DamageState::All)
    }

    /// 是否无脏区（静态画面：可以完全跳过重绘）。
    pub fn is_empty(&self) -> bool {
        matches!(self.state, DamageState::Clean)
    }

    /// tile 范围 `[tx0, tx1) × [ty0, ty1)`（半开区间）。
    ///
    /// `None` 表示**整幅全脏**（无边界可表达）；`Some` 且区间为空表示无脏区。
    pub fn tiles(&self) -> Option<(u32, u32, u32, u32)> {
        let d = match self.state {
            DamageState::All => return None,
            DamageState::Clean => return Some((0, 0, 0, 0)),
            DamageState::Rect(d) => d,
        };
        let tiles_x = self.width.div_ceil(DAMAGE_TILE);
        let tiles_y = self.height.div_ceil(DAMAGE_TILE);
        // 向外扩到 tile 边界：floor 起、ceil 止，再钳到画布 tile 数
        let tx0 = (d.x0.max(0.0) as u32 / DAMAGE_TILE).min(tiles_x);
        let ty0 = (d.y0.max(0.0) as u32 / DAMAGE_TILE).min(tiles_y);
        let tx1 = ((d.x1.max(0.0).ceil() as u32).div_ceil(DAMAGE_TILE)).min(tiles_x);
        let ty1 = ((d.y1.max(0.0).ceil() as u32).div_ceil(DAMAGE_TILE)).min(tiles_y);
        Some((tx0, ty0, tx1.max(tx0), ty1.max(ty0)))
    }

    /// 脏 tile 数量（全脏时 = 画布全部 tile，无脏区时 = 0）。
    pub fn tile_count(&self) -> u32 {
        match self.state {
            DamageState::All => self.total_tiles(),
            DamageState::Clean => 0,
            DamageState::Rect(_) => match self.tiles() {
                Some((tx0, ty0, tx1, ty1)) => (tx1 - tx0) * (ty1 - ty0),
                None => self.total_tiles(),
            },
        }
    }

    /// 画布总 tile 数。
    pub fn total_tiles(&self) -> u32 {
        self.width.div_ceil(DAMAGE_TILE) * self.height.div_ceil(DAMAGE_TILE)
    }
}

#[cfg(test)]
mod layer_damage_tests {
    use super::*;

    fn layered(dx: f32, sigma: f32) -> Scene {
        let mut s = Scene::new(200, 160);
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(40, 30, 60, 60),
                scene: Scene::new(200, 160),
                filter: Filter::Blur { sigma },
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(50.0 + dx, 40.0, 90.0 + dx, 80.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 0, 0)));
            },
        );
        s
    }

    /// **层内容变化必须产生脏区。**
    ///
    /// 这条曾经是缺失的：`edit_damage` 只遍历 `draws`，而层内绘制不在根 draw 列表里，
    /// 于是层内改动产生**零脏区** —— 增量渲染永远看不到它，画面停在旧内容上，
    /// 而且不报错。这是典型的静默错误。
    #[test]
    fn layer_content_change_dirties_its_inflated_region() {
        let a = layered(0.0, 4.0);
        let b = layered(20.0, 4.0);
        let d = edit_damage(&a, &b);
        assert!(!d.is_full(), "层内改动不应退化为全脏（那说明没算，只是兜底）");
        let (tx0, ty0, tx1, ty1) = d.tiles().expect("应有脏区");
        // 作用范围 = bounds(40,30,60,60) 外扩 3σ=12 → (28,18,112,102)
        // 转成 tile（16px）：tx 1..7，ty 1..7
        assert_eq!(
            (tx0, ty0, tx1, ty1),
            (1, 1, 7, 7),
            "脏区应等于层的作用范围（bounds 外扩 3σ）"
        );
    }

    /// 层内容未变 ⇒ 不产生脏区（否则"静态界面"就拿不到收益）。
    #[test]
    fn unchanged_layer_dirties_nothing() {
        let a = layered(0.0, 4.0);
        let d = edit_damage(&a, &a.clone());
        assert!(d.tiles().is_some(), "全清空也是有效的脏区表达");
        let mut probe = Damage::new(200, 160);
        probe.clear();
        assert_eq!(
            d.tiles(),
            probe.tiles(),
            "层与图元都没变时不应标脏（这是静态界面零开销的前提）"
        );
    }

    /// 只有 sigma 变化（层内场景相同）也要标脏 —— 滤镜改了，作用范围与结果都变。
    #[test]
    fn filter_change_dirties_layer() {
        let a = layered(0.0, 2.0);
        let b = layered(0.0, 9.0);
        let d = edit_damage(&a, &b);
        assert!(!d.is_full(), "滤镜变化不应退化为全脏");
        let (tx0, ty0, tx1, ty1) = d.tiles().expect("应有脏区");
        // 3σ 从 6 涨到 27 → 作用范围 (13,3,127,117) → tile 0..8
        assert_eq!((tx0, ty0), (0, 0), "应覆盖更大的外扩范围");
        assert_eq!((tx1, ty1), (8, 8), "应覆盖更大的外扩范围");
    }

    /// 层被删除 ⇒ 旧层的作用范围要标脏（否则旧画面的层会永远残留）。
    #[test]
    fn removed_layer_dirties_old_region() {
        let a = layered(0.0, 4.0);
        let mut b = Scene::new(200, 160);
        b.draws = a.draws.clone(); // 图元相同，只少了层
        let d = edit_damage(&a, &b);
        assert!(!d.is_full(), "删除层不应退化为全脏");
        assert_eq!(
            d.tiles(),
            Some((1, 1, 7, 7)),
            "应标脏被删层的作用范围"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn damage_defaults_to_full() {
        let d = Damage::new(128, 64);
        assert!(d.is_full(), "新建 Damage 必须是全脏（无可复用内容）");
        assert!(!d.is_empty());
        assert_eq!(d.tiles(), None, "全脏用 None 表达");
        assert_eq!(d.total_tiles(), 8 * 4);
        assert_eq!(d.tile_count(), d.total_tiles());
    }

    #[test]
    fn damage_clear_means_skip() {
        let mut d = Damage::new(128, 64);
        d.clear();
        assert!(d.is_empty(), "clear 后应无脏区（静态画面可完全跳过重绘）");
        assert!(!d.is_full());
        assert_eq!(d.tile_count(), 0);
    }

    #[test]
    fn damage_rect_quantizes_outward() {
        let mut d = Damage::new(128, 64);
        d.clear();
        // 1 像素的改动必须把**覆盖它的那个 tile** 整个标记为脏
        d.invalidate_rect(20.0, 20.0, 21.0, 21.0);
        assert_eq!(d.tiles(), Some((1, 1, 2, 2)), "1px 改动应量化到 tile(1,1)");
        assert_eq!(d.tile_count(), 1);
    }

    #[test]
    fn damage_union_of_rects() {
        let mut d = Damage::new(256, 256);
        d.clear();
        d.invalidate_rect(10.0, 10.0, 20.0, 20.0);   // tile (0,0)
        d.invalidate_rect(200.0, 200.0, 210.0, 210.0); // tile (12,12)
        // 并集是外接包围盒：覆盖 0..14 共 14x14 个 tile（保守，但保证不漏）
        let (tx0, ty0, tx1, ty1) = d.tiles().unwrap();
        assert_eq!((tx0, ty0), (0, 0));
        assert_eq!((tx1, ty1), (14, 14));
        assert_eq!(d.tile_count(), 14 * 14);
        // 保守性的关键断言：两块脏区所在的 tile 都必须落在范围内
        // （注意 tx0/ty0 是 u32，用 == 而非 <= 0，否则断言恒真、毫无价值）
        assert_eq!((tx0, ty0), (0, 0));
        assert!(tx1 > 13 && ty1 > 13, "并集必须覆盖到第二块脏区所在的 tile(13,13)");
    }

    #[test]
    fn damage_clamps_to_canvas() {
        let mut d = Damage::new(70, 40); // tiles_x = ceil(70/16) = 5, tiles_y = 3
        d.clear();
        // 越出画布右/下的改动必须被钳到画布 tile 数内
        d.invalidate_rect(60.0, 30.0, 9999.0, 9999.0);
        let (tx0, ty0, tx1, ty1) = d.tiles().unwrap();
        assert_eq!((tx0, ty0), (3, 1));
        assert_eq!((tx1, ty1), (5, 3), "必须钳到画布 tile 边界");
        // 完全在画布外的改动不产生脏区
        let mut d2 = Damage::new(70, 40);
        d2.clear();
        d2.invalidate_rect(-100.0, -100.0, -50.0, -50.0);
        assert_eq!(d2.tile_count(), 0, "画布外的改动不应产生脏 tile");
    }

    #[test]
    fn damage_invalidate_all_wins() {        let mut d = Damage::new(64, 64);
        d.clear();
        d.invalidate_rect(0.0, 0.0, 5.0, 5.0);
        assert!(!d.is_full());
        d.invalidate_all();
        assert!(d.is_full(), "invalidate_all 后必须回到全脏");
    }

    #[test]
    fn stream_round_trip() {
        let mut scene = Scene::new(800, 600);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(100.0, 100.0), 60.0, false);
        pb.circle(Vec2::new(100.0, 100.0), 30.0, true);
        let clip = scene.push_clip(Affine::rotate(0.3), pb.build());
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(400.0, 300.0), 120.0);
        scene.push_blend(
            Affine::translate(Vec2::new(10.0, 20.0)),
            pb.build(),
            Paint::Solid(Color::rgba8(255, 0, 0, 200)),
            BlendMode::Multiply,
        );
        scene.draws.last_mut().unwrap().clips = vec![clip];
        let mut pb = PathBuilder::new();
        pb.quad_to(Vec2::new(0.0, 100.0), Vec2::new(100.0, 0.0));
        pb.cubic_to(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0), Vec2::new(5.0, 6.0));
        pb.close();
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::TRANSPARENT));

        let bytes = encode_scene(&scene);
        let back = decode_scene(&bytes).expect("解码失败");
        assert_eq!(back.width, 800);
        assert_eq!(back.height, 600);
        assert_eq!(back.clips.len(), 1);
        assert_eq!(back.clips[0].path, scene.clips[0].path);
        assert_eq!(back.draws.len(), 2);
        assert_eq!(back.draws[0].path, scene.draws[0].path);
        assert_eq!(back.draws[0].blend, BlendMode::Multiply);
        assert_eq!(back.draws[0].clips, vec![clip]);
        assert_eq!(back.draws[1].path, scene.draws[1].path);
        assert_eq!(back.draws[1].paint, scene.draws[1].paint);
    }

    #[test]
    fn stream_rejects_garbage() {
        assert!(decode_scene(b"XXXX").is_err());
        let mut bytes = encode_scene(&Scene::new(10, 10));
        bytes[4] = 99; // 版本破坏
        assert!(decode_scene(&bytes).is_err());
    }

    /// AMK1 v2（0.8.0）：`fill_rule` 必须往返，且**混排顺序**不变。
    #[test]
    fn stream_v2_round_trip_keeps_fill_rule() {
        let mut scene = Scene::new(64, 64);
        for (i, rule) in [FillRule::NonZero, FillRule::EvenOdd, FillRule::NonZero, FillRule::EvenOdd]
            .into_iter()
            .enumerate()
        {
            let mut pb = PathBuilder::new();
            pb.rect(i as f32 * 8.0, 2.1, i as f32 * 8.0 + 6.0, 8.3);
            scene.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgb8(0, 0, 0)),
            );
            scene.draws.last_mut().unwrap().fill_rule = rule;
        }
        // 顺带放一条非 Normal 的混合模式，验证标志位没污染 blend
        scene.draws[1].blend = BlendMode::Multiply;

        let bytes = encode_scene(&scene);
        assert_eq!(&bytes[0..4], STREAM_MAGIC);
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            STREAM_VERSION,
            "编码器必须写 v2"
        );

        let back = decode_scene(&bytes).expect("v2 解码失败");
        assert_eq!(back.draws.len(), 4);
        for (i, r) in [FillRule::NonZero, FillRule::EvenOdd, FillRule::NonZero, FillRule::EvenOdd]
            .into_iter()
            .enumerate()
        {
            assert_eq!(back.draws[i].fill_rule, r, "第 {i} 条 draw 的填充规则应往返一致");
        }
        assert_eq!(back.draws[1].blend, BlendMode::Multiply, "blend 不应被标志位污染");
        assert_eq!(back.draws[0].blend, BlendMode::Normal);
        eprintln!("AMK1 v2：4 条 draw（含 2 条 EvenOdd）往返一致，blend 未被污染");
    }

    /// **v1 只解不编**：把版本改回 1 后，即使字节里 bit 12 是 1，也必须解成 NonZero。
    #[test]
    fn stream_v1_bytes_decode_as_nonzero() {
        let mut scene = Scene::new(32, 32);
        let mut pb = PathBuilder::new();
        pb.rect(2.3, 2.1, 28.7, 28.9);
        pb.rect(11.2, 11.3, 21.8, 20.7);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 0, 0)));
        scene.draws.last_mut().unwrap().fill_rule = FillRule::EvenOdd;

        let mut bytes = encode_scene(&scene);
        // 版本字段改回 1（布局相同，所以这就是一份合法的 v1 流）；
        // bit 12 仍然留着 1 —— v1 里它是保留位，必须被忽略。
        bytes[4..8].copy_from_slice(&STREAM_VERSION_V1.to_le_bytes());
        let back = decode_scene(&bytes).expect("v1 必须能解");
        assert_eq!(
            back.draws[0].fill_rule,
            FillRule::NonZero,
            "v1 的 bit 12 是保留位 ⇒ 必须解成 NonZero（不静默错解）"
        );
        eprintln!("AMK1 v1：同一份字节按 v1 解出 NonZero（保留位被忽略）");
    }

    /// 未知版本必须**显式报错**，而不是静默按某个版本解读。
    #[test]
    fn stream_rejects_unknown_version() {
        let mut bytes = encode_scene(&Scene::new(8, 8));
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        let e = decode_scene(&bytes).expect_err("版本 3 必须被拒绝");
        assert!(e.contains("版本"), "错误信息应说明版本问题：{e}");
        eprintln!("AMK1：未知版本被显式拒绝（{e}）");
    }

    // ===================== M5：文本（字形轮廓即路径） =====================

    /// 一个矩形字形：em 单位下覆盖 [0,0.1]×[0,0.6]（模拟一个"竖条"字形）。
    fn bar_glyph() -> GlyphOutline {
        GlyphOutline::new(vec![
            PathEl::MoveTo(Vec2::new(0.0, 0.0)),
            PathEl::LineTo(Vec2::new(0.1, 0.0)),
            PathEl::LineTo(Vec2::new(0.1, 0.6)),
            PathEl::LineTo(Vec2::new(0.0, 0.6)),
            PathEl::Close,
        ])
    }

    /// 一个方形字形（不同占位，用于验证多字形各自独立展开）。
    fn box_glyph() -> GlyphOutline {
        GlyphOutline::new(vec![
            PathEl::MoveTo(Vec2::new(0.0, 0.0)),
            PathEl::LineTo(Vec2::new(0.5, 0.0)),
            PathEl::LineTo(Vec2::new(0.5, 0.6)),
            PathEl::LineTo(Vec2::new(0.0, 0.6)),
            PathEl::Close,
        ])
    }

    #[test]
    fn text_run_expands_to_one_path_per_glyph() {
        let mut run = TextRun::new(20.0);
        run.push_glyph(1, 0.0, 0.0);
        run.push_glyph(2, 0.8, 0.0);
        run.push_glyph(1, 1.6, 0.0);

        let paths = expand_text_run(&run, |id| match id {
            1 => Some(bar_glyph()),
            2 => Some(box_glyph()),
            _ => None,
        });
        assert_eq!(paths.len(), 3, "三个字形应展开成三条独立路径");

        // 第 3 个字形与第 1 个同形，但位置应为 1.6 em × 20px = 32px
        let p0 = paths[0].1.map_point(Vec2::ZERO);
        let p2 = paths[2].1.map_point(Vec2::ZERO);
        assert!((p0.x - 0.0).abs() < 1e-4, "第 1 个字形应在 x=0，实际 {}", p0.x);
        assert!((p2.x - 32.0).abs() < 1e-4, "第 3 个字形应在 x=32，实际 {}", p2.x);

        // 字号应折进坐标：字形高度 0.6 em × 20 px = 12 px
        let top = paths[0].1.map_point(Vec2::new(0.0, 0.6));
        assert!((top.y - 12.0).abs() < 1e-4, "字形高度应为 12px，实际 {}", top.y);
    }

    #[test]
    fn missing_glyph_is_skipped_not_panicking() {
        let mut run = TextRun::new(16.0);
        run.push_glyph(1, 0.0, 0.0);
        run.push_glyph(99, 1.0, 0.0); // 轮廓缺失
        run.push_glyph(2, 2.0, 0.0);
        let paths = expand_text_run(&run, |id| match id {
            1 => Some(bar_glyph()),
            2 => Some(box_glyph()),
            _ => None,
        });
        assert_eq!(paths.len(), 2, "缺失轮廓的字形应被跳过而不是 panic");
    }

    /// 字体回退（0.7.0）：展开时要按**字体下标**取轮廓 ——
    /// 同一个 glyph id 在不同字体里是不同字形。
    #[test]
    fn expand_text_run_with_passes_font_index_to_lookup() {
        let mut run = TextRun::new(20.0);
        run.push_glyph_in(0, 1, 0.0, 0.0, 0); // 字体 0 的字形 1
        run.push_glyph_in(1, 1, 1.0, 0.0, 3); // 字体 1 的**同一个**字形 id

        let mut seen: Vec<(usize, u32)> = Vec::new();
        let paths = expand_text_run_with(&run, |fi, id| {
            seen.push((fi, id));
            // 两个字体对同一个 id 给出**不同**轮廓
            match fi {
                0 => Some(bar_glyph()),
                _ => Some(box_glyph()),
            }
        });
        assert_eq!(seen, vec![(0, 1), (1, 1)], "lookup 必须收到 (字体下标, 字形 id)");
        assert_eq!(paths.len(), 2);
        assert_ne!(
            paths[0].0.els, paths[1].0.els,
            "同一 glyph id 在两个字体下必须得到不同轮廓（否则等于没做字体回退）"
        );
    }

    /// 缓存必须按**字体**分键：同一个 glyph id 换了字体不能命中旧条目。
    #[test]
    fn text_cache_separates_fonts() {
        let mut cache = TextCache::new();
        let mut run = TextRun::new(20.0);
        run.push_glyph_in(0, 1, 0.0, 0.0, 0);
        run.push_glyph_in(1, 1, 1.0, 0.0, 0);

        let fonts = [FontId::from_bits(0xAAAA), FontId::from_bits(0xBBBB)];
        let mut lookups = 0;
        let out = cache.expand_with(
            &run,
            |fi| fonts[fi],
            |fi, _id| {
                lookups += 1;
                if fi == 0 {
                    Some(bar_glyph())
                } else {
                    Some(box_glyph())
                }
            },
        );
        assert_eq!(out.len(), 2);
        assert_eq!(lookups, 2, "不同字体必须各提取一次，不能共用缓存");
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.hits(), 0);
        assert_ne!(out[0].0.els, out[1].0.els, "两次展开的轮廓必须不同");

        // 同一个字体再展开一次：应当命中
        let again = cache.expand_with(&run, |fi| fonts[fi], |_fi, _id| {
            lookups += 1;
            Some(bar_glyph())
        });
        assert_eq!(again.len(), 2);
        assert_eq!(lookups, 2, "同一字体集合再展开不应重新提取");
        assert_eq!(cache.hits(), 2);
    }

    /// **零破坏**：老入口 `GlyphKey::new` 不带字体身份，且量化规则不变。
    #[test]
    fn glyph_key_new_keeps_legacy_semantics() {
        let k = GlyphKey::new(7, 10.0);
        assert_eq!(k.font_id, FontId::UNKNOWN);
        assert_eq!(k.id, 7);
        assert_eq!(k.size_q, 160, "10.0 × 16 = 160");
        // 带字体身份的键与老键**必须不同**，否则字体回退会跨字体误命中
        let with = GlyphKey::with_font(FontId::from_bits(42), 7, 10.0);
        assert_ne!(k, with);
        assert_eq!(with.size_q, k.size_q);
    }

    #[test]
    fn text_cache_hits_on_repeated_glyphs() {
        let mut cache = TextCache::new();
        let mut run = TextRun::new(20.0);
        for i in 0..10 {
            run.push_glyph(1, i as f32 * 0.5, 0.0);
        }
        let mut lookups = 0;
        let a = cache.expand(&run, |id| {
            lookups += 1;
            if id == 1 { Some(bar_glyph()) } else { None }
        });
        assert_eq!(a.len(), 10);
        assert_eq!(lookups, 1, "10 个相同字形只应提取一次轮廓");
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 9);

        // 第二次展开同一字号：全部命中
        let b = cache.expand(&run, |id| {
            lookups += 1;
            if id == 1 { Some(bar_glyph()) } else { None }
        });
        assert_eq!(b.len(), 10);
        assert_eq!(lookups, 1, "再次展开不应再提取轮廓");
        assert!(cache.hit_rate() > 0.9, "命中率应 > 0.9，实际 {}", cache.hit_rate());

        // 输出必须与未缓存路径**几何等价**（缓存不得改变渲染结果）。
        // 注意两者的表示不同：缓存把字号折进坐标、变换只含平移；
        // 直接展开则把缩放留在变换里。因此比较的是**变换后的顶点**，而非结构。
        let direct = expand_text_run(&run, |id| if id == 1 { Some(bar_glyph()) } else { None });
        assert_eq!(b.len(), direct.len());
        for (gi, (cached, plain)) in b.iter().zip(direct.iter()).enumerate() {
            assert_eq!(cached.0.els.len(), plain.0.els.len());
            let cp = el_points_all(&cached.0.els, &cached.1);
            let pp = el_points_all(&plain.0.els, &plain.1);
            for (pi, (a, c)) in cp.iter().zip(pp.iter()).enumerate() {
                assert!(
                    (a.x - c.x).abs() < 1e-4 && (a.y - c.y).abs() < 1e-4,
                    "字形 {gi} 点 {pi}：缓存展开 {a:?} 与直接展开 {c:?} 不一致\
                     （缓存不得改变几何；曾因此抓出位置未随字号缩放的 bug）"
                );
            }
        }
    }

    /// 把路径元素全部映射到最终坐标。
    fn el_points_all(els: &[PathEl], t: &Affine) -> Vec<Vec2> {
        els.iter()
            .flat_map(el_points)
            .map(|p| t.map_point(p))
            .collect()
    }

    /// 取出一个路径元素的控制点（用于几何比较，忽略表示差异）。
    fn el_points(el: &PathEl) -> Vec<Vec2> {
        match *el {
            PathEl::MoveTo(p) | PathEl::LineTo(p) => vec![p],
            PathEl::QuadTo(c, p) => vec![c, p],
            PathEl::CubicTo(a, b, p) => vec![a, b, p],
            PathEl::Close => vec![],
        }
    }

    #[test]
    fn text_cache_separates_quantized_sizes() {
        let mut cache = TextCache::new();
        let mk = |size: f32| {
            let mut r = TextRun::new(size);
            r.push_glyph(1, 0.0, 0.0);
            r
        };
        cache.expand(&mk(20.0), |_| Some(bar_glyph()));
        cache.expand(&mk(40.0), |_| Some(bar_glyph()));
        assert_eq!(cache.misses(), 2, "不同字号必须是不同缓存项");
        // 同一量化桶（20.0 与 20.01 → 1/16 粒度）应视为同一项
        cache.expand(&mk(20.0), |_| Some(bar_glyph()));
        assert_eq!(cache.misses(), 2, "同字号复现不应新增缓存项");
        assert_eq!(cache.len(), 2);
    }
}
