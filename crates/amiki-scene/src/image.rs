//! 位图：双线性采样、放置、九宫格与子图。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

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

/// 九宫格的**一块**：源图上的矩形 + 它在目标上应占的矩形。
///
/// 只给矩形、不给 `ImagePlacement`，是**有意的设计**（见 [`nine_patch`] 的说明）：
/// 调用方用最普通的"把子图画进矩形"的路径即可，不涉及 uv 子区间的任何约定。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NinePatchSlice {
    /// 源矩形（像素，左上闭、右下开）：`(x, y, w, h)`
    pub src: (u32, u32, u32, u32),
    /// 目标矩形（设备坐标）：`(x0, y0, x1, y1)`
    pub dst: (f32, f32, f32, f32),
}

/// 把源图按 `insets` 切成 **9 块**，算出每块的**源矩形**与**目标矩形**。
///
/// # 语义（**与 Skia `drawImageNine` / Android 9-patch 对齐**）
///
/// ```text
///   ┌───┬─────────┬───┐   四角：**保持源尺寸，不缩放**（1 源像素 = 1 设备像素）
///   │ 1 │    2    │ 3 │   四边：单向拉伸
///   ├───┼─────────┼───┤   中间：双向拉伸
///   │ 4 │    5    │ 6 │
///   ├───┼─────────┼───┤   ⇒ 圆角/边框装饰在任意尺寸下**不变形**（这才是它的用途）
///   │ 7 │    8    │ 9 │
///   └───┴─────────┴───┘
/// ```
///
/// # 为什么返回矩形而不是 `ImagePlacement`
///
/// 曾经用 `ImagePlacement` 的 "uv 子区间 → 设备矩形" 仿射来拼 9 块，结果与
/// 采样器的 texel 约定对不上：实测设备像素 `(1,1)` 取到的是源图 **texel (2,2)**
/// 处四色交汇的混合（`[128,101,76]` ≈ 等权四色 `[125,102,80]`）。与其猜约定，
/// 不如**绕开它**：把 9 块切成子图后用最普通的矩形放置画 —— 那条路径已被
/// "1:1 逐像素不变"的保证覆盖过。
///
/// # 边界处理（显式，不静默）
///
/// - `insets` 超出源图尺寸 ⇒ 按比例夹断到不超过源尺寸；
/// - `dst` 比两角之和还小 ⇒ 两角**等比压缩**（不产生负尺寸的块）；
/// - 两侧边距之和等于源尺寸 ⇒ 该行/列宽度为 0 ⇒ **跳过**（只剩四角）；
/// - `dst` 退化（宽或高 ≤ 0）或源图任一维为 0 ⇒ 返回**空**（什么都不画）。
pub fn nine_patch(
    src_w: u32,
    src_h: u32,
    insets: Insets,
    dst: (f32, f32, f32, f32),
) -> Vec<NinePatchSlice> {
    let (dx0, dy0, dx1, dy1) = dst;
    if src_w == 0 || src_h == 0 || dx1 <= dx0 || dy1 <= dy0 {
        return Vec::new();
    }
    let (mut il, mut ir) = (insets.left.min(src_w), insets.right.min(src_w));
    if il + ir > src_w {
        let k = src_w as f32 / (il + ir) as f32;
        il = (il as f32 * k).round() as u32;
        ir = src_w - il;
    }
    let (mut it, mut ib) = (insets.top.min(src_h), insets.bottom.min(src_h));
    if it + ib > src_h {
        let k = src_h as f32 / (it + ib) as f32;
        it = (it as f32 * k).round() as u32;
        ib = src_h - it;
    }
    let (dw, dh) = (dx1 - dx0, dy1 - dy0);
    // 角的设备边长 = 源 inset 原值（不缩放）；放不下两角时等比压缩
    let (mut dl, mut dr) = (il as f32, ir as f32);
    if dl + dr > dw {
        let k = dw / (dl + dr);
        dl *= k;
        dr *= k;
    }
    let (mut dt, mut db) = (it as f32, ib as f32);
    if dt + db > dh {
        let k = dh / (dt + db);
        dt *= k;
        db *= k;
    }

    let sx = [0u32, il, src_w - ir, src_w];
    let sy = [0u32, it, src_h - ib, src_h];
    let dx = [dx0, dx0 + dl, dx1 - dr, dx1];
    let dy = [dy0, dy0 + dt, dy1 - db, dy1];

    let mut out = Vec::with_capacity(9);
    for row in 0..3 {
        for col in 0..3 {
            let (sx0, sx1) = (sx[col], sx[col + 1]);
            let (sy0, sy1) = (sy[row], sy[row + 1]);
            let (x0, x1) = (dx[col], dx[col + 1]);
            let (y0, y1) = (dy[row], dy[row + 1]);
            if sx1 <= sx0 || sy1 <= sy0 || x1 <= x0 || y1 <= y0 {
                continue;
            }
            out.push(NinePatchSlice {
                src: (sx0, sy0, sx1 - sx0, sy1 - sy0),
                dst: (x0, y0, x1, y1),
            });
        }
    }
    out
}

/// 从源图取出一个子图（**复制**像素；九宫格每块很小，代价可忽略）。
///
/// 越界或空尺寸返回 `None` —— 显式，不产出半张图。
pub fn sub_image(src: &Image, rect: (u32, u32, u32, u32)) -> Option<Image> {
    let (x, y, w, h) = rect;
    if w == 0 || h == 0 || x + w > src.width || y + h > src.height {
        return None;
    }
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let off = (((y + row) * src.width + x) * 4) as usize;
        data.extend_from_slice(&src.data[off..off + (w * 4) as usize]);
    }
    Image::from_rgba8(w, h, data)
}

