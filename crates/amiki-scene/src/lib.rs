//! amiki-scene —— 路径、绘制与场景模型。
//!
//! Scene 是引擎的中间表示：上层 API 产出 Scene，后端（GPU / CPU）消费 Scene。
//! M1 采用直接结构化表示；M2 演进为 Vello 式 tag+data 紧凑编码流（接口保持稳定）。

#![forbid(unsafe_code)]

use amiki_math::{Affine, CubicBez, QuadBez, Vec2};

mod damage;
pub use damage::{changed_bounds, draw_device_bounds, edit_damage};

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

/// 填充规则。M1 仅 nonzero；even-odd 在路线图 M2。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillRule {
    NonZero,
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

/// 绘制样式。M1 支持实心色；渐变 / 图像在路线图 M4。
#[derive(Clone, Debug, PartialEq)]
pub enum Paint {
    Solid(Color),
}

/// 每 draw 最多引用的裁剪层数（WGSL 侧展开采样）。
pub const MAX_DRAW_CLIPS: usize = 4;

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

/// 场景：画布尺寸 + 有序绘制列表（画家算法，后绘制在上）+ 裁剪层。
///
/// 派生 `PartialEq` 供 M4 判定"本帧场景与上一帧是否相同"——相同则**一次着色都不做**。
/// 浮点字段按位比较：只要有一个比特变了就视为"变了"，宁可多算不可漏算。
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub width: u32,
    pub height: u32,
    pub draws: Vec<Draw>,
    pub clips: Vec<Clip>,
}

impl Scene {
    pub fn new(width: u32, height: u32) -> Scene {
        Scene { width, height, draws: Vec::new(), clips: Vec::new() }
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
//   每 draw：transform 6×f32 | color 4×f32 | blend u32 | n_refs u32 | refs | path
//   path = count u32 | count × (tag u8 + 坐标 f32)
//   tag：0 MoveTo, 1 LineTo, 2 QuadTo, 3 CubicTo, 4 Close

pub const STREAM_MAGIC: &[u8; 4] = b"AMK1";
pub const STREAM_VERSION: u32 = 1;

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
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
        match draw.paint {
            Paint::Solid(c) => {
                put_f32(&mut buf, c.r);
                put_f32(&mut buf, c.g);
                put_f32(&mut buf, c.b);
                put_f32(&mut buf, c.a);
            }
        }
        put_u32(&mut buf, draw.blend.id());
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
    if version != STREAM_VERSION {
        return Err(format!("编码流版本不支持: {version}"));
    }
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
        let c = r.f32x(4)?;
        let blend_id = r.u32()?;
        let n_refs = r.u32()? as usize;
        let mut clips = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            clips.push(r.u32()?);
        }
        let path = r.path()?;
        let paint = Paint::Solid(Color { r: c[0], g: c[1], b: c[2], a: c[3] });
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
            fill_rule: FillRule::NonZero,
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
    fn damage_invalidate_all_wins() {
        let mut d = Damage::new(64, 64);
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
}
