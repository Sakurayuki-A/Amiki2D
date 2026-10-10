//! 场景模型：`Draw` / `Layer` / `Filter` / `Scene` 与曲线辅助。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

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

