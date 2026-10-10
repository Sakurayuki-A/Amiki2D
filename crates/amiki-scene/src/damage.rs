//! M4：damage region 的**编辑契约**。
//!
//! # 为什么需要这个模块
//!
//! 增量渲染有一条**调用方必须遵守**的契约：
//!
//! > 改动一条 draw 时，必须把「旧包围盒 ∪ 新包围盒」标记为脏。
//!
//! 只标记新位置是**错的**：某 tile 可能从"有内容"变成"无内容"，而
//! CPU 侧的 `shade_tile` 在 tile 无绘制时是**保留原像素**（这正是复用上一帧的机制）。
//! 于是旧位置会永久残留陈旧像素。
//!
//! 把这条契约交给调用方手写，迟早会有人漏。本模块提供
//! [`edit_damage`]：**从旧场景与新场景自动推导脏区**，把"正确"变成默认行为。
//!
//! # 保守性
//!
//! 包围盒取**所有控制点的凸包**（贝塞尔曲线必落在控制点凸包内），
//! 因此只会**多**标脏、绝不会漏。多标只损失一点性能，漏标会产生错误画面。

use amiki_math::{Affine, Vec2};

use crate::{Draw, LayerBounds, Path, PathEl, Scene};

/// 把一次仿射与路径在设备空间的保守包围盒并入 `out`。
///
/// 返回 `false` 表示"没有有限包围盒"（空路径或非有限坐标），此时**不能**
/// 只并入一个空矩形——调用方应改用 `invalidate_all` 兜底（见 [`edit_damage`]）。
#[must_use]
fn path_bounds_in(transform: &Affine, path: &Path, out: &mut [f32; 4]) -> bool {
    let mut any = false;
    let mut cur: Option<Vec2> = None;
    let mut start: Option<Vec2> = None;

    // 依次并入一个点（已做设备变换）
    // 用普通 fn 而非闭包：闭包捕获 out 会与外层借用冲突
    fn put(p: Vec2, any: &mut bool, out: &mut [f32; 4]) {
        if !(p.x.is_finite() && p.y.is_finite()) {
            return;
        }
        if *any {
            out[0] = out[0].min(p.x);
            out[1] = out[1].min(p.y);
            out[2] = out[2].max(p.x);
            out[3] = out[3].max(p.y);
        } else {
            out[0] = p.x;
            out[1] = p.y;
            out[2] = p.x;
            out[3] = p.y;
            *any = true;
        }
    }

    for el in &path.els {
        match *el {
            PathEl::MoveTo(p) => {
                let m = transform.map_point(p);
                put(m, &mut any, out);
                cur = Some(m);
                start = Some(m);
            }
            PathEl::LineTo(p) => {
                let m = transform.map_point(p);
                put(m, &mut any, out);
                cur = Some(m);
            }
            PathEl::QuadTo(c, p) => {
                // 控制点与终点都在凸包内 → 全部并入即保守
                put(transform.map_point(c), &mut any, out);
                let m = transform.map_point(p);
                put(m, &mut any, out);
                cur = Some(m);
            }
            PathEl::CubicTo(c1, c2, p) => {
                put(transform.map_point(c1), &mut any, out);
                put(transform.map_point(c2), &mut any, out);
                let m = transform.map_point(p);
                put(m, &mut any, out);
                cur = Some(m);
            }
            PathEl::Close => {
                // 闭合段本身不引入新极值点（两端点已计入）
                cur = start;
            }
        }
    }
    // 填充时未闭合子路径会自动闭合，闭合段同样不引入新极值点
    let _ = (cur, start);
    any
}

/// 一条 draw 的设备空间保守包围盒：`Some([x0, y0, x1, y1])`，或 `None`（无法界定）。
pub fn draw_device_bounds(draw: &Draw) -> Option<[f32; 4]> {
    let mut b = [0.0f32; 4];
    if path_bounds_in(&draw.transform, &draw.path, &mut b) {
        Some(b)
    } else {
        None
    }
}

/// 两条 draw 的"变化影响范围"：旧包围盒 ∪ 新包围盒。
///
/// 任一侧无法界定（空路径 / 非有限坐标）时返回 `None`，调用方必须按全脏处理。
pub fn changed_bounds(old: &Draw, new: &Draw) -> Option<[f32; 4]> {
    let a = draw_device_bounds(old);
    let b = draw_device_bounds(new);
    match (a, b) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        // 两侧都无法界定 → 无法用矩形表达，交由调用方全脏兜底
        (None, None) => None,
    }
}

/// 计算从 `old` 到 `new` 的脏区。
///
/// 规则：
/// - 某个下标只存在于一侧（新增或删除）→ 该侧的包围盒；
/// - 两侧都有但**内容不同**（变换 / 路径 / 样式 / 混合 / 裁剪任一不同）→ 旧 ∪ 新；
/// - **层**：层内容/滤镜/范围变化时，标脏该层经滤镜后的**作用范围**
///   （见 [`LayerBounds::inflate_by_filter`]）。层是"整块滤波再合成"的，
///   所以层内任何改动都可能影响它作用范围内的一切 —— 这里取保守范围。
/// - 画布尺寸变化、或出现无法界定的 draw → **整幅全脏**（安全兜底）。
///
/// **只多标、不漏标**。多标只损失性能；漏标会产生错误画面。
pub fn edit_damage(old: &Scene, new: &Scene) -> Damage {
    let mut d = Damage::new(new.width, new.height);
    if old.width != new.width || old.height != new.height {
        return d; // 尺寸变了：全脏
    }
    d.clear();
    let n = old.draws.len().max(new.draws.len());
    for i in 0..n {
        match (old.draws.get(i), new.draws.get(i)) {
            (Some(a), Some(b)) => {
                if a == b {
                    continue;
                }
                match changed_bounds(a, b) {
                    Some(bb) => d.invalidate_bounds(bb),
                    None => return Damage::new(new.width, new.height),
                }
            }
            (Some(a), None) => match draw_device_bounds(a) {
                Some(bb) => d.invalidate_bounds(bb),
                None => return Damage::new(new.width, new.height),
            },
            (None, Some(b)) => match draw_device_bounds(b) {
                Some(bb) => d.invalidate_bounds(bb),
                None => return Damage::new(new.width, new.height),
            },
            (None, None) => unreachable!("i < max(len) 保证至少一侧存在"),
        }
    }

    // —— 层 ——
    //
    // 这一整块曾经是缺失的：`edit_damage` 只看 `draws`，于是**层内改动产生零脏区**，
    // 增量渲染永远看不到它（画面停在旧层内容上，且不报错）。
    let nl = old.layers.len().max(new.layers.len());
    for i in 0..nl {
        match (old.layers.get(i), new.layers.get(i)) {
            (Some(a), Some(b)) => {
                if a == b {
                    continue;
                }
                // 层的**作用范围**取两边的并集（滤镜/偏移可能同时变了）
                let ba = a.inflated_bounds();
                let bb = b.inflated_bounds();
                d.invalidate_bounds(bounds_rect(&ba));
                d.invalidate_bounds(bounds_rect(&bb));
            }
            (Some(a), None) => d.invalidate_bounds(bounds_rect(&a.inflated_bounds())),
            (None, Some(b)) => d.invalidate_bounds(bounds_rect(&b.inflated_bounds())),
            (None, None) => unreachable!("i < max(len) 保证至少一侧存在"),
        }
    }
    d
}

/// `LayerBounds` → `[x0, y0, x1, y1]`（`Damage` 的矩形约定）。
fn bounds_rect(b: &LayerBounds) -> [f32; 4] {
    [
        b.x as f32,
        b.y as f32,
        b.x as f32 + b.width as f32,
        b.y as f32 + b.height as f32,
    ]
}


// ============================================================================
// damage region 的**类型**（编辑契约的算法见本文件上半部分）
// ============================================================================

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

