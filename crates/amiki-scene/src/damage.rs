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

use crate::{Damage, Draw, LayerBounds, Path, PathEl, Scene};

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
