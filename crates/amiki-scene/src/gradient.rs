//! 渐变（`GradientStop` / `GradientKind` / `Gradient`）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

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

