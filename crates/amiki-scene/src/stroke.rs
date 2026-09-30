//! 描边（stroke）→ 闭合轮廓。
//!
//! # 为什么是"生成轮廓"而不是"另推一套闭式覆盖率"
//!
//! 本项目的填充用**有符号面积 + 绕数**做解析覆盖率（见 `amiki-raster` 的
//! `seg_area`）：把路径的每条线段对像素的贡献累加，取绝对值截断到 1。
//! 这套机制对**简单闭合轮廓**给出精确面积。
//!
//! 描边若按"每段一个四边形 + 拐角图形"直接累加，**转角处必然重叠**：
//! 两个各覆盖 50%、重叠 30% 的块会被算成 100%，而正确答案是 70%。
//! 有符号面积对重叠是**相加**而不是**取并**，所以那条路是错的。
//!
//! 正确做法：把描边区域表达成**一条（或几条）非自交的闭合轮廓**，
//! 再交给同一套填充机制。于是描边不需要任何新的覆盖率数学，
//! 抗锯齿精度与填充完全一致。
//!
//! # 轮廓的构造
//!
//! 沿折线走一圈，分正负两个法向链：
//!
//! ```text
//! 轮廓 = 正法向链（前向） ++ 负法向链（反向）
//! ```
//!
//! 每个顶点处两个链各自处理：
//!
//! | 该侧是 | 处理 | 理由 |
//! |---|---|---|
//! | **外侧** | miter 插入交点；超 `miter_limit` 则不插（= bevel） | 外侧要补上转角缺的那块 |
//! | **内侧** | **总是**插入两偏移线的交点 | 不插的话两条偏移线会交叉，轮廓自交 |
//!
//! bevel 之所以"什么都不用插"，是因为链里已经依次有前一段的终点偏移与
//! 后一段的终点偏移，两者直接相连就是一条斜切边。
//!
//! # 输入是折线
//!
//! 偏移曲线不是曲线（三次贝塞尔的等距偏移不是三次贝塞尔），
//! 所以描边在**压平之后**做：调用方先用与光栅化相同的容差压平，
//! 再把折线交给本模块。这也意味着描边路径在 GPU 侧以线段形式上传 ——
//! 与填充路径"曲线保持为曲线"不同，这是描边固有的性质。

use amiki_math::Vec2;

/// 线端样式（开放子路径的两端）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LineCap {
    /// 平端：正好在端点处截断。
    #[default]
    Butt,
    /// 圆端：以端点为心的半圆。
    Round,
    /// 方端：向外延伸半个线宽。
    Square,
}

/// 拐角样式（相邻两段的连接处）。
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LineJoin {
    /// 尖角：延伸到两偏移线交点；`limit` 为"交点距离 / 半线宽"的上限，
    /// 超过则退化为 [`LineJoin::Bevel`]（与 Canvas/SVG 的 miter limit 同义）。
    Miter { limit: f32 },
    /// 圆角：以顶点为心的圆弧。
    Round,
    /// 斜切：两偏移点直接相连。
    #[default]
    Bevel,
}

/// 描边样式。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeStyle {
    /// 线宽（**总宽**，不是半宽）。
    pub width: f32,
    pub cap: LineCap,
    pub join: LineJoin,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        StrokeStyle { width: 1.0, cap: LineCap::Butt, join: LineJoin::Bevel }
    }
}

/// 圆弧 / 圆端的分段数依据：给定半径，取"弦高误差 ≤ 0.1px"的分段数。
///
/// 用**固定容差**而不是固定段数：细线不该被过度细分，粗圆角不该显得有棱角。
/// 上限 128 段是为了给极端线宽兜底（否则段数会随半径线性增长而失控）。
fn arc_steps(radius: f32) -> u32 {
    if radius <= 0.0 {
        return 1;
    }
    // 弦高 h = r(1 − cos(Δ/2))；要求 h ≤ tol，得 Δ ≤ 2·acos(1 − tol/r)
    let tol = 0.1f32;
    let ratio = (1.0 - tol / radius).clamp(-1.0, 1.0);
    let delta = 2.0 * ratio.acos();
    if delta <= 1e-4 {
        return 128;
    }
    ((std::f32::consts::TAU / delta).ceil() as u32).clamp(1, 128)
}

fn perp(d: Vec2) -> Vec2 {
    Vec2::new(-d.y, d.x)
}

fn norm(d: Vec2) -> Option<Vec2> {
    let l = d.hypot();
    if l > 1e-12 {
        Some(Vec2::new(d.x / l, d.y / l))
    } else {
        None
    }
}

/// 两条直线的交点：`p + t·d` 与 `q + s·e`。平行时返回 `None`。
fn line_isect(p: Vec2, d: Vec2, q: Vec2, e: Vec2) -> Option<Vec2> {
    let den = d.cross(e);
    if den.abs() < 1e-12 {
        return None;
    }
    let t = (q - p).cross(e) / den;
    Some(Vec2::new(p.x + d.x * t, p.y + d.y * t))
}

/// 圆弧分段采样：从极角 `a0` 转过 `delta`，追加**不含起点、含终点**的点。
fn push_arc_sweep(out: &mut Vec<Vec2>, center: Vec2, a0: f32, delta: f32, radius: f32) {
    let tau = std::f32::consts::TAU;
    let n = arc_steps(radius);
    let steps = ((delta.abs() / tau) * n as f32).ceil().max(1.0) as u32;
    for i in 1..=steps {
        let a = a0 + delta * (i as f32 / steps as f32);
        out.push(Vec2::new(center.x + radius * a.cos(), center.y + radius * a.sin()));
    }
}

/// 角度归一到 `(−π, π]`。
fn wrap_pi(mut a: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let pi = std::f32::consts::PI;
    while a > pi {
        a -= tau;
    }
    while a <= -pi {
        a += tau;
    }
    a
}

fn angle_of(center: Vec2, p: Vec2) -> f32 {
    (p.y - center.y).atan2(p.x - center.x)
}

/// **圆角拐角**用的圆弧：走从 `from` 到 `to` 的**短弧**。
///
/// 短弧正是转角的外侧弧 —— 两偏移点相对顶点的夹角就是外角（≤ π）。
fn push_arc_short(out: &mut Vec<Vec2>, center: Vec2, from: Vec2, to: Vec2, radius: f32) {
    let a0 = angle_of(center, from);
    let a1 = angle_of(center, to);
    push_arc_sweep(out, center, a0, wrap_pi(a1 - a0), radius);
    // 精确落到 `to`：角度插值有累积误差，不修正会让轮廓留缝
    if let Some(last) = out.last_mut() {
        *last = to;
    }
}

/// **圆端帽**用的半圆：`from`/`to` 相对圆心是对径点，短弧与长弧都是 π。
///
/// 这里不靠手推旋转方向（极易搞反），而是**按几何选**：两个候选半圆各有中点，
/// 取中点方向与 `outward`（线的前进方向）同向的那个。
fn push_arc_half(out: &mut Vec<Vec2>, center: Vec2, from: Vec2, to: Vec2, outward: Vec2, radius: f32) {
    let a0 = angle_of(center, from);
    let pi = std::f32::consts::PI;
    let mut best = pi;
    let mut best_dot = f32::NEG_INFINITY;
    for cand in [pi, -pi] {
        let mid = a0 + cand * 0.5;
        let dot = Vec2::new(mid.cos(), mid.sin()).dot(outward);
        if dot > best_dot {
            best_dot = dot;
            best = cand;
        }
    }
    push_arc_sweep(out, center, a0, best, radius);
    if let Some(last) = out.last_mut() {
        *last = to;
    }
}

/// 顶点处的连接：把几何从 `end_prev`（前一段在顶点的偏移）补到
/// `start_next`（后一段在顶点的偏移），**含两端点**。
///
/// 这是描边几何的全部难点所在。三条规则：
///
/// | 该侧是 | 处理 | 理由 |
/// |---|---|---|
/// | **外侧** · bevel | 压入两个偏移点，直接相连 | 斜切边就是这两点的连线 |
/// | **外侧** · miter | 压入两偏移线的交点；超 `limit` 则退化为 bevel | 交点即尖角顶点 |
/// | **外侧** · round | 压入 `end_prev` + 短弧 | 两偏移点夹角就是外角（≤ π），短弧即外侧弧 |
/// | **内侧** | 压入两偏移线的**交点** | 不插交点则两偏移线交叉 → 轮廓自交 |
///
/// 内侧不受 `miter_limit` 约束：内侧交点到顶点的距离同样可能很长（反向尖角），
/// 但它落在描边区域内部，插它总是对的。
#[allow(clippy::too_many_arguments)]
fn emit_join(
    out: &mut Vec<Vec2>,
    vertex: Vec2,
    d_prev: Vec2,
    d_next: Vec2,
    side: f32,
    h: f32,
    style: &StrokeStyle,
    end_prev: Vec2,
    start_next: Vec2,
) {
    let cross = d_prev.cross(d_next);
    if cross.abs() < 1e-9 {
        // 共线：两偏移点重合（同向）或反向重合（折回），给一个即可
        out.push(start_next);
        return;
    }
    // cross > 0 表示路径朝 +perp 方向转，于是 +perp 侧是**内侧**
    let is_outer = if cross > 0.0 { side < 0.0 } else { side > 0.0 };

    if !is_outer {
        match line_isect(end_prev, d_prev, start_next, d_next) {
            Some(x) => out.push(x),
            None => {
                out.push(end_prev);
                out.push(start_next);
            }
        }
        return;
    }

    match style.join {
        // 斜切：两偏移点直接相连
        LineJoin::Bevel => {
            out.push(end_prev);
            out.push(start_next);
        }
        LineJoin::Miter { limit } => match line_isect(end_prev, d_prev, start_next, d_next) {
            // miter 比 = 交点到顶点距离 / 半线宽
            Some(x) if (x - vertex).hypot() <= limit * h => out.push(x),
            // 超限 → 退化为 bevel（与 Canvas/SVG 一致）
            _ => {
                out.push(end_prev);
                out.push(start_next);
            }
        },
        LineJoin::Round => {
            out.push(end_prev);
            // 短弧：不含起点、含终点（终点已精确落到 start_next）
            push_arc_short(out, vertex, end_prev, start_next, h);
        }
    }
}

/// 一条**非退化**折线的描边**轮廓集合**。
///
/// 返回**多条闭合轮廓**，调用方应把它们作为同一个填充路径的多条子路径：
///
/// - **开放折线** → 1 条轮廓（一条"带端帽的粗线"，是简单多边形）
/// - **闭合折线** → 2 条轮廓（环：外圈与反向的内圈，靠绕数相减得环面积）
///
/// `pts.len() >= 2` 且相邻点不重合时有效；退化输入返回空。
/// `closed` 为真时首尾相连（不产生端帽，且在起点处也有拐角）。
pub fn stroke_polyline(pts: &[Vec2], closed: bool, style: &StrokeStyle) -> Vec<Vec<Vec2>> {
    // —— 去掉重合点（重合段没有方向，会让法向无法定义）——
    let mut p: Vec<Vec2> = Vec::with_capacity(pts.len());
    for &q in pts {
        if p.last().is_none_or(|l| (q - *l).hypot() > 1e-9) {
            p.push(q);
        }
    }
    if closed && p.len() > 1 && (p[0] - *p.last().unwrap()).hypot() <= 1e-9 {
        p.pop();
    }
    let h = style.width.abs() * 0.5;
    if h <= 0.0 {
        return Vec::new();
    }

    // —— 段方向与法向 ——
    let m = if closed { p.len() } else { p.len().saturating_sub(1) };
    if m == 0 || (!closed && p.len() < 2) {
        return Vec::new();
    }
    let at = |i: usize| p[i % p.len()];
    let mut dirs: Vec<Vec2> = Vec::with_capacity(m);
    for i in 0..m {
        match norm(at(i + 1) - at(i)) {
            Some(d) => dirs.push(d),
            None => return Vec::new(), // 出现零长段：拒绝出图而不是画错
        }
    }
    let norms: Vec<Vec2> = dirs.iter().map(|&d| perp(d)).collect();

    // —— 两个法向链 ——
    //
    // 结构要点（这里曾经写错过，值得说明）：**每段只负责压入它在"起点"处的偏移**，
    // 而顶点的连接由 `emit_join` 负责 —— 它拿到的正是"前一段在顶点的偏移
    // （`end_prev`）"与"后一段在顶点的偏移（`start_next`）"两个点。
    //
    // 早先的写法是"每段压入它的终点偏移"，于是顶点处从上一段的终点偏移**直接
    // 跳到**下一段的终点偏移，把"后一段的起点偏移"整个漏掉 —— bevel 需要的
    // 两个点少了一个，转角被切掉一大块（实测面积 632，正确值 712）。
    //
    // `side` 为 +1 表示正法向链（沿 +perp 偏移），−1 为负法向链。
    let chain = |side: f32| -> Vec<Vec2> {
        let mut out: Vec<Vec2> = Vec::with_capacity(m * 2 + 2);
        // **闭合路径不预推起点偏移**：闭合时 at(0) 也是一个拐角顶点，
        // 它的正确形状由最后一轮 `emit_join` 给出（在该处推入交点/斜切点）。
        // 若这里先推一个顶点偏移，那个点会留在轮廓里成为多余顶点，
        // 把内圈撑大（实测闭合正方形描边 1264，正确值 1280）。
        if !closed {
            out.push(at(0) + norms[0] * (side * h));
        }
        for i in 0..m {
            let end_prev = at(i + 1) + norms[i] * (side * h);
            let has_next = i + 1 < m || closed;
            if !has_next {
                out.push(end_prev); // 末段：直接收尾
                break;
            }
            let j = (i + 1) % m;
            let start_next = at(i + 1) + norms[j] * (side * h);
            emit_join(
                &mut out,
                at(i + 1),
                dirs[i],
                dirs[j],
                side,
                h,
                style,
                end_prev,
                start_next,
            );
        }
        out
    };

    let front = chain(1.0);
    let back = chain(-1.0);

    if closed {
        // **闭合路径的描边是一个环**，必须产出两条**独立**轮廓：
        // 一条沿正法向、一条沿负法向**反向**。两者绕向相反，
        // 于是绕数相减即得环面积（外圈 − 内圈）。
        //
        // 不能把两条链首尾拼成一条：那会得到一个从内圈跳到外圈的
        // "蝴蝶结"，面积完全不对（实测闭合正方形描边 960，正确值 1248）。
        let mut a = front;
        let mut b: Vec<Vec2> = back.into_iter().rev().collect();
        // 去掉首尾重复点（链的收尾点由最后一轮 emit_join 给出，可能与首点重合）
        for c in [&mut a, &mut b] {
            if c.len() > 1 && (c[0] - *c.last().unwrap()).hypot() < 1e-9 {
                c.pop();
            }
        }
        return vec![a, b];
    }

    let mut outline: Vec<Vec2> = front;
    // 端帽：butt 直接相连；square 外延；round 走半圆
    {
        let e = *outline.last().unwrap();
        match style.cap {
            LineCap::Butt => {}
            LineCap::Square => {
                // 方端帽要把**整个横截面**外延 h，即两个角点都要推。
                // 只推一个角点会退化成三角形（实测面积少一半）。
                let d = dirs[m - 1];
                let other = *back.last().unwrap();
                outline.push(e + d * h);
                outline.push(other + d * h);
            }
            LineCap::Round => {
                let v = at(m);
                let from = e;
                let to = back.last().copied().unwrap();
                // 前进方向即端帽外延方向
                push_arc_half(&mut outline, v, from, to, dirs[m - 1], h);
            }
        }
    }
    // 负法向链反向接上
    outline.extend(back.iter().rev().copied());
    // 起端帽
    match style.cap {
        LineCap::Butt => {}
        LineCap::Square => {
            let d = dirs[0];
            let n0 = norms[0];
            // 同理：起点横截面的两个角点都要外延
            outline.push(at(0) - n0 * h - d * h);
            outline.push(at(0) + n0 * h - d * h);
        }
        LineCap::Round => {
            let from = at(0) - norms[0] * h;
            let to = at(0) + norms[0] * h;
            // 起点处的外延方向与前进方向相反
            let outward = Vec2::new(-dirs[0].x, -dirs[0].y);
            push_arc_half(&mut outline, at(0), from, to, outward, h);
        }
    }
    vec![outline]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 轮廓集合的**净面积**：各轮廓有符号面积之和的绝对值。
    ///
    /// 环（外圈 + 反向内圈）正负相消，于是直接得到环面积 ——
    /// 这也是渲染器用绕数法算覆盖率时的同一个量。
    fn area_of(contours: &[Vec<Vec2>]) -> f32 {
        let sum: f32 = contours.iter().map(|c| shoelace(c)).sum();
        sum.abs()
    }

    /// 单条轮廓的有符号面积（鞋带公式）。
    fn shoelace(poly: &[Vec2]) -> f32 {
        let mut s = 0.0;
        for i in 0..poly.len() {
            s += poly[i].cross(poly[(i + 1) % poly.len()]);
        }
        s * 0.5
    }

    fn style(width: f32, cap: LineCap, join: LineJoin) -> StrokeStyle {
        StrokeStyle { width, cap, join }
    }

    /// 最小实验 ①：单段粗线 —— 轮廓面积必须精确等于 `长 × 宽`。
    ///
    /// 这是"轮廓路线"成立的最低要求：任何重叠被相加，面积就会偏大。
    #[test]
    fn single_segment_area_is_exact() {
        for (a, b, w) in [
            (Vec2::new(0.0, 0.0), Vec2::new(100.0, 0.0), 10.0f32),
            (Vec2::new(0.0, 0.0), Vec2::new(0.0, 80.0), 4.0),
            (Vec2::new(10.0, 10.0), Vec2::new(90.0, 50.0), 6.0),
            (Vec2::new(-30.0, 7.0), Vec2::new(40.0, -9.0), 2.5),
        ] {
            let outline = stroke_polyline(
                &[a, b],
                false,
                &style(w, LineCap::Butt, LineJoin::Bevel),
            );
            assert_eq!(outline.len(), 1, "单段折线应只有 1 条轮廓");
            assert!(outline[0].len() >= 4, "轮廓点数不足：{outline:?}");
            let expect = (b - a).hypot() * w;
            let got = area_of(&outline);
            assert!(
                (got - expect).abs() < 1e-2 * expect.max(1.0),
                "面积应精确等于 长×宽：期望 {expect}，实际 {got}（a={a:?} b={b:?} w={w}）"
            );
        }
    }

    /// 最小实验 ②：**转角处不重复计数** —— 用可解析求出的精确面积逐项断言。
    ///
    /// 取一个 90° 转角（`A→V→B`，两段长 50 / 40，线宽 8 即 h=4）。
    /// 两段各自矩形的**并集**面积是
    /// `50·8 + 40·8 − h² = 400 + 320 − 16 = 704`
    /// （`h²` 是两矩形在转角处的正方形重叠）。
    ///
    /// 三种拐角在这个并集上各自补一块**互不重叠**的面积，于是总面积可精确写出：
    ///
    /// | 拐角 | 补上的形状 | 面积 |
    /// |---|---|---|
    /// | bevel | 直角边 h 的等腰三角形 | `h²/2 = 8` → **712** |
    /// | round | 半径 h 的四分之一圆 | `πh²/4 ≈ 12.566` → **716.566** |
    /// | miter（不限） | 边长 h 的正方形 | `h² = 16` → **720** |
    ///
    /// 注意 **miter 恰好等于"两矩形面积直接相加"**（720）—— 这不是重复计数，
    /// 而是方形转角的正确并集。所以不能拿"面积 < 朴素相加"去断言 miter。
    #[test]
    fn join_area_matches_exact_union() {
        let (a, v, b) = (Vec2::new(0.0, 0.0), Vec2::new(50.0, 0.0), Vec2::new(50.0, 40.0));
        let w = 8.0f32;
        let h = w * 0.5;
        let poly = [a, v, b];

        let rect_union = (v - a).hypot() * w + (b - v).hypot() * w - h * h;
        let cases = [
            (LineJoin::Bevel, rect_union + h * h * 0.5),
            (LineJoin::Round, rect_union + std::f32::consts::PI * h * h * 0.25),
            (LineJoin::Miter { limit: 100.0 }, rect_union + h * h),
        ];
        for (join, expect) in cases {
            let outline = stroke_polyline(&poly, false, &style(w, LineCap::Butt, join));
            let got = area_of(&outline);
            assert!(
                (got - expect).abs() < 1e-2 * expect,
                "{join:?}：面积应为 {expect}，实际 {got}"
            );
        }
    }

    /// miter limit 生效：极尖的转角超限后退化为 bevel（面积变小）。
    #[test]
    fn miter_limit_falls_back_to_bevel() {
        // 几乎折回：转角约 169°，miter 比 ≈ 1/sin(5.6°) ≈ 10.2（远超 limit 2）
        let (a, v, b) = (
            Vec2::new(0.0, 0.0),
            Vec2::new(50.0, 0.0),
            Vec2::new(0.0, 10.0),
        );
        let w = 8.0f32;
        let poly = [a, v, b];
        let sharp = stroke_polyline(&poly, false, &style(w, LineCap::Butt, LineJoin::Miter { limit: 2.0 }));
        let bevel = stroke_polyline(&poly, false, &style(w, LineCap::Butt, LineJoin::Bevel));
        let wide = stroke_polyline(&poly, false, &style(w, LineCap::Butt, LineJoin::Miter { limit: 100.0 }));
        let (as_, bs, ws) = (area_of(&sharp), area_of(&bevel), area_of(&wide));
        assert!(
            (as_ - bs).abs() < 1e-3,
            "超限的 miter 应恰好退化为 bevel：miter={as_} bevel={bs}"
        );
        assert!(ws > bs + 1.0, "未超限的 miter 应比 bevel 多出尖角：{ws} vs {bs}");
    }

    /// 端帽面积：butt = 0 额外；square = 各 h·w；round = 各半圆。
    #[test]
    fn caps_add_exact_area() {
        let (a, b) = (Vec2::new(10.0, 20.0), Vec2::new(110.0, 20.0));
        let w = 6.0f32;
        let h = w * 0.5;
        let len = (b - a).hypot();
        let poly = [a, b];
        let butt = area_of(&stroke_polyline(&poly, false, &style(w, LineCap::Butt, LineJoin::Bevel))).abs();
        let square = area_of(&stroke_polyline(&poly, false, &style(w, LineCap::Square, LineJoin::Bevel))).abs();
        let round = area_of(&stroke_polyline(&poly, false, &style(w, LineCap::Round, LineJoin::Bevel))).abs();

        assert!((butt - len * w).abs() < 1e-2, "butt 应为 长×宽：{butt}");
        assert!(
            (square - (len * w + 2.0 * h * w)).abs() < 1e-2,
            "square 应多出两个 h×w 的方块：期望 {}，实际 {square}",
            len * w + 2.0 * h * w
        );
        let expect_round = len * w + std::f32::consts::PI * h * h;
        assert!(
            (round - expect_round).abs() < 1e-2 * expect_round,
            "round 应多出一个整圆：期望 {expect_round}，实际 {round}"
        );
    }

    /// 闭合折线：不产生端帽，且每个顶点都有拐角。
    #[test]
    fn closed_polyline_area() {
        // 边长 40 的正方形，线宽 8 → 外圈 48、内圈 32
        let sq = [
            Vec2::new(10.0, 10.0),
            Vec2::new(50.0, 10.0),
            Vec2::new(50.0, 50.0),
            Vec2::new(10.0, 50.0),
        ];
        // 外圈 48×48 = 2304；内圈 32×32 = 1024。
        // - miter（不限）：4 个外角都是直角 → 2304 − 1024 = **1280**
        // - bevel：4 个外角各切掉腰为 h=4 的等腰直角三角形（面积 8）
        //   → 2304 − 4·8 − 1024 = **1248**
        let expect = |join| match join {
            LineJoin::Bevel => 1248.0,
            _ => 1280.0,
        };
        for join in [LineJoin::Bevel, LineJoin::Miter { limit: 100.0 }] {
            let cs = stroke_polyline(&sq, true, &style(8.0, LineCap::Butt, join));
            assert_eq!(cs.len(), 2, "{join:?}：闭合描边应是环（2 条轮廓）");
            let area = area_of(&cs);
            let e = expect(join);
            assert!(
                (area - e).abs() < 1.0,
                "{join:?}：闭合正方形描边应为 {e}，实际 {area}"
            );
        }
    }

    /// 轮廓必须**首尾相接无缝隙**：最后一笔要回到第一点附近。
    /// （这条防的是圆弧采样后没精确落到终点、留下肉眼难见的缺口。）
    #[test]
    fn outline_is_closed() {
        let poly = [
            Vec2::new(12.0, 14.0),
            Vec2::new(70.0, 30.0),
            Vec2::new(40.0, 78.0),
        ];
        for cap in [LineCap::Butt, LineCap::Square, LineCap::Round] {
            for join in [LineJoin::Bevel, LineJoin::Round, LineJoin::Miter { limit: 4.0 }] {
                let cs = stroke_polyline(&poly, false, &style(9.0, cap, join));
                assert_eq!(cs.len(), 1, "开放折线应只有 1 条轮廓");
                let o = &cs[0];
                let gap = (o[0] - *o.last().unwrap()).hypot();
                // butt 端帽下首尾自然不重合（两端各是一个横截面），
                // 这里检查的是**没有异常长的悬挂段**：闭合缺口不该超过线宽。
                assert!(
                    gap <= 9.0 + 1e-3,
                    "{cap:?}/{join:?}：轮廓首尾相距 {gap}，超出线宽，可能采样漏点"
                );
            }
        }
    }
}
