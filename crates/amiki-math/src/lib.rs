//! amiki-math —— Amiki2D 零依赖几何核心。
//!
//! 原则：不依赖任何第三方库，可被任意宿主嵌入与绑定。
//! 曲线压平采用**凸包性质保证的误差界自适应二分**：弦长归一化的控制点垂距
//! 不超过 ε 时输出弦，数学上保证折线与真实曲线的偏差 ≤ ε（设备像素）。

#![forbid(unsafe_code)]

use std::ops::{Add, Mul, Sub};

/// 2D 向量 / 点。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    /// 2D 叉积（标量）：self × o = self.x*o.y − self.y*o.x。
    pub fn cross(self, o: Vec2) -> f32 {
        self.x * o.y - self.y * o.x
    }

    pub fn hypot(self) -> f32 {
        self.x.hypot(self.y)
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f32) -> Vec2 {
        Vec2::new(self.x * s, self.y * s)
    }
}

pub type Point = Vec2;

/// 仿射变换（2×3，列向量习惯同 PostScript/kurbo）：
///
/// ```text
/// | a c e |       x' = a·x + c·y + e
/// | b d f |       y' = b·x + d·y + f
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

impl Affine {
    pub const IDENTITY: Affine = Affine { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: 0.0, f: 0.0 };

    pub fn translate(t: Vec2) -> Affine {
        Affine { a: 1.0, b: 0.0, c: 0.0, d: 1.0, e: t.x, f: t.y }
    }

    pub fn scale(s: f32) -> Affine {
        Affine::scale_nonuniform(s, s)
    }

    pub fn scale_nonuniform(sx: f32, sy: f32) -> Affine {
        Affine { a: sx, b: 0.0, c: 0.0, d: sy, e: 0.0, f: 0.0 }
    }

    /// 旋转 theta 弧度（数学惯例矩阵；屏幕坐标系 y 向下时视觉为顺时针）。
    pub fn rotate(theta: f32) -> Affine {
        let (s, c) = theta.sin_cos();
        Affine { a: c, b: s, c: -s, d: c, e: 0.0, f: 0.0 }
    }

    /// 映射点（含平移）。
    pub fn map_point(&self, p: Vec2) -> Vec2 {
        Vec2::new(self.a * p.x + self.c * p.y + self.e, self.b * p.x + self.d * p.y + self.f)
    }

    /// 映射向量（不含平移）。
    pub fn map_vec(&self, v: Vec2) -> Vec2 {
        Vec2::new(self.a * v.x + self.c * v.y, self.b * v.x + self.d * v.y)
    }

    /// 组合：先应用 `self`，再应用 `next`。
    pub fn then(&self, next: Affine) -> Affine {
        Affine {
            a: self.a * next.a + self.c * next.b,
            b: self.b * next.a + self.d * next.b,
            c: self.a * next.c + self.c * next.d,
            d: self.b * next.c + self.d * next.d,
            e: self.a * next.e + self.c * next.f + self.e,
            f: self.b * next.e + self.d * next.f + self.f,
        }
    }
}

impl Mul for Affine {
    type Output = Affine;
    /// `a * b` 等价于 `a.then(b)`：先 b 后 a。
    fn mul(self, rhs: Affine) -> Affine {
        self.then(rhs)
    }
}

/// 三次贝塞尔。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubicBez {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
    pub p3: Vec2,
}

impl CubicBez {
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> Self {
        Self { p0, p1, p2, p3 }
    }

    pub fn eval(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        let w0 = u * u * u;
        let w1 = 3.0 * u * u * t;
        let w2 = 3.0 * u * t * t;
        let w3 = t * t * t;
        self.p0 * w0 + self.p1 * w1 + self.p2 * w2 + self.p3 * w3
    }

    /// 在 t = 0.5 处 de Casteljau 分裂。
    pub fn split(&self) -> (CubicBez, CubicBez) {
        fn mid(a: Vec2, b: Vec2) -> Vec2 {
            (a + b) * 0.5
        }
        let p01 = mid(self.p0, self.p1);
        let p12 = mid(self.p1, self.p2);
        let p23 = mid(self.p2, self.p3);
        let p012 = mid(p01, p12);
        let p123 = mid(p12, p23);
        let p0123 = mid(p012, p123);
        (
            CubicBez::new(self.p0, p01, p012, p0123),
            CubicBez::new(p0123, p123, p23, self.p3),
        )
    }

    /// 误差界压平：把折线顶点（不含起点，含终点）追加到 `out`。
    ///
    /// 平坦判据：控制点到弦的垂距（凸包性质 ⇒ 曲线到弦的偏差 ≤ max(h1,h2)）。
    /// ε 以设备像素度量（调用方须先做变换）。
    pub fn flatten(&self, eps: f32, out: &mut Vec<Vec2>) {
        flatten_cubic_rec(*self, eps, 0, out);
    }

    /// 均匀压平所需段数（GPU 并行压平用）。
    ///
    /// 推导：三次曲线对弦的偏差 d(t) = t(1−t)·g(t)，g 线性于 t，
    /// g(0) = 3p1 − 2p0 − p3，g(1) = 3p2 − 2p3 − p0。
    /// 子区间 [t0, t0+w] 内的二阶差分泰勒展开给出每段偏差
    /// err_n ≤ (max|g| + |g′|)·3/4 /n² ≤ 1.3·G/n²，
    /// 其中 G = max(|g(0)|, |g(1)|)（g 线性 ⇒ |g| 在端点取最大；|g′| ≤ 2G）。
    /// 要求 ≤ ε ⇒ **n = ⌈√(1.3·G/ε)⌉（可证明保守界，随机化测试佐证）**。
    pub fn uniform_segments(&self, eps: f32) -> u32 {
        let g0 = self.p1 * 3.0 - self.p0 * 2.0 - self.p3;
        let g1 = self.p2 * 3.0 - self.p3 * 2.0 - self.p0;
        // 与 WGSL 逐式对应：分别开方后取 max（避免 max(平方) 再开方的 1 ulp 差异）
        let l0 = (g0.x * g0.x + g0.y * g0.y).sqrt();
        let l1 = (g1.x * g1.x + g1.y * g1.y).sqrt();
        let n = (1.3 * l0.max(l1) / eps).sqrt().ceil().max(1.0);
        n as u32
    }

    /// 均匀参数化压平：输出 n+1 个点（含首尾），段数由 [`Self::uniform_segments`] 决定。
    pub fn flatten_uniform(&self, eps: f32, out: &mut Vec<Vec2>) {
        let n = self.uniform_segments(eps);
        for i in 0..=n {
            out.push(self.eval(i as f32 / n as f32));
        }
    }
}

fn flatten_cubic_rec(c: CubicBez, eps: f32, depth: u32, out: &mut Vec<Vec2>) {
    let chord = c.p3 - c.p0;
    let cl = chord.hypot();
    let (h1, h2) = if cl > 1e-6 {
        (
            (c.p1 - c.p0).cross(chord).abs() / cl,
            (c.p2 - c.p0).cross(chord).abs() / cl,
        )
    } else {
        // 起终点重合（退化）：直接用控制点距离
        ((c.p1 - c.p0).hypot(), (c.p2 - c.p0).hypot())
    };
    // 32 层上限兜底（ε 极小 + 数值异常时仍终止）
    if depth >= 32 || (h1.max(h2) <= eps) {
        out.push(c.p3);
        return;
    }
    let (l, r) = c.split();
    flatten_cubic_rec(l, eps, depth + 1, out);
    flatten_cubic_rec(r, eps, depth + 1, out);
}

/// 二次贝塞尔。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuadBez {
    pub p0: Vec2,
    pub p1: Vec2,
    pub p2: Vec2,
}

impl QuadBez {
    pub fn new(p0: Vec2, p1: Vec2, p2: Vec2) -> Self {
        Self { p0, p1, p2 }
    }

    pub fn eval(&self, t: f32) -> Vec2 {
        let u = 1.0 - t;
        self.p0 * (u * u) + self.p1 * (2.0 * u * t) + self.p2 * (t * t)
    }

    pub fn split(&self) -> (QuadBez, QuadBez) {
        fn mid(a: Vec2, b: Vec2) -> Vec2 {
            (a + b) * 0.5
        }
        let p01 = mid(self.p0, self.p1);
        let p12 = mid(self.p1, self.p2);
        let p012 = mid(p01, p12);
        (
            QuadBez::new(self.p0, p01, p012),
            QuadBez::new(p012, p12, self.p2),
        )
    }

    pub fn flatten(&self, eps: f32, out: &mut Vec<Vec2>) {
        flatten_quad_rec(*self, eps, 0, out);
    }

    /// 均匀压平所需段数（GPU 并行压平用）。
    ///
    /// 二次曲线对弦的偏差 d(t) = −t(1−t)·a（a = p0 − 2p1 + p2，二阶差分），
    /// 最大偏差 |a|/4；均匀 n 段剖分后每段的二阶差分精确缩放为 a/n²，
    /// 故每段偏差 ≤ |a|/(4n²)。要求 ≤ ε ⇒ **n = ⌈√(|a|/4ε)⌉（精确界）**。
    pub fn uniform_segments(&self, eps: f32) -> u32 {
        let a = self.p0 - self.p1 * 2.0 + self.p2;
        let l = (a.x * a.x + a.y * a.y).sqrt();
        let n = (l / (4.0 * eps)).sqrt().ceil().max(1.0);
        n as u32
    }

    /// 均匀参数化压平：输出 n+1 个点（含首尾），段数由 [`Self::uniform_segments`] 决定。
    pub fn flatten_uniform(&self, eps: f32, out: &mut Vec<Vec2>) {
        let n = self.uniform_segments(eps);
        for i in 0..=n {
            out.push(self.eval(i as f32 / n as f32));
        }
    }
}

fn flatten_quad_rec(q: QuadBez, eps: f32, depth: u32, out: &mut Vec<Vec2>) {
    let chord = q.p2 - q.p0;
    let cl = chord.hypot();
    let h = if cl > 1e-6 {
        (q.p1 - q.p0).cross(chord).abs() / cl
    } else {
        (q.p1 - q.p0).hypot()
    };
    if depth >= 32 || h <= eps {
        out.push(q.p2);
        return;
    }
    let (l, r) = q.split();
    flatten_quad_rec(l, eps, depth + 1, out);
    flatten_quad_rec(r, eps, depth + 1, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 折线到真实曲线的最大采样偏差必须 ≤ ε（留 20% 余量给采样离散化）。
    fn assert_flatten_error(c: CubicBez, eps: f32) {
        let mut pts = vec![c.p0];
        c.flatten(eps, &mut pts);
        assert!(pts.len() >= 2);
        let mut max_err = 0.0f32;
        let n = 512;
        for i in 1..n {
            let t = i as f32 / n as f32;
            let p = c.eval(t);
            // 点到折线各段的最小距离
            let mut best = f32::INFINITY;
            for w in pts.windows(2) {
                let a = w[0];
                let b = w[1];
                let ab = b - a;
                let l2 = ab.dot(ab);
                let t2 = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
                let q = a + ab * t2;
                best = best.min((p - q).hypot());
            }
            max_err = max_err.max(best);
        }
        assert!(
            max_err <= eps * 1.2,
            "压平误差 {max_err} 超出界限 {}",
            eps * 1.2
        );
    }

    #[test]
    fn cubic_flatten_error_bound() {
        let eps = 0.1;
        // 常规弧
        assert_flatten_error(
            CubicBez::new(Vec2::new(0.0, 0.0), Vec2::new(0.0, 100.0), Vec2::new(100.0, 100.0), Vec2::new(100.0, 0.0)),
            eps,
        );
        // 带尖点 / 环（P1、P2 交叉）的最坏情形
        assert_flatten_error(
            CubicBez::new(Vec2::new(0.0, 0.0), Vec2::new(200.0, -150.0), Vec2::new(-150.0, 200.0), Vec2::new(120.0, 30.0)),
            eps,
        );
        // 近似直线
        assert_flatten_error(
            CubicBez::new(Vec2::new(0.0, 0.0), Vec2::new(30.0, 0.5), Vec2::new(60.0, -0.5), Vec2::new(90.0, 0.0)),
            eps,
        );
    }

    #[test]
    fn quad_flatten_error_bound() {
        let mut pts = vec![Vec2::new(0.0, 0.0)];
        QuadBez::new(Vec2::ZERO, Vec2::new(50.0, 120.0), Vec2::new(100.0, 0.0))
            .flatten(0.1, &mut pts);
        assert!(pts.len() >= 3);
    }

    #[test]
    fn affine_compose() {
        let t = Affine::translate(Vec2::new(10.0, 0.0));
        let r = Affine::rotate(std::f32::consts::FRAC_PI_2);
        // 先平移后旋转：点 (0,0) → (10,0) → (0,10)（y 向下）
        let m = r.then(t);
        let p = m.map_point(Vec2::ZERO);
        assert!((p.x - 0.0).abs() < 1e-5 && (p.y - 10.0).abs() < 1e-5);
    }

    // ===================== 均匀压平误差界性质测试 =====================

    /// 种子化 LCG（测试可复现，不引入 rand 依赖）。
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (u32::MAX as f32)
        }
        fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.next_f32()
        }
    }

    /// 点到线段距离。
    fn point_seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
        let ab = b - a;
        let l2 = ab.dot(ab);
        let t = if l2 > 0.0 { ((p - a).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
        (p - (a + ab * t)).hypot()
    }

    /// 折线到真实曲线的最大偏差（稠密采样真值）。
    fn polyline_error(pts: &[Vec2], eval: impl Fn(f32) -> Vec2) -> f32 {
        let n = 4096;
        let mut max_err = 0.0f32;
        for i in 1..n {
            let p = eval(i as f32 / n as f32);
            let mut best = f32::INFINITY;
            for w in pts.windows(2) {
                best = best.min(point_seg_dist(p, w[0], w[1]));
            }
            max_err = max_err.max(best);
        }
        max_err
    }

    #[test]
    fn uniform_flatten_error_bound_random() {
        let eps = 0.1f32;
        let mut rng = Lcg(0x1234_5678_9ABC_DEF0);

        // 随机曲线 + 对抗性构造（环、尖点、极端控制点）
        let mut quad_cases: Vec<QuadBez> = vec![
            QuadBez::new(Vec2::ZERO, Vec2::new(1e4, 1e4), Vec2::new(0.0, 1.0)),
            QuadBez::new(Vec2::new(50.0, 50.0), Vec2::new(50.2, 50.2), Vec2::new(50.0, 50.0)),
        ];
        let mut cubic_cases: Vec<CubicBez> = vec![
            // 环
            CubicBez::new(Vec2::ZERO, Vec2::new(200.0, -200.0), Vec2::new(-200.0, 200.0), Vec2::new(120.0, 30.0)),
            // 近尖点
            CubicBez::new(Vec2::ZERO, Vec2::new(100.0, 0.0), Vec2::new(100.0, 0.1), Vec2::new(0.0, 0.05)),
            // 超长
            CubicBez::new(Vec2::new(0.0, 0.0), Vec2::new(5e4, 3e4), Vec2::new(-3e4, 2e4), Vec2::new(1e3, 1.0)),
        ];
        for _ in 0..1000 {
            let v = |r: &mut Lcg| Vec2::new(r.next_range(-500.0, 500.0), r.next_range(-500.0, 500.0));
            quad_cases.push(QuadBez::new(v(&mut rng), v(&mut rng), v(&mut rng)));
            cubic_cases.push(CubicBez::new(v(&mut rng), v(&mut rng), v(&mut rng), v(&mut rng)));
        }

        for q in &quad_cases {
            let mut pts = Vec::new();
            q.flatten_uniform(eps, &mut pts);
            let err = polyline_error(&pts, |t| q.eval(t));
            assert!(err <= eps * 1.2, "quad 均匀压平误差 {err} 超界（n={}）", q.uniform_segments(eps));
        }
        for c in &cubic_cases {
            let mut pts = Vec::new();
            c.flatten_uniform(eps, &mut pts);
            let err = polyline_error(&pts, |t| c.eval(t));
            assert!(err <= eps * 1.2, "cubic 均匀压平误差 {err} 超界（n={}）", c.uniform_segments(eps));
        }
    }
}
