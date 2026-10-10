//! 路径与路径构造器（`PathEl` / `Path` / `PathBuilder`）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

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

