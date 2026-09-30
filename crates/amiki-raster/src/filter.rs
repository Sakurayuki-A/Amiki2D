//! M6：滤镜（层树）—— 高斯模糊与投影的 CPU 参考实现。
//!
//! # 为什么滤镜必须走"层"
//!
//! 模糊读取的是**周围像素**，而周围像素只有在整层画完之后才存在。
//! 因此不能逐图元计算，必须：
//!
//! ```text
//! 1) 把层内容渲进独立缓冲
//! 2) 对整层滤波
//! 3) 按变换/不透明度/混合模式合成回父场景
//! ```
//!
//! # 实现选择：三次盒式模糊近似高斯
//!
//! 真正的二维高斯是 O(r²) 每像素；而**三次盒式模糊**在 O(1) 每像素
//! （滑动窗口）下就能很好地逼近高斯（中心极限定理），且两次可分离
//! （先横向后纵向）。这是业界标准做法。
//!
//! 盒式宽度按 `w ≈ sqrt(12σ²/n + 1)`（n=3）选取，并强制为奇数以保证
//! 窗口对称 —— 不对称会让模糊中心偏移半像素。

use amiki_scene::{Filter, LayerBounds};

/// 一张离屏缓冲：**预乘** RGBA，f32 便于滤波中间结果不丢精度。
///
/// 为什么存预乘：模糊是对**颜色**做加权平均，若存直线 alpha，
/// 透明区域的"未定义颜色"会被平均进来，边缘出现暗边/亮边。
/// 预乘后颜色已在 alpha=0 处归零，加权平均才是正确的。
#[derive(Clone, Debug)]
pub struct LayerBuffer {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 4]>,
}

impl LayerBuffer {
    pub fn new(width: u32, height: u32) -> LayerBuffer {
        LayerBuffer { width, height, data: vec![[0.0; 4]; (width as usize) * (height as usize)] }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32) -> [f32; 4] {
        if x >= self.width || y >= self.height {
            return [0.0; 4];
        }
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: [f32; 4]) {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)] = v;
        }
    }
}

/// 按滤镜的实际需要扩张层范围。
///
/// 模糊/阴影会把影响扩散到内容之外，**必须**先扩张缓冲，
/// 否则边缘被切掉（而且因为只切一圈，很容易被忽略）。
pub fn inflated_bounds(bounds: LayerBounds, filter: Filter) -> LayerBounds {
    match filter {
        Filter::None => bounds,
        // 三倍标准差之外的高斯尾部可忽略（覆盖 99.7%）
        Filter::Blur { sigma } => bounds.inflate(sigma * 3.0),
        Filter::DropShadow { dx, dy, sigma, .. } => {
            let b = bounds.inflate(sigma * 3.0);
            LayerBounds {
                x: b.x + dx.floor() as i32,
                y: b.y + dy.floor() as i32,
                width: b.width + dx.abs().ceil() as u32,
                height: b.height + dy.abs().ceil() as u32,
            }
        }
    }
}

/// 三次盒式模糊近似高斯（可分离：先横向后纵向）。
pub fn blur(buf: &mut LayerBuffer, sigma: f32) {
    if sigma <= 0.0 || buf.is_empty() {
        return;
    }
    let w = box_width_for(sigma, 3);
    // 三次同宽盒式模糊
    for _ in 0..3 {
        box_blur_h(buf, w);
        box_blur_v(buf, w);
    }
}

/// 与 `n` 次盒式模糊近似等价于标准差 `sigma` 的高斯所需的盒宽（奇数）。
fn box_width_for(sigma: f32, n: u32) -> u32 {
    let ideal = (12.0 * sigma * sigma / n as f32 + 1.0).sqrt();
    let mut w = ideal.round() as u32;
    if w.is_multiple_of(2) {
        w += 1;
    }
    w.max(1)
}

/// 横向盒式模糊（滑动窗口，O(1) 每像素）。
fn box_blur_h(buf: &mut LayerBuffer, radius_w: u32) {
    let r = (radius_w / 2) as i64;
    if r <= 0 {
        return;
    }
    let (w, h) = (buf.width as i64, buf.height as i64);
    let mut row = vec![[0.0f32; 4]; w as usize];
    for y in 0..h {
        for x in 0..w {
            row[x as usize] = buf.data[(y * w + x) as usize];
        }
        // 初始窗口 [0-r, 0+r]
        let mut acc = [0.0f32; 4];
        for i in -r..=r {
            let c = row[i.clamp(0, w - 1) as usize];
            for (k, a) in acc.iter_mut().enumerate() {
                *a += c[k];
            }
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        for x in 0..w {
            for (k, a) in acc.iter().enumerate() {
                buf.data[(y * w + x) as usize][k] = a * inv;
            }
            // 窗口右移：去掉左边，加入右边
            let out = row[(x - r).clamp(0, w - 1) as usize];
            let inc = row[(x + r + 1).clamp(0, w - 1) as usize];
            for (k, a) in acc.iter_mut().enumerate() {
                *a += inc[k] - out[k];
            }
        }
    }
}

/// 纵向盒式模糊。
fn box_blur_v(buf: &mut LayerBuffer, radius_w: u32) {
    let r = (radius_w / 2) as i64;
    if r <= 0 {
        return;
    }
    let (w, h) = (buf.width as i64, buf.height as i64);
    let mut col = vec![[0.0f32; 4]; h as usize];
    for x in 0..w {
        for y in 0..h {
            col[y as usize] = buf.data[(y * w + x) as usize];
        }
        let mut acc = [0.0f32; 4];
        for i in -r..=r {
            let c = col[i.clamp(0, h - 1) as usize];
            for (k, a) in acc.iter_mut().enumerate() {
                *a += c[k];
            }
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        for y in 0..h {
            for (k, a) in acc.iter().enumerate() {
                buf.data[(y * w + x) as usize][k] = a * inv;
            }
            let out = col[(y - r).clamp(0, h - 1) as usize];
            let inc = col[(y + r + 1).clamp(0, h - 1) as usize];
            for (k, a) in acc.iter_mut().enumerate() {
                *a += inc[k] - out[k];
            }
        }
    }
}

/// 把滤镜应用到层缓冲（原地）。
///
/// `DropShadow` 的语义：**保留原内容**，并在其后方叠加一个模糊着色的副本。
/// 这与 CSS `box-shadow`/`drop-shadow` 的直觉一致：阴影不是替换，是附加。
pub fn apply(buf: &mut LayerBuffer, filter: Filter) {
    match filter {
        Filter::None => {}
        Filter::Blur { sigma } => blur(buf, sigma),
        Filter::DropShadow { dx, dy, sigma, color } => {
            // 1) 复制一份作为阴影源，先按颜色着色（保留其 alpha 形状）
            let mut shadow = buf.clone();
            for px in shadow.data.iter_mut() {
                // 阴影颜色固定，形状取源的 alpha
                let a = px[3];
                px[0] = color.r * color.a * a;
                px[1] = color.g * color.a * a;
                px[2] = color.b * color.a * a;
                px[3] = color.a * a;
            }
            // 2) 模糊
            blur(&mut shadow, sigma);
            // 3) 按 (dx,dy) 平移后叠加到原内容**下方**（源覆盖阴影）
            for y in 0..buf.height {
                for x in 0..buf.width {
                    let sx = x as i64 - dx.round() as i64;
                    let sy = y as i64 - dy.round() as i64;
                    let s = if sx >= 0 && sy >= 0 {
                        shadow.get(sx as u32, sy as u32)
                    } else {
                        [0.0; 4]
                    };
                    let d = buf.get(x, y);
                    // src-over：源 d 覆盖阴影 s（预乘）
                    let ia = 1.0 - d[3];
                    buf.set(
                        x,
                        y,
                        [
                            d[0] + s[0] * ia,
                            d[1] + s[1] * ia,
                            d[2] + s[2] * ia,
                            d[3] + s[3] * ia,
                        ],
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use amiki_scene::Color;

    fn single_white_pixel(sz: u32) -> LayerBuffer {
        let mut b = LayerBuffer::new(sz, sz);
        b.set(sz / 2, sz / 2, [1.0, 1.0, 1.0, 1.0]);
        b
    }

    #[test]
    fn box_width_is_odd_and_grows_with_sigma() {
        for s in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
            let w = box_width_for(s, 3);
            assert_eq!(w % 2, 1, "盒宽必须为奇数（对称窗口），sigma={s} 得 {w}");
        }
        assert!(box_width_for(8.0, 3) > box_width_for(1.0, 3), "盒宽应随 sigma 增大");
        assert_eq!(box_width_for(0.0, 3), 1, "sigma=0 应退化为 1（不模糊）");
    }

    #[test]
    fn blur_conserves_total_energy() {
        // 模糊是加权平均，总能量（各通道求和）应基本守恒 —— 这是"没有凭空
        // 增删亮度"的硬性检查，边界钳制会带来少量偏差故给 2% 容差。
        let mut b = single_white_pixel(33);
        let before: f32 = b.data.iter().map(|p| p[3]).sum();
        blur(&mut b, 2.0);
        let after: f32 = b.data.iter().map(|p| p[3]).sum();
        let rel = (after - before).abs() / before;
        assert!(rel < 0.02, "能量应守恒，前后 {before} → {after}（相对差 {rel:.4}）");
    }

    #[test]
    fn blur_spreads_and_center_dims() {
        let mut b = single_white_pixel(33);
        let center = b.get(16, 16);
        blur(&mut b, 3.0);
        let c2 = b.get(16, 16);
        assert!(c2[3] < center[3], "中心 alpha 应因扩散而下降");
        // 邻近像素应被点亮（这就是"扩散"）
        assert!(b.get(17, 16)[3] > 0.0, "相邻像素应获得能量");
        // 对称性：左右、上下应相等
        let l = b.get(15, 16)[3];
        let r = b.get(17, 16)[3];
        let u = b.get(16, 15)[3];
        let d = b.get(16, 17)[3];
        assert!((l - r).abs() < 1e-5, "横向应对称：{l} vs {r}");
        assert!((u - d).abs() < 1e-5, "纵向应对称：{u} vs {d}");
        assert!((l - u).abs() < 1e-5, "可分离模糊应对称：{l} vs {u}");
    }

    #[test]
    fn blur_of_constant_field_is_identity() {
        // 常数场在加权平均下不变 —— 若不成立，说明窗口归一化或边界处理有误
        let sz = 24u32;
        let mut b = LayerBuffer::new(sz, sz);
        for y in 0..sz {
            for x in 0..sz {
                b.set(x, y, [0.25, 0.5, 0.75, 0.5]);
            }
        }
        blur(&mut b, 2.5);
        for y in 1..sz - 1 {
            for x in 1..sz - 1 {
                let p = b.get(x, y);
                // 内部应严格保持（边界因钳制略有偏差，故只看内部）
                assert!((p[0] - 0.25).abs() < 1e-4, "R 应保持 0.25，实际 {}", p[0]);
                assert!((p[1] - 0.5).abs() < 1e-4, "G 应保持 0.5，实际 {}", p[1]);
                assert!((p[3] - 0.5).abs() < 1e-4, "A 应保持 0.5，实际 {}", p[3]);
            }
        }
    }

    #[test]
    fn drop_shadow_keeps_source_and_adds_shadow() {
        // 源：中心一个不透明白点。投影：向右下偏移、红色、无模糊。
        let sz = 21u32;
        let mut b = single_white_pixel(sz);
        apply(
            &mut b,
            Filter::DropShadow {
                dx: 4.0,
                dy: 4.0,
                sigma: 0.0,
                color: Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 },
            },
        );
        // 源仍在原处、保持白色且完全不透明（阴影在**下方**，不替换源）
        // 注意：白色是 [1,1,1,1]，早期版本这里误写成断言 blue≈0
        // （那是**阴影**的判据），把正确的实现判成了失败。
        let src = b.get(10, 10);
        assert!(
            (src[0] - 1.0).abs() < 1e-5 && (src[1] - 1.0).abs() < 1e-5 && (src[2] - 1.0).abs() < 1e-5,
            "源应保持白色 [1,1,1,1]：{src:?}"
        );
        assert!((src[3] - 1.0).abs() < 1e-5, "源应保持不透明：{src:?}");
        // 偏移处应出现红色阴影
        let sh = b.get(14, 14);
        assert!(sh[3] > 0.9, "偏移处应出现阴影，alpha={}", sh[3]);
        assert!(sh[0] > 0.9 && sh[2] < 0.1, "阴影应为红色：{sh:?}");
    }

    #[test]
    fn inflated_bounds_covers_blur_reach() {
        let b = LayerBounds::new(10, 10, 20, 20);
        let infl = inflated_bounds(b, Filter::Blur { sigma: 2.0 });
        assert!(infl.x <= 10 - 6 && infl.y <= 10 - 6, "应向外扩张 3σ：{infl:?}");
        assert!(infl.width >= 20 + 12 && infl.height >= 20 + 12);
        // None 不扩张
        assert_eq!(inflated_bounds(b, Filter::None), b);
    }
}
