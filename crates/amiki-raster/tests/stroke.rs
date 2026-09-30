//! P0-1 描边：轮廓经**解析覆盖率**渲染后必须正确。
//!
//! 验证链路是两段：`amiki_scene::stroke_polyline`（几何 → 轮廓）
//! → 解析覆盖率填充（轮廓 → 像素）。几何的精确面积已在 `amiki-scene` 内验证，
//! 本文件负责**渲染这一段**。
//!
//! # 参考实现为什么是"逐像素超采样 + 非零绕数"
//!
//! 渲染器的覆盖率先累加各段对像素的**有符号面积**，再取绝对值截断到 1。
//! 对一组子路径而言，这个量对应**非零绕数**语义。所以参考实现直接算绕数，
//! 语义与渲染器一致 —— 于是对拍结果反映的是**实现正确性**，
//! 而不是两套不同规则之间的系统性差异。
//!
//! 另外用**超采样**而不是解析求交：多子路径的并/差没有简单的解析覆盖率，
//! 而描边轮廓的点数不多，超采样足够精确（每像素 ss×ss 个样本）。

use amiki_math::{Affine, Vec2};
use amiki_scene::{
    stroke_polyline, Color, LineCap, LineJoin, Paint, PathBuilder, Scene, StrokeStyle,
};

fn style(width: f32, cap: LineCap, join: LineJoin) -> StrokeStyle {
    StrokeStyle { width, cap, join }
}

/// 非零绕数：点 `q` 是否在轮廓集合内部（含环的挖空）。
fn winding(q: Vec2, contours: &[Vec<Vec2>]) -> i32 {
    let mut wn = 0;
    for c in contours {
        for i in 0..c.len() {
            let a = c[i];
            let b = c[(i + 1) % c.len()];
            // 标准绕数算法（上穿 +1、下穿 −1）
            if a.y <= q.y {
                if b.y > q.y && (b - a).cross(q - a) > 0.0 {
                    wn += 1;
                }
            } else if b.y <= q.y && (b - a).cross(q - a) < 0.0 {
                wn -= 1;
            }
        }
    }
    wn
}

/// 超采样参考覆盖率。
fn reference(contours: &[Vec<Vec2>], w: u32, h: u32, ss: u32) -> Vec<f32> {
    let mut cov = vec![0.0f32; (w * h) as usize];
    for py in 0..h {
        for px in 0..w {
            let mut hits = 0u32;
            for sy in 0..ss {
                for sx in 0..ss {
                    let q = Vec2::new(
                        px as f32 + (sx as f32 + 0.5) / ss as f32,
                        py as f32 + (sy as f32 + 0.5) / ss as f32,
                    );
                    if winding(q, contours) != 0 {
                        hits += 1;
                    }
                }
            }
            cov[(py * w + px) as usize] = hits as f32 / (ss * ss) as f32;
        }
    }
    cov
}

/// 轮廓 → 场景 → 渲染 → 每像素覆盖率。
///
/// 画法：白底 + **不透明黑**描边。于是覆盖率 = `1 − R/255`，
/// 不必碰 alpha 合成的细节。
fn render_coverage(contours: &[Vec<Vec2>], w: u32, h: u32) -> Vec<f32> {
    let mut scene = Scene::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 255, 255)));

    let mut pb = PathBuilder::new();
    for c in contours {
        if c.len() < 3 {
            continue;
        }
        pb.move_to(c[0]);
        for p in &c[1..] {
            pb.line_to(*p);
        }
        pb.close();
    }
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 0, 0)));

    let img = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
    (0..(w * h) as usize)
        .map(|i| 1.0 - img[i * 4] as f32 / 255.0)
        .collect()
}

/// 对拍：返回（最大逐像素差，总面积差）。
fn compare(contours: &[Vec<Vec2>], w: u32, h: u32, ss: u32) -> (f32, f32) {
    let got = render_coverage(contours, w, h);
    let exp = reference(contours, w, h, ss);
    let max_px = got
        .iter()
        .zip(exp.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let sa: f32 = got.iter().sum();
    let sb: f32 = exp.iter().sum();
    (max_px, (sa - sb).abs() / sb.max(1.0))
}

/// 开放折线：三种 join × 三种 cap 的组合都要渲染正确。
///
/// **总面积差是主判据**（紧到 0.5%）：它能直接抓出重复计数、缺块、绕向错。
/// 逐像素差放宽到 0.06 —— 超采样参考本身有约 `1/(2·ss)` 的量化误差。
#[test]
fn open_stroke_renders_like_reference() {
    let poly = [
        Vec2::new(20.0, 18.0),
        Vec2::new(74.0, 34.0),
        Vec2::new(44.0, 82.0),
    ];
    let (w, h) = (96u32, 96u32);
    for cap in [LineCap::Butt, LineCap::Square, LineCap::Round] {
        for join in [LineJoin::Bevel, LineJoin::Round, LineJoin::Miter { limit: 4.0 }] {
            let cs = stroke_polyline(&poly, false, &style(11.0, cap, join));
            let (px, area) = compare(&cs, w, h, 16);
            assert!(
                area < 0.005,
                "{cap:?}/{join:?}：总面积相对差 {area:.4} 过大（应 < 0.005）"
            );
            assert!(
                px < 0.06,
                "{cap:?}/{join:?}：最大逐像素覆盖率差 {px:.4}（应 < 0.06）"
            );
        }
    }
}

/// 闭合折线：环（外圈 + 反向内圈）必须渲染成**中空**的环。
///
/// 这条专门防"两条轮廓绕向相同"的错误：那样内圈会被**填满**，
/// 总面积会从 `外−内` 变成 `外+内`，主判据立刻失败。
#[test]
fn closed_stroke_is_a_hollow_ring() {
    let sq = [
        Vec2::new(24.0, 24.0),
        Vec2::new(72.0, 24.0),
        Vec2::new(72.0, 72.0),
        Vec2::new(24.0, 72.0),
    ];
    let (w, h) = (96u32, 96u32);
    for join in [LineJoin::Bevel, LineJoin::Miter { limit: 4.0 }] {
        let cs = stroke_polyline(&sq, true, &style(10.0, LineCap::Butt, join));
        assert_eq!(cs.len(), 2, "{join:?}：闭合描边应是环（2 条轮廓）");

        // 环的中心必须是空的 —— 这是"中空"最直接的证据
        let cov = render_coverage(&cs, w, h);
        let center = cov[(48 * w + 48) as usize];
        assert!(
            center < 0.01,
            "{join:?}：环心应未被填充（内圈绕向可能与外圈相同），实际覆盖率 {center}"
        );

        let (px, area) = compare(&cs, w, h, 16);
        assert!(area < 0.005, "{join:?}：环总面积相对差 {area:.4} 过大");
        assert!(px < 0.06, "{join:?}：最大逐像素覆盖率差 {px:.4}");
    }
}

/// 细线也要正确：线宽 1px 时覆盖率完全落在解析范围里，
/// 逐像素差应**明显更小**（没有超采样量化带来的宽容差空间）。
#[test]
fn thin_stroke_is_precise() {
    let poly = [Vec2::new(10.0, 30.5), Vec2::new(86.0, 30.5)];
    let cs = stroke_polyline(&poly, false, &style(1.0, LineCap::Butt, LineJoin::Bevel));
    let (w, h) = (96u32, 48u32);
    let got = render_coverage(&cs, w, h);
    // 水平 1px 线，中心在 y=30.5：落在第 30 行的覆盖率应为 1，其余为 0
    for py in 0..h {
        for px in 10..86 {
            let c = got[(py * w + px) as usize];
            let expect = if py == 30 { 1.0 } else { 0.0 };
            assert!(
                (c - expect).abs() < 1e-3,
                "({px},{py}) 覆盖率应为 {expect}，实际 {c}"
            );
        }
    }
}

/// 退化输入必须**返回空**而不是画出错的东西。
#[test]
fn degenerate_inputs_yield_nothing() {
    let st = style(5.0, LineCap::Butt, LineJoin::Bevel);
    assert!(stroke_polyline(&[], false, &st).is_empty(), "空折线");
    assert!(stroke_polyline(&[Vec2::new(1.0, 1.0)], false, &st).is_empty(), "单点");
    // 所有点重合
    let same = [Vec2::new(3.0, 3.0); 4];
    assert!(stroke_polyline(&same, false, &st).is_empty(), "全部重合");
    // 零宽
    assert!(
        stroke_polyline(&[Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)], false, &style(0.0, LineCap::Butt, LineJoin::Bevel))
            .is_empty(),
        "零线宽"
    );
}
