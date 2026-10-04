//! even-odd 填充规则的端到端测试（0.8.0）。
//!
//! 参照实现是**独立**的：逐点算绕数 → 按 `wn != 0`（NonZero）或 `wn % 2 != 0`
//! （even-odd）判定 → 超采样得到覆盖率。它不共享引擎的任何一行覆盖率数学，
//! 所以能真正裁判实现。
//!
//! 形状一律**非整数对齐** —— 否则边缘落在像素边界上，一个部分覆盖像素都没有，
//! 测试会"因为没牙而通过"（spike 第一次就是这么假通过的）。

use amiki_math::{Affine, Vec2};
use amiki_raster::{bin, render};
use amiki_scene::{Color, FillRule, Paint, PathBuilder, Scene};

/// 点 `q` 的绕数（标准算法：上穿 +1、下穿 −1）。
fn winding(q: Vec2, contours: &[Vec<Vec2>]) -> i32 {
    let mut wn = 0;
    for c in contours {
        for i in 0..c.len() {
            let a = c[i];
            let b = c[(i + 1) % c.len()];
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
fn reference(contours: &[Vec<Vec2>], w: u32, h: u32, ss: u32, even_odd: bool) -> Vec<f32> {
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
                    let wn = winding(q, contours);
                    let inside = if even_odd { wn % 2 != 0 } else { wn != 0 };
                    if inside {
                        hits += 1;
                    }
                }
            }
            cov[(py * w + px) as usize] = hits as f32 / (ss * ss) as f32;
        }
    }
    cov
}

/// 轴对齐矩形；`ccw = true` 时反向（屏幕坐标 y 向下）。
fn rect(x0: f32, y0: f32, x1: f32, y1: f32, ccw: bool) -> Vec<Vec2> {
    let mut v = vec![
        Vec2::new(x0, y0),
        Vec2::new(x1, y0),
        Vec2::new(x1, y1),
        Vec2::new(x0, y1),
    ];
    if ccw {
        v.reverse();
    }
    v
}

/// **真自相交**星形：`{5/2}` 星多边形（隔一个顶点连），中心区绕数为 2。
///
/// 注意别写成"外/内半径交替"——那是**简单多边形**（不自交），两条规则结果一样。
fn star(cx: f32, cy: f32, r: f32, k: usize) -> Vec<Vec2> {
    let n = 2 * k + 1; // 5 → {5/2}
    let mut v = Vec::new();
    for i in 0..n {
        let a = 2.0 * std::f32::consts::PI * (i as f32) * 2.0 / (n as f32)
            - std::f32::consts::FRAC_PI_2;
        v.push(Vec2::new(cx + r * a.cos(), cy + r * a.sin()));
    }
    v
}

/// 白底 + 不透明黑填充 ⇒ 覆盖率 = `1 − R/255`。
fn render_coverage(contours: &[Vec<Vec2>], w: u32, h: u32, rule: FillRule) -> Vec<f32> {
    let mut scene = Scene::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(255, 255, 255)),
    );

    let mut pb = PathBuilder::new();
    for c in contours {
        pb.move_to(c[0]);
        for p in &c[1..] {
            pb.line_to(*p);
        }
        pb.close();
    }
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(0, 0, 0)),
    );
    scene.draws.last_mut().expect("刚 push 过").fill_rule = rule;

    let img = render(&bin(&scene, 0.05));
    (0..(w * h) as usize)
        .map(|i| 1.0 - img[i * 4] as f32 / 255.0)
        .collect()
}

/// 逐像素比对，返回 `(最大偏差, 位置, 部分覆盖像素数)`。
fn compare(got: &[f32], want: &[f32], w: u32) -> (f32, (u32, u32), usize) {
    let mut worst = 0.0f32;
    let mut at = (0u32, 0u32);
    let mut partial = 0usize;
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let (px, py) = (i as u32 % w, i as u32 / w);
        if b > 0.001 && b < 0.999 {
            partial += 1;
        }
        let d = (a - b).abs();
        if d > worst {
            worst = d;
            at = (px, py);
        }
    }
    (worst, at, partial)
}

// ================================================================ 测试

/// 简单形状：even-odd 与 NonZero **语义相同**，因此结果也必须与 NonZero 参考一致。
#[test]
fn even_odd_matches_reference_on_simple_shape() {
    let contours = vec![rect(2.3, 2.1, 6.4, 5.7, true)];
    let (w, h) = (8u32, 8u32);
    let got = render_coverage(&contours, w, h, FillRule::EvenOdd);
    let want = reference(&contours, w, h, 32, true);
    let (worst, at, partial) = compare(&got, &want, w);
    eprintln!("简单方块 even-odd：最大偏差 {worst:.4}（在 {at:?}），部分覆盖像素 {partial} 个");
    assert!(partial >= 4, "参照里必须有部分覆盖像素，否则测试没牙");
    assert!(worst <= 0.03, "even-odd 与参考不符：{worst:.4}");
}

/// **零破坏**：同一个简单形状下，两条规则给出同一张图（逐字节）。
#[test]
fn nonzero_and_even_odd_agree_on_simple_shape() {
    let contours = vec![rect(1.7, 2.2, 5.9, 6.3, true)];
    let (w, h) = (8u32, 8u32);
    let nz = render_coverage(&contours, w, h, FillRule::NonZero);
    let eo = render_coverage(&contours, w, h, FillRule::EvenOdd);
    let (worst, at, _) = compare(&eo, &nz, w);
    eprintln!("简单方块 两条规则差异：最大 {worst:.4}（在 {at:?}）");
    assert!(worst <= 0.02, "简单形状下两条规则应当一致：{worst:.4}");
}

/// **判别用例**：同向双环 —— NonZero 实心、EvenOdd 镂空。
///
/// 注意计划里写的是"外圈顺时针 + 内圈逆时针"：那种环两条规则**都**镂空，
/// 没有判别力（见 `fill_rule_semantics.rs` 的实测）。
#[test]
fn same_orientation_double_loop_discriminates() {
    let contours = vec![rect(1.4, 1.3, 6.7, 6.8, true), rect(3.2, 3.1, 4.9, 5.2, true)];
    let (w, h) = (8u32, 8u32);
    let nz = render_coverage(&contours, w, h, FillRule::NonZero);
    let eo = render_coverage(&contours, w, h, FillRule::EvenOdd);
    let want_eo = reference(&contours, w, h, 32, true);
    let want_nz = reference(&contours, w, h, 32, false);

    let center = (4 * 8 + 4) as usize; // 内环中心（**部分覆盖**像素：内环边界穿过它）
    eprintln!(
        "同向双环中心：NonZero 渲染 {:.3}（参考 {:.3}）、EvenOdd 渲染 {:.3}（参考 {:.3}）",
        nz[center], want_nz[center], eo[center], want_eo[center]
    );
    assert!(nz[center] > 0.99, "NonZero 下同向双环中心应实心");
    // 该像素被内环覆盖约 0.9，even-odd 只留下没被覆盖的那部分 ⇒ 明显变浅
    assert!(
        nz[center] - eo[center] > 0.5,
        "EvenOdd 下同向双环中心必须明显变浅（实测差 {:.3}）",
        nz[center] - eo[center]
    );

    let (worst, at, partial) = compare(&eo, &want_eo, w);
    eprintln!("同向双环 even-odd vs 参考：最大偏差 {worst:.4}（在 {at:?}），部分覆盖像素 {partial}");
    assert!(partial >= 8, "参照里必须有部分覆盖像素");
    assert!(worst <= 0.03, "even-odd 与参考不符：{worst:.4}");
}

/// 计划里的形状（外圈顺时针 + 内圈逆时针）：两条规则结果相同 —— 它没有判别力。
#[test]
fn opposite_orientation_ring_is_indistinguishable() {
    let contours = vec![rect(1.4, 1.3, 6.7, 6.8, true), rect(3.2, 3.1, 4.9, 5.2, false)];
    let (w, h) = (8u32, 8u32);
    let nz = render_coverage(&contours, w, h, FillRule::NonZero);
    let eo = render_coverage(&contours, w, h, FillRule::EvenOdd);
    let want_nz = reference(&contours, w, h, 32, false);
    let want_eo = reference(&contours, w, h, 32, true);

    let (d_nz, at_nz, _) = compare(&nz, &want_nz, w);
    let (d_eo, at_eo, _) = compare(&eo, &want_eo, w);
    let (worst, at, _) = compare(&eo, &nz, w);
    eprintln!(
        "反向双环：两条规则最大差 {worst:.4}（在 {at:?}）；各自 vs 参考 {d_nz:.4}（{at_nz:?}）/{d_eo:.4}（{at_eo:?}）"
    );
    // 反向环在两套语义下都是"环" ⇒ 两条规则必须给出同一张图（这就是它没判别力的原因）
    assert!(worst <= 0.02, "反向双环下两条规则应当一致：{worst:.4}");
    assert!(d_nz <= 0.03 && d_eo <= 0.03, "两条规则都要与各自参考一致");
}

/// **自相交**单环（`{5/2}` 五角星）：中心区绕数为 2 ⇒ 两条规则必须明显不同，且都与参考一致。
#[test]
fn self_intersecting_star_matches_reference() {
    let contours = vec![star(12.0, 12.0, 9.3, 2)];
    let (w, h) = (24u32, 24u32);
    let eo = render_coverage(&contours, w, h, FillRule::EvenOdd);
    let nz = render_coverage(&contours, w, h, FillRule::NonZero);
    let want_eo = reference(&contours, w, h, 32, true);
    let want_nz = reference(&contours, w, h, 32, false);

    let (worst_eo, at_eo, partial) = compare(&eo, &want_eo, w);
    let (worst_nz, at_nz, _) = compare(&nz, &want_nz, w);
    let (diff, at_d, _) = compare(&eo, &nz, w);
    eprintln!(
        "五角星：even-odd 偏差 {worst_eo:.4}（在 {at_eo:?}）、nonzero 偏差 {worst_nz:.4}（在 {at_nz:?}）"
    );
    eprintln!("        两条规则差异 {diff:.4}（在 {at_d:?}）；部分覆盖像素 {partial} 个");
    assert!(partial >= 20, "参照里必须有足够多的部分覆盖像素");
    assert!(diff > 0.2, "自相交星形下两条规则必须有明显差异（实测 {diff:.4}）");
    assert!(worst_eo <= 0.05, "五角星 even-odd 与参考不符：{worst_eo:.4}");
    // NonZero 在自相交路径上的偏差是**既有**行为（`min(|Σ|,1)` 在 |W| ≥ 2 区域边缘饱和），
    // 与 even-odd 的预处理无关（nonzero 路径一行未改）。这里只做数量级兜底，
    // 实测值记在 `docs/KNOWN_ISSUES.md` 的 B16。
    assert!(
        worst_nz <= 0.2,
        "五角星 NonZero 偏差突然变大（{worst_nz:.4}）—— 这可能是 nonzero 路径被改动了"
    );
}

/// 三层同向嵌套：even-odd 应当是"环 + 环"（交替），NonZero 是实心。
#[test]
fn nested_same_orientation_rings_alternate() {
    let contours = vec![
        rect(1.4, 1.4, 14.6, 14.6, true),
        rect(3.2, 3.2, 12.8, 12.8, true),
        rect(5.1, 5.1, 10.9, 10.9, true),
    ];
    let (w, h) = (16u32, 16u32);
    let eo = render_coverage(&contours, w, h, FillRule::EvenOdd);
    let want = reference(&contours, w, h, 32, true);
    let (worst, at, partial) = compare(&eo, &want, w);
    eprintln!("三层嵌套 even-odd：最大偏差 {worst:.4}（在 {at:?}），部分覆盖像素 {partial}");
    assert!(partial >= 8);
    assert!(worst <= 0.03, "三层嵌套 even-odd 与参考不符：{worst:.4}");
}

/// 退化输入不应 panic：空路径、单段、零面积路径。
#[test]
fn degenerate_paths_do_not_panic() {
    let (w, h) = (8u32, 8u32);
    // 空路径
    let mut scene = Scene::new(w, h);
    let pb = PathBuilder::new();
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(0, 0, 0)),
    );
    scene.draws.last_mut().unwrap().fill_rule = FillRule::EvenOdd;
    let _ = render(&bin(&scene, 0.05));

    // 只有一段、以及零面积"来回走"
    let mut pb = PathBuilder::new();
    pb.move_to(Vec2::new(2.0, 2.0));
    pb.line_to(Vec2::new(6.0, 6.0));
    pb.move_to(Vec2::new(1.0, 1.0));
    pb.line_to(Vec2::new(5.0, 5.0));
    pb.line_to(Vec2::new(1.0, 1.0));
    let mut scene = Scene::new(w, h);
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(0, 0, 0)),
    );
    scene.draws.last_mut().unwrap().fill_rule = FillRule::EvenOdd;
    let img = render(&bin(&scene, 0.05));
    assert_eq!(img.len(), (w * h * 4) as usize);
    eprintln!("退化路径：未 panic");
}
