//! **阶段 1 spike**：even-odd 的两件事，先用最小实验钉死，再动生产代码。
//!
//! 1. **边界约定**（计划 §2.2）：段与 tile / 像素的归属统一成左闭右开 `[lo, hi)`。
//! 2. **累加语义**（计划 §2.1 的前提）：even-odd 能不能"复用 `seg_area`、
//!    只把最终结果取奇偶"就算对？—— 用**独立超采样裁判**（逐点绕数取模 2）
//!    对拍，看它到底对不对。
//!
//! 本文件是**实验**，不改任何生产代码；结论写进报告与 `docs/`。

use amiki_math::Vec2;
use amiki_raster::seg_area;

// ---------------------------------------------------------------- 参照实现

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

/// 超采样参考覆盖率。`even_odd = true` 时用**奇偶**判定（这正是 even-odd 的定义）。
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

// ---------------------------------------------------------------- 段与内核

/// 轮廓 → 线段表（每条 `[x0, y0, x1, y1]`）。
fn segs_of(contours: &[Vec<Vec2>]) -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    for c in contours {
        for i in 0..c.len() {
            let a = c[i];
            let b = c[(i + 1) % c.len()];
            if a.y != b.y {
                // 水平段对覆盖率贡献恒为 0（`seg_area` 里 dy≈0 直接返回 0）
                out.push([a.x, a.y, b.x, b.y]);
            }
        }
    }
    out
}

/// **生产同式**的累加值：`Σ seg_area`（不做分箱/前缀，等价于"全部段都参与"）。
///
/// `shade_pixel` 里是 `carries[row] + Σ 本 tile 段`；把所有段都给出来，
/// 前缀项自然包含在内，因此这里与生产是同一个量。
fn area_of(segs: &[[f32; 4]], cx: f32, cy: f32) -> f32 {
    segs.iter().map(|s| seg_area(s[0], s[1], s[2], s[3], cx, cy)).sum()
}

/// 现有 NonZero 覆盖率：`min(|area|, 1)`（与 `shade_pixel` 逐字一致）。
fn cov_nonzero(segs: &[[f32; 4]], cx: f32, cy: f32) -> f32 {
    area_of(segs, cx, cy).abs().min(1.0)
}

/// 计划 §2.1 设想的 even-odd：**取 Σ 的奇偶性**（实数按 `rem_euclid(2)` 落在 [1,2) 视为"奇"）。
fn cov_evenodd_by_parity(segs: &[[f32; 4]], cx: f32, cy: f32) -> f32 {
    if area_of(segs, cx, cy).rem_euclid(2.0) >= 1.0 {
        1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------- 形状

/// 轴对齐矩形，`ccw = false` 表示屏幕坐标下的顺时针（y 向下）。
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

// ================================================================ 1) 边界约定

/// 段是否属于 tile `[t_l, t_r)`（左闭右开）。
///
/// 规则（计划 §2.2）：`x_min < T_r && x_max > T_l`；桶前缀排除用 `x_max <= T_l`。
fn segment_belongs_to_tile(x_min: f32, x_max: f32, t_l: f32, t_r: f32) -> bool {
    x_min < t_r && x_max > t_l
}

/// 桶前缀是否应排除该段（即"整段都在 tile 左边"）。
fn excluded_by_prefix(x_max: f32, t_l: f32) -> bool {
    x_max <= t_l
}

#[test]
fn boundary_convention() {
    let t_l = 8.0;
    let t_r = 16.0;

    // 段正好在 tile 左边界结束 → 不归属
    assert!(!segment_belongs_to_tile(0.0, t_l, t_l, t_r));
    // 段越过左边界 → 归属
    assert!(segment_belongs_to_tile(0.0, t_l + 0.001, t_l, t_r));
    // 段正好在 tile 右边界开始 → 不归属
    assert!(!segment_belongs_to_tile(t_r, t_r + 8.0, t_l, t_r));
    // 段越过右边界 → 归属
    assert!(segment_belongs_to_tile(t_r - 0.001, t_r + 8.0, t_l, t_r));
    // 段跨整个 tile → 归属
    assert!(segment_belongs_to_tile(0.0, 32.0, t_l, t_r));
    // 退化段 → 不归属
    assert!(!segment_belongs_to_tile(t_l, t_l, t_l, t_r));

    // 前缀排除与归属必须**互补**：正好在左边界结束的段，既不归属也不参与前缀
    assert!(excluded_by_prefix(t_l, t_l));
    assert!(!segment_belongs_to_tile(0.0, t_l, t_l, t_r));
}

// ================================================================ 2) 累加语义

/// 先确认实验装置本身可信：**生产的累加式**能复现 NonZero 的参考覆盖率。
///
/// 形状**故意非整数对齐** —— 否则边缘落在像素边界上，参考里一个部分覆盖像素都没有，
/// 后面两条断言会"因为没牙而通过"（本 spike 第一次就是这么假通过的）。
#[test]
fn harness_reproduces_nonzero_reference() {
    let contours = vec![rect(2.3, 2.1, 6.4, 5.7, true)];
    let segs = segs_of(&contours);
    let (w, h) = (8u32, 8u32);
    let ref_nz = reference(&contours, w, h, 32, false);

    let mut worst = 0.0f32;
    let mut partial = 0;
    for py in 0..h {
        for px in 0..w {
            let got = cov_nonzero(&segs, px as f32 + 0.5, py as f32 + 0.5);
            let want = ref_nz[(py * w + px) as usize];
            if want > 0.001 && want < 0.999 {
                partial += 1;
            }
            worst = worst.max((got - want).abs());
        }
    }
    eprintln!("装置自检：NonZero 最大偏差 {worst:.4}；参考里 {partial} 个部分覆盖像素");
    assert!(partial >= 4, "参考必须真的包含部分覆盖像素，否则本 spike 没有判别力");
    assert!(worst < 0.05, "实验装置与参考不符（{worst:.4}），结论不可信");
}

/// **判据 1**：简单形状下 even-odd 与 NonZero 完全等价 ⇒ 奇偶规则必须给出同样的覆盖率。
///
/// 这是对"取 Σ 的奇偶性"最轻的一击：一个普通方块，边界像素的覆盖率应当**连续**。
#[test]
fn parity_rule_cannot_do_antialiasing() {
    let contours = vec![rect(2.3, 2.1, 6.4, 5.7, true)];
    let segs = segs_of(&contours);
    let ref_eo = reference(&contours, 8, 8, 32, true);

    let mut worst = 0.0f32;
    let mut worst_at = (0u32, 0u32);
    let mut non_binary = 0;
    for py in 0..8u32 {
        for px in 0..8u32 {
            let want = ref_eo[(py * 8 + px) as usize];
            if want > 0.001 && want < 0.999 {
                non_binary += 1;
            }
            let got = cov_evenodd_by_parity(&segs, px as f32 + 0.5, py as f32 + 0.5);
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = (px, py);
            }
        }
    }
    eprintln!(
        "奇偶规则 vs 参考：最大偏差 {worst:.4}（在 {worst_at:?}）；参考里有 {non_binary} 个边界像素是连续覆盖率"
    );
    assert!(non_binary >= 4, "参考必须真的包含部分覆盖像素");
    assert!(
        worst > 0.4,
        "预期奇偶规则会把连续覆盖率变成 0/1（偏差应很大），实际 {worst:.4} —— 前提需要重新检查"
    );
}

/// **判据 2（判别力）**：同向双环 —— NonZero 实心、EvenOdd 镂空。
///
/// 注意：计划里写的是"外圈顺时针 + **内圈逆时针**"，那种环在两条规则下**都是环**，
/// **没有判别力**；真正能区分的是**同向**双环。
#[test]
fn same_orientation_double_loop_is_the_discriminating_case() {
    let outer = rect(1.0, 1.0, 7.0, 7.0, true);
    let inner_same = rect(3.0, 3.0, 5.0, 5.0, true); // 与 outer 同向
    let inner_opp = rect(3.0, 3.0, 5.0, 5.0, false); // 反向

    let same = vec![outer.clone(), inner_same];
    let opp = vec![outer.clone(), inner_opp];

    let nz_same = reference(&same, 8, 8, 32, false);
    let eo_same = reference(&same, 8, 8, 32, true);
    let nz_opp = reference(&opp, 8, 8, 32, false);
    let eo_opp = reference(&opp, 8, 8, 32, true);

    // 内环中心像素 (4,4)
    let i = 4 * 8 + 4;
    eprintln!(
        "同向双环 中心：NonZero={:.3} EvenOdd={:.3}",
        nz_same[i], eo_same[i]
    );
    eprintln!(
        "反向双环 中心：NonZero={:.3} EvenOdd={:.3}",
        nz_opp[i], eo_opp[i]
    );

    // 同向：NonZero 实心（1.0）而 EvenOdd 镂空（0.0）⇒ 有判别力
    assert!(nz_same[i] > 0.99, "同向双环中心 NonZero 应为实心");
    assert!(eo_same[i] < 0.01, "同向双环中心 EvenOdd 应为镂空");
    // 反向：两条规则都镂空 ⇒ 无判别力
    assert!(nz_opp[i] < 0.01 && eo_opp[i] < 0.01, "反向双环两条规则都应是空的");
    eprintln!("结论：判别用例必须是**同向**双环；计划里的反向环无法区分两条规则");
}

/// **判据 3**：即使换成同向双环（整数区域），奇偶规则在**边界像素**上仍然错。
#[test]
fn parity_rule_fails_on_double_loop_edges() {
    let contours = vec![rect(1.4, 1.3, 6.7, 6.8, true), rect(3.2, 3.1, 4.9, 5.2, true)];
    let segs = segs_of(&contours);
    let ref_eo = reference(&contours, 8, 8, 32, true);

    let mut worst = 0.0f32;
    let mut worst_at = (0u32, 0u32);
    for py in 0..8u32 {
        for px in 0..8u32 {
            let want = ref_eo[(py * 8 + px) as usize];
            let got = cov_evenodd_by_parity(&segs, px as f32 + 0.5, py as f32 + 0.5);
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = (px, py);
            }
        }
    }
    eprintln!("同向双环：奇偶规则最大偏差 {worst:.4}（在 {worst_at:?}）");
    assert!(worst > 0.4, "预期边界像素偏差很大，实际 {worst:.4}");
}

/// **判据 4（出路）**：把 even-odd 的形状**预处理成等价的 NonZero 几何**
/// （即把内环反向），现有 `min(|area|,1)` 就能给出正确的 even-odd 覆盖率。
///
/// 这条是"修正方案"的原理证明：不需要新的覆盖率数学，只需要**改几何**。
#[test]
fn equivalent_nonzero_geometry_reproduces_even_odd() {
    let contours = vec![rect(1.4, 1.3, 6.7, 6.8, true), rect(3.2, 3.1, 4.9, 5.2, true)];
    let ref_eo = reference(&contours, 8, 8, 32, true);

    // 等价几何：外圈保持、内圈反向 ⇒ 内区绕数从 2 变 0
    let converted = vec![rect(1.4, 1.3, 6.7, 6.8, true), rect(3.2, 3.1, 4.9, 5.2, false)];
    let segs = segs_of(&converted);

    let mut worst = 0.0f32;
    let mut worst_at = (0u32, 0u32);
    let mut partial = 0;
    for py in 0..8u32 {
        for px in 0..8u32 {
            let want = ref_eo[(py * 8 + px) as usize];
            if want > 0.001 && want < 0.999 {
                partial += 1;
            }
            let got = cov_nonzero(&segs, px as f32 + 0.5, py as f32 + 0.5);
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = (px, py);
            }
        }
    }
    eprintln!(
        "等价 NonZero 几何 vs even-odd 参考：最大偏差 {worst:.4}（在 {worst_at:?}），部分覆盖像素 {partial} 个"
    );
    assert!(partial >= 4, "参考必须真的包含部分覆盖像素");
    assert!(
        worst < 0.05,
        "等价几何应当逐像素复现 even-odd（容差 0.05），实际 {worst:.4}"
    );
}
