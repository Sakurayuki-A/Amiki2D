//! `ColorMatrix`（0.10.4）：纯函数层的完整验证。
//!
//! 判据原则：**给互不相同的输入**，别去猜输出 ——
//! 这里用"已知颜色的已知结果"（Rec.709 权重、恒等、钳制）做断言。

use amiki_scene::{Color, ColorMatrix};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}
fn same(a: Color, b: Color, tol: f32) -> bool {
    close(a.r, b.r, tol) && close(a.g, b.g, tol) && close(a.b, b.b, tol) && close(a.a, b.a, tol)
}

/// 恒等：任意颜色不变（含 alpha）。
#[test]
fn identity_is_a_no_op() {
    let c = Color { r: 0.13, g: 0.57, b: 0.91, a: 0.42 };
    assert!(same(ColorMatrix::IDENTITY.apply(c), c, 1e-6));
    assert!(same(ColorMatrix::default().apply(c), c, 1e-6));
    eprintln!("恒等：颜色与 alpha 都不变");
}

/// 去色：纯红/纯绿/纯蓝分别变成同一个亮度灰；alpha 不变。
#[test]
fn greyscale_uses_rec709_weights() {
    let g = ColorMatrix::greyscale();
    let red = g.apply(Color { r: 1.0, g: 0.0, b: 0.0, a: 0.7 });
    let green = g.apply(Color { r: 0.0, g: 1.0, b: 0.0, a: 0.7 });
    let blue = g.apply(Color { r: 0.0, g: 0.0, b: 1.0, a: 0.7 });
    assert!(close(red.r, 0.2126, 1e-4), "红 → 0.2126，实际 {}", red.r);
    assert!(close(green.r, 0.7152, 1e-4), "绿 → 0.7152，实际 {}", green.r);
    assert!(close(blue.r, 0.0722, 1e-4), "蓝 → 0.0722，实际 {}", blue.r);
    // 灰：三通道相等
    assert!(close(red.r, red.g, 1e-6) && close(red.g, red.b, 1e-6));
    assert!(close(red.a, 0.7, 1e-6), "alpha 不受去色影响");
    eprintln!("去色：红 {:.4} / 绿 {:.4} / 蓝 {:.4}（Rec.709）", red.r, green.r, blue.r);
}

/// 饱和度：0 = 去色、1 = 恒等；两者可由 matrix 组合关系交叉验证。
#[test]
fn saturate_endpoints_match_greyscale_and_identity() {
    let c = Color { r: 0.2, g: 0.6, b: 0.9, a: 1.0 };
    assert!(same(ColorMatrix::saturate(0.0).apply(c), ColorMatrix::greyscale().apply(c), 1e-5));
    assert!(same(ColorMatrix::saturate(1.0).apply(c), c, 1e-5));
    // s > 1 会把颜色推离灰，但仍在 [0,1]
    let vivid = ColorMatrix::saturate(2.0).apply(c);
    assert!(vivid.r >= 0.0 && vivid.r <= 1.0 && vivid.g <= 1.0);
    eprintln!("饱和度：s=0 等价去色，s=1 等价恒等，s=2 越艳但不越界");
}

/// `then` 的组合语义 = "先 inner 再 self"（用可交换性做判别：缩放+偏移）。
#[test]
fn then_composes_in_the_right_order() {
    // 先提亮 0.2，再整体乘 0.5 ⇒ 结果 (c+0.2)*0.5，而不是 c*0.5+0.2
    let brighten = ColorMatrix::offset(0.2, 0.2, 0.2, 0.0);
    let dim = ColorMatrix::scale(0.5, 0.5, 0.5, 1.0);
    let composed = dim.then(&brighten);
    let c = Color { r: 0.4, g: 0.4, b: 0.4, a: 1.0 };
    let got = composed.apply(c);
    assert!(close(got.r, (0.4 + 0.2) * 0.5, 1e-5), "应为 0.30，实际 {}", got.r);
    // 反序结果不同（0.4*0.5+0.2 = 0.40）—— 证明组合不是对易的、顺序被正确实现
    let other = brighten.then(&dim).apply(c);
    assert!(close(other.r, 0.4 * 0.5 + 0.2, 1e-5), "反序应为 0.40，实际 {}", other.r);
    eprintln!("then：先提亮后变暗 = 0.30；反序 = 0.40（顺序语义正确）");
}

/// 钳制与非有限输入：**绝不**产出 NaN/∞。
#[test]
fn clamps_and_never_produces_nan() {
    let big = ColorMatrix::scale(10.0, 10.0, 10.0, 10.0);
    let c = big.apply(Color { r: 0.9, g: 0.9, b: 0.9, a: 0.9 });
    assert!(c.r == 1.0 && c.a == 1.0, "应钳到 1.0");
    let neg = ColorMatrix::offset(-5.0, -5.0, -5.0, -5.0).apply(Color { r: 0.5, g: 0.5, b: 0.5, a: 0.5 });
    assert!(neg.r == 0.0 && neg.a == 0.0, "应钳到 0.0");
    // NaN 系数：结果按钳制路径落 0，而不是 NaN 扩散
    let nan = ColorMatrix::scale(f32::NAN, 1.0, 1.0, 1.0).apply(Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 });
    assert!(!nan.r.is_nan() && !nan.g.is_nan(), "不得产出 NaN");
    eprintln!("钳制：越界落到 [0,1]；NaN 系数不扩散（r={}）", nan.r);
}

/// sepia 与 scale 的基本性状。
#[test]
fn sepia_and_scale_behave() {
    // ⚠️ 用**中灰**而不是纯白：经典 sepia 系数会饱和（白色 r、g 都被钳到 1.0，
    //    r>g 不成立）—— 用白色当输入时这条断言无法成立。
    let s = ColorMatrix::sepia().apply(Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 });
    assert!(
        s.r > s.g && s.g > s.b,
        "sepia 应偏暖（r>g>b），实际 {:.3}/{:.3}/{:.3}",
        s.r,
        s.g,
        s.b
    );
    let d = ColorMatrix::scale(0.5, 0.5, 0.5, 1.0).apply(Color { r: 0.8, g: 0.6, b: 0.4, a: 1.0 });
    assert!(same(d, Color { r: 0.4, g: 0.3, b: 0.2, a: 1.0 }, 1e-6));
    eprintln!("sepia 偏暖（{:.3}/{:.3}/{:.3}）；scale 通道乘算正确", s.r, s.g, s.b);
}
