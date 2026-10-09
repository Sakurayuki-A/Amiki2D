//! 图像采样约定：**测出来的**，不是猜的（交接文档第 11 条）。
//!
//! 实测方法：4×1 的图（红/绿/蓝/黄，互不相同）铺进已知设备矩形，打印原始 RGB。
//! 实测结论：
//!
//! ```text
//!   设备 5.5 → texel 0.05     设备 9.5 → texel 0.45     设备 13.5 → texel 0.85
//!   ⇒ texel = device / texel_px − 0.5   ⇒ **texel 中心在 (i + 0.5) × texel_px**
//! ```
//!
//! 即标准 texel 中心约定（采样器一直是对的）。**推论**：texel_px 为偶数时，
//! 任何整数像素中心都落在两个 texel 中心之间 ⇒ 必然出现双线性混色；用**奇数**
//! texel_px（下面用 9）就能让像素中心正好落在 texel 中心上 ⇒ 取到**纯色**。
//!
//! 这条约定曾经害我误判：九宫格第一版把"设备 (1,1) 取到四色交汇"当成
//! "uv 约定不符"，其实 2 texel 铺 2 设备像素时每 texel 恰好 1px，设备 1.5
//! **正好落在 texel 边界**上 —— 采样器没问题，是我的采样点选得巧。
//!
//! 用法：`cargo test -p amiki --test image_uv_convention -- --nocapture`

use amiki::Canvas;
use amiki_math::Vec2;
use amiki_scene::{Image, ImagePlacement};

const COLORS: [[u8; 3]; 4] = [
    [255, 0, 0],   // texel 0
    [0, 255, 0],   // texel 1
    [0, 0, 255],   // texel 2
    [255, 255, 0], // texel 3
];

fn src4() -> Image {
    let mut d = Vec::new();
    for c in COLORS {
        d.extend_from_slice(&[c[0], c[1], c[2], 255]);
    }
    Image::from_rgba8(4, 1, d).expect("4x1")
}

fn rgb(px: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
    let i = ((y * w + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2]]
}

/// 核心断言：texel 中心落在 `(i + 0.5) × texel_px`。
///
/// 用**奇数** texel_px = 9（4 texel 铺 36px）⇒ texel 中心正好是像素 4/13/22/31 的
/// 中心 ⇒ 这四个像素必须是**纯色**（无任何混色），就是约定的硬证据。
#[test]
fn texel_centers_are_at_half_offsets() {
    let (w, h) = (36u32, 9u32); // 每 texel 9px（奇数）
    let mut c = Canvas::new(w, h);
    c.image(std::sync::Arc::new(src4()), 0.0, 0.0, w as f32, h as f32);
    let px = amiki::render_cpu(&c, 0.05);

    for (i, want) in COLORS.iter().enumerate() {
        let center_px = (i as u32) * 9 + 4; // (i + 0.5) × 9 = 4.5 ⇒ 像素 4
        let got = rgb(&px, w, center_px, 4);
        assert_eq!(
            &got, want,
            "texel {i} 的中心像素 {center_px} 必须是纯色 {want:?}，实测 {got:?}"
        );
    }
    // 相邻像素之间才是混色：像素 3 与 5 应当都不纯（在 texel 0 内但偏离中心）
    let a = rgb(&px, w, 3, 4);
    let b = rgb(&px, w, 5, 4);
    assert!(a != COLORS[0] || b != COLORS[0], "偏离中心处不应同时为纯色");
    eprintln!(
        "texel 中心约定成立：36px / 4 texel = 9px ⇒ 像素 4/13/22/31 为纯色 \
         {:?}/{:?}/{:?}/{:?}",
        rgb(&px, w, 4, 4),
        rgb(&px, w, 13, 4),
        rgb(&px, w, 22, 4),
        rgb(&px, w, 31, 4)
    );
}

/// 偶数 texel_px（40px / 4 = 10）时，**任何**整数像素中心都取不到纯色 —— 这是
/// 上一条约定的必然推论，也是我曾误判"约定不符"的地方。这里把它固定下来。
#[test]
fn even_texel_size_never_samples_a_pure_texel() {
    let (w, h) = (40u32, 10u32);
    let mut c = Canvas::new(w, h);
    c.image(std::sync::Arc::new(src4()), 0.0, 0.0, w as f32, h as f32);
    let px = amiki::render_cpu(&c, 0.05);
    // 精确断言：**内部**不得出现纯色（两端因 clamp 取边缘 texel，必然纯 —— 实测
    // 两端各约 5 个共 10 个，我第一版笼统断言"一个都没有"，被实测打回）。
    let pure_inside: Vec<u32> = (5..35)
        .filter(|&x| COLORS.iter().any(|c| *c == rgb(&px, w, x, 5)))
        .collect();
    assert!(
        pure_inside.is_empty(),
        "内部像素不得为纯色（11px 内应全在中心之间），实测纯色于 {pure_inside:?}"
    );
    let pure_edges: u32 = (0..w)
        .filter(|&x| COLORS.iter().any(|c| *c == rgb(&px, w, x, 5)))
        .count() as u32;
    assert!(pure_edges <= 12, "两端 clamp 区之外的纯色数异常：{pure_edges}");
    eprintln!(
        "40px / 4 texel = 10px（偶）⇒ 内部纯色 0 个；两端 clamp 区纯色 {pure_edges} 个"
    );
}

/// `uv_at` 是官方反解：设备点 → uv，和上面的采样约定必须自洽。
#[test]
fn uv_at_is_consistent_with_sampling() {
    let (w, h) = (36u32, 9u32);
    let place = ImagePlacement::rect(0.0, 0.0, w as f32, h as f32);
    // 像素 4 的中心 4.5 ⇒ 应为 texel 0 中心 ⇒ uv = 0.5/4 = 0.125
    let uv = place.uv_at(Vec2::new(4.5, 4.5)).expect("非退化");
    assert!(
        (uv.x - 0.125).abs() < 1e-4,
        "像素 4 的中心应对应 uv 0.125（texel 0 中心），实测 {}",
        uv.x
    );
    // 反过来：uv 0.125 ⇒ 设备 x = 4.5
    let back = place.origin.x + uv.x * place.u_axis.x;
    assert!((back - 4.5).abs() < 1e-4, "反推设备坐标应为 4.5，实测 {back}");
    eprintln!("uv_at 自洽：设备 4.5 ↔ uv {:.4}（texel 0 中心）", uv.x);
}
