//! 九宫格（0.10.3）：判据是**逐区域的颜色**。
//!
//! 教训（本文件第一版）：我曾用"面积和 / 轴长"这类从结构推的量当判据 —— 算错了，
//! 还掩盖了"角被缩放"这个真 bug；接着又用"某像素等于纯色"当判据 —— 采样点落在
//! 半 texel 混色带上，一样不可靠。现在改成：**源图 9 个区域用 9 种互不相同的颜色**，
//! 于是"哪个区域画到了哪里、角的尺寸是多少"都能从颜色上无歧义地判定。

use amiki::Canvas;
use amiki_scene::{nine_patch, Color, Image, Insets, Paint};

/// 9 个区域的颜色：**互不相同**，且与背景（白）都不接近。
const C: [[u8; 3]; 9] = [
    [200, 30, 30],   // 0 左上角
    [230, 140, 20],  // 1 上边
    [250, 230, 20],  // 2 右上角
    [30, 150, 60],   // 3 左边
    [40, 90, 210],   // 4 中间
    [140, 40, 190],  // 5 右边
    [20, 190, 190],  // 6 左下角
    [120, 120, 120], // 7 下边
    [20, 20, 20],    // 8 右下角
];

/// 8×8 源图：外圈 2px 分四角与四边，中间 4×4。
fn src9() -> Image {
    let mut d = vec![0u8; 8 * 8 * 4];
    for y in 0..8u32 {
        for x in 0..8u32 {
            let col = if x < 2 { 0 } else if x < 6 { 1 } else { 2 };
            let row = if y < 2 { 0 } else if y < 6 { 1 } else { 2 };
            let c = C[row * 3 + col];
            let i = ((y * 8 + x) * 4) as usize;
            d[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    Image::from_rgba8(8, 8, d).expect("8x8")
}

fn render(dst: (f32, f32, f32, f32), insets: Insets, size: (u32, u32)) -> (u32, Vec<u8>) {
    let mut c = Canvas::new(size.0, size.1);
    let mut pb = amiki_scene::PathBuilder::new();
    pb.rect(0.0, 0.0, size.0 as f32, size.1 as f32);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(255, 255, 255)));
    c.image_nine(std::sync::Arc::new(src9()), insets, dst);
    (size.0, amiki::render_cpu(&c, 0.05))
}

fn at(img: &(u32, Vec<u8>), x: u32, y: u32) -> [u8; 3] {
    let i = ((y * img.0 + x) * 4) as usize;
    [img.1[i], img.1[i + 1], img.1[i + 2]]
}

/// 9 个区域各自画到了正确的位置；**角的尺寸 = 源尺寸（2px），没有被拉伸**。
#[test]
fn nine_regions_land_where_they_should() {
    // dst 100×100（源只有 8×8，放大 12.5×）
    let img = render((0.0, 0.0, 100.0, 100.0), Insets::all(2), (100, 100));
    // 四角：各 2×2 设备像素，采样中心
    assert_eq!(at(&img, 1, 1), C[0], "左上角 = 区域 0");
    assert_eq!(at(&img, 98, 1), C[2], "右上角 = 区域 2");
    assert_eq!(at(&img, 1, 98), C[6], "左下角 = 区域 6");
    assert_eq!(at(&img, 98, 98), C[8], "右下角 = 区域 8");
    // 四边：中心处采样
    assert_eq!(at(&img, 50, 1), C[1], "上边 = 区域 1");
    assert_eq!(at(&img, 50, 98), C[7], "下边 = 区域 7");
    assert_eq!(at(&img, 1, 50), C[3], "左边 = 区域 3");
    assert_eq!(at(&img, 98, 50), C[5], "右边 = 区域 5");
    // 中间
    assert_eq!(at(&img, 50, 50), C[4], "中间 = 区域 4");
    // **角尺寸探针**：第 3 个像素就已经不属角（角只有 2px）
    assert_eq!(at(&img, 3, 3), C[4], "角只占 2px ⇒ (3,3) 已是中间色");
    assert_eq!(at(&img, 1, 3), C[3], "(1,3) 在左边带上");
    assert_eq!(at(&img, 3, 1), C[1], "(3,1) 在上边带上");
    eprintln!("9 区域全部就位；角保持 2px（dst 放大 12.5× 仍不拉伸）");
}

/// 同一张贴图放到不同尺寸：角尺寸**不随尺寸变化**。
#[test]
fn corner_size_is_independent_of_dst() {
    for (w, h) in [(100u32, 100u32), (40, 30), (300, 80)] {
        let img = render((0.0, 0.0, w as f32, h as f32), Insets::all(2), (w, h));
        assert_eq!(at(&img, 1, 1), C[0], "{w}×{h}: 左上角");
        assert_eq!(at(&img, 3, 3), C[4], "{w}×{h}: (3,3) 仍是中间色");
        assert_eq!(at(&img, 1, 3), C[3], "{w}×{h}: 左边带");
        assert_eq!(at(&img, w - 2, h - 2), C[8], "{w}×{h}: 右下角");
    }
    eprintln!("三种尺寸（100×100 / 40×30 / 300×80）角都是 2px、四边带位置一致");
}

/// 边距为 0 ⇒ 只剩 1 块，整幅拉伸（四角颜色会被拉大，这是预期行为）。
#[test]
fn zero_insets_is_a_single_stretch() {
    let places = nine_patch(8, 8, Insets::all(0), (0.0, 0.0, 80.0, 80.0));
    assert_eq!(places.len(), 1, "边距 0 ⇒ 只剩中间一块");
    let img = render((0.0, 0.0, 80.0, 80.0), Insets::all(0), (80, 80));
    assert_eq!(at(&img, 40, 40), C[4], "中心是中间色");
    assert_eq!(at(&img, 2, 2), C[0], "左上仍是角色（被拉大了，符合预期）");
    eprintln!("边距 0：1 块整幅拉伸，中心/左上颜色正确");
}

/// 退化与越界：显式、安全。
#[test]
fn degenerate_cases_are_safe() {
    assert!(nine_patch(0, 8, Insets::all(2), (0.0, 0.0, 10.0, 10.0)).is_empty());
    assert!(nine_patch(8, 8, Insets::all(2), (5.0, 5.0, 5.0, 9.0)).is_empty());
    // dst 比两角之和还小 ⇒ 等比压缩，仍有限、不 panic
    let p = nine_patch(8, 8, Insets::all(2), (0.0, 0.0, 3.0, 3.0));
    assert!(!p.is_empty());
    for q in &p {
        assert!(q.dst.0.is_finite() && q.dst.2.is_finite() && q.dst.2 > q.dst.0);
    }
    let _ = render((0.0, 0.0, 3.0, 3.0), Insets::all(2), (8, 8));
    // 两侧边距之和 = 源尺寸 ⇒ 中间行/列退化 ⇒ 只剩四角
    let p = nine_patch(8, 8, Insets::all(4), (0.0, 0.0, 40.0, 40.0));
    assert_eq!(p.len(), 4, "只剩四角，实际 {}", p.len());
    eprintln!("退化输入：源 0 / dst 0 返回空；dst 小于两角之和时压缩；边距占满时只剩四角");
}

/// 演示图：同一张贴图拉到两种尺寸，看角落不变形。
#[test]
fn nine_patch_demo_image() {
    let img = std::sync::Arc::new(src9());
    let mut c = Canvas::new(320, 140);
    let mut pb = amiki_scene::PathBuilder::new();
    pb.rect(0.0, 0.0, 320.0, 140.0);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(250, 250, 252)));
    c.image_nine(img.clone(), Insets::all(2), (16.0, 16.0, 200.0, 70.0)); // 放大
    c.image_nine(img.clone(), Insets::all(2), (216.0, 16.0, 300.0, 124.0)); // 竖着拉长
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/nine_patch_demo.png");
    let px = amiki::render_cpu(&c, 0.05);
    if let Some(im) = image::RgbaImage::from_raw(320, 140, px) {
        let _ = std::fs::create_dir_all(out.parent().unwrap());
        im.save(&out).expect("写演示图");
        eprintln!("演示图 -> {}", out.display());
    }
}
