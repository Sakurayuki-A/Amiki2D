//! 效果图：把 0.10.x 落地的东西一次用上（只画图 + 存 PNG，无断言）。
//!
//! 用到：九宫格面板（0.10.3）· 虚线焦点框（0.10.2）· 调用方持有的字形缓存
//! （0.10.1）· 颜色矩阵滤镜灰化"禁用态"（0.10.5）· 普通位图绘制（M6）。
//!
//! 用法：`cargo test -p amiki --test showcase -- --nocapture`

use amiki::Canvas;
use amiki_math::Vec2;
use amiki_scene::{Color, ColorMatrix, Image, Insets, LineCap, LineJoin, Paint, PathBuilder, StrokeStyle};

/// 8×8 "面板贴图"：外圈 2px 边框 + 内部浅色（九宫格会保持四角不缩放）
fn panel_tex() -> Image {
    let mut d = vec![0u8; 8 * 8 * 4];
    for y in 0..8u32 {
        for x in 0..8u32 {
            let i = ((y * 8 + x) * 4) as usize;
            let edge = x < 2 || y < 2 || x >= 6 || y >= 6;
            let c = if edge { [46u8, 84, 150] } else { [226u8, 236, 250] };
            d[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    Image::from_rgba8(8, 8, d).expect("8x8")
}

/// 16×16 "图标"：一圈点阵（用于验证图像绘制）
fn icon_tex() -> Image {
    let mut d = vec![0u8; 16 * 16 * 4];
    for y in 0..16u32 {
        for x in 0..16u32 {
            let i = ((y * 16 + x) * 4) as usize;
            let dx = x as f32 - 7.5;
            let dy = y as f32 - 7.5;
            let r = (dx * dx + dy * dy).sqrt();
            let c: [u8; 4] = if (5.0..=6.5).contains(&r) {
                [216, 92, 40, 255]
            } else {
                [0, 0, 0, 0]
            };
            d[i..i + 4].copy_from_slice(&c);
        }
    }
    Image::from_rgba8(16, 16, d).expect("16x16")
}

fn stroke() -> StrokeStyle {
    StrokeStyle { width: 2.0, cap: LineCap::Butt, join: LineJoin::Bevel }
}

#[test]
fn render_showcase() {
    let (w, h) = (420u32, 240u32);
    let mut c = Canvas::new(w, h);

    // 背景
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(246, 247, 250)));

    // ① 九宫格面板（左）：四角保持源尺寸 ⇒ 边框粗细不随尺寸变化
    c.image_nine(std::sync::Arc::new(panel_tex()), Insets::all(2), (16.0, 16.0, 236.0, 130.0));
    // 同一贴图放大成竖长条（右）：角仍然不变形
    c.image_nine(std::sync::Arc::new(panel_tex()), Insets::all(2), (256.0, 16.0, 404.0, 224.0));

    // ② 文字（走调用方持有的字形缓存）
    let font_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/fonts/Inter-Regular.ttf");
    let mut cache = amiki_scene::TextCache::new();
    if let Ok(font) = amiki_text::Font::from_file(font_path.to_string_lossy().as_ref()) {
        if let Ok(run) = amiki_text::shape(&font, "amiki 0.10 · nine-patch + dash + cache", 15.0) {
            c.text_cached(
                &mut cache,
                &run,
                Vec2::new(30.0, 58.0),
                Paint::Solid(Color::rgb8(28, 48, 92)),
                |id| font.outline(id).ok(),
            );
        }
        if let Ok(run) = amiki_text::shape(&font, "cached text", 13.0) {
            c.text_cached(
                &mut cache,
                &run,
                Vec2::new(280.0, 52.0),
                Paint::Solid(Color::rgb8(28, 48, 92)),
                |id| font.outline(id).ok(),
            );
        }
    }

    // ③ 图标（普通位图绘制）
    c.image(std::sync::Arc::new(icon_tex()), 30.0, 74.0, 54.0, 98.0);

    // ④ 虚线焦点框（0.10.2）
    let mut pb = PathBuilder::new();
    pb.rect(24.0, 68.0, 228.0, 118.0);
    c.stroke_path_dashed(&pb.build(), stroke(), Color::rgb8(216, 92, 40), &[10.0, 6.0]);

    // ⑤ 颜色滤镜：右下面板做"禁用态"灰化（0.10.5）
    let mut pb = PathBuilder::new();
    pb.rect(272.0, 84.0, 388.0, 150.0);
    let region = pb.build();
    c.set_color_filter(Some(ColorMatrix::greyscale()));
    c.fill_path(&region, Paint::Solid(Color::rgb8(226, 120, 60)));
    c.set_color_filter(None);
    // 同一颜色、未灰化的对照条
    let mut pb = PathBuilder::new();
    pb.rect(272.0, 162.0, 388.0, 196.0);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(226, 120, 60)));

    let px = amiki::render_cpu(&c, 0.05);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../out/showcase.png");
    if let Some(im) = image::RgbaImage::from_raw(w, h, px) {
        let _ = std::fs::create_dir_all(out.parent().unwrap());
        im.save(&out).expect("写效果图");
        eprintln!("效果图 -> {}", out.display());
        eprintln!("字形缓存：hits={} misses={}", cache.hits(), cache.misses());
    }
}
