//! B4：mipmap 效果对比图 —— `out/mipmap_demo.png`
//!
//! 同一张 512×512 的 8px 黑白棋盘格，按 5 档缩小倍率各画一遍：
//! **上排 = 无 mip 链**（只有双线性，0.5.0 的行为）——缩小后是摩尔纹 / 椒盐噪声；
//! **下排 = 带 mip 链**（BOX 盒式滤波 + 三线性采样）——缩小后是平滑的平均灰。
//! 渲染走门面 Canvas（GPU 可用则 GPU，否则 CPU）。
//!
//! 运行：`cargo run --release -p amiki --example mipmap_demo`

use amiki::scene::{Color, Image, Paint};
use amiki::Canvas;
use amiki_math::Vec2;

/// 1/8x 那一格的画布位置（与下方布局严格对应，改布局要同步改这里）
const CELL_MOIRE: (usize, usize, usize) = (512, 70, 27); // 1/19x 格 (x, y, 边长)，LOD 4.2 已过奈奎斯特

fn main() {
    // ── 源图：512×512，8px 黑白棋盘格 ──
    // 同一份像素构造两个 Image：一个带 mip 链，一个不带。
    // 带链与否只改采样层级，不改像素数据 —— 上下两排画的是"同一张图"。
    let (w, h) = (512u32, 512u32);
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 255u8 } else { 0u8 };
            px.extend([v, v, v, 255]);
        }
    }
    let with_mip =
        std::sync::Arc::new(Image::from_rgba8_with_mipmap(w, h, px.clone()).expect("图像数据合法"));
    let without_mip = std::sync::Arc::new(Image::from_rgba8(w, h, px).expect("图像数据合法"));

    // ── 画布 ──
    const W: u32 = 780;
    const H: u32 = 660;
    let mut canvas = Canvas::new(W, H);
    canvas.rect(0.0, 0.0, W as f32, H as f32, Paint::Solid(Color::rgb8(24, 26, 32)));

    // 5 档缩小倍率：**刻意避开 2 的整数次幂** —— 整倍率下双线性的采样点
    // 恰好落在格点上，走样会被"伪装"成均匀色块；非整倍率才会拍频出
    // 真正看得见的摩尔纹（0.22x 起）。
    let sizes = [224.0f32, 111.0, 53.0, 27.0, 13.0];
    let labels = ["0.44x", "0.22x", "0.10x", "1/19x", "1/39x"];

    // 随仓库分发的字体（中文标签用 Noto，倍率数字用 Inter；缺了就静默跳过标签）
    let noto = std::fs::read("assets/fonts/NotoSansSC-Regular.ttf")
        .ok()
        .and_then(|b| amiki_text::Font::from_bytes(b).ok());
    let inter = std::fs::read("assets/fonts/Inter-Regular.ttf")
        .ok()
        .and_then(|b| amiki_text::Font::from_bytes(b).ok());

    let draw_label = |canvas: &mut Canvas, font: &amiki_text::Font, text: &str, x: f32, y: f32, size: f32, color: Color| {
        if let Ok(run) = amiki_text::shape(font, text, size) {
            canvas.text(&run, Vec2::new(x, y), Paint::Solid(color), |id| font.outline(id).ok());
        }
    };

    // ── 两排对比 ──
    let rows = [(&without_mip, 70.0f32), (&with_mip, 390.0f32)];
    for (img, y0) in rows {
        let mut x = 40.0;
        for (size, label) in sizes.iter().zip(labels.iter()) {
            canvas.image(img.clone(), x, y0, x + size, y0 + size);
            if let Some(f) = inter.as_ref().or(noto.as_ref()) {
                draw_label(&mut canvas, f, label, x, y0 + size + 18.0, 14.0, Color::rgb8(160, 164, 176));
            }
            x += size + 28.0;
        }
    }
    // 分隔线（画在两排之间，避免"上排的下边框"与"下排的上边框"混淆视线）
    canvas.rect(32.0, 344.0, W as f32 - 32.0, 345.5, Paint::Solid(Color::rgb8(60, 64, 76)));

    // 排标题与脚注
    if let Some(f) = noto.as_ref().or(inter.as_ref()) {
        draw_label(&mut canvas, f, "无 mip 链（双线性，0.5.0 行为）—— 摩尔纹 / 椒盐噪声。源图：512x512 的 8px 黑白棋盘格", 40.0, 42.0, 18.0, Color::rgb8(235, 235, 240));
        draw_label(&mut canvas, f, "mip 链（BOX 滤波 + 三线性，0.6.0）—— 缩小后平滑平均灰", 40.0, 362.0, 18.0, Color::rgb8(120, 220, 160));
    }

    // ── 渲染 + 写 PNG ──
    let t0 = std::time::Instant::now();
    let rgba = match amiki::render_to_rgba(&canvas, 0.05) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("GPU 渲染不可用（{e}），走 CPU 引擎");
            amiki::render_cpu(&canvas, 0.05)
        }
    };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    std::fs::create_dir_all("out").expect("创建 out 目录");
    image::save_buffer("out/mipmap_demo.png", &rgba, W, H, image::ColorType::Rgba8)
        .expect("写出 PNG");

    // ── 量化证据：1/19x 那格（LOD 4.2，格子已"死透"）的中心区域 ──
    // 无 mip：0/255 椒盐噪声（跨度 255）；带 mip：均匀 188 ——
    // 恰好是 sRGB(线性 0.5)，即线性空间平均的理论正确值。
    println!("mipmap 对比图已生成：out/mipmap_demo.png（{ms:.0} ms）");
    println!("量化证据（1/19x 缩小的棋盘中心区域，R 通道范围）：");
    let (x0, _, s) = CELL_MOIRE;
    for (name, y0) in [("无 mip", 70usize), ("带 mip", 390usize)] {
        let (mut mn, mut mx) = (u8::MAX, 0u8);
        for yy in (y0 + s / 4..y0 + 3 * s / 4).step_by(2) {
            for xx in (x0 + s / 4..x0 + 3 * s / 4).step_by(2) {
                let v = rgba[(yy * W as usize + xx) * 4];
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        println!("  {name}: {mn}..{mx}（跨度 {}）", mx - mn);
    }
}
