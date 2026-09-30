//! 离屏渲染示例：构建演示场景 → GPU（或 CPU 兜底）渲染 → 输出 PNG。
//!
//! ```bash
//! cargo run --release -p amiki --example headless_png
//! ```

use amiki::color::*;
use amiki::math::Vec2;
use amiki::raster;
use amiki::scene::{BlendMode, Paint, PathBuilder};
use amiki::{Canvas, gpu};
use std::time::Instant;

const W: u32 = 1280;
const H: u32 = 720;

fn build_scene(t: f32) -> amiki::scene::Scene {
    let mut canvas = Canvas::new(W, H);

    // 背景
    canvas.rect(0.0, 0.0, W as f32, H as f32, Paint::Solid(INK));

    // 1) 旋转五角星（自相交，nonzero：中央五边形 W=2 → clamp 1）
    canvas.translate(Vec2::new(320.0, 300.0));
    canvas.rotate(t * 0.4);
    canvas.pentagram(Vec2::ZERO, 210.0, Paint::Solid(RED));

    // 2) 有洞的圆环（外圈 + 反绕内圈），缓慢平移
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(700.0, 220.0), 120.0, false);
    pb.circle(Vec2::new(700.0, 220.0), 58.0, true);
    canvas.reset();
    canvas.fill_path(&pb.build(), Paint::Solid(GREEN));

    // 3) 半透明大圆叠加（src-over alpha 合成）
    canvas.circle(
        Vec2::new(780.0, 300.0 + 24.0 * t.sin()),
        140.0,
        Paint::Solid(Color { r: BLUE.r, g: BLUE.g, b: BLUE.b, a: 0.55 }),
    );

    // 4) 裁剪展示（M2d）：固定圆窗内嵌旋转星
    let clip_c = Vec2::new(1010.0, 180.0);
    let mut pb = PathBuilder::new();
    pb.circle(clip_c, 105.0, false);
    canvas.push_clip(&pb.build());
    canvas.translate(clip_c);
    canvas.rotate(-t * 0.7);
    canvas.pentagram(
        Vec2::ZERO,
        150.0,
        Paint::Solid(Color { r: PURPLE.r, g: PURPLE.g, b: PURPLE.b, a: 0.95 }),
    );
    canvas.reset();
    canvas.pop_clip();
    let mut pb = PathBuilder::new();
    pb.circle(clip_c, 105.0, false);
    pb.circle(clip_c, 101.0, true);
    canvas.fill_path(&pb.build(), Paint::Solid(Color { r: 1.0, g: 1.0, b: 1.0, a: 0.12 }));

    // 5) 混合模式画廊（M2e）：彩色底条 + 8 种模式色块
    let gy = 500.0;
    let gh = 150.0;
    let cols = [RED, ORANGE, GREEN, BLUE, PURPLE];
    let bw = (W as f32 - 80.0) / 5.0;
    for (i, c) in cols.iter().enumerate() {
        let x0 = 40.0 + i as f32 * bw;
        canvas.rect(x0, gy, x0 + bw, gy + gh, Paint::Solid(*c));
    }
    let modes = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Luminosity,
    ];
    let tw = (W as f32 - 80.0) / modes.len() as f32;
    for (i, m) in modes.iter().enumerate() {
        let x0 = 40.0 + i as f32 * tw;
        canvas.set_blend(*m);
        canvas.rect(
            x0 + tw * 0.12,
            gy + gh * 0.15,
            x0 + tw * 0.88,
            gy + gh * 0.85,
            Paint::Solid(Color { r: 0.92, g: 0.86, b: 0.25, a: 0.85 }),
        );
    }
    canvas.set_blend(BlendMode::Normal);

    // 6) 沿正弦线的一串小圆：海量路径 + AA 质量展示
    for i in 0..50 {
        let x = 50.0 + i as f32 * 24.0;
        let y = 680.0 + (i as f32 * 0.45 + t).sin() * 18.0;
        let r = 5.0 + 4.0 * (i as f32 * 0.3).sin();
        canvas.circle(Vec2::new(x, y), r, Paint::Solid(ORANGE));
    }

    canvas.scene
}

fn main() {
    std::fs::create_dir_all("out").ok();

    // ── M4：增量渲染收益的实测（静态内容 + 少量动元素 是 UI 的典型形态）──
    // 同一场景按动画参数推进，逐帧用 edit_damage 推导脏区，统计实际需要重画的 tile 占比。
    {
        use amiki::scene::{edit_damage, Damage};
        let mut prev: Option<amiki::scene::Scene> = None;
        let mut total_dirty = 0u64;
        let mut total_all = 0u64;
        let mut frames = 0u32;
        let steps = 30;
        for i in 0..=steps {
            let scene = build_scene(i as f32 / steps as f32 * 2.0);
            let d = match &prev {
                Some(p) => edit_damage(p, &scene),
                None => Damage::new(scene.width, scene.height),
            };
            total_dirty += d.tile_count() as u64;
            total_all += d.total_tiles() as u64;
            frames += 1;
            prev = Some(scene);
        }
        let ratio = total_dirty as f64 / total_all.max(1) as f64;
        println!(
            "M4 增量渲染（全屏动画场景）：{frames} 帧平均重画 {:.1}% 的 tile（省下 {:.1}%）",
            ratio * 100.0,
            (1.0 - ratio) * 100.0
        );
    }

    // ── M4 真正的目标场景：静态界面 + 少量元素变动（UI / 图表 / 文档的典型形态）──
    {
        use amiki::scene::{edit_damage, Damage};
        let (w, h) = (1280u32, 720u32);
        let mut prev: Option<amiki::scene::Scene> = None;
        let mut total_dirty = 0u64;
        let mut total_all = 0u64;
        for i in 0..=30u32 {
            let mut c = amiki::Canvas::new(w, h);
            // 静态：背景 + 侧栏 + 两张面板 + 一行分隔线
            c.rect(0.0, 0.0, w as f32, h as f32, Paint::Solid(INK));
            c.rect(0.0, 0.0, 220.0, h as f32, Paint::Solid(amiki::scene::Color::rgb8(28, 34, 42)));
            c.rect(248.0, 64.0, 616.0, 400.0, Paint::Solid(amiki::scene::Color::rgb8(38, 46, 56)));
            c.rect(648.0, 64.0, 1240.0, 400.0, Paint::Solid(amiki::scene::Color::rgb8(38, 46, 56)));
            c.rect(248.0, 440.0, 1240.0, 444.0, Paint::Solid(amiki::scene::Color::rgb8(60, 70, 84)));
            // 侧栏 12 个静态条目
            for k in 0..12 {
                let y = 90.0 + k as f32 * 34.0;
                c.rect(24.0, y, 196.0, y + 22.0, Paint::Solid(amiki::scene::Color::rgb8(44, 52, 64)));
            }
            // 静态折线图骨架
            for k in 0..24 {
                let x = 268.0 + k as f32 * 14.0;
                c.rect(x, 340.0, x + 8.0, 392.0, Paint::Solid(amiki::scene::Color::rgb8(52, 62, 76)));
            }
            // 唯一在动的东西：一个进度点沿图表滑动（约 26×26 px）
            let px = 270.0 + (i as f32 / 30.0) * 320.0;
            c.circle(Vec2::new(px, 300.0), 13.0, Paint::Solid(ORANGE));

            let scene = c.scene;
            let d = match &prev {
                Some(p) => edit_damage(p, &scene),
                None => Damage::new(scene.width, scene.height),
            };
            total_dirty += d.tile_count() as u64;
            total_all += d.total_tiles() as u64;
            prev = Some(scene);
        }
        let ratio = total_dirty as f64 / total_all.max(1) as f64;
        println!(
            "M4 增量渲染（静态界面场景）：31 帧平均重画 {:.1}% 的 tile（省下 {:.1}%）← 这才是 damage region 的目标场景",
            ratio * 100.0,
            (1.0 - ratio) * 100.0
        );
    }

    let t0 = Instant::now();
    let scene = build_scene(0.7);
    let binned = raster::bin(&scene, 0.1);
    let bin_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // ── M5：文本走同一管线（字形轮廓即路径，引擎无文本专用代码）──
    // 演示用的极简字形由若干**有宽度的矩形笔画**组成（真实的 TrueType 轮廓
    // 是填充区域而非单线，这里必须同样是区域，否则面积为零、渲染不出像素）。
    {
        use amiki::scene::{expand_text_run, Color, GlyphOutline, Paint, PathEl, TextRun};
        /// 一段有宽度的笔画（矩形），坐标为归一化 [0,1]
        fn stroke(x0: f32, y0: f32, x1: f32, y1: f32, w: f32) -> Vec<PathEl> {
            let (dx, dy) = (x1 - x0, y1 - y0);
            let len = (dx * dx + dy * dy).sqrt().max(1e-6);
            // 垂直方向半宽
            let (px, py) = (-dy / len * w * 0.5, dx / len * w * 0.5);
            vec![
                PathEl::MoveTo(Vec2::new(x0 + px, y0 + py)),
                PathEl::LineTo(Vec2::new(x1 + px, y1 + py)),
                PathEl::LineTo(Vec2::new(x1 - px, y1 - py)),
                PathEl::LineTo(Vec2::new(x0 - px, y0 - py)),
                PathEl::Close,
            ]
        }
        // 演示字形：id 0 = "IⅤ" 竖+斜、id 1 = 竖、id 2 = 横
        let glyphs: Vec<GlyphOutline> = vec![
            GlyphOutline::new({
                let mut v = stroke(0.0, 0.0, 0.0, 1.0, 0.16);
                v.extend(stroke(0.0, 1.0, 1.0, 0.0, 0.16));
                v
            }),
            GlyphOutline::new(stroke(0.0, 0.0, 0.0, 1.0, 0.16)),
            GlyphOutline::new(stroke(0.0, 0.0, 1.0, 0.0, 0.16)),
        ];
        // 位置以 em 为单位（相对字号），故按设备坐标换算：设备位置 ÷ 字号 = em 位置
        let size = 150.0f32;
        let mut run = TextRun::new(size);
        for (i, id) in [1u32, 0, 1, 2].iter().enumerate() {
            let dev_x = 60.0 + i as f32 * 130.0; // 设备像素
            let dev_y = 80.0;
            run.push_glyph(*id, dev_x / size, dev_y / size);
        }
        let paths = expand_text_run(&run, |id| glyphs.get(id as usize).cloned());
        let mut text = amiki::scene::Scene::new(1280, 720);
        for (p, t) in paths {
            text.push(t, p, Paint::Solid(Color::rgba8(255, 255, 255, 235)));
        }
        let t0 = Instant::now();
        let tb = raster::bin(&text, 0.1);
        let timg = raster::render(&tb);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let px = timg.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
        image::save_buffer(
            "out/amiki_text_demo.png",
            &timg,
            1280,
            720,
            image::ColorType::Rgba8,
        )
        .expect("保存文本演示 PNG 失败");
        println!(
            "M5 文本演示：字形轮廓走解析覆盖率管线，{px} 个不透明像素，{ms:.1} ms → out/amiki_text_demo.png"
        );
    }

    let t1 = Instant::now();
    let img = match pollster::block_on(gpu::GpuRenderer::new()) {
        Ok(r) => match r.render_blocking(&scene, 0.1) {
            Ok(px) => {
                println!("后端: GPU (wgpu)");
                px
            }
            Err(e) => {
                println!("GPU 渲染失败（{e}），使用 CPU 引擎");
                raster::render(&binned)
            }
        },
        Err(e) => {
            println!("后端: CPU（GPU 不可用: {e}）");
            raster::render(&binned)
        }
    };
    let render_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let path = "out/amiki_headless.png";
    image::save_buffer(path, &img, W, H, image::ColorType::Rgba8).expect("保存 PNG 失败");
    println!("分箱 {bin_ms:.2} ms | 渲染 {render_ms:.2} ms | 输出 {path}");
}
