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

/// 加载根目录下的 `img.jpg` 作为演示用位图（存在才加载）。
///
/// 解码放在示例里而不是引擎里：引擎只认"线性 RGBA8"这一种内存布局，
/// 因此不需要引入任何图像解码库。
fn load_demo_image() -> Option<std::sync::Arc<amiki::scene::Image>> {
    let p = std::path::Path::new("img.jpg");
    if !p.exists() {
        return None;
    }
    // 解码失败**必须区分于文件不存在**：早前这里用 `.ok()?` 把两者混为一谈，
    // 结果"JPEG 解码未启用"被误报成"未找到 img.jpg"，白白浪费一轮排查。
    let dynimg = match image::open(p) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("加载 img.jpg 失败（文件存在）：{e}");
            return None;
        }
    };
    let rgba = dynimg.to_rgba8();
    let (w, h) = rgba.dimensions();
    amiki::scene::Image::from_rgba8(w, h, rgba.into_raw()).map(std::sync::Arc::new)
}

/// 加载随仓库分发的 Inter 字体（`assets/fonts/Inter-Regular.ttf`，SIL OFL 1.1）。
///
/// 找不到时返回 `None` 并提示，而不是让示例失败 —— 示例应当能在没有字体的
/// 环境里照常演示图形能力。
fn load_inter() -> Option<amiki_text::Font> {
    // 示例运行时的工作目录是仓库根（cargo 约定）
    let p = std::path::Path::new("assets/fonts/Inter-Regular.ttf");
    if !p.exists() {
        return None;
    }
    match amiki_text::Font::from_file(p) {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("加载 Inter 失败：{e}");
            None
        }
    }
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

    // ── M5：真实字体文本（Inter，SIL OFL 1.1）──
    // 字形轮廓 → 既有解析覆盖率管线。引擎无文本专用代码；
    // 抗锯齿来自真实面积积分，因此放大不糊（不依赖字形位图）。
    if let Some(font) = load_inter() {
        use amiki::scene::{expand_text_run, Color, Gradient, GradientStop, Paint, Scene};
        let size = 110.0f32;
        let text = "Amiki2D";
        match amiki_text::shape(&font, text, size) {
            Ok(run) => {
                let paths = expand_text_run(&run, |id| font.outline(id).ok());
                let mut text_scene = Scene::new(1280, 240);
                // 浅色底：让文字可读（否则白字在透明背景上几乎看不见）
                let mut bg = PathBuilder::new();
                bg.rect(0.0, 0.0, 1280.0, 240.0);
                text_scene.push(
                    amiki::math::Affine::IDENTITY,
                    bg.build(),
                    Paint::Solid(Color::rgb8(248, 246, 242)),
                );
                let shift = amiki::math::Affine::translate(Vec2::new(60.0, 160.0));
                // M6：文字用**渐变**着色 —— 同时验证 M5 与 M6 正交可组合
                let grad = std::sync::Arc::new(
                    Gradient::linear(
                        Vec2::new(0.0, 0.0),
                        Vec2::new(760.0, 60.0),
                        vec![
                            GradientStop::new(0.0, Color::rgb8(28, 34, 46)),
                            GradientStop::new(0.45, Color::rgb8(196, 78, 40)),
                            GradientStop::new(1.0, Color::rgb8(228, 158, 40)),
                        ],
                    )
                    .expect("渐变色标有效"),
                );
                for (p, t) in paths.iter() {
                    // 每个字形一条独立 draw：可被分箱与 damage region 独立处理
                    text_scene.push(shift.then(*t), p.clone(), Paint::Gradient(grad.clone()));
                }
                let tb = raster::bin(&text_scene, 0.1);
                let timg = raster::render(&tb);
                let px = timg.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count();
                image::save_buffer(
                    "out/amiki_text_demo.png",
                    &timg,
                    1280,
                    240,
                    image::ColorType::Rgba8,
                )
                .expect("保存文本演示 PNG 失败");
                println!(
                    "M5 文本：真实字体 Inter（{text:?}，{size}px）→ {px} 个不透明像素，\
                     字形数 {} → out/amiki_text_demo.png",
                    run.glyphs.len()
                );
            }
            Err(e) => println!("M5 文本：shaping 失败（{e}）"),
        }
    } else {
        println!("M5 文本：未找到 assets/fonts/Inter-Regular.ttf，跳过真字演示");
    }

    // ── M6：综合演示 —— 位图 + 渐变 + 文字 同场 ──
    // 用示例自带的 img.jpg（存在才画），展示图像与图形/文字混合。
    if let Some(img) = load_demo_image() {
        use amiki::scene::{Color, Gradient, GradientStop, Paint};
        let (cw, ch) = (1280u32, 720u32);
        let mut c = amiki::Canvas::new(cw, ch);
        c.rect(0.0, 0.0, cw as f32, ch as f32, Paint::Solid(Color::rgb8(246, 243, 238)));

        // 位图：等比缩放 + 裁剪，并套一层**投影**（M6 层树）
        let (ix, iy, iw) = (64.0f32, 96.0f32, 420.0f32);
        let ih = iw * (img.height as f32) / (img.width as f32);
        c.push_layer_filtered(
            amiki::scene::LayerBounds::new(
                (ix - 10.0) as i32,
                (iy - 10.0) as i32,
                (iw + 20.0) as u32,
                (ih + 20.0) as u32,
            ),
            amiki::scene::Filter::DropShadow {
                dx: 10.0,
                dy: 12.0,
                sigma: 9.0,
                color: Color::rgba8(30, 26, 22, 190),
            },
            1.0,
            amiki::scene::BlendMode::Normal,
            |lc| {
                let mut clip = PathBuilder::new();
                clip.rect(ix, iy, ix + iw, iy + ih);
                lc.push_clip(&clip.build());
                lc.image(img.clone(), ix, iy, ix + iw, iy + ih);
                lc.pop_clip();
            },
        );

        // 渐变横幅（线性 + 径向各一条，展示两种几何）
        let linear = std::sync::Arc::new(
            Gradient::linear(
                Vec2::new(560.0, 0.0),
                Vec2::new(1220.0, 0.0),
                vec![
                    GradientStop::new(0.0, Color::rgb8(28, 34, 46)),
                    GradientStop::new(0.5, Color::rgb8(186, 72, 44)),
                    GradientStop::new(1.0, Color::rgb8(232, 166, 46)),
                ],
            )
            .expect("渐变色标有效"),
        );
        c.rect(560.0, 120.0, 1220.0, 208.0, Paint::Gradient(linear.clone()));
        let radial = std::sync::Arc::new(
            Gradient::radial(
                Vec2::new(760.0, 420.0),
                190.0,
                vec![
                    GradientStop::new(0.0, Color::rgba8(90, 150, 210, 255)),
                    GradientStop::new(1.0, Color::rgba8(18, 30, 52, 255)),
                ],
            )
            .expect("渐变色标有效"),
        );
        c.rect(600.0, 260.0, 920.0, 580.0, Paint::Gradient(radial));

        // 模糊（M6 层树）：把一组几何整体模糊，展示"逐图元无法实现"的效果
        c.push_layer_filtered(
            amiki::scene::LayerBounds::new(950, 260, 250, 250),
            amiki::scene::Filter::Blur { sigma: 14.0 },
            1.0,
            amiki::scene::BlendMode::Normal,
            |lc| {
                lc.circle(Vec2::new(1030.0, 350.0), 52.0, Paint::Solid(Color::rgb8(52, 120, 196)));
                lc.circle(Vec2::new(1130.0, 400.0), 62.0, Paint::Solid(Color::rgb8(196, 92, 52)));
                lc.circle(Vec2::new(1075.0, 460.0), 44.0, Paint::Solid(Color::rgb8(96, 170, 96)));
            },
        );

        // 文字：用渐变着色，并叠在位图与渐变之上
        if let Some(font) = load_inter() {
            if let Ok(run) = amiki_text::shape(&font, "Amiki2D", 96.0) {
                c.text(
                    &run,
                    Vec2::new(600.0, 560.0),
                    Paint::Gradient(linear),
                    |id| font.outline(id).ok(),
                );
            }
        }

        let scene = c.scene;
        let t0 = Instant::now();
        // 后端要**如实报告实际用了哪个**，不能写死。
        // 这里曾经硬编码"滤镜走 CPU 参考实现"，而在 GPU 层合成实现之后，
        // 那句话就变成了假信息（会让人以为滤镜没走 GPU）。
        let backend;
        let mut img_rgba = match pollster::block_on(gpu::GpuRenderer::new()) {
            Ok(r) => match r.render_blocking(&scene, 0.1) {
                Ok(px) => {
                    backend = "GPU (wgpu)";
                    px
                }
                Err(e) => {
                    println!("综合演示：GPU 渲染失败（{e}），改用 CPU");
                    backend = "CPU（GPU 拒绝）";
                    raster::render(&raster::bin(&scene, 0.1))
                }
            },
            Err(e) => {
                println!("综合演示：GPU 不可用（{e}），改用 CPU");
                backend = "CPU（无 GPU 适配器）";
                raster::render(&raster::bin(&scene, 0.1))
            }
        };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        image::save_buffer("out/amiki_showcase.png", &img_rgba, cw, ch, image::ColorType::Rgba8)
            .expect("保存综合演示 PNG 失败");
        println!(
            "M6 综合演示：位图（{}×{}，投影）+ 线性/径向渐变 + 渐变文字 + 模糊层\n\
             \x20           后端: {backend} → out/amiki_showcase.png（{ms:.0} ms 含设备创建）",
            img.width, img.height
        );
    } else {
        println!("M6：跳过图像演示（未找到 img.jpg，或文件存在但解码失败——原因见上方提示）");
    }

    // ── M5.2：排版（断行 + 行高 + 对齐）──
    // 排版是**独立的一层**：渲染器只接收"已定位的字形"，不需要排版概念。
    if let Some(font) = load_inter() {
        let body = "Amiki2D renders text through the same analytic-coverage pipeline used for shapes, \
so glyph outlines stay crisp at any zoom. This paragraph is laid out by amiki-layout: \
greedy line breaking at word boundaries, line height derived from the font's own metrics, \
and left alignment within the given width.";
        let opts = amiki_layout::LayoutOptions {
            align: amiki_layout::Align::Left,
            line_spacing: 1.35,
            ..amiki_layout::LayoutOptions::new(26.0, 640.0)
        };
        match amiki_layout::layout(&font, body, &opts) {
            Ok(res) => {
                let mut c = amiki::Canvas::new(1280, 720);
                c.rect(0.0, 0.0, 1280.0, 720.0, Paint::Solid(Color::rgb8(250, 248, 244)));
                // 按排版结果逐行绘制：引擎只接收"已定位的字形"，不需要排版概念
                let ox = 64.0;
                let oy = 60.0;
                for line in &res.lines {
                    let origin = Vec2::new(ox + line.baseline.x, oy + line.baseline.y);
                    c.text(
                        &line.run,
                        origin,
                        Paint::Solid(Color::rgb8(28, 32, 40)),
                        |id| font.outline(id).ok(),
                    );
                }
                let out = raster::render(&raster::bin(&c.scene, 0.1));
                image::save_buffer(
                    "out/amiki_paragraph.png",
                    &out,
                    1280,
                    720,
                    image::ColorType::Rgba8,
                )
                .expect("保存排版 PNG 失败");
                println!(
                    "M5.2 排版：{} 字符 → {} 行（行距 1.35×，左对齐，宽上限 {:.0}px）→ out/amiki_paragraph.png",
                    body.chars().count(),
                    res.lines.len(),
                    opts.max_width
                );
                println!(
                    "           排版块尺寸 {:.1}×{:.1}px，overflow={}",
                    res.size.x, res.size.y, res.overflow
                );
            }
            Err(e) => println!("M5.2 排版失败：{e}"),
        }
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
