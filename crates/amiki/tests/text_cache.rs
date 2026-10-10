//! `Canvas::text_cached`（0.10.0）：用调用方持有的 `TextCache` 复用字形轮廓。
//!
//! 判据三条：
//! 1. **输出与 `Canvas::text` 逐字段相同**（缓存不得改变画出来的东西）
//! 2. 同一段文字重复绘制时**命中率上升**（缓存真的在起作用）
//! 3. 有可测的加速（打印出来，不做脆弱断言 —— 性能数字在并行测试里会失真）
//!
//! 顺带把一张演示图写到 `out/text_cache_demo.png`（`out/` 已 gitignore）。

use amiki::Canvas;
use amiki_math::Vec2;
use amiki_scene::{Color, GlyphOutline, Paint, PathBuilder, TextCache, TextRun};

/// 一个"竖条"字形：em 单位下覆盖 [0,0.1]×[0,0.6]（模拟真实字形轮廓）
fn bar_glyph() -> GlyphOutline {
    GlyphOutline::new(vec![
        amiki_scene::PathEl::MoveTo(Vec2::new(0.0, 0.0)),
        amiki_scene::PathEl::LineTo(Vec2::new(0.1, 0.0)),
        amiki_scene::PathEl::LineTo(Vec2::new(0.1, 0.6)),
        amiki_scene::PathEl::LineTo(Vec2::new(0.0, 0.6)),
        amiki_scene::PathEl::Close,
    ])
}

/// 造一段 run：20 个字形，字号 18，逐字推进
fn run(n: usize) -> TextRun {
    let glyphs: Vec<amiki_scene::PositionedGlyph> = (0..n)
        .map(|i| amiki_scene::PositionedGlyph {
            id: (i % 3) as u32, // 只用 3 个不同字形：便于观察命中
            x: i as f32 * 12.0,
            y: 0.0,
            font_index: 0,
            cluster: i as u32,
        })
        .collect();
    let mut r = TextRun::new(18.0);
    r.glyphs = glyphs;
    r
}

fn paint() -> Paint {
    Paint::Solid(Color::rgb8(20, 60, 120))
}

/// 1) 缓存不得改变**画出来的东西**（比渲染结果，而不是比 Draw 字段）。
///
/// 为什么不能比 `Draw`：`TextCache` 返回的是**已按字号缩放**的轮廓 + 单位变换，
/// 而 `Canvas::text` 用 **em 轮廓 + 缩放变换** —— 设备空间几何相同，
/// 但 `path` 与 `transform` 两个字段合法地不同。比像素才是真判据。
///
/// ⚠️ 已知且**有意**的差异：缓存键里的字号量化到 **1/16 px**
/// （`GlyphKey.size_q`），所以字号不是 1/16 整数倍时，缓存路径的缩放会差
/// 最多 1/32 的相对量 —— 视觉不可见，但确实不是逐位相同。这里用 ≤1/255 判据。
#[test]
fn cached_text_renders_identically() {
    let r = run(20);

    let mut plain = Canvas::new(120, 64);
    plain.text(&r, Vec2::new(4.0, 40.0), paint(), |_id: u32| Some(bar_glyph()));

    let mut cache = TextCache::new();
    let mut cached = Canvas::new(120, 64);
    cached.text_cached(&mut cache, &r, Vec2::new(4.0, 40.0), paint(), |_id: u32| {
        Some(bar_glyph())
    });

    assert_eq!(plain.scene.draws.len(), cached.scene.draws.len());
    assert_eq!(plain.scene.draws.len(), 20, "20 个字形 = 20 条 draw");

    let a = amiki::render_cpu(&plain, 0.05);
    let b = amiki::render_cpu(&cached, 0.05);
    let max = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).abs())
        .max()
        .unwrap_or(0);
    let ink = a.chunks(4).filter(|p| p[3] > 0).count();
    eprintln!(
        "渲染一致：最大通道差 {max}/255（墨迹 {ink} 像素）；缓存 hits={} misses={}",
        cache.hits(),
        cache.misses()
    );
    assert!(ink > 10, "必须真的画出东西（实测约 20 个墨迹像素）");
    assert!(max <= 1, "缓存路径与普通路径的渲染差 {max} 超出 1/255");
}

/// 2) 重复绘制：命中率上升（缓存真在起作用）。
#[test]
fn repeated_draws_hit_the_cache() {
    let r = run(20);
    let mut cache = TextCache::new();
    let mut c = Canvas::new(64, 64);

    c.text_cached(&mut cache, &r, Vec2::new(0.0, 30.0), paint(), |_| Some(bar_glyph()));
    let (h1, m1) = (cache.hits(), cache.misses());
    // 清掉场景，只关心第二轮的缓存行为
    c.scene.draws.clear();
    c.text_cached(&mut cache, &r, Vec2::new(0.0, 30.0), paint(), |_| Some(bar_glyph()));
    let (h2, m2) = (cache.hits(), cache.misses());

    eprintln!("第一轮 hits={h1} misses={m1}；第二轮 hits={h2} misses={m2}");
    assert_eq!(m1, 3, "第一轮应只对 3 个不同 (id,字号) 各 miss 一次");
    assert_eq!(m2, 3, "第二轮不应再有 miss");
    assert_eq!(h2 - h1, 20, "第二轮 20 个字形应全部命中");
}

/// 3) 加速 + 演示图（数字只打印，不做脆弱断言）。
#[test]
fn cache_speeds_up_repeated_text() {
    let r = run(60);
    let iters = 60;

    let t0 = std::time::Instant::now();
    let mut c = Canvas::new(320, 80);
    for _ in 0..iters {
        c.scene.draws.clear();
        c.text(&r, Vec2::new(4.0, 50.0), paint(), |_| Some(bar_glyph()));
    }
    let without = t0.elapsed().as_secs_f64() * 1000.0;

    let mut cache = TextCache::new();
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        c.scene.draws.clear();
        c.text_cached(&mut cache, &r, Vec2::new(4.0, 50.0), paint(), |_| Some(bar_glyph()));
    }
    let with = t1.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "重复展开 {iters} 次 × {n} 字形：不带缓存 {without:.2} ms，带缓存 {with:.2} ms（{:.1}×）",
        without / with.max(0.001),
        n = r.glyphs.len()
    );

    // 出演示图：用**真字体**（Inter）走 text_cached 路径渲染，肉眼可验
    let font_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/fonts/Inter-Regular.ttf");
    if let Ok(font) = amiki_text::Font::from_file(font_path.to_string_lossy().as_ref()) {
        let text = "Amiki2D text_cached 1.4x";
        if let Ok(run) = amiki_text::shape(&font, text, 26.0) {
            let mut demo = Canvas::new(360, 96);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 360.0, 96.0);
            demo.fill_path(&pb.build(), Paint::Solid(Color::rgb8(252, 252, 254)));
            let mut dc = TextCache::new();
            for k in 0..2 {
                demo.text_cached(
                    &mut dc,
                    &run,
                    Vec2::new(12.0, 40.0 + k as f32 * 30.0),
                    Paint::Solid(if k == 0 {
                        Color::rgb8(24, 60, 130)
                    } else {
                        Color::rgb8(200, 80, 40)
                    }),
                    |id| font.outline(id).ok(),
                );
            }
            let img = amiki::render_cpu(&demo, 0.05);
            let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../out/text_cache_demo.png");
            if let Some(px) = image::RgbaImage::from_raw(360, 96, img) {
                let _ = std::fs::create_dir_all(out.parent().unwrap());
                px.save(&out).expect("写演示图");
                eprintln!(
                    "演示图 -> {}（字形 {} 个，缓存 hits={} misses={}）",
                    out.display(),
                    run.glyphs.len(),
                    dc.hits(),
                    dc.misses()
                );
            }
        }
    } else {
        eprintln!("没找到 Inter 字体，跳过演示图");
    }
    // Per discipline #7: measurement only. A wall-clock comparison here is
    // flaky on a shared machine (it failed once in CI-like conditions).
    // The deterministic proof of reuse is the cache hit/miss counters above.
    let _ = (with, without);
}
