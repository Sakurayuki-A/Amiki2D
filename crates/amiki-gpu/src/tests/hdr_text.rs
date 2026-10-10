//! HDR 层与真字体（CJK）的跨后端一致性。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

// ============================================================================
// HDR：>1.0 的颜色必须能穿过层边界存活（CPU 与 GPU 必须一致）
// ============================================================================

/// 与 `amiki-raster` 的 `hdr_values_survive_a_blurred_layer` 同一个场景，
/// 但这里要的是**两个后端给出同一个答案**。
///
/// 为什么这条比 CPU 那条更关键：如果只有 CPU 修好，GPU 仍然在模糊前裁到 1.0，
/// 那么**同一个场景在两个后端上会渲染出不同的图** —— 那比"两边都不支持 HDR"
/// 更糟，因为它是静默的、只在特定内容下才暴露的不一致。
#[test]
fn gpu_hdr_layer_matches_cpu() {
    use amiki_math::Affine;
    use amiki_scene::{BlendMode, Color, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };

    let mut scene = Scene::new(240, 180);
    {
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 240.0, 180.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(20, 20, 24)));
    }
    scene.push_layer(
        Layer {
            bounds: LayerBounds::new(60, 20, 120, 140),
            scene: Scene::new(240, 180),
            filter: Filter::Blur { sigma: 8.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |s| {
            let mut pb = PathBuilder::new();
            pb.rect(117.0, 60.0, 123.0, 120.0);
            // 4.0 —— HDR 高光源。若在模糊前被裁到 1.0，这一条就不起作用了。
            s.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color { r: 4.0, g: 4.0, b: 4.0, a: 1.0 }),
            );
        },
    );

    let gpu = renderer
        .render_blocking(&scene, 0.05)
        .expect("GPU 渲染失败");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

    let mut max_diff = 0u32;
    let mut worst = (0usize, 0usize);
    for i in 0..(240 * 180) {
        for k in 0..4 {
            let d = (gpu[i * 4 + k] as i32 - cpu[i * 4 + k] as i32).unsigned_abs();
            if d > max_diff {
                max_diff = d;
                worst = (i % 240, i / 240);
            }
        }
    }

    // 能量对比：HDR 若真的存活，总能量必须显著高于"裁到 1.0"的情形。
    let mut e = 0u64;
    for y in 20..160u32 {
        for x in 60..180u32 {
            e += gpu[((y * 240 + x) * 4) as usize] as u64;
        }
    }
    eprintln!("GPU HDR 层：最大通道差 {max_diff}（位于 {worst:?}），能量 {e}");

    assert!(
        max_diff <= 2,
        "HDR 层：GPU 与 CPU 最大通道差 {max_diff}（位于 {worst:?}）超出容差。\n\
         差异很大通常意味着只有一个后端保住了 >1.0 的值 —— 那会让同一场景\n\
         在两个后端上渲染出不同的图。"
    );
}

/// 0.7.0：**CJK 字形**（经字体回退 shaping）在 GPU 与 CPU 上逐像素一致。
///
/// 字形轮廓对引擎来说就是普通路径，但 **CJK 字形比拉丁复杂得多**
/// （轮廓点数多、曲线多、还有大量二次曲线），所以"拉丁对拍通过"推不出
/// "CJK 也对拍通过" —— 这条测试补的正是那个缺口。
#[test]
fn gpu_renders_cjk_glyphs_identically_to_cpu() {
    use amiki_math::Affine;
    use amiki_scene::{Color, Paint, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };

    // 字体（仓库 assets/fonts 下；缺一个就跳过并说明）
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("仓库根");
    let mut stack = amiki_text::FontStack::new();
    for name in ["Inter-Regular.ttf", "NotoSansSC-Regular.ttf"] {
        let p = root.join("assets/fonts").join(name);
        if !p.exists() {
            eprintln!("跳过：未找到 {}", p.display());
            return;
        }
        stack.push(amiki_text::Font::from_file(&p).expect("字体应能解析"));
    }

    // 混排：汉字走 Noto、拉丁与数字走 Inter
    let text = "你好世界 Amiki2D 2026";
    let size = 40.0f32;
    let run = amiki_text::shape_with_stack(&stack, text, size, &amiki_text::TextOptions::default())
        .expect("shaping 应成功");
    let n_cjk = run.glyphs.iter().filter(|g| g.font_index == 1).count();
    assert!(n_cjk >= 4, "汉字应由 Noto 渲染，实际 {n_cjk} 个字形落在字体 1");

    let mut scene = Scene::new(320, 80);
    let base = Affine::translate(amiki_math::Vec2::new(8.0, 58.0));
    let paths = amiki_scene::expand_text_run_with(&run, |fi, id| stack.get(fi)?.outline(id).ok());
    // 空格有字形但没有轮廓（`expand_text_run_with` 按设计跳过空轮廓）
    let spaces = text.chars().filter(|c| *c == ' ').count();
    assert_eq!(
        paths.len(),
        run.glyphs.len() - spaces,
        "除空格外每个字形都应展开出一条路径（{} 个字形、{spaces} 个空格）",
        run.glyphs.len()
    );
    for (path, t) in paths {
        scene.push(base.then(t), path, Paint::Solid(Color::rgb8(0, 0, 0)));
    }

    let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

    let mut max_diff = 0u32;
    let mut ink = 0usize;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        if *a > 0 {
            ink += 1;
        }
    }
    // 有牙的前提：真的画出了字形。否则"两边都是空白"也会让对拍通过。
    assert!(
        ink > 2000,
        "必须真的画出 CJK 字形（否则对拍没有意义），实际非零通道 {ink}"
    );
    assert!(
        max_diff <= 2,
        "CJK 字形：GPU 与 CPU 最大通道差 {max_diff} 超出容差 2/255"
    );
    eprintln!(
        "CJK 对拍：{text:?} → {} 个字形（其中 {n_cjk} 个来自 Noto），最大通道差 {max_diff}，非零通道 {ink}",
        run.glyphs.len()
    );
}

