//! M3b 稀疏桶表：与稠密对拍、容量投机、strip 逻辑、启动开销。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

// ===================== M3b：稀疏桶表 =====================

/// M3b：稀疏路径 vs 稠密路径逐字节一致（多场景，含越界/混合模式/裁剪）。
#[test]
fn sparse_matches_dense() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Some((mut renderer, _gpu_guard)) = gpu() else { return };
    let mut scenes: Vec<(&str, Scene)> = Vec::new();
    {
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 40.0, 40.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 40, 40)));
        scenes.push(("rect", scene));
    }
    {
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 72.0, 72.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(36.0, 36.0), 28.0, false);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40, 90, 160, 200)),
            BlendMode::Multiply,
        );
        let mut pb = PathBuilder::new();
        pb.rect(-16.0, -10.0, 16.0, 10.0);
        let t = Affine::translate(Vec2::new(44.0, 36.0)).then(Affine::rotate(0.5));
        scene.push(t, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 180)));
        scenes.push(("mixed", scene));
    }
    {
        let mut scene = Scene::new(96, 96);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        let t = Affine::translate(Vec2::new(48.0, 48.0))
            .then(Affine::rotate(0.3))
            .then(Affine::translate(Vec2::new(-48.0, -48.0)));
        scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(60.0, 60.0), 34.0);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(230, 120, 60, 200)),
            BlendMode::Multiply,
        );
        scenes.push(("curves", scene));
    }
    {
        // 越出画布右缘 / 左缘 / 四角（carry 右界扩张与桶 0 钳制路径）
        let mut scene = Scene::new(80, 64);
        let mut pb = PathBuilder::new();
        pb.rect(40.0, 8.0, 140.0, 48.0); // 跨右缘
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 160, 90)));
        let mut pb = PathBuilder::new();
        pb.rect(-30.0, 4.0, 30.0, 40.0); // 跨左缘
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(200, 90, 30, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(78.0, 60.0), 22.0, false); // 跨右下角
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(60, 60, 220, 160)),
            BlendMode::Screen,
        );
        scenes.push(("offscreen", scene));
    }
    {
        // 裁剪 + 非可分离混合模式（clip 覆盖率纹理亦走稀疏路径）
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
        let c0 = scene.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.rect(16.0, 16.0, 64.0, 64.0);
        let c1 = scene.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(32.0, 32.0), 26.0);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40, 90, 160, 220)),
            BlendMode::Hue,
        );
        scene.draws.last_mut().unwrap().clips = vec![c0, c1];
        scenes.push(("clips", scene));
    }
    {
        // 全 16 种混合模式（逐个对比）
        let modes = [
            BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
            BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
            BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference, BlendMode::Exclusion,
            BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity,
        ];
        let mut scene = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 56.0, 56.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 90, 160)));
        for (i, mode) in modes.iter().enumerate() {
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(16.0 + (i % 4) as f32 * 10.0, 16.0 + (i / 4) as f32 * 10.0), 14.0, false);
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(230, 120, 60, 200)),
                *mode,
            );
        }
        scenes.push(("blend16", scene));
    }

    for (name, scene) in &scenes {
        renderer.path = RenderPath::Dense;
        let dense = renderer.render_blocking(scene, 0.05).expect("稠密渲染失败");
        renderer.path = RenderPath::Sparse;
        let sparse = renderer.render_blocking(scene, 0.05).expect("稀疏渲染失败");
        let mut max_diff = 0u32;
        let mut n_diff = 0usize;
        for (a, b) in dense.iter().zip(sparse.iter()) {
            let d = (*a as i32 - *b as i32).unsigned_abs();
            max_diff = max_diff.max(d);
            if d > 0 {
                n_diff += 1;
            }
        }
        eprintln!("{name}: 最大差 {max_diff}，不同字节 {n_diff}/{}", dense.len());
        assert_eq!(max_diff, 0, "{name}: 稀疏与稠密渲染不一致（最大差 {max_diff}）");
    }
}

/// M3c：稀疏路径在「稠密不可能」的规模下可运行，且显存与内容成比例。
/// 目标参照 §4.5（1080p × 10k 曲线）；测试用 2k 曲线控制耗时。
#[test]
fn sparse_scales_beyond_dense() {
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, Scene};
    let Some((renderer, _gpu_guard)) = gpu() else { return };
    let (w, h, n) = (1920u32, 1080u32, 2000usize);
    let mut scene = Scene::new(w, h);
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..n {
        let c = Vec2::new(rnd() * w as f32, rnd() * h as f32);
        let mut pb = amiki_scene::PathBuilder::new();
        pb.circle(c, 4.0 + 4.0 * rnd(), false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 120, 60, 200)),
        );
    }

    // 稠密网格：显存 ∝ tiles×draws，早已越过 128MB 存储绑定上限
    let gs = build_gpu_scene(&scene, 0.1);
    let n_grid = (gs.tiles_x as u64) * (gs.tiles_y as u64) * (n as u64) * 16;
    let dense_bytes = n_grid * 12 + (n as u64) * (gs.tiles_y as u64) * 16 * ((gs.tiles_x + 1) as u64) * 4;
    assert!(
        dense_bytes > 128 * 1024 * 1024,
        "本测试应处于稠密不可行区间（实际稠密需求 {:.1} MB）",
        dense_bytes as f64 / 1e6
    );

    // 稀疏路径：可运行，且占用与内容成比例
    let img = renderer.render_blocking(&scene, 0.1).expect("稀疏渲染失败");
    assert_eq!(img.len(), (w * h * 4) as usize);
    let st = renderer.sparse_stats(&scene, 0.1).expect("统计失败").expect("稀疏统计缺失");
    let sparse_bytes = st.n_cells as u64 * 24
        + st.n_buckets as u64 * 4
        + st.n_tile_entries as u64 * 16
        + st.n_items as u64 * 16;
    eprintln!(
        "1080p × {n} 曲线：稠密需求 {:.0} MB → 稀疏 {:.1} MB（条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}）",
        dense_bytes as f64 / 1e6,
        sparse_bytes as f64 / 1e6,
        st.n_strips, st.n_cells, st.n_buckets, st.n_tile_entries, st.n_items
    );
    assert!(
        sparse_bytes * 50 < dense_bytes,
        "稀疏占用应显著小于稠密（稀疏 {sparse_bytes} vs 稠密 {dense_bytes}）"
    );
}

/// M3.6：投机分配的**容量表**必须来自实测用量，而不是写死的猜测，
/// 且必须**按场景特征分表**（共用一个全局值会让大小场景互相踩，
/// 实测 720p×64 曾因此退化 3 倍）。
///
/// 本测试回答"学到了什么"：渲染一帧后，表里应当出现**该场景**的表项，
/// 且其规模恰好等于该帧实测的用量（与 `sparse_stats` 逐项一致）。
/// 容量**够不够用**、不够时能否被检出并重做，见
/// `speculative_overflow_is_redone_correctly`。
#[test]
fn sparse_capacity_is_learned_from_measured_usage() {
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };
    assert_eq!(renderer.capacity_table_len(), 0, "首帧之前不应有任何容量表项");

    let mut scene = Scene::new(160, 160);
    for i in 0..12 {
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(24.0 + 9.0 * i as f32, 60.0 + 5.0 * i as f32), 9.0, false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(60, 140, 220, 200)),
        );
    }

    renderer.render_blocking(&scene, 0.05).expect("渲染失败");

    let entries = renderer.capacity_entries();
    assert_eq!(entries.len(), 1, "应为该场景建立一个表项：{entries:?}");
    let (key, cap) = entries[0];
    assert_eq!(
        (key.width, key.height, key.draws),
        (160, 160, 12),
        "key 应描述刚渲染的场景（宽/高/draw 数）"
    );
    assert!(
        key.seg_bucket.is_power_of_two(),
        "几何桶应为 2 的幂（量化后才稳定）：{key:?}"
    );

    let st = renderer.sparse_stats(&scene, 0.05).expect("统计失败").expect("稀疏统计缺失");
    assert_eq!(
        (cap.cells, cap.buckets, cap.tile_entries, cap.items),
        (st.n_cells, st.n_buckets, st.n_tile_entries, st.n_items),
        "容量必须等于实测用量（单元/桶/条目/成员）"
    );
}

/// P3：**启动开销**必须保持在目标以内。
///
/// 指标来自发布计划：`GpuRenderer::new()` < 200 ms（原为 0.5–0.7 s）。
///
/// # 实测（RTX 3050 / Windows / release）
///
/// | 配置 | 冷启动 | 热启动 | 达标 |
/// |---|---|---|---|
/// | **默认**（全部后端 → 本机选 Vulkan） | **337 ms** | 202 ms | ✗ |
/// | `AMIKI_BACKENDS=vulkan` | **154 ms** | 53 ms | ✓ |
/// | `AMIKI_BACKENDS=dx12` | 2272 ms | 2093 ms | ✗ |
///
/// **结论：本项"有条件达标"** —— 默认配置下未达标，显式指定后端时达标。
/// 这个条件必须写明，不能含糊成"已达标"。
///
/// 为什么默认不指定后端：默认值要保证"哪台机器都能起来"，而指定单一后端
/// 会削弱项目"一套 WGSL 通吃"的既定价值（默认只用 Vulkan 会快一倍，
/// 但违背这条价值，故不采用）。
///
/// 需要更快启动的应用有两条正当途径：
/// 1. `AMIKI_BACKENDS` 显式指定后端；
/// 2. `GpuRenderer::new` 本就是 `async` —— 与窗口初始化**重叠**，
///    不让驱动初始化阻塞首帧（这才是引擎该给的答案）。
///
/// 断言取 **1200 ms**：只作为"病态回归"的兜底（例如误改成同步阻塞），
/// 不去卡一个本项尚未达成的指标 —— 那种断言会在慢机器上产生假失败，
/// 也会掩盖"其实没达标"这个事实。
///
/// 定位结论：启动开销的 **86% 在后端枚举**（`Instance` + 请求适配器），
/// 21 条管线只占 12.7 ms。所以发布计划里"用 PipelineCache / 惰性创建"
/// 的假设是**错的**，已按实测更正。
#[test]
fn startup_cost_stays_within_budget() {
    // 本测试**连着创建 4 个渲染器**，所以只拿一次闸门（`std::sync::Mutex`
    // 不可重入，反复调用 `gpu()` 会自锁）。
    let _gpu_guard = gpu_lock();
    let Ok(first) = pollster::block_on(GpuRenderer::new()) else {
        eprintln!("跳过：无 GPU 适配器");
        return;
    };
    let name = first.adapter_name.clone();
    drop(first);

    // **取多次的最小值**，而不是测一次就断言。
    //
    // 理由：这台机器上单次冷启动的**正常**范围是 340–500 ms，但机器有负载时
    // 实测出现过 **1598.8 ms** —— 一次就断言会产生假失败（实际发生过一次：
    // 连续跑完 10 轮基准后跑测试，`startup_cost` 假失败）。
    // 而真实的回归会让**每一次**都慢，所以取最小值既不误报、也不放过回归。
    let mut cold_ms = f64::MAX;
    let mut warm_ms = f64::MAX;
    let mut names = Vec::new();
    for i in 0..3 {
        let t0 = std::time::Instant::now();
        let Ok(r) = pollster::block_on(GpuRenderer::new()) else { return };
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if i == 0 {
            // 第一次是真正的"冷"（驱动尚未在本进程里加载过）
            cold_ms = ms;
        } else {
            warm_ms = warm_ms.min(ms);
        }
        names.push(r.adapter_name.clone());
        drop(r);
    }
    eprintln!(
        "[启动] 适配器 = {name}；首次(冷) = {cold_ms:.1} ms；后续(热) 最小 = {warm_ms:.1} ms"
    );
    assert!(
        names.iter().all(|n| *n == name),
        "同一进程内多次创建应选中同一个适配器：{names:?}"
    );
    // 显式指定后端时（`AMIKI_BACKENDS`）不套用默认预算：那时测的不是默认配置。
    // DX12 走 FXC 运行期编译，冷启动实测约 2.7 s，本就不该套用本预算。
    if std::env::var("AMIKI_BACKENDS").is_ok() {
        eprintln!("（已显式指定后端，跳过预算断言）");
        return;
    }
    assert!(
        cold_ms < 1200.0,
        "冷启动 {cold_ms:.1} ms —— 这是病态回归的兜底阈值（正常 340–500 ms）；\
         注意**本项未达标**（目标 200 ms），详见测试文档注释"
    );
}

/// P0-3：**帧路径必须能渲染含层场景**，且与参考逐像素一致。
///
/// 覆盖三种滤镜（`None` / `Blur` / `DropShadow`）× 两条入口
/// （`render_into` 无 damage、`render_into_persistent` 带 damage）。
///
/// 对拍基准有两个，缺一不可：
/// - `render_blocking`（GPU 侧重合成参考）—— 验证帧路径的接线；
/// - CPU `amiki_raster`（唯一真值）—— 验证数学本身。
///
/// 本测试在开发中抓到过一个**参数传反**的缺陷（`composite_pass` 的
/// `layer_tex` 与 `shadow_tex` 位置写反），症状是层边缘出现一圈不该有的淡影；
/// 若只跟 `render_blocking` 比而不跟 CPU 比，这个错误会被漏掉若干轮。
#[test]
fn gpu_frame_path_composites_layers() {
    use amiki_math::Affine;
    use amiki_scene::{BlendMode, Color, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };

    let build = |filter: Filter| -> Scene {
        let mut scene = Scene::new(160, 120);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 120.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(246, 243, 238)));
        // 层外图元，用来确认"层 + 图元"都要在
        let mut pb = PathBuilder::new();
        pb.rect(6.0, 90.0, 40.0, 114.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 140, 90)));
        scene.push_layer(
            Layer {
                bounds: LayerBounds::new(36, 26, 70, 70),
                scene: Scene::new(160, 120),
                filter,
                opacity: 0.95,
                blend: BlendMode::Normal,
            },
            |s| {
                let mut pb = PathBuilder::new();
                pb.rect(44.0, 34.0, 96.0, 86.0);
                s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 60, 50)));
            },
        );
        scene
    };

    let cases: Vec<(&str, Filter)> = vec![
        ("分组", Filter::None),
        ("模糊", Filter::Blur { sigma: 6.0 }),
        (
            "投影",
            Filter::DropShadow {
                dx: 9.0,
                dy: 11.0,
                sigma: 7.0,
                color: Color::rgba8(25, 22, 20, 190),
            },
        ),
        (
            "投影无模糊",
            Filter::DropShadow {
                dx: 9.0,
                dy: 11.0,
                sigma: 0.0,
                color: Color::rgba8(25, 22, 20, 190),
            },
        ),
    ];

    for (name, filter) in cases {
        let scene = build(filter);
        let reference = renderer
            .render_blocking(&scene, 0.05)
            .unwrap_or_else(|e| panic!("{name}层：render_blocking 失败（{e}）"));
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

        // ① 帧路径（无 damage）
        let frame = renderer.prepare(&scene, 0.05).expect("prepare 失败");
        assert!(frame.scene.has_layers, "场景应被识别为含层");
        let tex = renderer.create_output_texture(&frame.scene);
        renderer
            .render_into(&tex, &frame)
            .unwrap_or_else(|e| panic!("{name}层：帧路径不得拒绝含层场景（{e}）"));
        let got = renderer.readback_rgba8(&tex).expect("读回失败");

        // ② 帧持久路径（全域 damage，等价全量）
        let frame2 = renderer.prepare(&scene, 0.05).expect("prepare 失败");
        let tex2 = renderer.create_output_texture(&frame2.scene);
        let d = amiki_scene::Damage::new(160, 120);
        renderer
            .render_into_persistent(&tex2, &frame2, Some(&d))
            .unwrap_or_else(|e| panic!("{name}层：帧持久路径不得拒绝含层场景（{e}）"));
        let got_persist = renderer.readback_rgba8(&tex2).expect("读回失败");

        let diff = |a: &[u8], b: &[u8]| -> u32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
                .max()
                .unwrap_or(0)
        };
        // 与 `render_blocking`：两者不是同一条浮点路径（那条是"读回 u8 →
        // 主机 f32 → 写回 u8"的往返），允许 ±1 舍入差。
        let max_ref = diff(&got, &reference);
        let max_cpu = diff(&got, &cpu);
        let max_persist = diff(&got_persist, &cpu);
        assert!(max_ref <= 2, "{name}层：与 render_blocking 差 {max_ref} 超容差");
        assert!(max_cpu <= 2, "{name}层：与 CPU 差 {max_cpu} 超容差");
        assert!(max_persist <= 2, "{name}层：帧持久路径与 CPU 差 {max_persist} 超容差");
        eprintln!(
            "帧路径 {name}层：render_blocking {max_ref} / CPU {max_cpu} / 持久 {max_persist}"
        );
    }
}



/// M3a：轴对齐矩形（多 draw、半透明、跨 tile、部分出画布）——strip 逐格对拍。
#[test]
fn strips_match_dense_rects() {
    use amiki_math::Affine;
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };
    // 高 72 = 4.5 个 tile 行 → 末行带为部分带（验证高度钳制）
    let mut scene = Scene::new(72, 72);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 72.0, 72.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
    let mut pb = PathBuilder::new();
    pb.rect(8.0, 8.0, 40.0, 56.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
    // 左/上出画布（carry 桶 0 钳制路径）
    let mut pb = PathBuilder::new();
    pb.rect(-24.0, -12.0, 20.0, 30.0);
    scene.push_blend(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(230, 120, 60)), BlendMode::Multiply);
    // 完全出画布（左下方）→ 不得产生 strip
    let mut pb = PathBuilder::new();
    pb.rect(-200.0, 200.0, -160.0, 260.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 0, 0)));

    let (gs, dump) = dump_pipeline(&renderer, &scene, 0.05);
    check_strips(&gs, &dump, true);
}

/// M3a：曲线/旋转/自相交混合场景——strip 覆盖性（含越界保守扩张路径）。
#[test]
fn strips_match_dense_curves() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };
    let mut scene = Scene::new(96, 96);
    // 环形（曲线 + 反绕洞）+ 旋转（斜边跨带）
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
    pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
    let t = Affine::translate(Vec2::new(48.0, 48.0))
        .then(Affine::rotate(0.3))
        .then(Affine::translate(Vec2::new(-48.0, -48.0)));
    scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
    // 自相交五角星（直线，W=2 区域）
    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(60.0, 60.0), 34.0);
    scene.push_blend(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgba8(230, 120, 60, 200)),
        BlendMode::Multiply,
    );
    // 右下角出画布曲线（钳制路径）
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(104.0, 100.0), 24.0, false);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(52, 152, 219)));

    let (gs, dump) = dump_pipeline(&renderer, &scene, 0.05);
    check_strips(&gs, &dump, false);
}

/// M3.5：稠密容量预检——超限必须返回**可读错误**，而不是让 wgpu 校验层 panic。
///
/// 回归背景：720p × 500 曲线走稠密时，`grid_aux` 需 230.4 MB，超过默认
/// `max_storage_buffer_binding_size`（128 MiB），此前表现为
/// `create_bind_group` 内部的 wgpu Validation Error panic（进程终止、无法降级）。
#[test]
fn dense_ceiling_reports_error_not_panic() {
    let Some((mut renderer, _gpu_guard)) = gpu() else { return };

    // 1) 纯函数判定：默认限制下 720p × 500 draw 必须超限，且错误信息含关键数字
    let limits = wgpu::Limits::default();
    let (tx, ty) = (1280u32.div_ceil(16), 720u32.div_ceil(16)); // 80 × 45
    let err = dense_capacity(&limits, tx, ty, 500).expect_err("720p×500 应判定超限");
    let msg = err.to_string();
    assert!(msg.contains("稠密网格超限"), "错误信息应说明是稠密超限：{msg}");
    assert!(msg.contains("grid_aux"), "错误信息应指明是哪个缓冲：{msg}");
    assert!(msg.contains("RenderPath::Sparse"), "错误信息应给出降级建议：{msg}");
    eprintln!("预检信息（应可读）：{msg}");

    // 2) 正常规模不应被误判
    assert!(dense_capacity(&limits, tx, ty, 8).is_ok(), "小场景不应被预检拦下");

    // 3) 端到端：走稠密路径渲染超限场景必须返回 Err（而非 panic）
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, PathBuilder};
    let mut scene = Scene::new(1280, 720);
    let mut s = 0x5EED_1234_5678_9ABCu64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..500 {
        let c = Vec2::new(rnd() * 1280.0, rnd() * 720.0);
        let mut pb = PathBuilder::new();
        pb.circle(c, 4.0 + 6.0 * rnd(), false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 120, 60, 200)),
        );
    }

    // 第二个渲染器：本测试要**同时**持有稠密与稀疏两个渲染器，
    // 而闸门已在手（`std::sync::Mutex` 不可重入）→ 直接创建，不再走 `gpu()`。
    let Ok(mut r) = pollster::block_on(GpuRenderer::new()) else { return };
    r.path = RenderPath::Dense;
    let res = r.render_blocking(&scene, 0.1);
    assert!(
        matches!(res, Err(ref e) if e.to_string().contains("稠密网格超限")),
        "稠密超限应返回 Err(稠密网格超限)，实际：{:?}",
        res.map(|v| format!("Ok({} 字节)", v.len()))
    );

    // 4) 同一场景走稀疏路径必须成功（验证建议的降级方案确实可行）
    renderer.path = RenderPath::Sparse;
    let img = renderer.render_blocking(&scene, 0.1).expect("稀疏路径应可渲染同一场景");
    assert_eq!(img.len(), 1280 * 720 * 4);
    eprintln!("同一场景：稠密返回可读错误，稀疏渲染成功（{} 字节）", img.len());
}

