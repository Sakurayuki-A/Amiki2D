//! M6 滤镜（层树）测试：模糊 / 投影 / 分组不透明度。
//!
//! 滤镜的"正确"有两条独立判据：
//! 1. **数学性质**：能量守恒、常数场不变、对称性（已由 `filter` 模块单元测试覆盖）；
//! 2. **端到端可见性**：经完整渲染管线后，像素确实呈现出模糊/投影该有的样子。
//!
//! 这里做第二条：只有走完整管线才能发现"边界裁错""层没合上""混合模式用错"这类问题。

use amiki_math::Affine;
use amiki_raster::{bin, render};
use amiki_scene::{
    BlendMode, Color, Filter, LayerBounds, Paint, PathBuilder, Scene,
};

/// 画一个不透明方块，返回场景。
fn square_scene(cx: f32, cy: f32, half: f32, color: Color) -> Scene {
    let mut s = Scene::new(240, 180);
    let mut pb = PathBuilder::new();
    pb.rect(cx - half, cy - half, cx + half, cy + half);
    s.push(Affine::IDENTITY, pb.build(), Paint::Solid(color));
    s
}

#[test]
fn layer_without_filter_matches_direct_draw() {
    // 层的基准性质：`Filter::None` + 不透明 + Normal 下，
    // 层的结果必须与"直接画在同一位置"逐像素一致。
    // 若不成立，说明层合成路径本身有偏差（偏移、预乘、混合任一处）。
    let (cx, cy, half) = (80.0f32, 70.0f32, 30.0f32);
    let color = Color::rgba8(200, 60, 40, 255);

    let direct = square_scene(cx, cy, half, color);

    let mut layered = Scene::new(240, 180);
    layered.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(20, 10, 160, 150),
            scene: Scene::new(240, 180),
            filter: Filter::None,
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |s| {
            let mut pb = PathBuilder::new();
            pb.rect(cx - half, cy - half, cx + half, cy + half);
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(color));
        },
    );

    let a = render(&bin(&direct, 0.05));
    let b = render(&bin(&layered, 0.05));
    let mut max_diff = 0u32;
    let mut worst = (0usize, 0usize);
    for i in 0..(240 * 180) {
        for k in 0..4 {
            let d = (a[i * 4 + k] as i32 - b[i * 4 + k] as i32).unsigned_abs();
            if d > max_diff {
                max_diff = d;
                worst = (i % 240, i / 240);
            }
        }
    }
    assert!(
        max_diff <= 1,
        "无滤镜层应与直接绘制一致，最大差 {max_diff}（位于 {worst:?}）"
    );
    eprintln!("层（无滤镜）与直接绘制一致（最大差 {max_diff}）");
}

#[test]
fn blur_actually_spreads_energy_outward() {
    // 用**细条**而不是方块：方块若比模糊半径宽得多，其中心在模糊后仍会饱和
    // （邻域全是同色），"中心变暗"就不成立 —— 这条断言必须换输入才有牙。
    // 细条的横向能量必然外流，效果无歧义。
    let (cx, cy) = (120.0f32, 90.0f32);
    let (hw, hh) = (4.0f32, 40.0f32);
    let color = Color::rgba8(255, 255, 255, 255);

    let bar = |scene: &mut Scene| {
        let mut pb = PathBuilder::new();
        pb.rect(cx - hw, cy - hh, cx + hw, cy + hh);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(color));
    };

    let mut sharp = Scene::new(240, 180);
    bar(&mut sharp);

    let mut blurred = Scene::new(240, 180);
    blurred.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(80, 30, 80, 120),
            scene: Scene::new(240, 180),
            filter: Filter::Blur { sigma: 6.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        bar,
    );

    let a = render(&bin(&sharp, 0.05));
    let b = render(&bin(&blurred, 0.05));
    let lum = |img: &[u8], x: usize, y: usize| img[(y * 240 + x) * 4] as u32;
    let alpha = |img: &[u8], x: usize, y: usize| img[(y * 240 + x) * 4 + 3];

    // 1) 条内：模糊后应更暗（能量横向外流）
    assert!(
        lum(&b, 120, 90) < lum(&a, 120, 90),
        "细条内应因能量外流而变暗：清晰 {} vs 模糊 {}",
        lum(&a, 120, 90),
        lum(&b, 120, 90)
    );
    // 2) 条外：模糊后应出现能量（这就是"扩散"）
    let ox = 120 + 12;
    assert_eq!(alpha(&a, ox, 90), 0, "清晰图中条外应完全透明");
    assert!(
        alpha(&b, ox, 90) > 8,
        "模糊后条外应出现能量，实际 alpha={}",
        alpha(&b, ox, 90)
    );
    // 3) 横向剖面单调不增（从中心向外），说明扩散无振荡
    let mut prev = u32::MAX;
    for d in 0..40 {
        let v = lum(&b, 120 + d, 90);
        assert!(v <= prev, "横向剖面应单调不增：d={d} 处 {v} > {prev}");
        prev = v;
    }
    // 4) 远到 3σ 之外应几乎为 0（扩散有界，没有糊满整幅）
    assert!(
        alpha(&b, 120 + 70, 90) < 4,
        "3σ 之外应几乎为 0，实际 alpha={}",
        alpha(&b, 120 + 70, 90)
    );
    eprintln!(
        "模糊：条内 {}→{}，条外 alpha {}→{}，剖面单调不增且有界",
        lum(&a, 120, 90),
        lum(&b, 120, 90),
        alpha(&a, ox, 90),
        alpha(&b, ox, 90)
    );
}

#[test]
fn drop_shadow_appears_offset_and_behind() {
    let (cx, cy, half) = (100.0f32, 80.0f32, 22.0f32);
    let red = Color::rgba8(220, 50, 50, 255);
    let (dx, dy) = (18.0f32, 14.0f32);

    let mut s = Scene::new(240, 180);
    s.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(40, 30, 140, 120),
            scene: Scene::new(240, 180),
            filter: Filter::DropShadow {
                dx,
                dy,
                sigma: 3.0,
                color: Color::rgba8(0, 0, 0, 200),
            },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |sc| {
            let mut pb = PathBuilder::new();
            pb.rect(cx - half, cy - half, cx + half, cy + half);
            sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(red));
        },
    );

    let img = render(&bin(&s, 0.05));
    let px = |x: usize, y: usize| {
        let o = (y * 240 + x) * 4;
        [img[o], img[o + 1], img[o + 2], img[o + 3]]
    };

    // 1) 源方块中心仍是红色（阴影在下方，不替换源）
    let c = px(100, 80);
    assert!(c[0] > 200 && c[1] < 80, "源应保持红色：{c:?}");
    // 2) 偏移方向（右下）的方块外应出现深色阴影
    let sh = px(100 + half as usize + 8, 80 + half as usize + 6);
    assert!(sh[3] > 20, "偏移处应出现阴影，实际 {sh:?}");
    assert!(sh[0] < 120, "阴影应为暗色，实际 {sh:?}");
    // 3) 反方向（左上）方块外不应有阴影（验证偏移方向正确）
    let nosh = px(100 - half as usize - 10, 80 - half as usize - 8);
    assert_eq!(nosh[3], 0, "左上不应有阴影（偏移方向错误？）实际 {nosh:?}");
    eprintln!("投影：源 {c:?}，右下阴影 {sh:?}，左上无阴影 ✓");
}

#[test]
fn layer_opacity_scales_alpha() {
    let color = Color::rgba8(0, 0, 0, 255);
    let mk = |op: f32| {
        let mut s = Scene::new(160, 120);
        s.push_layer(
            amiki_scene::Layer {
                bounds: LayerBounds::new(20, 20, 120, 80),
                scene: Scene::new(160, 120),
                filter: Filter::None,
                opacity: op,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(40.0, 40.0, 120.0, 80.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(color));
            },
        );
        render(&bin(&s, 0.05))
    };
    let full = mk(1.0);
    let half = mk(0.5);
    let a_full = full[(60 * 160 + 80) * 4 + 3];
    let a_half = half[(60 * 160 + 80) * 4 + 3];
    assert_eq!(a_full, 255, "不透明度 1.0 应为全不透明");
    assert!(
        (a_half as i32 - 128).abs() <= 2,
        "不透明度 0.5 应得 alpha≈128，实际 {a_half}"
    );
    eprintln!("层不透明度：1.0 → alpha {a_full}，0.5 → alpha {a_half}");
}

#[test]
fn blur_bounds_that_are_too_small_are_clipped_as_documented() {
    // 这个测试把"边界由调用方给出"的后果**固定下来**：
    // 若 bounds 紧贴内容、不预留模糊外扩，扩散出去的部分会被裁掉。
    // 这不是 bug，而是该设计的必然语义 —— 写进测试以免日后被误认为缺陷。
    let (cx, cy, half) = (120.0f32, 90.0f32, 20.0f32);
    let color = Color::rgba8(255, 255, 255, 255);
    let mut s = Scene::new(240, 180);
    s.push_layer(
        amiki_scene::Layer {
            // 紧贴内容：不预留任何外扩
            bounds: LayerBounds::new(
                (cx - half) as i32,
                (cy - half) as i32,
                (half * 2.0) as u32,
                (half * 2.0) as u32,
            ),
            scene: Scene::new(240, 180),
            filter: Filter::Blur { sigma: 6.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |sc| {
            let mut pb = PathBuilder::new();
            pb.rect(cx - half, cy - half, cx + half, cy + half);
            sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(color));
        },
    );
    let img = render(&bin(&s, 0.05));
    // 膨胀后仍会向外扩 3σ，故边缘外**仍应**有能量；
    // 但若调用方完全不给余量（bounds 小于内容），才会真正被切。
    let alpha = |x: usize, y: usize| img[(y * 240 + x) * 4 + 3];
    assert!(
        alpha(120 + 28, 90) > 0,
        "inflate 应已补偿 3σ，边缘外应有能量，实际 {}",
        alpha(120 + 28, 90)
    );
    eprintln!("紧凑 bounds + inflate(3σ)：边缘外 alpha = {}", alpha(120 + 28, 90));
}

#[test]
fn layer_clips_content_to_bounds_plus_filter_reach() {    // 内容画到 bounds 之外 → 应被裁掉（除滤镜外扩部分）
    let mut s = Scene::new(200, 160);
    s.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(20, 20, 60, 60),
            scene: Scene::new(200, 160),
            filter: Filter::None,
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |sc| {
            // 一个远超 bounds 的大矩形
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 200.0, 160.0);
            sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(0, 0, 0, 255)));
        },
    );
    let img = render(&bin(&s, 0.05));
    let alpha = |x: usize, y: usize| img[(y * 200 + x) * 4 + 3];
    assert!(alpha(50, 50) > 200, "bounds 内应被绘制，实际 {}", alpha(50, 50));
    assert_eq!(alpha(150, 130), 0, "bounds 外应被裁掉（Filter::None 不扩张）");
    assert_eq!(alpha(5, 5), 0, "bounds 外应被裁掉（左上）");
    eprintln!("层裁剪：内部 alpha {}，外部 {} ✓", alpha(50, 50), alpha(150, 130));
}

#[test]
fn incremental_path_does_not_lose_layers() {
    // 层合成需要整层的完整内容，因此含层的场景在增量路径上**回落到全量渲染**。
    // 这个测试锁定该行为：增量结果必须与全量逐字节一致（尤其**不能丢掉滤镜**）。
    use amiki_scene::Damage;

    let (cx, cy, half) = (100.0f32, 80.0f32, 22.0f32);
    let mut s = Scene::new(200, 160);
    s.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(40, 30, 120, 100),
            scene: Scene::new(200, 160),
            filter: Filter::Blur { sigma: 5.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |sc| {
            let mut pb = PathBuilder::new();
            pb.rect(cx - half, cy - half, cx + half, cy + half);
            sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(255, 255, 255, 255)));
        },
    );

    let b = bin(&s, 0.05);
    let full = render(&b);

    // 只标一个很小的脏区，迫使走增量分支
    let mut d = Damage::new(200, 160);
    d.clear();
    d.invalidate_rect(0.0, 0.0, 8.0, 8.0);
    let mut inc = vec![0u8; full.len()];
    amiki_raster::render_damage(&b, &mut inc, &d);
    assert_eq!(inc, full, "含层场景的增量渲染必须与全量一致（不得丢掉滤镜）");

    // 同时确认滤镜确实生效了（否则上面的相等可能是"两边都没模糊"）
    let alpha = |img: &[u8], x: usize, y: usize| img[(y * 200 + x) * 4 + 3];
    assert!(
        alpha(&full, 100 + 28, 80) > 0,
        "模糊应扩散到方块之外，实际 alpha={}",
        alpha(&full, 100 + 28, 80)
    );
    eprintln!("含层场景增量渲染与全量一致，且滤镜确实生效");
}

/// M4/P0-2：**持久渲染器的增量帧不得丢掉层**。
///
/// 这是一个真实的静默错误：`PersistentRenderer::update` 的增量分支只做
/// `clear_tile` + `shade_tile`，而**层合成发生在 `render` 里**（图元之后）。
/// 于是首帧（走全量）正确，**第二帧起层整个消失** —— 而且不报错。
///
/// 既有的 `incremental_path_does_not_lose_layers` 覆盖的是 `render_damage`，
/// 它含层时回落到全量渲染，所以那条路是对的；**`update` 是另一条路**，
/// 这个洞就是被漏掉的地方。
#[test]
fn persistent_renderer_keeps_layers_on_incremental_frames() {
    use amiki_raster::PersistentRenderer;

    // 含一个模糊层的场景。`inner_dx` 用来**改动层内容本身** ——
    // 这一点是必须的：若只改层外的图元，层的陈旧像素恰好仍是正确的，
    // 测试会因为"侥幸"而通过（那样就暴露不了问题）。
    let build = |inner_dx: f32| -> Scene {
        let mut s = Scene::new(200, 160);
        s.push_layer(
            amiki_scene::Layer {
                bounds: LayerBounds::new(40, 30, 120, 100),
                scene: Scene::new(200, 160),
                filter: Filter::Blur { sigma: 5.0 },
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(78.0 + inner_dx, 58.0, 122.0 + inner_dx, 102.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(255, 60, 40, 255)));
            },
        );
        s
    };

    let mut r = PersistentRenderer::new(0.05);

    let s1 = build(0.0);
    r.update(&s1);
    let full1 = render(&bin(&s1, 0.05));
    assert_eq!(r.image(), full1.as_slice(), "首帧（全量）应与 render 一致");

    let s2 = build(14.0); // 层内方块挪动 14px → 层内容变化
    r.update(&s2);
    let full2 = render(&bin(&s2, 0.05));
    let stats = r.last_stats();
    assert!(
        stats.shaded_tiles < stats.total_tiles,
        "本测试要的是增量分支，但着色的 tile 数是 {} / {}（说明走了全量）",
        stats.shaded_tiles,
        stats.total_tiles
    );
    // 用差异摘要而不是 assert_eq!：整幅字节数组被打印出来会让失败输出无法阅读
    let got = r.image();
    let mut diff_px = 0usize;
    let mut worst = 0u8;
    for i in 0..(200 * 160) {
        let a = &got[i * 4..i * 4 + 4];
        let b = &full2[i * 4..i * 4 + 4];
        if a != b {
            diff_px += 1;
            for k in 0..4 {
                worst = worst.max(a[k].abs_diff(b[k]));
            }
        }
    }
    assert!(
        diff_px == 0,
        "增量帧与全量渲染必须逐字节一致 —— 尤其**不能丢掉层**；\
         实际有 {diff_px} 个像素不同（最大通道差 {worst}，画布共 {} 像素）",
        200 * 160
    );
    eprintln!(
        "持久渲染器的增量帧保住了层（重画 {}/{} tile）",
        stats.shaded_tiles, stats.total_tiles
    );
}


// ============================================================================
// HDR（高动态范围）：>1.0 的颜色必须能穿过层边界存活
// ============================================================================

/// 把一个细条的**亮度**放进带模糊的层里，返回渲染结果。
///
/// 用细条而不是方块：方块要是比模糊半径宽得多，中心在模糊后仍然饱和，
/// "更亮"就看不出来（`blur_actually_spreads_energy_outward` 里踩过同一个坑）。
fn blurred_bar_energy(level: f32) -> u64 {
    let (cx, cy) = (120.0f32, 90.0f32);
    let (hw, hh) = (3.0f32, 30.0f32);
    // 直接构造：`Color` 的字段是公开的 f32，`rgba()` 也不裁剪 ——
    // 也就是说**引擎在类型上允许 HDR 值**，问题只在于管线会不会把它丢掉。
    let color = Color { r: level, g: level, b: level, a: 1.0 };

    let mut s = Scene::new(240, 180);
    s.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(60, 20, 120, 140),
            scene: Scene::new(240, 180),
            filter: Filter::Blur { sigma: 8.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |s| {
            let mut pb = PathBuilder::new();
            pb.rect(cx - hw, cy - hh, cx + hw, cy + hh);
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(color));
        },
    );

    let img = render(&bin(&s, 0.05));
    // 只统计层作用的区域，把 R 通道求和当"总能量"。
    // 峰值在两个 level 下都会饱和到 255，所以看的是**扩散出去的那部分**。
    let mut energy = 0u64;
    for y in 20..160u32 {
        for x in 60..180u32 {
            energy += img[((y * 240 + x) * 4) as usize] as u64;
        }
    }
    energy
}

#[test]
fn hdr_values_survive_a_blurred_layer() {
    // 这一条是 HDR 的核心可观测判据。
    //
    // level = 1.0 与 level = 4.0 的**几何完全相同**，只有亮度不同。
    // 若管线在模糊**之前**就把值裁到 1.0（当前行为：层内容先渲染成 8 位
    // `Vec<u8>`，再拷进 f32 的 LayerBuffer —— 裁切在那一步已经发生），
    // 那么两次的能量会**完全一样**，4.0 那个高光源凭空消失。
    //
    // 若 HDR 值真的存活，4.0 的条在模糊后会向外扩散**多得多的能量**。
    let e1 = blurred_bar_energy(1.0);
    let e4 = blurred_bar_energy(4.0);

    assert!(
        e4 > e1 * 2,
        "HDR 值没有穿过层边界：level=1.0 能量 {e1}，level=4.0 能量 {e4}。\n\
         两者几乎相同说明 >1.0 的部分在模糊前就被裁掉了 —— 高光源会消失。"
    );
    eprintln!("HDR 存活：level=1.0 能量 {e1}，level=4.0 能量 {e4}（比值 {:.2}）", e4 as f64 / e1 as f64);
}


// ============================================================================
// f32 层合成与 u8 层合成必须给出同一个答案（防漂移）
// ============================================================================

#[test]
fn layer_blend_f32_matches_u8_where_both_are_valid() {
    // HDR 需要一条 f32 的层合成路径（未裁剪），于是同一个 `composite_premul`
    // 有了两个外壳：u8 版与 f32 版。两份外壳迟早会漂移 —— 这条测试就是
    // 拦住它的东西：**在 `[0,1]` 范围内两者必须逐字节一致**。
    use amiki_raster::{blend_layer_rect, blend_layer_rect_f32, LayerBuffer};

    let (w, h) = (24u32, 16u32);
    let mut buf = LayerBuffer::new(w, h);
    // 半透明 + 渐变 + 全透明混合，覆盖多个分支
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / w as f32;
            let a = if (x + y) % 5 == 0 { 0.0 } else { 0.6 };
            buf.set(x, y, [t, 1.0 - t, 0.35, a]);
        }
    }

    // 目标：一个同样有变化的底图（都在 `[0,1]` 内）
    let mut u8_root = vec![0u8; (w * h * 4) as usize];
    let mut f32_root = vec![[0.0f32; 4]; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let c = [
                (x as f32 / w as f32) * 0.8,
                (y as f32 / h as f32) * 0.6,
                0.25,
                if (x * 3 + y) % 7 == 0 { 0.0 } else { 0.9 },
            ];
            f32_root[i] = c;
            for k in 0..4 {
                u8_root[i * 4 + k] = (c[k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    let backup = f32_root.clone();

    blend_layer_rect(&mut u8_root, w, h, &buf, 0, 0, 1.0, 0, (0, 0, w, h));
    blend_layer_rect_f32(&mut f32_root, w, h, &buf, 0, 0, 1.0, 0, (0, 0, w, h));

    let mut max_diff = 0i32;
    let mut worst = (0usize, 0usize);
    for i in 0..(w * h) as usize {
        for k in 0..4 {
            let a = u8_root[i * 4 + k] as i32;
            let b = (f32_root[i][k] * 255.0 + 0.5).clamp(0.0, 255.0) as i32;
            if (a - b).abs() > max_diff {
                max_diff = (a - b).abs();
                worst = ((i as u32 % w) as usize, (i as u32 / w) as usize);
            }
        }
    }

    // 只在**没被写到的像素**上确认 f32 版是原样保留的（不被清零）
    let untouched = (0..(w * h) as usize).filter(|&i| f32_root[i] == backup[i]).count();

    assert!(
        max_diff <= 1,
        "u8 与 f32 层合成在 `[0,1]` 范围内必须一致，最大差 {max_diff}（位于 {worst:?}）。\n\
         两者不一致说明两份外壳已经漂移 —— 那会让 CPU 与 GPU 后端给出不同的图。"
    );
    eprintln!("层合成 u8/f32 一致（最大差 {max_diff}，原样保留的像素 {untouched}/{})", w * h);
}


// ============================================================================
// 色调映射：CPU 出口也必须能把它用上
// ============================================================================

#[test]
fn render_with_tone_reaches_the_cpu_output() {
    // 这条测试的存在理由：色调映射**实现**在 `amiki-scene`，但只有**接进出口**
    // 才有意义。若 `render_with_tone` 忘了用 tone（或快路径判错），
    // 三种映射会给出完全一样的图 —— 那正是这条要抓的。
    use amiki_raster::{bin, render, render_with_tone};
    use amiki_scene::ToneMap;

    // 一个带模糊层的 HDR 高光源：正是发光效果需要的那种内容
    let mut s = Scene::new(240, 180);
    {
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 240.0, 180.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(20, 20, 24)));
    }
    s.push_layer(
        amiki_scene::Layer {
            bounds: LayerBounds::new(60, 20, 120, 140),
            scene: Scene::new(240, 180),
            filter: Filter::Blur { sigma: 8.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        |s| {
            let mut pb = PathBuilder::new();
            pb.rect(114.0, 60.0, 126.0, 120.0);
            // 1.8 —— sRGB 空间里的"高光源"量级（见 docs/COLOR.md：4.0 已经是天文数字）
            s.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color { r: 1.8, g: 1.8, b: 1.8, a: 1.0 }),
            );
        },
    );

    let b = bin(&s, 0.05);
    let raw = render(&b);
    let raw_explicit = render_with_tone(&b, ToneMap::Raw);
    let reinhard = render_with_tone(&b, ToneMap::Reinhard);
    let aces = render_with_tone(&b, ToneMap::Aces);

    // 快路径必须与 render() 逐字节一致（否则"Raw 不多花钱"就是假的）
    assert_eq!(raw, raw_explicit, "ToneMap::Raw 应当与 render() 完全一致");

    let diff = |a: &[u8], b: &[u8]| -> (u32, usize) {
        let mut m = 0u32;
        let mut n = 0usize;
        for i in 0..a.len() {
            let d = (a[i] as i32 - b[i] as i32).unsigned_abs();
            if d > 0 {
                n += 1;
            }
            m = m.max(d);
        }
        (m, n)
    };
    let (d_rei, n_rei) = diff(&raw, &reinhard);
    let (d_ace, n_ace) = diff(&raw, &aces);
    let (d_ra, n_ra) = diff(&reinhard, &aces);

    eprintln!("色调映射对 CPU 输出的影响：Reinhard 最大差 {d_rei}/255（{n_rei} 像素），ACES {d_ace}（{n_ace}），两者互差 {d_ra}（{n_ra}）");

    assert!(
        d_rei > 0 && d_ace > 0,
        "色调映射没有影响 CPU 输出（Reinhard 差 {d_rei}，ACES 差 {d_ace}）——\n\
         说明 render_with_tone 没有真的用上 tone。"
    );
    assert!(
        d_ra > 0,
        "Reinhard 与 ACES 给出完全相同的输出 —— 那等于只有一个映射"
    );
    // alpha 不该被色调映射改动（它是覆盖率，不是光）
    let mut alpha_diff = 0u32;
    for i in 0..(240 * 180) {
        alpha_diff = alpha_diff.max(
            (raw[i * 4 + 3] as i32 - aces[i * 4 + 3] as i32).unsigned_abs(),
        );
    }
    assert_eq!(alpha_diff, 0, "色调映射改了 alpha");
}
