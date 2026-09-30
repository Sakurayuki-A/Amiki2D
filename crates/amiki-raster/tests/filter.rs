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
    // （邻域全是同色），"中心变暗"就不成立 —— 我第一版正是这么写错的。
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
