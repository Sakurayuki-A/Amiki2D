//! P0-2：**含层场景的增量渲染**。
//!
//! 背景：M4 的 damage region 是核心卖点（静态界面省 96.5% tile 着色），但真实 UI
//! 几乎都有阴影或模糊。早先 `PersistentRenderer` 的增量分支**从不合成层**，
//! 而 `edit_damage` **不看层** —— 两者合起来的效果是"有滤镜的界面从第二帧起
//! 悄悄丢掉滤镜"，且不报错。
//!
//! 本文件锁定修复后的行为，重点是**逐字节一致性**（与全量渲染对拍），
//! 而不是"看起来对"。

use amiki_math::{Affine, Vec2};
use amiki_raster::{bin, render, PersistentRenderer};
use amiki_scene::{
    BlendMode, Color, Damage, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene,
};

// ───────────────────────── 随机数（与 coverage.rs 同款，测试自足） ─────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next_u32() % n.max(1)
    }
}

const W: u32 = 240;
const H: u32 = 180;

/// 一个"典型界面"：背景 + 若干图元 + 一两个带滤镜的层。
///
/// 参数化到足以让随机序列产生各种变化：图元增删与移动、层内容变化、
/// 滤镜参数变化、层增删、层顺序变化。
/// 一个矩形：`(x, y, w, h)`。
type Rect = (f32, f32, f32, f32);

/// 一层的规格 + 它的内容图元。
type LayerWithShapes = (LayerSpec, Vec<Rect>);

fn build(root_rects: &[Rect], layers: &[LayerWithShapes]) -> Scene {
    let mut s = Scene::new(W, H);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, W as f32, H as f32);
    s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(246, 243, 238)));
    for &(x, y, w, h) in root_rects {
        let mut pb = PathBuilder::new();
        pb.rect(x, y, x + w, y + h);
        s.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(60, 120, 200, 200)),
        );
    }
    for (spec, shapes) in layers {
        s.push_layer(
            Layer {
                bounds: spec.bounds,
                scene: Scene::new(W, H),
                filter: spec.filter,
                opacity: spec.opacity,
                blend: spec.blend,
            },
            |sc| {
                for &(x, y, w, h) in shapes {
                    let mut pb = PathBuilder::new();
                    pb.rect(x, y, x + w, y + h);
                    sc.push(
                        Affine::IDENTITY,
                        pb.build(),
                        Paint::Solid(Color::rgba8(220, 80, 60, 230)),
                    );
                }
            },
        );
    }
    s
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LayerSpec {
    bounds: LayerBounds,
    filter: Filter,
    opacity: f32,
    blend: BlendMode,
}

fn random_rect(rng: &mut Lcg, w: f32, h: f32) -> Rect {
    let x = rng.range(220) as f32;
    let y = rng.range(160) as f32;
    (x, y, 6.0 + rng.range(w as u32) as f32, 6.0 + rng.range(h as u32) as f32)
}

fn random_layer(rng: &mut Lcg) -> LayerSpec {
    let x = rng.range(160) as i32;
    let y = rng.range(120) as i32;
    let filter = match rng.range(3) {
        0 => Filter::Blur { sigma: 2.0 + rng.range(10) as f32 },
        1 => Filter::DropShadow {
            dx: -8.0 + rng.range(17) as f32,
            dy: -8.0 + rng.range(17) as f32,
            sigma: 2.0 + rng.range(8) as f32,
            color: Color::rgba8(20, 20, 30, 160),
        },
        _ => Filter::None,
    };
    LayerSpec {
        bounds: LayerBounds::new(x, y, 40 + rng.range(80), 40 + rng.range(80)),
        filter,
        opacity: 0.4 + rng.range(60) as f32 / 100.0,
        blend: if rng.range(4) == 0 { BlendMode::Multiply } else { BlendMode::Normal },
    }
}

fn random_scene(rng: &mut Lcg) -> Scene {
    let n_root = rng.range(4);
    let root: Vec<_> = (0..n_root).map(|_| random_rect(rng, 40.0, 30.0)).collect();
    let n_layers = rng.range(3);
    let mut layers = Vec::new();
    for _ in 0..n_layers {
        let spec = random_layer(rng);
        let n_shapes = 1 + rng.range(3);
        let shapes: Vec<_> = (0..n_shapes).map(|_| random_rect(rng, 50.0, 40.0)).collect();
        layers.push((spec, shapes));
    }
    build(&root, &layers)
}

/// **核心正确性**：含层场景下，任意编辑序列的增量结果必须与全量**逐字节一致**。
///
/// 这是整块功能的地基。早先的 bug（增量不合成层、edit_damage 不看层）
/// 都会在这里被抓住 —— 而且不是"看起来不对"，是精确的字节比较。
#[test]
fn layer_incremental_matches_full_random_sequences() {
    let mut rng = Lcg(0x51ED_2A17_C0DE_0001);
    let mut r = PersistentRenderer::new(0.05);

    // 先给一个初始场景
    let mut scene = random_scene(&mut rng);
    r.update(&scene);
    assert_eq!(
        r.image(),
        render(&bin(&scene, 0.05)).as_slice(),
        "首帧必须与全量一致"
    );

    let mut layer_change_frames = 0;
    for round in 0..60 {
        scene = random_scene(&mut rng);
        r.update(&scene);
        let full = render(&bin(&scene, 0.05));
        let got = r.image();

        let mut diff_px = 0usize;
        let mut worst = 0u8;
        for i in 0..(W * H) as usize {
            let a = &got[i * 4..i * 4 + 4];
            let b = &full[i * 4..i * 4 + 4];
            if a != b {
                diff_px += 1;
                for k in 0..4 {
                    worst = worst.max(a[k].abs_diff(b[k]));
                }
            }
        }
        if !scene.layers.is_empty() {
            layer_change_frames += 1;
        }
        assert!(
            diff_px == 0,
            "第 {round} 轮：增量与全量不一致 —— {diff_px} 个像素不同（最大通道差 {worst}）。\
             该场景有 {} 个层，本帧重画 {}/{} tile",
            scene.layers.len(),
            r.last_stats().shaded_tiles,
            r.last_stats().total_tiles
        );
    }
    assert!(layer_change_frames > 10, "随机序列应产生足够多有层的帧，否则测试没有判别力");
    eprintln!("60 轮含层随机编辑全部与全量逐字节一致（其中 {layer_change_frames} 帧含层）");
}

/// **验收标准**：带阴影的静态界面，只改层之外的一小块时，重画 tile 必须 **< 10%**。
///
/// 这是 P0-2 的目标场景 —— 也是这个东西存在的理由：真实 UI 都有阴影，
/// 而早先一有层就全量重画（收益归零）。
#[test]
fn static_ui_with_shadow_keeps_damage_benefit() {
    let mut r = PersistentRenderer::new(0.05);

    // 静态界面：一整块背景 + 一个带阴影的卡片区（层）+ 一个小指示条
    let build_ui = |marker_x: f32| -> Scene {
        let mut s = Scene::new(320, 240);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 320.0, 240.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(246, 243, 238)));

        // 阴影层：内容是个"卡片"，放在左下
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(30, 150, 60, 60),
                scene: Scene::new(320, 240),
                filter: Filter::DropShadow {
                    dx: 6.0,
                    dy: 6.0,
                    sigma: 6.0,
                    color: Color::rgba8(20, 20, 30, 170),
                },
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(40.0, 160.0, 90.0, 210.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(90, 140, 200)));
            },
        );

        // 右上角的小指示条：**唯一变化的东西**，且远离层
        let mut pb = PathBuilder::new();
        pb.rect(marker_x, 16.0, marker_x + 8.0, 24.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 60, 40)));
        s
    };

    r.update(&build_ui(280.0));
    let total = r.last_stats().total_tiles;

    // 连续 30 帧只挪动那个小指示条
    let mut worst_ratio = 0.0f64;
    for f in 0..30 {
        let x = 280.0 - (f as f32);
        let scene = build_ui(x);
        r.update(&scene);
        let st = r.last_stats();
        let work = st.shaded_tiles.max(st.composited_tiles) as f64 / total as f64;
        worst_ratio = worst_ratio.max(work);
        assert_eq!(
            r.image(),
            render(&bin(&scene, 0.05)).as_slice(),
            "第 {f} 帧增量与全量不一致"
        );
    }
    assert!(
        worst_ratio < 0.10,
        "带阴影的静态界面里，只改一小块时的重画率应 < 10%，实际最差 {:.1}%",
        worst_ratio * 100.0
    );
    eprintln!(
        "带阴影静态界面：连续 30 帧最大重画 {:.1}% tile（共 {total} tile）",
        worst_ratio * 100.0
    );
}

/// 层**内容本身**变化时，脏区必须覆盖该层的作用范围，且结果正确。
#[test]
fn layer_content_change_is_picked_up() {
    let mut r = PersistentRenderer::new(0.05);
    let mk = |dx: f32| -> Scene {
        let mut s = Scene::new(160, 160);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 160.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 255, 255)));
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(30, 30, 60, 60),
                scene: Scene::new(160, 160),
                filter: Filter::Blur { sigma: 5.0 },
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(40.0 + dx, 40.0, 80.0 + dx, 80.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 40, 40)));
            },
        );
        s
    };

    r.update(&mk(0.0));
    let before = r.image().to_vec();
    r.update(&mk(18.0));
    let scene = mk(18.0);
    assert_ne!(before, r.image(), "层内容挪了 18px，画面应当变化");
    assert_eq!(
        r.image(),
        render(&bin(&scene, 0.05)).as_slice(),
        "层内容变化后增量必须与全量一致"
    );
}

/// **层下方**的图元变化：层贡献必须保持（不能因为复位成 `base` 就丢掉层）。
#[test]
fn primitive_under_layer_is_recomposited_with_layer() {
    let mut r = PersistentRenderer::new(0.05);
    let mk = |cx: f32| -> Scene {
        let mut s = Scene::new(160, 160);
        // 层下方的方块：它变化时所在的 tile 会被复位成 base
        let mut pb = PathBuilder::new();
        pb.rect(cx, 60.0, cx + 20.0, 80.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 180, 90)));

        // 一个覆盖该区域的层（无滤镜，纯分组）—— 它必须仍然可见
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(40, 40, 80, 80),
                scene: Scene::new(160, 160),
                filter: Filter::None,
                opacity: 0.5,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(40.0, 40.0, 120.0, 120.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(220, 60, 40)));
            },
        );
        s
    };

    r.update(&mk(60.0));
    for f in 0..6 {
        let scene = mk(60.0 + f as f32 * 2.0);
        r.update(&scene);
        assert_eq!(
            r.image(),
            render(&bin(&scene, 0.05)).as_slice(),
            "第 {f} 帧：层下方的图元变化后，层贡献必须仍在"
        );
    }
}

/// **增删层**：删除时旧层区域要被擦掉，新增时要在正确位置出现。
#[test]
fn layer_add_remove_is_handled() {
    let with_layer = |n: usize| -> Scene {
        let mut s = Scene::new(160, 160);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 160.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 255, 255)));
        for i in 0..n {
            let ox = i as i32 * 20;
            s.push_layer(
                Layer {
                    bounds: LayerBounds::new(20 + ox, 20, 50, 50),
                    scene: Scene::new(160, 160),
                    filter: Filter::Blur { sigma: 4.0 },
                    opacity: 1.0,
                    blend: BlendMode::Normal,
                },
                |sc| {
                    let mut pb = PathBuilder::new();
                    pb.rect(30.0 + ox as f32, 30.0, 60.0 + ox as f32, 60.0);
                    sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 90, 200)));
                },
            );
        }
        s
    };

    let mut r = PersistentRenderer::new(0.05);
    for n in [0usize, 1, 2, 1, 0, 2, 0] {
        let scene = with_layer(n);
        r.update(&scene);
        assert_eq!(
            r.image(),
            render(&bin(&scene, 0.05)).as_slice(),
            "层数 = {n} 时增量与全量不一致（增删层必须正确处理）"
        );
        assert_eq!(r.cached_layer_count(), n, "缓存的层数应与场景一致");
    }
}

/// 只改**层的不透明度**：不必重新滤波，但结果必须正确。
#[test]
fn opacity_change_recomposites_correctly() {
    let mk = |op: f32| -> Scene {
        let mut s = Scene::new(160, 160);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 160.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 255, 255)));
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(30, 30, 80, 80),
                scene: Scene::new(160, 160),
                filter: Filter::Blur { sigma: 6.0 },
                opacity: op,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(40.0, 40.0, 100.0, 100.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(20, 20, 20)));
            },
        );
        s
    };

    let mut r = PersistentRenderer::new(0.05);
    r.update(&mk(1.0));
    for (i, op) in [0.9f32, 0.5, 0.2, 0.75, 1.0].into_iter().enumerate() {
        let scene = mk(op);
        r.update(&scene);
        assert_eq!(
            r.image(),
            render(&bin(&scene, 0.05)).as_slice(),
            "第 {i} 次改不透明度后增量与全量不一致"
        );
    }
}

/// 两个**重叠**的层：合成顺序必须与全量一致（顺序错了重叠区就会不同）。
#[test]
fn overlapping_layers_keep_order() {
    let order = |swap: bool| -> Scene {
        let mut s = Scene::new(160, 160);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 160.0, 160.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 255, 255)));
        let specs = [
            (Color::rgba8(220, 40, 40, 200), BlendMode::Normal),
            (Color::rgba8(40, 40, 220, 200), BlendMode::Multiply),
        ];
        let idx: [usize; 2] = if swap { [1, 0] } else { [0, 1] };
        for i in idx {
            let (c, b) = specs[i];
            s.push_layer(
                Layer {
                    bounds: LayerBounds::new(30, 30, 80, 80),
                    scene: Scene::new(160, 160),
                    filter: Filter::None,
                    opacity: 0.8,
                    blend: b,
                },
                |sc| {
                    let mut pb = PathBuilder::new();
                    pb.rect(40.0, 40.0, 100.0, 100.0);
                    sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(c));
                },
            );
        }
        s
    };

    let mut r = PersistentRenderer::new(0.05);
    for (i, swap) in [false, true, false].into_iter().enumerate() {
        let scene = order(swap);
        r.update(&scene);
        assert_eq!(
            r.image(),
            render(&bin(&scene, 0.05)).as_slice(),
            "第 {i} 次（swap={swap}）重叠层的合成顺序必须与全量一致"
        );
    }
}

/// 场景完全未变 ⇒ **零工作**（这是静态界面零开销的前提，含层时同样成立）。
#[test]
fn unchanged_layered_scene_does_no_work() {
    let scene = build(
        &[(20.0, 20.0, 40.0, 40.0)],
        &[(
            LayerSpec {
                bounds: LayerBounds::new(60, 60, 80, 80),
                filter: Filter::Blur { sigma: 6.0 },
                opacity: 0.9,
                blend: BlendMode::Normal,
            },
            vec![(70.0, 70.0, 50.0, 50.0)],
        )],
    );
    let mut r = PersistentRenderer::new(0.05);
    r.update(&scene);
    let first = r.image().to_vec();

    r.update(&scene);
    let st = r.last_stats();
    assert!(st.skipped, "场景未变应完全跳过");
    assert_eq!(st.shaded_tiles, 0);
    assert_eq!(st.composited_tiles, 0);
    assert_eq!(r.image(), first.as_slice(), "跳过后像素不应改变");
}

/// 走一遍 `render_damage`（无跨帧层缓存的那条路）：含层时**必须回落全量**。
///
/// 这条锁定"没有缓存就不能假装增量"：`render_damage` 拿不到已滤波的层缓冲，
/// 若它试图只重算部分 tile，层就会残缺。
#[test]
fn render_damage_still_falls_back_with_layers() {
    let scene = build(
        &[(10.0, 10.0, 30.0, 30.0)],
        &[(
            LayerSpec {
                bounds: LayerBounds::new(50, 50, 70, 70),
                filter: Filter::Blur { sigma: 5.0 },
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            vec![(60.0, 60.0, 40.0, 40.0)],
        )],
    );
    let b = bin(&scene, 0.05);
    let full = render(&b);
    let mut d = Damage::new(W, H);
    d.clear();
    d.invalidate_rect(0.0, 0.0, 8.0, 8.0);
    let mut img = vec![0u8; full.len()];
    amiki_raster::render_damage(&b, &mut img, &d);
    assert_eq!(img, full, "含层时 render_damage 应回落全量并给出正确结果");
}

/// 层内容变化**不得**让 `edit_damage` 退化为全脏 —— 否则收益归零。
#[test]
fn layer_damage_is_scoped_not_full() {
    let a = build(&[], &[(
        LayerSpec {
            bounds: LayerBounds::new(100, 100, 40, 40),
            filter: Filter::Blur { sigma: 3.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        vec![(110.0, 110.0, 20.0, 20.0)],
    )]);
    let b = build(&[], &[(
        LayerSpec {
            bounds: LayerBounds::new(100, 100, 40, 40),
            filter: Filter::Blur { sigma: 3.0 },
            opacity: 1.0,
            blend: BlendMode::Normal,
        },
        vec![(120.0, 110.0, 20.0, 20.0)],
    )]);
    let d = amiki_scene::edit_damage(&a, &b);
    assert!(!d.is_full(), "层内改动不应退化为全脏");
    let (tx0, ty0, tx1, ty1) = d.tiles().expect("应有脏区");
    // bounds(100,100,40,40) 外扩 3σ=9 → (91,91,149,149) → tile 5..10
    assert_eq!((tx0, ty0, tx1, ty1), (5, 5, 10, 10), "脏区应等于层的作用范围");
}

/// 一个简单的"图标 + 阴影"场景，用来确认层在**曲线上**也正常（非矩形图元）。
#[test]
fn curved_layer_content_is_consistent() {
    let mk = |r0: f32| -> Scene {
        let mut s = Scene::new(180, 180);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 180.0, 180.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(250, 250, 250)));
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(40, 40, 100, 100),
                scene: Scene::new(180, 180),
                filter: Filter::Blur { sigma: 7.0 },
                opacity: 0.95,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.circle(Vec2::new(90.0, 90.0), r0, false);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 110, 190)));
            },
        );
        s
    };
    let mut r = PersistentRenderer::new(0.05);
    for (i, rr) in [30.0f32, 34.0, 28.0, 36.0, 30.0].into_iter().enumerate() {
        let scene = mk(rr);
        r.update(&scene);
        assert_eq!(
            r.image(),
            render(&bin(&scene, 0.05)).as_slice(),
            "第 {i} 帧（半径 {rr}）曲线层内容增量与全量不一致"
        );
    }
}
