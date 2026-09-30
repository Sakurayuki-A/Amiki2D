//! 正确性测试：解析式覆盖率光栅器 vs brute-force 超采样绕数判定。
//!
//! brute-force 裁判对每个像素取 N×N 采样点，逐点计算绕数（射线向 +x：
//! 计数 x_cross > x 的穿越，delta = sign(dy)），覆盖 = mean([W ≠ 0])。
//! 解析算法与之独立推导，二者一致即互证正确。

use amiki_raster::{bin, flatten_draw, render, render_damage, render_tiles, PersistentRenderer};
use amiki_math::{Affine, Vec2};
use amiki_scene::{edit_damage, Color, Damage, Paint, PathBuilder, Scene};

const SS: u32 = 16; // 超采样率（顶点行量化误差 ~2/SS）

/// brute-force 覆盖率（单条路径的设备线段）。
fn coverage_brute(segs: &[[f32; 4]], px: u32, py: u32) -> f32 {
    let mut hits = 0u32;
    for l in 0..SS {
        for k in 0..SS {
            let sx = px as f32 + (k as f32 + 0.5) / SS as f32;
            let sy = py as f32 + (l as f32 + 0.5) / SS as f32;
            let mut w = 0.0f32;
            for &[x0, y0, x1, y1] in segs {
                let dy = y1 - y0;
                if dy.abs() < 1e-9 {
                    continue;
                }
                let (ya, yb) = if dy > 0.0 { (y0, y1) } else { (y1, y0) };
                if sy >= ya && sy < yb {
                    let x_c = x0 + (x1 - x0) * (sy - y0) / dy;
                    if x_c > sx {
                        w += dy.signum();
                    }
                }
            }
            if w != 0.0 {
                hits += 1;
            }
        }
    }
    hits as f32 / (SS * SS) as f32
}

/// 对单个不透明 draw 的场景：CPU 渲染 alpha 通道 vs brute-force 覆盖率。
fn assert_scene_matches(scene: &Scene, tol: f32) {
    assert_eq!(scene.draws.len(), 1, "对拍仅支持单 draw 场景（混合由独立测试覆盖）");
    let binned = bin(scene, 0.05);
    let img = render(&binned);

    let d = &scene.draws[0];
    let segs = flatten_draw(&d.transform, &d.path.els, 0.05);

    let w = scene.width;
    let h = scene.height;
    let mut max_err = 0.0f32;
    for py in 0..h {
        for px in 0..w {
            let expect = coverage_brute(&segs, px, py);
            let got = img[(py * w + px) as usize * 4 + 3] as f32 / 255.0;
            let err = (expect - got).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err <= tol,
        "最大覆盖率误差 {max_err:.4} 超出容差 {tol}"
    );
}

fn solid(color: Color) -> Paint {
    Paint::Solid(color)
}

#[test]
fn axis_aligned_rect() {
    let mut pb = PathBuilder::new();
    pb.rect(20.0, 10.0, 60.0, 50.0);
    let mut scene = Scene::new(80, 64);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 0, 0)));
    assert_scene_matches(&scene, 0.02);
}

#[test]
fn rect_touching_tile_boundaries() {
    // 边恰好落在 tile 边界（16 的倍数）上：分区规则的精确性测试
    let mut pb = PathBuilder::new();
    pb.rect(16.0, 32.0, 48.0, 80.0);
    let mut scene = Scene::new(80, 96);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(0, 255, 0)));
    assert_scene_matches(&scene, 0.02);
}

#[test]
fn rotated_rect() {
    let mut pb = PathBuilder::new();
    pb.rect(-20.0, -14.0, 20.0, 14.0);
    let mut scene = Scene::new(100, 100);
    let t = Affine::translate(Vec2::new(50.0, 50.0))
        .then(Affine::rotate(0.6));
    scene.push(t, pb.build(), solid(Color::rgb8(0, 0, 255)));
    assert_scene_matches(&scene, 0.03);
}

#[test]
fn circle_kappa_cubics() {
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(40.0, 40.0), 28.0, false);
    let mut scene = Scene::new(80, 80);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 128, 0)));
    // 曲线压平误差 + 离散化：容差略宽
    assert_scene_matches(&scene, 0.045);
}

#[test]
fn pentagram_self_intersecting() {
    // 自相交多边形：nonzero 填充（中央五边形 W=2 也必须被 clamp 到 1）
    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(48.0, 48.0), 40.0);
    let mut scene = Scene::new(96, 96);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(200, 0, 200)));
    // 容差 0.25：自相交顶点行中，W=2 区域与 W=0 缺口共存于同一像素时，
    // clamp(|∫W|) 会高估覆盖（见 RENDERING_STRATEGY §1.2 的精确性条件与
    // §4.3 的 M2 修复计划）。W∈{0,1} 的常规情形仍数学精确。
    assert_scene_matches(&scene, 0.25);
}

#[test]
fn donut_with_reversed_hole() {
    // 外圈 + 反绕内圈 = 有洞：洞内 G≈0，必须不被填充
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
    pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
    let mut scene = Scene::new(96, 96);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(0, 200, 100)));
    assert_scene_matches(&scene, 0.045);
}

#[test]
fn scaled_transform() {
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(10.0, 10.0), 5.0, false);
    let mut scene = Scene::new(120, 80);
    let t = Affine::translate(Vec2::new(20.0, 10.0)).then(Affine::scale(4.0));
    scene.push(t, pb.build(), solid(Color::rgb8(255, 255, 0)));
    assert_scene_matches(&scene, 0.045);
}

#[test]
fn two_draws_blend_order() {
    // 红矩形 + 蓝矩形重叠：重叠区必须为蓝（后绘制在上）
    let mut scene = Scene::new(64, 64);
    let mut p1 = PathBuilder::new();
    p1.rect(8.0, 8.0, 40.0, 40.0);
    scene.push(Affine::IDENTITY, p1.build(), solid(Color::rgb8(255, 0, 0)));
    let mut p2 = PathBuilder::new();
    p2.rect(24.0, 8.0, 56.0, 40.0);
    scene.push(Affine::IDENTITY, p2.build(), solid(Color::rgb8(0, 0, 255)));

    let binned = bin(&scene, 0.05);
    let img = render(&binned);
    let at = |x: u32, y: u32| -> [u8; 4] {
        let o = ((y * 64 + x) * 4) as usize;
        [img[o], img[o + 1], img[o + 2], img[o + 3]]
    };
    // 纯红区
    let r = at(16, 24);
    assert_eq!((r[0], r[1], r[2], r[3]), (255, 0, 0, 255), "左侧应为纯红");
    // 重叠区：蓝在上
    let b = at(32, 24);
    assert_eq!((b[0], b[1], b[2], b[3]), (0, 0, 255, 255), "重叠区应为蓝");
    // 画布外无绘制
    let o = at(60, 60);
    assert_eq!(o, [0, 0, 0, 0]);
}

#[test]
fn translucent_blend() {
    // alpha=0.5 的黑条盖在白底上：中心像素 ≈ 0.5 灰
    let mut scene = Scene::new(64, 64);
    let mut p1 = PathBuilder::new();
    p1.rect(0.0, 0.0, 64.0, 64.0);
    scene.push(Affine::IDENTITY, p1.build(), solid(Color::rgb8(255, 255, 255)));
    let mut p2 = PathBuilder::new();
    p2.rect(16.0, 16.0, 48.0, 48.0);
    scene.push(Affine::IDENTITY, p2.build(), solid(Color::rgba8(0, 0, 0, 128)));
    let binned = bin(&scene, 0.05);
    let img = render(&binned);
    let g = img[(32 * 64 + 32) * 4];
    assert!((g as i32 - 128).abs() <= 2, "半透明混合偏差过大: {g}");
}

#[test]
fn seg_area_closed_path_sums() {
    use amiki_raster::seg_area;
    // seg_area 是含折叠项的 tile 局部量，单段无独立语义；
    // 闭合路径求和（carry=0 的 tile）才对应 G。
    // 方形 (0,0)-(10,10)，像素中心 (5.5,5.5)：G 应为 1。
    let g = seg_area(10.0, 0.0, 10.0, 10.0, 5.5, 5.5)   // 右边
        + seg_area(0.0, 10.0, 0.0, 0.0, 5.5, 5.5);       // 左边（闭合绕向）
    assert!((g - 1.0).abs() < 1e-5, "内部像素 G 应为 1：{g}");
    // 像素在方形右侧（中心 15.5）：G 应为 0
    let g = seg_area(10.0, 0.0, 10.0, 10.0, 15.5, 5.5)
        + seg_area(0.0, 10.0, 0.0, 0.0, 15.5, 5.5);
    assert!(g.abs() < 1e-5, "外部像素 G 应为 0：{g}");
    // 边恰在像素中心 → 半覆盖：矩形 (4.5,0)-(20,100)，像素中心 (4.5,50.5)
    let g = seg_area(20.0, 0.0, 20.0, 100.0, 4.5, 50.5)
        + seg_area(4.5, 100.0, 4.5, 0.0, 4.5, 50.5);
    assert!((g - 0.5).abs() < 1e-5, "中心穿越应半覆盖：{g}");
}

#[test]
fn reversed_circle_is_the_same_disk() {
    // 语义回归：反绕圆必须与正向圆填充同一圆盘（控制点互换正确性）。
    let render_disk = |reversed: bool| -> Vec<u8> {
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, reversed);
        let mut scene = Scene::new(96, 96);
        scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 0, 0)));
        render(&bin(&scene, 0.05))
    };
    let fwd = render_disk(false);
    let rev = render_disk(true);
    let mut max_diff = 0u32;
    for (a, b) in fwd.iter().zip(rev.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
    }
    assert!(max_diff <= 2, "反绕圆与正向圆渲染差异 {max_diff} 超出容差");
}

#[test]
fn blend_known_values() {
    use amiki_raster::composite;
    // multiply：红 × 白 = 红；红 × 绿 = 黑
    let r = [1.0f32, 0.0, 0.0];
    assert!(composite([1.0, 1.0, 1.0, 1.0], [r[0], r[1], r[2], 1.0], 1.0, 1)[..3]
        .iter().zip([1.0, 0.0, 0.0]).all(|(a, b)| (a - b).abs() < 1e-5));
    // screen：黑 screen X = X
    assert!(composite([0.2, 0.4, 0.6, 1.0], [0.0, 0.0, 0.0, 1.0], 1.0, 2)[..3]
        .iter().zip([0.2, 0.4, 0.6]).all(|(a, b)| (a - b).abs() < 1e-5));
    // difference：白 difference 灰0.5 = 0.5
    assert!(composite([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 1.0], 1.0, 10)[..3]
        .iter().all(|a| (a - 0.5).abs() < 1e-5));
    // alpha 合成不受模式影响的 α 通道：αo = αs + αb(1−αs)
    let o = composite([0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5], 0.5, 1);
    assert!((o[3] - 0.75).abs() < 1e-5, "αo={}", o[3]);
}

#[test]
fn blend_scene_render() {
    // 白底 + multiply 红圆 = 红色（multiply 与白底相乘保持源色）
    let mut scene = Scene::new(48, 48);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 48.0, 48.0);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 255, 255)));
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(24.0, 24.0), 16.0, false);
    scene.push_blend(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 0, 0)), amiki_scene::BlendMode::Multiply);
    let img = render(&bin(&scene, 0.05));
    let c = img[(24 * 48 + 24) * 4..(24 * 48 + 24) * 4 + 4].to_vec();
    assert!(c[0] > 240 && c[1] < 15 && c[2] < 15 && c[3] == 255, "multiply 中心应为纯红: {c:?}");
}

#[test]
fn clip_rect_masks_circle() {
    // 圆被矩形裁剪：clip 外的圆部分必须透明
    let mut scene = Scene::new(64, 64);
    let mut pb = PathBuilder::new();
    pb.rect(8.0, 8.0, 56.0, 56.0);
    let clip_idx = scene.push_clip(Affine::IDENTITY, pb.build());
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(32.0, 32.0), 28.0, false);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(0, 120, 255)));
    scene.draws.last_mut().unwrap().clips = vec![clip_idx];
    let img = render(&bin(&scene, 0.05));
    let at = |x: u32, y: u32| -> u8 { img[((y * 64 + x) * 4) as usize + 3] };
    assert_eq!(at(6, 32), 0, "clip 外（左）应透明");
    assert_eq!(at(58, 32), 0, "clip 外（右）应透明");
    assert_eq!(at(32, 12), 255, "clip 内应填充");
}

#[test]
fn nested_clips_intersect() {
    // 两层裁剪取交集：左半矩形 ∩ 上半矩形 = 左上角
    let mut scene = Scene::new(64, 64);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 32.0, 64.0);
    let c0 = scene.push_clip(Affine::IDENTITY, pb.build());
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 64.0, 32.0);
    let c1 = scene.push_clip(Affine::IDENTITY, pb.build());
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 64.0, 64.0);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(0, 120, 255)));
    scene.draws.last_mut().unwrap().clips = vec![c0, c1];
    let img = render(&bin(&scene, 0.05));
    let at = |x: u32, y: u32| -> u8 { img[((y * 64 + x) * 4) as usize + 3] };
    assert_eq!(at(16, 16), 255, "交集内应填充");
    assert_eq!(at(48, 16), 0, "clip0 之外应透明");
    assert_eq!(at(16, 48), 0, "clip1 之外应透明");
}

#[test]
fn stream_round_trip_renders_identically() {    // M2c：编码→解码→渲染 必须与原场景逐字节一致
    let mut scene = Scene::new(72, 72);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 72.0, 72.0);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(240, 220, 200)));
    let mut pb = PathBuilder::new();
    pb.rect(12.0, 12.0, 60.0, 60.0);
    let clip = scene.push_clip(Affine::rotate(0.2), pb.build());
    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(36.0, 36.0), 30.0);
    scene.push_blend(
        Affine::IDENTITY,
        pb.build(),
        solid(Color::rgba8(255, 0, 0, 220)),
        amiki_scene::BlendMode::Multiply,
    );
    scene.draws.last_mut().unwrap().clips = vec![clip];
    let bytes = amiki_scene::encode_scene(&scene);
    let back = amiki_scene::decode_scene(&bytes).expect("解码失败");
    let img_a = render(&bin(&scene, 0.05));
    let img_b = render(&bin(&back, 0.05));
    assert_eq!(img_a, img_b, "编码往返后渲染不一致");
}

// ============================================================================
// M4：damage region 增量渲染的正确性铁律
// ============================================================================

/// 覆盖各类 tile 边界的场景（矩形 / 曲线+洞 / 自相交 / 越界 / 裁剪+混合）。
fn damage_scene() -> Scene {
    let mut scene = Scene::new(96, 96);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 96.0, 96.0);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(240, 220, 200)));

    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(30.0, 30.0), 22.0, false);
    pb.circle(Vec2::new(30.0, 30.0), 10.0, true);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgba8(30, 160, 90, 220)));

    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(64.0, 64.0), 28.0);
    scene.push_blend(
        Affine::IDENTITY,
        pb.build(),
        solid(Color::rgba8(230, 120, 60, 200)),
        amiki_scene::BlendMode::Multiply,
    );

    // 越出画布四缘（tile 钳制路径）
    let mut pb = PathBuilder::new();
    pb.rect(-20.0, 40.0, 20.0, 70.0);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(52, 152, 219)));

    // 裁剪 + 旋转（斜边跨带）
    let mut pb = PathBuilder::new();
    pb.rect(10.0, 10.0, 86.0, 60.0);
    let clip = scene.push_clip(Affine::rotate(0.2), pb.build());
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(48.0, 48.0), 30.0, false);
    scene.push(Affine::IDENTITY, pb.build(), solid(Color::rgba8(200, 60, 140, 200)));
    scene.draws.last_mut().unwrap().clips = vec![clip];

    scene
}

/// 种子化 LCG（测试可复现，不引入 rand 依赖）。
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// **INV-3**：任意 invalidate 序列下，增量渲染必须与全量渲染**逐字节一致**。
///
/// 为什么必须用**随机序列**而不是几个样例：damage region 最典型的 bug 是
/// "特定序列下才错"（例如某次并集恰好没覆盖到某个像素所在的 tile）。
/// 样例测试抓不住这类漏覆盖，随机化 + 多轮才能。
#[test]
fn damage_incremental_matches_full_random_sequences() {
    let scene = damage_scene();
    let binned = bin(&scene, 0.05);
    // 参考：全量渲染（唯一真值）
    let full = render(&binned);
    let (w, h) = (scene.width, scene.height);
    assert_eq!(full.len(), (w * h * 4) as usize);

    let mut rng = Lcg(0xDEAD_BEEF_1234_5678);
    let mut worst_tiles = 0u32;
    for round in 0..200 {
        let mut damage = Damage::new(w, h);
        damage.clear(); // 进入增量模式（无脏区）
        // 每次 1..=6 个随机矩形，可能重叠、可能越界、可能退化
        let n_rects = 1 + (rng.next_u32() % 6);
        for _ in 0..n_rects {
            let x0 = (rng.next_u32() % (w + 40)) as f32 - 20.0;
            let y0 = (rng.next_u32() % (h + 40)) as f32 - 20.0;
            let rw = (rng.next_u32() % 40) as f32;
            let rh = (rng.next_u32() % 40) as f32;
            damage.invalidate_rect(x0, y0, x0 + rw, y0 + rh);
        }
        worst_tiles = worst_tiles.max(damage.tile_count());

        // 增量：起点取全量结果（模拟"上一帧已经是对的"）
        let mut img = full.clone();
        render_damage(&binned, &mut img, &damage);

        assert_eq!(
            img, full,
            "第 {round} 轮：增量渲染与全量不一致（脏 tile {} 个）—— 存在漏覆盖",
            damage.tile_count()
        );
    }
    assert!(worst_tiles > 0, "随机序列应至少产生过脏 tile，否则测试没有判别力");
    eprintln!("200 轮随机 invalidate 序列全部与全量逐字节一致（最大脏 tile {worst_tiles}）");
}

/// 边界情形：无脏区时必须**完全跳过**（结果 = 传入的上一帧缓冲原样）。
#[test]
fn damage_empty_skips_everything() {
    let scene = damage_scene();
    let binned = bin(&scene, 0.05);
    let full = render(&binned);
    let (w, h) = (scene.width, scene.height);

    let mut damage = Damage::new(w, h);
    damage.clear();
    assert!(damage.is_empty());

    // 故意传入"被污染的"缓冲：若无脏区却有任何写入，断言会立刻失败
    let mut img = vec![7u8; full.len()];
    render_damage(&binned, &mut img, &damage);
    assert!(img.iter().all(|&b| b == 7), "无脏区时不得写入任何像素");
}

/// 全脏时必须与全量渲染等价（`Damage::new` 的默认态）。
#[test]
fn damage_full_equals_full_render() {
    let scene = damage_scene();
    let binned = bin(&scene, 0.05);
    let full = render(&binned);
    let (w, h) = (scene.width, scene.height);

    let damage = Damage::new(w, h);
    assert!(damage.is_full());
    let mut img = vec![0u8; full.len()];
    render_damage(&binned, &mut img, &damage);
    assert_eq!(img, full, "全脏渲染必须等于全量渲染");
}

// ============================================================================
// M4 编辑契约：随机改动场景，用 edit_damage 推导脏区并验证增量 == 全量
// ============================================================================

const EDIT_W: u32 = 112;
const EDIT_H: u32 = 112;

/// 一条可随机增删改的绘制。
#[derive(Clone, Copy, Debug)]
enum Edit {
    Rect { x: f32, y: f32, w: f32, h: f32, c: Color },
    Circle { x: f32, y: f32, r: f32, c: Color },
    Pentagram { x: f32, y: f32, r: f32, c: Color },
}

impl Edit {
    fn push_into(&self, scene: &mut Scene) {
        let mut pb = PathBuilder::new();
        let (t, paint) = match *self {
            Edit::Rect { x, y, w, h, c } => {
                pb.rect(x, y, x + w, y + h);
                (Affine::IDENTITY, Paint::Solid(c))
            }
            Edit::Circle { x, y, r, c } => {
                pb.circle(Vec2::new(x, y), r, false);
                (Affine::IDENTITY, Paint::Solid(c))
            }
            Edit::Pentagram { x, y, r, c } => {
                pb.pentagram(Vec2::new(x, y), r);
                (Affine::IDENTITY, Paint::Solid(c))
            }
        };
        scene.push(t, pb.build(), paint);
    }
}

fn build_edit_scene(edits: &[Edit]) -> Scene {
    let mut scene = Scene::new(EDIT_W, EDIT_H);
    // 底：整幅不透明背景，保证"删除"也有可观察的变化
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, EDIT_W as f32, EDIT_H as f32);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(250, 246, 240)));
    for e in edits {
        e.push_into(&mut scene);
    }
    scene
}

fn random_edit(rng: &mut Lcg, w: u32, h: u32) -> Edit {
    let x = (rng.next_u32() % w) as f32;
    let y = (rng.next_u32() % h) as f32;
    let r = 6.0 + (rng.next_u32() % 22) as f32;
    let c = Color::rgba8(
        (rng.next_u32() % 256) as u8,
        (rng.next_u32() % 256) as u8,
        (rng.next_u32() % 256) as u8,
        120 + (rng.next_u32() % 136) as u8,
    );
    match rng.next_u32() % 3 {
        0 => Edit::Rect { x, y, w: 8.0 + (rng.next_u32() % 30) as f32, h: 8.0 + (rng.next_u32() % 30) as f32, c },
        1 => Edit::Circle { x, y, r, c },
        _ => Edit::Pentagram { x, y, r, c },
    }
}

/// `edit_damage` 的**覆盖性**：任何发生变化的像素，都必须落在脏 tile 范围内。
///
/// 这是整个增量方案的地基——若此断言失败，说明有人会漏标、画面会残留陈旧像素。
#[test]
fn edit_damage_covers_all_changed_pixels() {
    let mut rng = Lcg(0x1234_ABCD_5678_EF01);
    let mut covered_rounds = 0;

    for round in 0..40 {
        // 随机生成"旧"—"新"两个场景（同尺寸，所以比较是有意义的）
        let n_old = rng.next_u32() % 5;
        let n_new = rng.next_u32() % 5;
        let old_edits: Vec<Edit> = (0..n_old).map(|_| random_edit(&mut rng, EDIT_W, EDIT_H)).collect();
        let new_edits: Vec<Edit> = (0..n_new).map(|_| random_edit(&mut rng, EDIT_W, EDIT_H)).collect();
        let old = build_edit_scene(&old_edits);
        let new = build_edit_scene(&new_edits);

        let damage = edit_damage(&old, &new);
        let (w, h) = (new.width, new.height);

        let b_old = bin(&old, 0.05);
        let b_new = bin(&new, 0.05);
        let img_old = render(&b_old);
        let img_new = render(&b_new);

        // 1) 覆盖性：所有变化像素必须落在脏 tile 内
        let tiles: Vec<(u32, u32)> = match damage.tiles() {
            // 全脏 = 覆盖一切
            None => (0..h.div_ceil(16)).flat_map(|ty| (0..w.div_ceil(16)).map(move |tx| (tx, ty))).collect(),
            Some((tx0, ty0, tx1, ty1)) => {
                (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect()
            }
        };
        let set: std::collections::HashSet<(u32, u32)> = tiles.iter().copied().collect();
        let mut changed_px = 0u32;
        let mut uncovered = 0u32;
        for py in 0..h {
            for px in 0..w {
                let o = ((py * w + px) * 4) as usize;
                if img_old[o..o + 4] != img_new[o..o + 4] {
                    changed_px += 1;
                    if !set.contains(&(px / 16, py / 16)) {
                        uncovered += 1;
                    }
                }
            }
        }
        if changed_px > 0 {
            covered_rounds += 1;
        }
        assert_eq!(
            uncovered, 0,
            "第 {round} 轮：有 {uncovered} 个变化像素不在脏 tile 内（共变化 {changed_px}）——\
             edit_damage 漏覆盖，增量渲染会残留陈旧像素"
        );

        // 2) 增量正确性：以"上一帧结果"为底，只重算脏 tile，必须与全量逐字节一致。
        //    （不能用 render_tiles 返回的全新缓冲去比全量渲染——那在未脏 tile 上是
        //     零初始化的透明像素，与"复用上一帧"无关，比较本身没有意义。）
        let mut carried = img_old.clone();
        render_damage(&b_new, &mut carried, &damage);
        assert_eq!(
            carried, img_new,
            "第 {round} 轮：在上帧结果上按脏区增量更新后与全量不一致"
        );

        // 3) `render_tiles` 本身的自洽性：以全量为底、只重算"全部 tile"，
        //    必须仍是全量（验证 clear+shade 对每个 tile 都是幂等的）
        let all: Vec<(u32, u32)> = (0..h.div_ceil(16))
            .flat_map(|ty| (0..w.div_ceil(16)).map(move |tx| (tx, ty)))
            .collect();
        let mut base = img_new.clone();
        let tiles_full = render_tiles(&b_new, Some(&all));
        // render_tiles 返回的是"只含这些 tile"的图，未列出的 tile 为透明；
        // 这里只需验证它列出的 tile 与全量一致
        for &(tx, ty) in &all {
            for py in ty * 16..((ty + 1) * 16).min(h) {
                for px in tx * 16..((tx + 1) * 16).min(w) {
                    let o = ((py * w + px) * 4) as usize;
                    assert_eq!(
                        tiles_full[o..o + 4],
                        img_new[o..o + 4],
                        "第 {round} 轮：render_tiles 在 tile ({tx},{ty}) 像素 ({px},{py}) 与全量不一致"
                    );
                }
            }
        }
        let _ = &mut base;
    }

    assert!(covered_rounds >= 20, "至少一半轮次应有真实像素变化，否则测试没有判别力（实际 {covered_rounds}）");
    eprintln!("40 轮随机场景编辑：覆盖性与增量一致性全部通过（{covered_rounds} 轮有真实像素变化）");
}

// ============================================================================
// M4 帧间持久化：真的省掉工作，同时结果与"从零重算"逐字节一致
// ============================================================================

/// 场景未变时必须**零着色**（静态画面的收益就在这里）。
#[test]
fn persistence_skips_unchanged_scene() {
    let scene = damage_scene();
    let mut r = PersistentRenderer::new(0.05);

    r.update(&scene);
    let first = r.last_stats();
    assert!(!first.skipped, "首帧必须真实渲染");
    assert!(first.shaded_tiles > 0);
    let img_first = r.image().to_vec();

    // 再提交同一场景若干次：每次都必须完全跳过
    for round in 0..5 {
        r.update(&scene);
        let st = r.last_stats();
        assert!(st.skipped, "第 {round} 次重复提交未跳过（场景未变）");
        assert_eq!(st.shaded_tiles, 0, "跳过的帧不得着色任何 tile");
        assert_eq!(st.saved_ratio(), 1.0);
        assert_eq!(r.image(), img_first.as_slice(), "跳过的帧不得改变任何像素");
    }
    eprintln!(
        "静态场景：首帧着色 {}/{} tile，之后 5 帧全部零着色",
        first.shaded_tiles, first.total_tiles
    );
}

/// 随机编辑序列下，持久渲染器的输出必须**逐帧**等于"从零全量重算"。
#[test]
fn persistence_matches_full_render_across_edits() {
    let mut rng = Lcg(0xFEED_FACE_CAFE_BEEF);
    let mut edits: Vec<Edit> = (0..3).map(|_| random_edit(&mut rng, EDIT_W, EDIT_H)).collect();
    let mut r = PersistentRenderer::new(0.05);

    let mut frames_with_change = 0;
    let mut total_shaded = 0u64;
    let mut total_full = 0u64;

    for round in 0..60 {
        // 每帧随机做一次编辑：增 / 删 / 改
        let op = rng.next_u32() % 3;
        match (op, edits.len()) {
            (0, _) => edits.push(random_edit(&mut rng, EDIT_W, EDIT_H)),
            (1, n) if n > 0 => {
                let i = (rng.next_u32() as usize) % n;
                edits.remove(i);
            }
            (_, n) if n > 0 => {
                let i = (rng.next_u32() as usize) % n;
                edits[i] = random_edit(&mut rng, EDIT_W, EDIT_H);
            }
            _ => edits.push(random_edit(&mut rng, EDIT_W, EDIT_H)),
        }

        let scene = build_edit_scene(&edits);
        r.update(&scene);

        // 参照：从零全量重算
        let reference = render(&bin(&scene, 0.05));
        assert_eq!(
            r.image(),
            reference.as_slice(),
            "第 {round} 轮（op={op}, {} draws）：持久渲染器输出与全量重算不一致",
            edits.len()
        );

        let st = r.last_stats();
        total_shaded += st.shaded_tiles as u64;
        total_full += st.total_tiles as u64;
        if !st.skipped {
            frames_with_change += 1;
        }
    }

    assert!(frames_with_change >= 40, "绝大多数帧应有变化（实际 {frames_with_change}）");
    let saved = 1.0 - (total_shaded as f64 / total_full as f64);
    eprintln!(
        "60 帧随机编辑：逐帧与全量重算一致；累计着色 {total_shaded}/{total_full} tile，省下 {:.1}%",
        saved * 100.0
    );
    // 每次只改一条 draw，脏区远小于全画布 → 必须省下可观的工作
    assert!(saved > 0.30, "每次只改一条 draw，应省下 >30% 的 tile 着色（实际 {:.1}%）", saved * 100.0);
}
