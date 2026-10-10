//! 派发维度与脏 tile 表（含 0.10.9 的 4K 尾部回归钉子）。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

/// 0.6.1：脏 tile 索引表的构造（**纯函数**，不需要 GPU 适配器）。
///
/// 这张表是 fine 1D 派发的唯一输入，算错就等于画错 tile，因此把它的语义钉死：
/// 全量 = 整幅画布 row-major；局部 = 脏矩形覆盖的 tile；空脏区 = 空表。
#[test]
fn dirty_tile_list_covers_exactly_the_dirty_rect() {
    use amiki_scene::Damage;

    // 打包 (tx, ty) → 表项（与 `dirty_tile_list` 的约定一致）
    let p = |tx: u32, ty: u32| tx | (ty << 16);
    let full = vec![p(0, 0), p(1, 0), p(2, 0), p(0, 1), p(1, 1), p(2, 1)];
    // 全量（damage = None）：整幅画布，row-major
    assert_eq!(dirty_tile_list(None, 3, 2), full);

    // 全脏（`Damage::new` 的默认态）与 None 等价
    let d_all = Damage::new(48, 32);
    assert!(d_all.is_full());
    assert_eq!(dirty_tile_list(Some(&d_all), 3, 2), full);

    // 局部矩形 tile (1,0)..(3,1)：只有两项，且**起点不是 0**
    let mut d = Damage::new(48, 32);
    d.clear();
    d.invalidate_rect(16.0, 0.0, 48.0, 16.0);
    assert_eq!(d.tiles(), Some((1, 0, 3, 1)));
    assert_eq!(dirty_tile_list(Some(&d), 3, 2), vec![1, 2]);

    // 空脏区 → 空表（调用方据此**不派发**任何 workgroup）
    let mut d0 = Damage::new(48, 32);
    d0.clear();
    assert!(dirty_tile_list(Some(&d0), 3, 2).is_empty());

    // 越界脏区被钳到画布范围内（防御性：调用方已由 check_damage_bounds 拦过）
    let mut d_out = Damage::new(48, 32);
    d_out.clear();
    d_out.invalidate_rect(-100.0, -100.0, 1000.0, 1000.0);
    assert_eq!(dirty_tile_list(Some(&d_out), 3, 2).len(), 6);
}

/// 0.6.1：1D 派发维度 —— x 每批最多 65535 个 workgroup，超出用 z 维分批。
///
/// 这条路径在测试里跑不到 z > 1（需要 > 4096×4096 的画布），所以按纯函数钉住
/// 算术：否则 8K 以上的画布会在**没有任何测试覆盖**的情况下静默少算一批。
#[test]
fn fine_dispatch_dims_batches_beyond_the_1d_workgroup_limit() {
    assert_eq!(fine_dispatch_dims(1), (1, 1));
    assert_eq!(fine_dispatch_dims(65535), (65535, 1));
    assert_eq!(fine_dispatch_dims(65536), (65535, 2));
    assert_eq!(fine_dispatch_dims(65535 * 2), (65535, 2));
    assert_eq!(fine_dispatch_dims(65535 * 2 + 1), (65535, 3));
    // 8K 画布：7680×4320 → 480×270 = 129 600 个 tile
    assert_eq!(fine_dispatch_dims(480 * 270), (65535, 2));
    // 0 也必须给出合法维度（workgroup 数不能为 0）
    assert_eq!(fine_dispatch_dims(0), (1, 1));
}

/// **纯算术**：单维派发的分批必须让每个元素**恰好被处理一次**。
///
/// 这是那个 4K 静默错误的钉子：宿主按"每行 65535 个**工作组**"推进，而内核的
/// `lin()` 按"每行 65535 个**元素**"解码 —— 行与行重叠 95%，尾部 62 385 个元素
/// 从未被处理。本测试不需要 GPU，所以它对任何环境都有效。
#[test]
fn dispatch_dims_1d_covers_every_element_exactly_once() {
    /// 与内核 `lin()` 的行距一致（见 `dispatch_dims_1d` 的文档）。
    const ROW: u64 = 65_536;
    const WG: u64 = 64;

    // 取值覆盖：小尺寸、行边界两侧、旧的 65535 工作组边界、以及旧 bug 的确切触发点。
    for n in [
        1u32,
        63,
        64,
        65,
        1_024 * 64,
        1_024 * 64 + 1,
        65_535,
        65_536,
        65_537,
        4_194_240,
        4_194_241,
        4_259_775,
        4_322_160,
        10_000_000,
        65_535 * 65_536,
    ] {
        let (x, y) = dispatch_dims_1d(n);
        assert!(x >= 1 && y >= 1, "n={n}：派发维度不能为 0");
        assert!(x <= 1_024, "n={n}：x={x} 超过一行");
        assert!(x <= 65_535 && y <= 65_535, "n={n}：超出单维上限");

        let per_row = x as u64 * WG;
        if y == 1 {
            assert!(per_row >= n as u64, "n={n}：单行没盖住");
        } else {
            assert_eq!(per_row, ROW, "n={n}：多行时每行必须正好一行，否则会重叠或留空隙");
            assert!(
                (y as u64 - 1) * ROW + per_row >= n as u64,
                "n={n}：最后一行没盖到尾部（尾部元素会永远不被处理）"
            );
        }
        // 不做无谓的放大：总量最多是需要的两倍（只在行边界附近发生）
        let wg = (n as u64).div_ceil(WG);
        assert!((x as u64) * (y as u64) <= 2 * wg, "n={n}：派发的总量浪费过多");
    }

    // 旧 bug 的确切形状：3840×2160 / 2000 矩形 ⇒ carry 需要 4 322 160 个元素
    assert_eq!(dispatch_dims_1d(4_322_160), (1_024, 66));
    assert_eq!(dispatch_dims_1d(0), (0, 0), "0 元素 ⇒ 调用方跳过派发");
}

/// **回归钉子（0.10.9）**：`draws × tile 行` 够大时，稀疏路径的 1D 派发会拆成二维；
/// 只要宿主与内核的行距不一致，**尾部 strip 就永远不会被处理** —— 表现是画面最下面
/// 若干行瓦片整片空白（0.10.8 把它记成了"GPU 在 `y ≥ 2048` 会算错"）。
///
/// 这里用最小的复现形状：256×2160（135 个 tile 行）× 2000 个矩形 ⇒
/// `draws × tile 行 = 270 135 > 262 140`（= `65535×64/16`），carry 需要
/// 4 322 160 个元素 = 67 534 个工作组 ⇒ **必然触发二维分批**。
#[test]
fn tall_canvas_with_many_draws_matches_cpu_in_the_bottom_rows() {
    use amiki_math::Affine;
    use amiki_scene::{Color, Paint, PathBuilder, Scene};

    let Some((renderer, _gpu_guard)) = gpu() else { return };

    let (w, h, n) = (256u32, 2160u32, 2000u32);
    let mut scene = Scene::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(250, 250, 252)));
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let col = i % side;
        let row = i / side;
        let cw = w as f32 / side as f32;
        let ch = h as f32 / side as f32;
        let (x0, y0) = (col as f32 * cw + cw * 0.15, row as f32 * ch + ch * 0.15);
        let mut pb = PathBuilder::new();
        pb.rect(x0, y0, x0 + cw * 0.7, y0 + ch * 0.7);
        let t = i as f32 / n as f32;
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color { r: t, g: 1.0 - t, b: (t * 3.0) % 1.0, a: 1.0 }),
        );
    }

    let gpu_px = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
    let cpu_px = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

    let max_diff = gpu_px
        .iter()
        .zip(cpu_px.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 2,
        "GPU 与 CPU 最大通道差 {max_diff} 超出容差 —— 尾部 tile 可能又被丢掉了"
    );

    // 最下面一行必须真的画了东西（旧 bug 时这里是全透明）
    let bottom = ((h - 1) * w + w / 2) as usize * 4;
    assert!(
        gpu_px[bottom + 3] > 0,
        "最下面一行是透明的：尾部 strip 又被丢掉了（1D 派发的行距与 lin() 不一致？）"
    );
    eprintln!("4K 尾部回归：{w}x{h}/{n} 矩形，最大通道差 {max_diff}");
}

