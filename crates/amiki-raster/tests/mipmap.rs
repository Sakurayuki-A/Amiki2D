//! B4：mipmap 的光栅器级验证。
//!
//! 这些测试走**真实的 bin + render 路径**（不是直接调采样函数），
//! 钉住五件事：
//! 1. 大幅缩小时摩尔纹被消除（任务书验收第 3 条，量化判定而非目视）
//! 2. 平铺的 LOD 按**单个平铺单元**算，不被平铺次数放大（任务书 2.4）
//! 3. `mip_bias` 生效且可回退到 level 0
//! 4. 1:1 渲染与无 mip **逐字节一致**（硬约束"1:1 行为不变"）
//! 5. 帧间持久化路径与全量路径用**同一个 bias**（含"改了 bias 必须重画"）

use amiki_math::{Affine, Vec2};
use amiki_raster::{bin, render, PersistentRenderer};
use amiki_scene::{Image, ImagePlacement, Paint, PathBuilder, Scene};

/// 8×8 像素黑白棋盘格，拼成 `cells × cells` 格的大图。**带 mip 链**。
fn checker_image(cells: u32) -> Image {
    let w = cells * 8;
    let h = cells * 8;
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / 8) + (y / 8)) % 2 == 0;
            let v = if on { 255u8 } else { 0u8 };
            data.extend([v, v, v, 255]);
        }
    }
    Image::from_rgba8_with_mipmap(w, h, data).expect("图像数据合法")
}

fn render_image_scene(img: &std::sync::Arc<Image>, place: ImagePlacement, size: u32) -> Vec<u8> {
    let mut scene = Scene::new(size, size);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, size as f32, size as f32);
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Image { image: img.clone(), placement: place },
    );
    render(&bin(&scene, 0.05))
}

/// 缩小到**奈奎斯特极限之后**（格子周期死于链的深层）：
/// 8px 棋盘在 level 4（每纹素 2×2 格）起变成均匀灰。采样落在 lod ≈ 4.5
/// 时，三线性只会在两个均匀层之间插值 → 整幅均匀灰；
/// 无 mip 时每个屏幕像素砸在某个原纹素上 → 0/255 椒盐噪声（摩尔纹）。
#[test]
fn downscale_past_nyquist_kills_the_moire() {
    // 256×256 图（32×32 格）缩到 11×11：LOD = log2(256/11) ≈ 4.54
    let img = std::sync::Arc::new(checker_image(32));
    let place = ImagePlacement::rect(0.0, 0.0, 11.0, 11.0);

    let with_mip = render_image_scene(&img, place, 11);
    let img_plain =
        std::sync::Arc::new(Image::from_rgba8(img.width, img.height, img.data.clone()).unwrap());
    let without_mip = render_image_scene(&img_plain, place, 11);

    // 有 mip：整幅 R 通道一个值，且是**线性空间**平均出来的灰
    // （sRGB(线性 0.5) ≈ 188 —— 若是 ~128 就说明有人在编码空间平均）
    let (mn, mx) = (0..11 * 11)
        .map(|i| with_mip[i * 4])
        .fold((u8::MAX, 0u8), |a, b| (a.0.min(b), a.1.max(b)));
    assert_eq!(mn, mx, "格子死于 level 4 → 输出应均匀，实际范围 {mn}..{mx}");
    assert!(
        (186..=190).contains(&mn),
        "均匀灰应 ≈ 188（线性平均），实际 {mn}（~128 即编码空间平均，~0/255 即未滤波）"
    );

    // 无 mip：椒盐噪声
    let (mn, mx) = (0..11 * 11)
        .map(|i| without_mip[i * 4])
        .fold((u8::MAX, 0u8), |a, b| (a.0.min(b), a.1.max(b)));
    assert!(mx - mn > 100, "无 mip 应存在摩尔纹，实际范围 {mn}..{mx}");
}

/// 平铺的 LOD 按**单个平铺单元**算（任务书 2.4）。
///
/// `u_axis` 是**一个平铺单元**的屏幕跨度。同一张 64×64 图：
/// A：平铺，单元 16px，铺满 32×32 画布（uv 走 0..2，取模回绕两次）
/// B：单次放置 16×16（uv 走 0..1）
/// 两者的 uv 梯度相同（每屏幕像素 1/16 uv）→ LOD 相同 →
/// 同一 uv 处（各自的 (0.5,0.5)）像素值必须一致。
#[test]
fn tiled_lod_is_computed_per_tile_not_per_screen() {
    let mut data = Vec::with_capacity(64 * 64 * 4);
    for y in 0..64u32 {
        for x in 0..64u32 {
            data.extend([(x * 4) as u8, (y * 4) as u8, 128, 255]);
        }
    }
    let img = std::sync::Arc::new(Image::from_rgba8_with_mipmap(64, 64, data).unwrap());

    // A：平铺，单元 16×16，2×2 铺满 32px（LOD 2）
    let tiled = ImagePlacement {
        origin: Vec2::new(0.0, 0.0),
        u_axis: Vec2::new(16.0, 0.0),
        v_axis: Vec2::new(0.0, 16.0),
        repeat: true,
    };
    let out_tiled = render_image_scene(&img, tiled, 32);

    // B：单个单元 16×16 放在画布中央（LOD 2，同一梯度）
    let single = ImagePlacement::rect(8.0, 8.0, 24.0, 24.0);
    let out_single = render_image_scene(&img, single, 32);

    // A 的 (8,8)：uv = (0.5,0.5)；B 的 (16,16)：uv = (0.5,0.5)
    let a = &out_tiled[(8 * 32 + 8) * 4..][..4];
    let b = &out_single[(16 * 32 + 16) * 4..][..4];
    let diff =
        a.iter().zip(b.iter()).map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs()).max().unwrap();
    assert!(diff <= 2, "平铺与单单元的同一 uv 处应一致（LOD 不得被平铺次数放大），最大差 {diff}");

    // 反向对照：若 LOD 真被"总覆盖范围"放大（错误实现按 32px 算出更深的层），
    // A 的像素会取到更糊的层 → 与 B 不同。渐变图对此敏感。
    let wrong_lod_pixel = &out_tiled[..4]; // 像素 (0,0)，uv=(0,0) 回绕处
    let at_uv0 = &out_single[(8 * 32 + 8) * 4..][..4]; // B 的 uv=(0,0)
    let d = wrong_lod_pixel
        .iter()
        .zip(at_uv0.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
        .max()
        .unwrap();
    assert!(d <= 2, "平铺取模处的 uv=(0,0) 也应与单单元一致，最大差 {d}");
}

/// `mip_bias`：±1 采不同层（输出必须变化）；够大的负 bias 回到 level 0，
/// 与无 mip 的双线路径**逐字节一致**。
///
/// 用**伪随机噪声图**：线性渐变、棋盘这类自相似/线性的图案，不同层的
/// 面积平均可能落在同一画布值上，分辨不出 bias 是否生效；噪声对各层的
/// 响应真正不同。
#[test]
fn mip_bias_shifts_levels_and_negative_bias_falls_back_to_level_zero() {
    let mut data = Vec::with_capacity(64 * 64 * 4);
    for i in 0..64 * 64 {
        let v = (i * 7 % 251) as u8;
        data.extend([v, v, v, 255]);
    }
    let img = std::sync::Arc::new(Image::from_rgba8_with_mipmap(64, 64, data).unwrap());
    let place = ImagePlacement::rect(0.0, 0.0, 16.0, 16.0); // 缩小 4× → LOD 2

    let render_with_bias = |bias: f32| {
        let mut scene = Scene::new(16, 16);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 16.0, 16.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Image { image: img.clone(), placement: place });
        let mut b = bin(&scene, 0.05);
        b.mip_bias = bias;
        render(&b)
    };

    let base = render_with_bias(0.0);
    let sharper = render_with_bias(-1.0);
    let blurrier = render_with_bias(1.0);
    assert_ne!(base[..100], sharper[..100], "bias -1（更锐）应改变输出");
    assert_ne!(base[..100], blurrier[..100], "bias +1（更糊）应改变输出");

    // 极大负 bias：钳回 level 0 = 双线性，与无 mip 图像逐字节一致
    let img_plain =
        std::sync::Arc::new(Image::from_rgba8(64, 64, img.data.clone()).unwrap());
    let mut scene = Scene::new(16, 16);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 16.0, 16.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Image { image: img_plain, placement: place });
    let plain = render(&bin(&scene, 0.05));
    let back_to_l0 = render_with_bias(-100.0);
    assert_eq!(back_to_l0, plain, "bias 钳回 level 0 后必须与无 mip 路径逐字节一致");
}

/// 硬约束：1:1（LOD 0）时，带 mip 与不带 mip 的整幅输出**逐字节一致**。
#[test]
fn one_to_one_render_is_byte_identical_with_and_without_mip() {
    let img = std::sync::Arc::new(checker_image(4)); // 32×32
    let place = ImagePlacement::rect(2.0, 3.0, 34.0, 35.0); // 原尺寸放置（含非整数偏移）
    let with_mip = render_image_scene(&img, place, 40);
    let img_plain =
        std::sync::Arc::new(Image::from_rgba8(img.width, img.height, img.data.clone()).unwrap());
    let without_mip = render_image_scene(&img_plain, place, 40);
    assert_eq!(with_mip, without_mip, "1:1 渲染不得因 mip 链的存在而改变");
}

/// 非整数缩放也应有 mip 收益：8px 格子在 level 4（每纹素 2×2 格）处
/// 死成均匀灰，采样落在那之后 → 内部取值集中在少数几档。
#[test]
fn non_integer_downscale_stays_smooth() {
    // 128×128（16×16 格）缩到 6×6：LOD = log2(128/6) ≈ 4.4
    // level 4 与 level 5 都是均匀灰 → 输出均匀
    let img = std::sync::Arc::new(checker_image(16));
    let place = ImagePlacement::rect(0.0, 0.0, 6.0, 6.0);
    let with_mip = render_image_scene(&img, place, 6);

    let mut values = std::collections::BTreeMap::new();
    for y in 1..5 {
        for x in 1..5 {
            *values.entry(with_mip[(y * 6 + x) * 4]).or_insert(0) += 1;
        }
    }
    assert!(
        values.len() <= 3,
        "LOD 4.4 落在格子死透的层级之后，内部应接近均匀灰，实际 {} 档：{:?}",
        values.len(),
        values
    );
}

/// 平铺 + 旋转（任务书 §7 黄金用例）：不崩溃，且带 mip 输出与不带 mip 不同
/// （说明 mip 路径真的在采样），同时 LOD 来自轴向量而非画布位置。
#[test]
fn tiled_and_rotated_placement_goes_through_the_mip_path() {
    let img = std::sync::Arc::new(checker_image(16));
    let rotated = ImagePlacement {
        origin: Vec2::new(40.0, 10.0),
        u_axis: Vec2::new(28.0, 14.0), // 缩小 + 旋转 26.6°
        v_axis: Vec2::new(-14.0, 28.0),
        repeat: true,
    };
    let with_mip = render_image_scene(&img, rotated, 64);
    let img_plain =
        std::sync::Arc::new(Image::from_rgba8(img.width, img.height, img.data.clone()).unwrap());
    let without_mip = render_image_scene(&img_plain, rotated, 64);
    assert_ne!(
        with_mip[..512],
        without_mip[..512],
        "缩小 1.4×+ 的旋转平铺应有 mip 参与（两路输出必须不同）"
    );
}

/// 帧间持久化路径（`PersistentRenderer`）必须与全量路径用**同一个** `mip_bias`。
///
/// 这一条钉住两个在 0.6.0 修复前都真实存在的缺陷：
///
/// 1. `mip_bias` 只传给了**层内容**（`render_inner_f32` 的入参），主画布走
///    `bin()` 后直接渲染，而 `bin()` 把 bias 固定为 0 —— 于是含位图的场景里
///    增量结果与全量结果静默不一致；
/// 2. 改了 `mip_bias` 而场景没变时，`update` 会走"场景未变 ⇒ 零工作"的早退
///    分支，画面停留在**旧 bias** 上 —— 同样不报错。
///
/// 用噪声图而不是渐变/棋盘：自相似图案在不同层可能落到同一个画布值上，
/// 那样第 2 条断言没有判别力（bias 没生效也照样"相等"）。
#[test]
fn persistent_renderer_uses_the_same_mip_bias_as_the_full_path() {
    let mut data = Vec::with_capacity(64 * 64 * 4);
    for i in 0..64 * 64 {
        let v = (i * 7 % 251) as u8;
        data.extend([v, v, v, 255]);
    }
    let img = std::sync::Arc::new(Image::from_rgba8_with_mipmap(64, 64, data).unwrap());
    let place = ImagePlacement::rect(0.0, 0.0, 16.0, 16.0); // 缩小 4× → LOD 2

    let mut scene = Scene::new(16, 16);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 16.0, 16.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Image { image: img, placement: place });

    let full = |bias: f32| {
        let mut b = bin(&scene, 0.05);
        b.mip_bias = bias;
        render(&b)
    };

    // 对照：bias 必须真的改变画面，否则下面的相等断言没有判别力
    assert_ne!(full(1.0)[..64], full(0.0)[..64], "bias +1 应改变输出");
    assert_ne!(full(-2.0)[..64], full(1.0)[..64], "bias -2 与 +1 应不同");

    // ① 首帧就带 bias：持久化路径必须与全量逐字节一致
    let mut r = PersistentRenderer::new(0.05);
    r.mip_bias = 1.0;
    r.update(&scene);
    assert_eq!(r.image(), full(1.0).as_slice(), "首帧的 bias 必须传到主画布");

    // ② 场景不变、只改 bias：必须整幅重画，而不是"零工作"早退到旧画面
    r.mip_bias = -2.0;
    r.update(&scene);
    assert_eq!(r.image(), full(-2.0).as_slice(), "改 bias 后必须重画，否则停在旧 bias 上");

    // ③ 场景与 bias 都没变：仍应零工作 —— M4 的卖点不能被这次修复吃掉
    r.update(&scene);
    let st = r.last_stats();
    assert!(st.skipped && st.shaded_tiles == 0, "bias 未变时不应重新着色，实际 {st:?}");
}
