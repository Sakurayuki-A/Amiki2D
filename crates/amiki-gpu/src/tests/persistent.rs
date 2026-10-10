//! 增量/持久化路径：只重算脏区，结果与全量逐字节相同。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

/// M4：**GPU 侧只重算脏 tile** 必须与全量派发**逐字节一致**。
///
/// # 这个测试必须让纹理停在**上一帧**
///
/// 早先的版本先把纹理渲成"全量正确"的内容，再用脏区重算一批 tile，然后断言
/// 等于全量 —— 那样**无论重算的是哪一批 tile 都会通过**，它检验的是"重算 tile
/// 的结果与全量一致"，而不是"重算的确实是脏 tile"。结果是一个真 bug 藏了很久：
/// 派发只用了脏区的**大小**、丢掉了**起点**，于是脏区不在左上角时，被重算的是
/// `[0,fw) × [0,fh)`，真正的脏 tile 一个都没重算，画面静默保留上一帧像素。
/// （0.6.1 已改为 1D 派发 + 脏 tile 索引表，从结构上消除"起点"这个参数。）
///
/// 因此本测试的结构是：
/// 1. 纹理 `tex` 先渲成 **A 帧**的全量（= 上一帧）；
/// 2. 参考 `tex_full` 渲成 **B 帧**的全量；
/// 3. 用 `edit_damage(A, B)` 在 `tex` 上做增量（B 只改了**右下角**一个元素）；
/// 4. 断言 `tex == tex_full` —— 脏 tile 若没被重算，旧像素会立刻暴露。
///
/// 并且断言脏区**不在原点起步**：否则这个测试就退化成恒真。
#[test]
fn gpu_damage_dispatch_matches_full() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{edit_damage, Color, Damage, Paint, PathBuilder};

    let Some((mut renderer, _gpu_guard)) = gpu() else { return };
    renderer.path = RenderPath::Sparse;

    // A：左上角固定红圆 + 右下角蓝圆；B：只把蓝圆上移 30px（改动**远离原点**）
    let build = |blue_y: f32| -> Scene {
        let mut s = Scene::new(128, 96);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(20.0, 20.0), 12.0, false);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(220, 40, 40, 255)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(100.0, blue_y), 12.0, false);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 220, 210)));
        s
    };
    let a = build(70.0);
    let b = build(40.0);

    // ① 上一帧：A 的全量
    let frame_a = renderer.prepare(&a, 0.1).expect("prepare A 失败");
    let tex = renderer.create_output_texture(&frame_a.scene);
    renderer
        .render_into_persistent(&tex, &frame_a, None)
        .expect("A 全量渲染失败");

    // ② 参照：B 的全量（同时把 B 的容量表预热到稳态，避免首帧投机分配干扰）
    let frame_b = renderer.prepare(&b, 0.1).expect("prepare B 失败");
    let tex_full = renderer.create_output_texture(&frame_b.scene);
    renderer
        .render_into_persistent(&tex_full, &frame_b, None)
        .expect("B 全量渲染失败");
    let px_full = renderer.readback_rgba8(&tex_full).expect("读回 B 全量失败");

    // ③ 增量：纹理此刻是 A，只重算 A→B 的脏 tile
    let d = edit_damage(&a, &b);
    let (tx0, ty0, tx1, ty1) = d.tiles().expect("A→B 应是有界脏区");
    assert!(
        tx0 > 0,
        "本测试的脏区必须**不在原点起步**（否则派发起点被丢弃也照样通过）；tile 矩形 = {tx0},{ty0}..{tx1},{ty1}"
    );
    eprintln!(
        "脏区 tile 矩形 = ({tx0}, {ty0})..({tx1}, {ty1})；画布 {}×{} tiles",
        frame_b.scene.tiles_x, frame_b.scene.tiles_y
    );
    let skipped = renderer
        .render_into_persistent(&tex, &frame_b, Some(&d))
        .expect("增量渲染失败");
    assert!(!skipped, "有脏区时不得跳过");

    let after = renderer.readback_rgba8(&tex).expect("读回增量失败");
    let n_diff = after.iter().zip(px_full.iter()).filter(|(x, y)| x != y).count();
    assert_eq!(
        after, px_full,
        "脏区不在原点时，增量结果必须等于 B 的全量（不同字节 = {n_diff}）—— \
         若不等，说明被重算的不是脏 tile"
    );

    // ④ 无脏区：必须完全跳过且不改变像素
    let mut d2 = Damage::new(b.width, b.height);
    d2.clear();
    assert!(d2.is_empty());
    let skipped = renderer
        .render_into_persistent(&tex, &frame_b, Some(&d2))
        .expect("空脏区渲染失败");
    assert!(skipped, "无脏区时必须跳过");
    let after2 = renderer.readback_rgba8(&tex).expect("读回失败");
    assert_eq!(after2, px_full, "跳过的帧不得改变任何像素");

    // ⑤ 全脏：也必须等于全量
    let d3 = Damage::new(b.width, b.height);
    assert!(d3.is_full());
    renderer
        .render_into_persistent(&tex, &frame_b, Some(&d3))
        .expect("全脏渲染失败");
    let after3 = renderer.readback_rgba8(&tex).expect("读回失败");
    assert_eq!(after3, px_full, "全脏渲染与全量派发不一致");

    // ⑥ 脏区 tile 数校验：尺寸不匹配的 damage 必须被断言拦下（防"全脏被误判为局部"）
    let wrong = Damage::new(b.width + 64, b.height);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        renderer.render_into_persistent(&tex, &frame_b, Some(&wrong))
    }))
    .is_err();
    assert!(caught, "画布尺寸不匹配的 damage 应被断言拦下");

    eprintln!("GPU 脏区派发：A→B 增量 / 空脏区 / 全脏 / 尺寸校验 四类均正确");
}

/// GPU 帧持久路径的**多帧滚动序列**必须逐帧等于"从零全量重算"。
///
/// 这是 `window_demo` 用法的缩影：每帧 `edit_damage(上一帧, 本帧)` →
/// `prepare_into` → `render_into_persistent`。
///
/// 与单帧的 `gpu_damage_dispatch_matches_full` 互补：单帧只能证明"一次改动落到
/// 了正确位置"，本测试证明**连续多帧、脏区一路移动**时不会累积残影 ——
/// 滚动时脏区从一个条带移到下一个条带，**起点每帧都不同**，正是"派发起点被
/// 丢弃"这类 bug 最容易积累可见错误的情形。
#[test]
fn gpu_persistent_matches_full_across_scroll_like_frames() {
    use amiki_math::Affine;
    use amiki_scene::{edit_damage, Color, Paint, PathBuilder};

    let Some((mut renderer, _gpu_guard)) = gpu() else { return };
    renderer.path = RenderPath::Sparse;

    const W: u32 = 160;
    const H: u32 = 160;

    // 不动的背景 + 一列会"滚动"的条目。条目 x 从 40 起（tile 列 2），
    // 因此脏区**永远不会**从 tile 列 0 起步 —— 这是本测试的判别力所在。
    let build = |offset: f32| -> Scene {
        let mut s = Scene::new(W, H);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, W as f32, H as f32);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(248, 246, 242)));
        for i in 0..5u8 {
            let y = 16.0 + ((i as f32 * 30.0 + offset) % 130.0);
            let mut pb = PathBuilder::new();
            pb.rect(40.0, y, 140.0, y + 22.0);
            s.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgb8(60, 90 + i * 20, 200)),
            );
        }
        s
    };

    // 两个复用的 Frame（真实帧路径的形态），一份输出纹理 + 一份全量参照纹理
    let mut frame = Frame::default();
    let mut frame_ref = Frame::default();
    let mut prev = build(0.0);
    renderer.prepare_into(&prev, 0.1, &mut frame).expect("prepare 失败");
    let tex = renderer.create_output_texture(&frame.scene);
    renderer
        .render_into_persistent(&tex, &frame, None)
        .expect("首帧全量失败");
    let tex_ref = renderer.create_output_texture(&frame.scene);

    for step in 1..=8 {
        let cur = build(step as f32 * 26.0); // 每帧滚动 26px（非 tile 对齐）
        let d = edit_damage(&prev, &cur);
        let (tx0, ty0, tx1, ty1) = d.tiles().expect("滚动应产生有界脏区");
        assert!(
            tx0 > 0,
            "第 {step} 帧：滚动条带必须不在 tile 列 0（否则本测试抓不到起点 bug）；\
             实际 tile 矩形 ({tx0},{ty0})..({tx1},{ty1})"
        );

        // 增量：`tex` 里装的是上一帧
        renderer.prepare_into(&cur, 0.1, &mut frame).expect("prepare 失败");
        renderer
            .render_into_persistent(&tex, &frame, Some(&d))
            .expect("增量渲染失败");
        let got = renderer.readback_rgba8(&tex).expect("读回失败");

        // 参照：从零全量
        renderer
            .prepare_into(&cur, 0.1, &mut frame_ref)
            .expect("prepare 失败");
        renderer
            .render_into_persistent(&tex_ref, &frame_ref, None)
            .expect("全量渲染失败");
        let want = renderer.readback_rgba8(&tex_ref).expect("读回失败");

        let n_diff = got.iter().zip(want.iter()).filter(|(x, y)| x != y).count();
        assert_eq!(
            got, want,
            "第 {step} 帧（滚动 {:.0}px，脏区 tile ({tx0},{ty0})..({tx1},{ty1})）\
             增量结果与全量不一致，不同字节 = {n_diff} —— 残影",
            step as f32 * 26.0
        );
        prev = cur;
    }
    eprintln!("GPU 帧持久路径：8 帧滚动序列逐帧等于全量（无残影）");
}

