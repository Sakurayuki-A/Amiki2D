//! M3.5 稀疏自洽检查：装置自检 + 故意损坏必须被抓到。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

/// M3.5：稀疏路径自洽——**不依赖稠密路径**的结构正确性验证。
///
/// 这是 M3 之后最重要的安全网：稠密参照在大场景（1080p 约 31 draw 起）跑不起来，
/// 稀疏此前没有独立裁判。
#[test]
fn sparse_self_consistency() {
    let Some((renderer, _gpu_guard)) = gpu() else { return };
    for (name, scene) in consistency_scenes() {
        let (gs, dump) = renderer.dump_sparse(&scene, 0.05).expect("稀疏管线失败");
        if let Err(e) = check_sparse_consistency(&gs, &dump) {
            panic!("[{name}] 稀疏自洽检查失败：{e}");
        }
        let s = dump.sparse_stats.expect("统计缺失");
        eprintln!(
            "[{name}] 自洽 ✓  条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}",
            s[4], s[0], s[1], s[2], s[3]
        );
    }
}

/// M3.5 反向验证：**故意破坏稀疏结构，检查器必须抓住**。
///
/// 没有这一步，"检查通过"可能只是因为检查器什么都没查。
#[test]
fn sparse_consistency_detects_corruption() {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, Paint, PathBuilder};

    let Some((renderer, _gpu_guard)) = gpu() else { return };
    let mut scene = Scene::new(96, 96);
    let mut pb = PathBuilder::new();
    pb.circle(Vec2::new(40.0, 40.0), 30.0, false);
    pb.circle(Vec2::new(40.0, 40.0), 14.0, true);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 160, 90, 220)));
    let mut pb = PathBuilder::new();
    pb.pentagram(Vec2::new(60.0, 60.0), 28.0);
    scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 200)));

    let (gs, dump) = renderer.dump_sparse(&scene, 0.05).expect("稀疏管线失败");
    check_sparse_consistency(&gs, &dump).expect("基准场景应自洽");

    // 破坏 1：tile_count 人为 +1 → 逐 tile 桶表条目数应不符
    let mut bad = clone_dump(&dump);
    let tile = bad
        .tile_count
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|c| **c > 0)
        .expect("本场景应有非空 tile");
    *tile += 1;
    let e1 = check_sparse_consistency(&gs, &bad).expect_err("应检出桶表条目数不符");
    assert!(e1.to_string().contains("桶表条目数不一致"), "错误信息不符：{e1}");
    eprintln!("破坏 tile_count → 已检出：{e1}");

    // 破坏 2：cell_meta 的 carry 位模式扰动 → carry 与桶前缀应不符
    let mut bad = clone_dump(&dump);
    let cm = bad
        .cell_meta
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|c| f32::from_bits(c[0]) != 0.0)
        .expect("本场景应有非零 carry 的单元");
    cm[0] = (f32::from_bits(cm[0]) + 0.5).to_bits();
    let e2 = check_sparse_consistency(&gs, &bad).expect_err("应检出 carry 不符");
    assert!(e2.to_string().contains("carry"), "错误信息不符：{e2}");
    eprintln!("破坏 carry 位模式 → 已检出：{e2}");

    // 破坏 3：段数量 +1 → 与 emit 成员计数应不符
    let mut bad = clone_dump(&dump);
    let cm = bad
        .cell_meta
        .as_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c[2] > 0)
        .expect("本场景应有非空单元");
    cm[2] += 1;
    let e3 = check_sparse_consistency(&gs, &bad).expect_err("应检出段数量不符");
    assert!(
        e3.to_string().contains("段数量与 emit 成员计数不符")
            || e3.to_string().contains("段范围越界")
            || e3.to_string().contains("CSR 非连续"),
        "错误信息不符：{e3}"
    );
    eprintln!("破坏段数量 → 已检出：{e3}");

    // 破坏 4：tr_list 交换两项 → draw 序错乱应被检出（若该行有 ≥2 个活跃 strip）
    let mut bad = clone_dump(&dump);
    let tr_list = bad.tr_list.as_mut().unwrap();
    if let Some(tr) = bad.tr_offsets.as_ref().unwrap().iter().position(|o| o[1] >= 2) {
        let start = bad.tr_offsets.as_ref().unwrap()[tr][0] as usize;
        tr_list.swap(start, start + 1);
        let e4 = check_sparse_consistency(&gs, &bad).expect_err("交换 tr_list 应被检出");
        eprintln!("交换 tr_list 相邻项 → 已检出：{e4}");
    }
}

/// M3.5：**大场景**（稠密跑不起来）的全链路转储必须可完成且自洽。
///
/// 这是 ③ 的存在理由：稠密参照在 1080p 约 31 draw 起就撞 128 MiB 绑定上限，
/// 稀疏此前在大场景没有裁判。本测试证明：该规模下稀疏**有自己的裁判**，
/// 且全字段转储不会因缓冲过大而失败。
#[test]
fn sparse_consistency_at_scale() {
    use amiki_math::Vec2;
    use amiki_scene::{Color, Paint, PathBuilder};

    let Some((renderer, _gpu_guard)) = gpu() else { return };
    let (w, h, n) = (1920u32, 1080u32, 600usize);
    let mut scene = Scene::new(w, h);
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..n {
        let c = Vec2::new(rnd() * w as f32, rnd() * h as f32);
        let mut pb = PathBuilder::new();
        pb.circle(c, 4.0 + 6.0 * rnd(), false);
        scene.push(
            amiki_math::Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(200, 120, 60, 200)),
        );
    }

    // 该规模下稠密应当已被预检拦下（坐实"稠密跑不起来"这一前提）
    let (tx, ty) = (w.div_ceil(16), h.div_ceil(16));
    if dense_capacity(&wgpu::Limits::default(), tx, ty, n as u32).is_ok() {
        eprintln!("注意：本规模稠密仍可运行，测试未处于预期区间");
    }

    let (gs, dump) = renderer.dump_sparse(&scene, 0.1).expect("大场景稀疏转储失败");
    let st = dump.sparse_stats.expect("统计缺失");
    eprintln!(
        "1080p × {n} 曲线：转储完成（条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}）",
        st[4], st[0], st[1], st[2], st[3]
    );
    match check_sparse_consistency(&gs, &dump) {
        Ok(()) => eprintln!("大场景稀疏自洽 ✓"),
        Err(e) => panic!("1080p × {n} 曲线稀疏自洽检查失败：{e}"),
    }
}

