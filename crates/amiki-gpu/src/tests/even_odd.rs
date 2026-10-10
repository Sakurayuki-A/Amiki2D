//! even-odd 填充规则在两条路径下逐字节一致。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

/// even-odd：**GPU 与 CPU 逐像素一致**（判据 ≤2/255），且确实画出了东西。
#[test]
fn gpu_even_odd_matches_cpu() {
    let Some((renderer, _gpu_guard)) = gpu() else { return };
    let scene = even_odd_scene(64, 64);

    let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
    let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

    let mut max_diff = 0u32;
    let mut ink = 0usize;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        if *a < 250 {
            ink += 1;
        }
    }
    assert!(ink > 2000, "必须真的画出 even-odd 内容，实际非白通道 {ink}");
    assert!(max_diff <= 2, "even-odd：GPU 与 CPU 最大通道差 {max_diff} 超出容差");
    eprintln!("even-odd 对拍：最大通道差 {max_diff}，非白通道 {ink}");
}

/// even-odd：**稀疏与稠密两条分箱路径逐字节一致**。
#[test]
fn sparse_and_dense_agree_on_even_odd() {
    let Some((mut renderer, _gpu_guard)) = gpu() else { return };
    let scene = even_odd_scene(64, 64);

    renderer.path = RenderPath::Sparse;
    let sparse = renderer.render_blocking(&scene, 0.05).expect("稀疏渲染失败");
    renderer.path = RenderPath::Dense;
    let dense = renderer.render_blocking(&scene, 0.05).expect("稠密渲染失败");

    let diff = sparse.iter().zip(dense.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(
        diff, 0,
        "even-odd 在稀疏/稠密两条路径下必须逐字节一致，实际不同字节 {diff}"
    );
    eprintln!("even-odd：稀疏 ⇄ 稠密逐字节一致（{} 字节）", sparse.len());
}

// ───────────────────── 1D 派发契约与 4K 尾部回归（0.10.9）─────────────────────

