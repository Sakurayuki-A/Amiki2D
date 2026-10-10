//! amiki-gpu 的集成/对拍测试。
//!
//! 从 `lib.rs` 拆出：这部分占原文件约 43%（2466/5719 行），
//! 且与实现代码的改动节奏完全不同（实现改一行，测试经常改几十行）。

// 测试通过 `crate::` 全路径引用被测项，不需要 `use super::*`。

use super::*;

// ───────────────────────── GPU 测试的串行闸门 ─────────────────────────
//
// 实测（RTX 3050 / Vulkan / Windows）：**多个测试线程同时**
// `GpuRenderer::new()`（枚举适配器 + 创建 `wgpu::Device`）会让驱动层崩掉
// 整个测试进程 —— 症状是 `test exited abnormally` + `0xC0000005`（访问违例），
// **没有任何 panic 信息**。这与 `RELEASE_PLAN.md` §8.5 记的那次"不可复现的
// 测试瞬时失败"是同一件事，现在有了可复现的触发条件（0.6.1）：
//
// | 运行方式 | 崩溃次数 |
// |---|---|
// | `cargo test -p amiki-gpu --lib`（默认线程，~16） | **4 / 6** |
// | `--test-threads=4` | 0 / 4 |
// | `--test-threads=2` | 0 / 4 |
// | `--test-threads=1` | 0 / 12 |
//
// 因此所有需要 GPU 设备的测试都先领取这把锁：**同一时刻只有一个 GPU 设备**。
// 代价是这台 crate 的测试串行跑（整套仍只有 ~8 s），换来的是**确定性** ——
// 而"性能断言在并行环境里失真"这个老问题也一并消失（性能数字只做测量，
// 严格判据交给 `--test-threads=1` 的单跑或独立基准）。
//
// 注意：`std::sync::Mutex` **不可重入**。一个测试若要创建/持有**多个**渲染器
// （`startup_cost_stays_within_budget`、`dense_ceiling_reports_error_not_panic`
// 都是），请自己用 [`gpu_lock`] 拿一次锁，其余渲染器直接
// `pollster::block_on(GpuRenderer::new())` —— 反复调用 [`gpu`] 会自锁
// （表现为测试进程 CPU 时间≈0、墙钟无限增长）。
static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 领取串行闸门（中毒也继续 —— 一个测试 panic 不该让后面全部失败）。
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// 领取闸门并创建渲染器。`None` = 无适配器（测试应当直接返回，即"跳过"）。
fn gpu() -> Option<(GpuRenderer, std::sync::MutexGuard<'static, ()>)> {
    let guard = gpu_lock();
    match pollster::block_on(GpuRenderer::new()) {
        Ok(r) => {
            // 打一行适配器：CI runner 上只有软件适配器（WARP，类型 Cpu），
            // 而性能类断言在那里没有意义 —— 日志里能一眼看出是哪条路。
            eprintln!(
                "[gpu] 适配器 = {}，类型 = {:?}{}",
                r.adapter_name,
                r.adapter_device_type,
                if r.adapter_device_type == wgpu::DeviceType::Cpu {
                    "（软件实现：性能类断言会跳过，正确性断言按 1 LSB 容忍）"
                } else {
                    ""
                }
            );
            Some((r, guard))
        }
        Err(_) => {
            eprintln!("跳过：无 GPU 适配器");
            None
        }
    }
}


use helpers::*;

mod helpers;
mod core;
mod sparse;
mod consistency;
mod dispatch;
mod persistent;
mod profile;
mod hdr_text;
mod even_odd;
