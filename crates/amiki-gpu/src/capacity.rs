//! 稀疏容量投机表与限额（含 0.7.0 的容量学习）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

#[doc(hidden)] // 仅基准与诊断用，见 docs/API.md「实验面」
/// 稠密网格（`(tile, draw, row)`）所需规模：`(元素数, 桶数)`。
///
/// 布局见 `bins.wgsl`：`b = ((tr * tiles_x + col) * n_draws + draw) * 16 + row_local`。
pub fn dense_grid_shape(tiles_x: u32, tiles_y: u32, n_draws: u32) -> (u64, u64) {
    let n_grid = (tiles_x as u64) * (tiles_y as u64) * (n_draws as u64) * 16;
    let n_bucket = (n_draws as u64) * (tiles_y as u64) * 16 * (tiles_x as u64 + 1);
    (n_grid, n_bucket)
}

#[doc(hidden)] // 仅基准与诊断用，见 docs/API.md「实验面」
/// 稠密路径容量预检：`Ok((n_grid, n_bucket))` 或带具体数字的可读错误。
///
/// # 为什么需要它
///
/// 稠密网格显存 ∝ `tiles × draws`，会**远早于其他限制**撞上
/// `max_storage_buffer_binding_size`（wgpu 默认 128 MiB）。若不预检，失败发生在
/// `create_bind_group` 内部，表现为 wgpu 校验层 panic——调用方拿不到可读原因，
/// 也无法在生产代码里优雅降级到稀疏路径。
///
/// # 两条天花板：**绑定尺寸墙先到**
///
/// - **绑定墙**：`grid_aux`（8 B/元素）与 `bucket_sums`（4 B/桶）各占一个 storage 绑定；
/// - **dispatch 墙**：`clear` 按 `⌈max(n_grid, n_bucket)/64⌉` 个工作组**线性 2D** 派发，
///   上限 `65535²`——极高，实际不是瓶颈。
///
/// 实测（RTX 3050 / Vulkan，默认 limits `max_storage_buffer_binding_size = 128 MiB`）：
/// **720p 约 292 draw、1080p 约 129 draw** 即撞绑定墙（由本函数公式逐 draw 复算）。
/// 注意 `benches/render.rs` 里打印的"稠密网格上限：draw 31/71"是**按单维
/// 65535 工作组**估算的保守值（见 `dense_draw_ceiling`），远小于线性 2D 的真实容量；
/// 两条口径不同，勿混用。
pub fn dense_capacity(
    limits: &wgpu::Limits,
    tiles_x: u32,
    tiles_y: u32,
    n_draws: u32,
) -> Result<(u64, u64), RenderError> {
    let (n_grid, n_bucket) = dense_grid_shape(tiles_x, tiles_y, n_draws);
    // 实际进 bind group 的 storage 缓冲（见 run_pipeline_impl 的分配）
    let grid_aux_bytes = n_grid * 8; // 绑定 4：counters + fill 前缀区
    let bucket_bytes = n_bucket * 4; // 绑定 5
    // 单缓冲上限取两者较小值（wgpu 校验两条都查）
    let binding_cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);

    for (name, bytes) in [
        ("grid_aux（counters+fill 前缀区）", grid_aux_bytes),
        ("bucket_sums", bucket_bytes),
    ] {
        if bytes > binding_cap {
            return err(format!(
                "稠密网格超限（RenderPath::Dense）：{name} 需 {:.1} MB，单缓冲上限 {:.1} MB（max_storage_buffer_binding_size / max_buffer_size 取小）—— 本场景 tiles={tiles_x}x{tiles_y} draws={n_draws}。请改用 RenderPath::Sparse（显存 ∝ 画面内容，无此上限）；稠密路径仅适用于小场景对拍与诊断。",
                bytes as f64 / 1e6,
                binding_cap as f64 / 1e6,
            ));
        }
    }

    // dispatch 墙：clear 的线性 2D 拆分上限为 65535² 个工作组
    let clear_wg = n_grid.max(n_bucket).div_ceil(64);
    if clear_wg > 65_535 * 65_535 {
        return err(format!(
            "稠密网格 clear dispatch 超限：需 {clear_wg} 个工作组，线性 2D 上限 {}（65535²）——请改用 RenderPath::Sparse。",
            65_535u64 * 65_535
        ));
    }
    Ok((n_grid, n_bucket))
}

/// 引擎需要的**设备上限**。自己创建 `wgpu::Device` 的调用方**必须**用它。
///
/// # 为什么需要这个函数（dogfooding 抓到的真问题）
///
/// 引擎的 `amiki_fine_bgl` 有 **10 个 storage buffer**（0.6.1 起；第 10 个是
/// 脏 tile 索引表，在此之前是 9 个、第 9 个是图像放置参数），
/// 而 `wgpu::Limits::default()` 只保证 **8** 个（WebGPU 基线）。
///
/// `GpuRenderer::new()` 会按适配器能力上调这个上限，但**自己建设备的调用方
/// 不会** —— 那样会在 `create_bind_group_layout` 里抛出 wgpu 校验错误：
///
/// ```text
/// In Device::create_bind_group_layout, label = 'amiki_fine_bgl'
///   Too many bindings of type StorageBuffers in Stage COMPUTE, limit is 8, count was 10
/// ```
///
/// 这个错误**深在 wgpu 内部**，调用方几乎无法从消息定位到"我该改 limits"。
/// 窗口示例（`window_demo`）就踩了这个坑，而且因为示例不在测试覆盖里，
/// **一直没被发现** —— 它是靠 dogfooding（真去跑那个应用）才暴露的。
///
/// 所以现在有两道防线：
/// 1. 本函数给出正确的上限，调用方直接用它请求设备；
/// 2. [`GpuRenderer::from_device`] **校验**传入的设备并**显式报错**，
///    不再把问题留给 wgpu 深处的校验失败。
pub fn required_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: REQUIRED_STORAGE_BUFFERS,
        ..wgpu::Limits::default()
    }
}

/// 引擎在单个着色阶段里需要的 storage buffer 数量（见 [`required_limits`]）。
///
/// 0.6.1 起是 **10**：fine 新增了**脏 tile 索引表**（binding 18，`array<u32>`）。
/// 这个数字是**对外契约**的一部分 —— 自行创建 `wgpu::Device` 的调用方必须用
/// [`required_limits`] 请求，否则 `from_device` 会显式报错。
pub const REQUIRED_STORAGE_BUFFERS: u32 = 10;

impl GpuRenderer {
    /// M3c：渲染一帧并报告稀疏分箱的内容规模（显存与内容成比例的直接量化）。
    /// 与 `render` 走同一条管线，只多一次 32B 读回；稠密路径或空画布返回 `None`。
    pub fn sparse_stats(&self, scene: &Scene, eps: f32) -> Result<Option<SparseStats>, RenderError> {
        if self.path != RenderPath::Sparse {
            return Ok(None);
        }
        let frame = self.prepare(scene, eps)?;
        if frame.scene.colors.is_empty() {
            return Ok(None);
        }
        let tex = self.create_output_texture(&frame.scene);
        let mut dump = PipelineDump { sparse_stats: Some([0; 8]), ..Default::default() };
        let mut prof = FrameProfile::default();
        self.run_pipeline_sparse(
            &frame.scene,
            &tex,
            false,
            &frame.clip_views,
            None,
            Some(&mut dump),
            None,
            &mut prof,
        )?;
        let s = dump.sparse_stats.unwrap_or([0; 8]);
        Ok(Some(SparseStats {
            n_cells: s[0],
            n_buckets: s[1],
            n_tile_entries: s[2],
            n_items: s[3],
            n_strips: s[4],
        }))
    }

    /// M3.6：取某场景的投机容量。`None` 表示该场景应走**精确测量路径**
    /// （从未见过，或已因反复溢出放弃投机）。
    ///
    /// **运行时报告**：测试与诊断据此断言"这一帧究竟走没走投机路径"，
    /// 而不是靠注释描述。
    pub fn sparse_capacity(&self, key: SceneKey) -> Option<SparseCapacity> {
        self.sparse_capacity.lock().ok().and_then(|mut t| t.get(key))
    }

    /// 累计溢出次数（只增）。测试用它断言"溢出确实发生并被处理"。
    pub fn capacity_overflow_total(&self) -> u64 {
        self.sparse_capacity
            .lock()
            .map(|t| t.overflow_total())
            .unwrap_or(0)
    }

    /// 诊断：容量表的项数（长跑里不该无限增长）。
    pub fn capacity_table_len(&self) -> usize {
        self.sparse_capacity.lock().map(|t| t.len()).unwrap_or(0)
    }

    /// 诊断：某场景的容量是否因反复溢出而被放弃投机。
    pub fn capacity_gave_up(&self, key: SceneKey) -> bool {
        self.sparse_capacity.lock().is_ok_and(|t| t.gave_up(key))
    }

    /// 诊断：列出容量表的全部表项（key 与容量）。
    ///
    /// 测试用 —— 避免在测试里硬编码场景特征（例如几何桶的具体取值），
    /// 那些属于实现的内部细节。
    pub fn capacity_entries(&self) -> Vec<(SceneKey, SparseCapacity)> {
        self.sparse_capacity
            .lock()
            .map(|t| {
                t.entries
                    .iter()
                    .map(|(k, e)| (*k, e.cap))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// M3.6：记录该场景的实测用量（`overflowed` 表示这一帧因容量不足作废）。
    ///
    /// 锁中毒时静默放弃 —— 容量只是加速手段，丢了只会退回精确测量路径
    /// （更慢，但结果不变）。
    ///
    /// `pub(crate)`：`sparse.rs` 在容量学习路径上直接调用它（原来同处一个文件，
    /// 所以是私有的）。
    pub(crate) fn store_sparse_capacity(&self, key: SceneKey, usage: SparseCapacity, overflowed: bool) {
        if let Ok(mut t) = self.sparse_capacity.lock() {
            t.record(key, usage, overflowed);
        }
    }

}
