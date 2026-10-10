//! amiki-gpu —— wgpu 计算着色器渲染器（M3：稀疏桶表全 GPU 分箱）。
//!
//! CPU 每帧只做：场景 → 控制点/直线提取（不压平、不分箱）→ 上传。
//!
//! **稀疏路径（默认，M3b）**：显存与画面实际内容成比例
//! ```text
//! enc1a: clear strips → flatten+strip_extract → measure → 3×scan →
//!        tile 行活跃列表 → 逐 tile 桶表        〖读回 32B 规模〗
//! enc1b: clear_sparse → emit（成员计数 + carry 定点桶）〖读回 32B 成员数〗
//! enc2:  copy+scan(单元 CSR) → tile_fill → scatter → carry → fine
//! ```
//! **稠密路径（M2f，对拍基准）**：`(tile, draw, row)` 稠密网格，显存 ∝ tiles×draws；
//! 与稀疏路径渲染结果逐字节一致（见 `sparse_matches_dense` 测试）。
//!
//! 两条路径共用：GPU 均匀压平（闭式段数界）、通用三级前缀和、active tile 元数据
//! （[`PipelineDump`] 可逐级读出）、GPU 时间戳分析（[`PassTimings`]）。

//! API 稳定性：本 crate 的稳定面为 `GpuRenderer` / `RenderError` /
//! `RenderPath` / `Frame` / `Blitter` / `GpuScene` / `build_gpu_scene`；
//! `profile`、`consistency`、容量表与各内部模块属**实验面**。
//! 完整划分见仓库 `docs/API.md`。

#![forbid(unsafe_code)]

use amiki_scene::Scene;
use std::sync::Mutex;
use std::time::Instant;
use wgpu::util::DeviceExt;

/// M6：GPU 离屏层滤镜（可分离盒式模糊）。
use amiki_scene::rgba8_from_f32_with;
pub mod blit;
pub mod blur;
pub mod consistency;
pub mod dense;
pub mod extract;
pub mod layer;
pub mod profile;
pub mod readback;
pub mod sparse;
pub use blit::Blitter;
pub use consistency::check_sparse_consistency;
pub use extract::{build_gpu_scene, build_gpu_scene_into, GpuScene};
pub(crate) use extract::{BinsParams, FineParams, FlattenParams, ScanParams};
pub(crate) use profile::TimestampRecorder;
pub use profile::{
    FrameProfile, PassTimings, PipelineDump, SceneKey, SparseCapacity, SparseCapacityTable,
    SparseStats,
};

/// `GpuScene` 各类图元的数量（诊断与容量换算共用）。
fn nq_len(gs: &GpuScene) -> usize {
    gs.quads.len()
}
fn nc_len(gs: &GpuScene) -> usize {
    gs.cubics.len()
}
fn nl_len(gs: &GpuScene) -> usize {
    gs.lines.len()
}


// ---- 0.10.9：按职责分文件（原 lib.rs 2101 行）----
pub mod shaders;
pub mod device;
pub mod pipeline;
pub mod frame;
pub mod capacity;
pub mod pending;
pub mod dispatch;

// 搬到子模块的公开项在这里重导出，保持 `amiki_gpu::xxx` 路径不变
pub use pending::{PendingFrame, PollStatus};
pub use capacity::{dense_capacity, dense_grid_shape, required_limits, REQUIRED_STORAGE_BUFFERS};
// 跨模块使用的 crate 内 helper 同理（sparse.rs / dense.rs / extract.rs 不用改）
pub(crate) use device::{pack_gradients, pack_image_place};
pub(crate) use dispatch::{check_damage_bounds, dirty_tile_list, fine_dispatch_dims, dispatch_dims_1d};
pub(crate) use profile::{ts_slot, ts_writes};
pub(crate) use shaders::*;

/// 渲染路径：稠密网格（M2f，正确性基准）或稀疏桶表（M3b，主线）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderPath {
    /// 稠密 (tile, draw, row) 网格；显存与 tiles×draws 成比例，仅限小场景
    Dense,
    /// 稀疏 strip/桶表；显存与画面实际内容成比例（M3c 验证后为默认）
    #[default]
    Sparse,
}

#[derive(Debug)]
pub struct RenderError(pub String);

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RenderError {}

fn err<T>(msg: impl Into<String>) -> Result<T, RenderError> {
    Err(RenderError(msg.into()))
}

/// M3a：逐帧资源。主场景 + clip 覆盖率纹理视图。
#[derive(Default)]
pub struct Frame {
    scene: GpuScene,
    /// **clip 覆盖率纹理用的复用缓冲**。每个 clip 都要提取一次场景，
    /// 若每帧每个 clip 都新建缓冲，大块分配 + 缺页的代价会照常付
    /// （见 `GpuScene::reset_for_reuse` 的实测）。
    clip_scratch: GpuScene,
    /// 只用于**持有** clip 纹理的生命周期 —— `clip_views` 借用它们，
    /// 纹理一旦 drop，视图即失效。字段本身从不被读，所以需要 allow。
    #[allow(dead_code)]
    clip_texs: Vec<wgpu::Texture>,
    pub clip_views: Vec<wgpu::TextureView>,
    /// `prepare` 时用的压平容差。层内容的渲染要用**同一个**容差，
    /// 否则层内几何的压平精度会与主场景不一致。
    pub eps: f32,
}

/// GPU 渲染器（设备级状态，可跨帧复用）。
pub struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// 分箱路径：稠密网格（M2f 对拍基准）或稀疏桶表（M3b 主线）
    pub path: RenderPath,
    /// 适配器名称/后端（基准报告用；`new` 填充，`from_device` 为 "unknown"）
    pub adapter_name: String,
    /// 适配器类型（`new` 填充；`from_device` 为 `Other` = 未知）。
    ///
    /// 存在的理由：CI runner **没有真卡**，只有软件适配器
    /// （`Microsoft Basic Render Driver` / WARP，wgpu 报 `DeviceType::Cpu`）。
    /// 在那条路上性能预算与"逐字节相等"的判据都不成立 —— 首次 CI 实测：
    /// 单帧 GPU pass 795 ms（真卡 <5 ms）、冷启动 1502 ms（真卡 340–500 ms）、
    /// 稀疏↔稠密出现 1 LSB 差。测试据此分别处理，而不是把判据整体放松。
    pub adapter_device_type: wgpu::DeviceType,
    /// 实际使用的 wgpu 后端。`new` 填充；`from_device` 为 `None`（未知 ⇒ 不做后端相关判定）。
    pub backend: Option<wgpu::Backend>,
    /// 适配器是否支持 GPU 时间戳（`profile` 的前提）
    pub has_timestamps: bool,
    /// 时间戳 tick → 纳秒
    pub timestamp_period_ns: f32,
    /// 设备限制（稠密路径容量预检用：提前返回可读错误而非让 wgpu 校验层 panic）
    pub limits: wgpu::Limits,
    /// M3.6：MAP_READ 暂存缓冲池（读回复用，避免每次读回都新建 + 冷映射）。
    /// 用 `Mutex` 而非 `RefCell`：`&self` 方法需要内部可变性，且便于将来多线程提交。
    readback_pool: Mutex<Vec<wgpu::Buffer>>,
    /// M3.6：上一帧实测的稀疏容量（投机分配）。锁中毒时按"无容量"处理
    /// ——退回精确路径，只会更慢，不会更错。
    sparse_capacity: Mutex<SparseCapacityTable>,
    fine_pipeline: wgpu::ComputePipeline,
    coverage_pipeline: wgpu::ComputePipeline,
    fine_bgl: wgpu::BindGroupLayout,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scan_sums_pipeline: wgpu::ComputePipeline,
    scan_add_pipeline: wgpu::ComputePipeline,
    scan_bgl: wgpu::BindGroupLayout,
    flatten_pipeline: wgpu::ComputePipeline,
    flatten_bgl: wgpu::BindGroupLayout,
    bins_clear_pipeline: wgpu::ComputePipeline,
    bins_emit_pipeline: wgpu::ComputePipeline,
    bins_scatter_pipeline: wgpu::ComputePipeline,
    bins_carry_pipeline: wgpu::ComputePipeline,
    bins_bgl: wgpu::BindGroupLayout,
    strip_clear_pipeline: wgpu::ComputePipeline,
    strip_extract_pipeline: wgpu::ComputePipeline,
    strip_bgl: wgpu::BindGroupLayout,
    sparse_bgl: wgpu::BindGroupLayout,
    sparse_measure_pipeline: wgpu::ComputePipeline,
    sparse_tr_scatter_pipeline: wgpu::ComputePipeline,
    sparse_tr_offsets_pipeline: wgpu::ComputePipeline,
    sparse_tile_count_pipeline: wgpu::ComputePipeline,
    sparse_tile_fill_pipeline: wgpu::ComputePipeline,
    sparse_clear_pipeline: wgpu::ComputePipeline,
    sparse_emit_pipeline: wgpu::ComputePipeline,
    sparse_scatter_pipeline: wgpu::ComputePipeline,
    sparse_carry_pipeline: wgpu::ComputePipeline,
    sparse_fine_pipeline: wgpu::ComputePipeline,
    sparse_coverage_pipeline: wgpu::ComputePipeline,
    dummy_clip_view: wgpu::TextureView,
    _dummy_clip_tex: wgpu::Texture,
    /// M6：图像绑定的占位（稠密路径不用图像，但共用同一 bind group layout）
    dummy_img_view: wgpu::TextureView,
    _dummy_img_tex: wgpu::Texture,
    dummy_img_sampler: wgpu::Sampler,
    /// M6：放置参数的占位缓冲（layout 共用，需要合法非空 storage buffer）
    dummy_img_place: wgpu::Buffer,
    /// M6：离屏层滤镜资源（可分离盒式模糊）。`blur_rgba` 用它。
    blur: blur::BlurResources,
    /// P0-3：离屏层的 **GPU 合成**资源（帧路径用；compute pass，不读回）。
    layer_composite: layer::LayerCompositeResources,
}

/// 临时 bind group 构建期间的 16B 垫片（满足 vec4 数组的最小绑定尺寸）。
const PAD16: [u8; 16] = [0u8; 16];

#[cfg(test)]
mod tests;
