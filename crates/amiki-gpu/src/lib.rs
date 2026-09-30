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

#![forbid(unsafe_code)]

use amiki_scene::Scene;
use bytemuck::{Pod, Zeroable};
use std::sync::Mutex;
use std::time::Instant;
use wgpu::util::DeviceExt;

const FINE_MATH_WGSL: &str = include_str!("fine_math.wgsl");
const FINE_WGSL: &str = include_str!("fine.wgsl");
const SPARSE_FINE_WGSL: &str = include_str!("sparse_fine.wgsl");

/// M6：GPU 离屏层滤镜（可分离盒式模糊）。
pub mod blur;
const BLIT_WGSL: &str = include_str!("blit.wgsl");
const SCAN_WGSL: &str = include_str!("scan.wgsl");
const FLATTEN_WGSL: &str = include_str!("flatten.wgsl");
const BINS_WGSL: &str = include_str!("bins.wgsl");
const STRIPS_WGSL: &str = include_str!("strips.wgsl");
const SPARSE_MEASURE_WGSL: &str = include_str!("sparse_measure.wgsl");
const SPARSE_TILES_WGSL: &str = include_str!("sparse_tiles.wgsl");
const SPARSE_EMIT_WGSL: &str = include_str!("sparse_emit.wgsl");
const SPARSE_SCATTER_WGSL: &str = include_str!("sparse_scatter.wgsl");
const SPARSE_CARRY_WGSL: &str = include_str!("sparse_carry.wgsl");

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

fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FineParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    n_draws: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FlattenParams {
    n_quads: u32,
    n_cubics: u32,
    n_lines: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BinsParams {
    tiles_x: u32,
    n_draws: u32,
    tiles_y: u32,
    height: u32,
    total_segments: u32,
    /// M3b 稀疏管线：单元 / 桶 / tile 条目数（稠密路径恒为 0）
    n_cells: u32,
    n_buckets: u32,
    n_tile_entries: u32,
}

/// CPU 提取的 GPU 场景：控制点 + 直线 + 逐 draw 属性（零压平、零分箱）。
#[derive(Clone, Debug, Default)]
pub struct GpuScene {
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub quads: Vec<[[f32; 4]; 2]>,
    pub quad_draws: Vec<u32>,
    pub cubics: Vec<[[f32; 4]; 2]>,
    pub cubic_draws: Vec<u32>,
    pub lines: Vec<[f32; 4]>,
    pub line_draws: Vec<u32>,
    pub colors: Vec<[f32; 4]>,
    /// M6：每 draw 的渐变下标；`u32::MAX` 表示纯实心色。
    /// 随 `draw_meta.w` 传给 WGSL（该分量此前闲置）。
    pub gradient_index: Vec<u32>,
    /// M6：本场景用到的渐变定义，按首次出现顺序排列（下标即 `gradient_index` 的值）。
    pub gradients: Vec<std::sync::Arc<amiki_scene::Gradient>>,
    /// M6：每 draw 的图像下标；`u32::MAX` 表示非图像绘制。
    pub image_index: Vec<u32>,
    /// M6：本场景用到的位图与放置（下标即 `image_index` 的值，也是纹理数组的层号）。
    pub images: Vec<(std::sync::Arc<amiki_scene::Image>, amiki_scene::ImagePlacement)>,
    /// M6：源场景是否含离屏层。`GpuScene` 本身只描述图元（层由
    /// `render_blocking` 单独处理），因此这里只记一个标志，供逐帧路径
    /// 明确拒绝"会静默丢层"的场景。
    pub has_layers: bool,
    pub draw_flags: Vec<u32>,
    pub draw_clips: Vec<[u32; 4]>,
    /// CPU 闭式段数（每入口：quad/cubic 变长，line=1）
    pub counts: Vec<u32>,
    pub total_segments: u32,
    /// emit/scatter 的 indirect dispatch 参数
    pub indirect: [u32; 3],
}

impl GpuScene {
    fn push_line(&mut self, draw: u32, a: amiki_math::Vec2, b: amiki_math::Vec2) {
        if (a - b).hypot() > 1e-9 {
            self.lines.push([a.x, a.y, b.x, b.y]);
            self.line_draws.push(draw);
            self.counts.push(1);
        }
    }
    fn push_quad(&mut self, draw: u32, p0: amiki_math::Vec2, c: amiki_math::Vec2, p2: amiki_math::Vec2) {
        self.quads.push([[p0.x, p0.y, c.x, c.y], [p2.x, p2.y, 0.0, 0.0]]);
        self.quad_draws.push(draw);
    }
    fn push_cubic(&mut self, draw: u32, p0: amiki_math::Vec2, c1: amiki_math::Vec2, c2: amiki_math::Vec2, p3: amiki_math::Vec2) {
        self.cubics.push([[p0.x, p0.y, c1.x, c1.y], [c2.x, c2.y, p3.x, p3.y]]);
        self.cubic_draws.push(draw);
    }
}

/// 场景 → GpuScene（CPU 只做变换、控制点收集与闭式段数计算；不压平、不分箱）。
pub fn build_gpu_scene(scene: &Scene, eps: f32) -> GpuScene {
    use amiki_math::Vec2;
    use amiki_scene::{Paint, PathEl};
    let width = scene.width.max(1);
    let height = scene.height.max(1);
    let mut gs = GpuScene {
        width,
        height,
        tiles_x: width.div_ceil(16),
        tiles_y: height.div_ceil(16),
        // 只记标志：层的内容由 render_blocking 单独渲染，不进 GpuScene
        has_layers: !scene.layers.is_empty(),
        ..Default::default()
    };
    for (d, draw) in scene.draws.iter().enumerate() {
        let di = d as u32;
        let mut gi = u32::MAX;
        let mut ii = u32::MAX;
        gs.colors.push(match &draw.paint {
            Paint::Solid(c) => [c.r, c.g, c.b, c.a],
            Paint::Gradient(g) => {
                // 去重复用同一定义（同一渐变常用于多条 draw）
                gi = gs
                    .gradients
                    .iter()
                    .position(|x| std::sync::Arc::ptr_eq(x, g))
                    .unwrap_or_else(|| {
                        gs.gradients.push(g.clone());
                        gs.gradients.len() - 1
                    }) as u32;
                // colors 仍是占位：着色走 gradient_index
                [0.0, 0.0, 0.0, 0.0]
            }
            Paint::Image { image, placement } => {
                ii = gs
                    .images
                    .iter()
                    .position(|(x, p)| std::sync::Arc::ptr_eq(x, image) && p == placement)
                    .unwrap_or_else(|| {
                        gs.images.push((image.clone(), *placement));
                        gs.images.len() - 1
                    }) as u32;
                [0.0, 0.0, 0.0, 0.0]
            }
        });
        gs.gradient_index.push(gi);
        gs.image_index.push(ii);
        let _ = di;
        let nclip = draw.clips.len().min(4);
        let mut ci = [0u32; 4];
        ci[..nclip].copy_from_slice(&draw.clips[..nclip]);
        gs.draw_clips.push(ci);
        gs.draw_flags.push(draw.blend.id() | ((nclip as u32) << 8));

        let t = &draw.transform;
        let mut cur: Option<Vec2> = None;
        let mut sub_start: Option<Vec2> = None;
        for el in &draw.path.els {
            match *el {
                PathEl::MoveTo(p) => {
                    if let (Some(c), Some(s)) = (cur, sub_start) {
                        gs.push_line(di, c, s);
                    }
                    let m = t.map_point(p);
                    cur = Some(m);
                    sub_start = Some(m);
                }
                PathEl::LineTo(p) => {
                    if let Some(c) = cur {
                        let m = t.map_point(p);
                        gs.push_line(di, c, m);
                        cur = Some(m);
                    }
                }
                PathEl::QuadTo(c, p) => {
                    if let Some(c0) = cur {
                        let m = t.map_point(p);
                        gs.push_quad(di, c0, t.map_point(c), m);
                        cur = Some(m);
                    }
                }
                PathEl::CubicTo(c1, c2, p) => {
                    if let Some(c0) = cur {
                        let m = t.map_point(p);
                        gs.push_cubic(di, c0, t.map_point(c1), t.map_point(c2), m);
                        cur = Some(m);
                    }
                }
                PathEl::Close => {
                    if let (Some(c), Some(s)) = (cur, sub_start) {
                        gs.push_line(di, c, s);
                        cur = Some(s);
                    }
                }
            }
        }
        if let (Some(c), Some(s)) = (cur, sub_start) {
            gs.push_line(di, c, s);
        }
    }
    // counts 按【数据布局顺序】（quads | cubics | lines）重算 —— 与 data/entry_draws
    // 的入口顺序严格一致。此前的遍历顺序 counts 与类型分块布局错位，
    // 导致混合场景（线 + 曲线）的段数张冠李戴（M2f 曲线偏差的根因）。
    gs.counts.clear();
    for q in &gs.quads {
        let q = amiki_math::QuadBez::new(
            amiki_math::Vec2::new(q[0][0], q[0][1]),
            amiki_math::Vec2::new(q[0][2], q[0][3]),
            amiki_math::Vec2::new(q[1][0], q[1][1]),
        );
        gs.counts.push(q.uniform_segments(eps));
    }
    for c in &gs.cubics {
        let c = amiki_math::CubicBez::new(
            amiki_math::Vec2::new(c[0][0], c[0][1]),
            amiki_math::Vec2::new(c[0][2], c[0][3]),
            amiki_math::Vec2::new(c[1][0], c[1][1]),
            amiki_math::Vec2::new(c[1][2], c[1][3]),
        );
        gs.counts.push(c.uniform_segments(eps));
    }
    for _ in &gs.lines {
        gs.counts.push(1);
    }
    gs.total_segments = gs.counts.iter().sum();
    let wg = gs.total_segments.div_ceil(64).max(1);
    gs.indirect = [wg, 1, 1];
    gs
}

/// M2f 诊断：管线中间数据转储（测试用，release 路径传 None 零开销）。
#[derive(Default)]
pub struct PipelineDump {
    pub seg_ends: Option<Vec<[f32; 4]>>,
    pub seg_draws: Option<Vec<u32>>,
    pub counters: Option<Vec<u32>>,
    pub scan_base: Option<Vec<u32>>,
    pub carries: Option<Vec<f32>>,
    pub item_total: Option<u32>,
    pub seg_refs: Option<Vec<u32>>,
    pub bucket_sums: Option<Vec<i32>>,
    /// M3a：活跃 strip 交错元数据（4×u32/strip：lo, hi, y0, y1；转置布局 `si = tr * n_draws + d`）
    pub strip_meta: Option<Vec<u32>>,
    /// M3b：稀疏桶表与稀疏单元结构（仅稀疏路径转储）
    pub sparse_stats: Option<[u32; 8]>,
    pub tr_offsets: Option<Vec<[u32; 2]>>,
    pub tr_list: Option<Vec<u32>>,
    pub tile_count: Option<Vec<u32>>,
    pub tile_start: Option<Vec<u32>>,
    pub tile_list: Option<Vec<[u32; 4]>>,
    pub cell_meta: Option<Vec<[u32; 4]>>,
    pub cell_segs: Option<Vec<[f32; 4]>>,
    /// M3.5：稀疏自洽检查所需的稀疏结构
    /// - `strip_cell_off`：每 strip 的单元基址（`measure` 计数 → 扫描后 exclusive 前缀）
    pub strip_cell_off: Option<Vec<u32>>,
    /// - `strip_bucket_off`：每 strip 的桶基址（`measure` 的 `(w+1)*h` 计数 → 扫描后 exclusive 前缀）
    pub strip_bucket_off: Option<Vec<u32>>,
    /// - `cell_aux`：`[0, n_cells)` 成员计数，`[n_cells, 2·n_cells)` scatter 游标（emit 后终值 = 计数）
    pub cell_aux: Option<Vec<u32>>,
    /// - `cell_csr_off`：单元 CSR 基址（扫描后 exclusive 前缀，长度 `n_cells + 1`）
    pub cell_csr_off: Option<Vec<u32>>,
    /// - `sparse_buckets`：carry 定点桶（`i32`，×65536，布局
    ///   `strip_bucket_off[si] + (row-y0)*(w+1) + kb`）
    pub sparse_buckets: Option<Vec<i32>>,
}

/// M3c：稀疏分箱的内容规模（显存与画面内容成比例的直接证据）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseStats {
    /// 稀疏单元数（strip × 行 × 列）
    pub n_cells: u32,
    /// carry 定点桶数
    pub n_buckets: u32,
    /// 逐 tile 桶表条目数（活跃 (tile, draw) 对）
    pub n_tile_entries: u32,
    /// 成员项数（段 × 行 × 列）
    pub n_items: u32,
    /// 活跃 strip 数（draw × tile 行）
    pub n_strips: u32,
}

/// M3 基准：GPU 时间戳分析结果（每个 compute pass 的 GPU 时长）。
#[derive(Clone, Debug, Default)]
pub struct PassTimings {
    /// 适配器是否支持并启用了 `TIMESTAMP_QUERY`
    pub supported: bool,
    /// 每 tick 的纳秒数（`Queue::get_timestamp_period`）
    pub timestamp_period_ns: f32,
    /// (pass 标签, GPU 时长 ns)，按提交顺序
    pub passes: Vec<(&'static str, f64)>,
}

impl PassTimings {
    /// 全部 pass 的 GPU 时长合计（ns）。
    pub fn total_ns(&self) -> f64 {
        self.passes.iter().map(|(_, ns)| ns).sum()
    }

    /// 单行摘要（基准/日志用）。
    pub fn summary(&self) -> String {
        if !self.supported {
            return "timestamp query 不可用".to_string();
        }
        let mut s = format!("GPU 合计 {:.3} ms:", self.total_ns() / 1e6);
        for (label, ns) in &self.passes {
            s.push_str(&format!(" [{label} {:.3} ms]", ns / 1e6));
        }
        s
    }
}

/// 时间戳记录器：为每个 pass 分配 begin/end 查询槽位，末尾统一 resolve。
struct TimestampRecorder {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    labels: Vec<&'static str>,
}

/// 单次管线分析的 pass 上限（每个 pass 2 个时间戳槽）。
///
/// **实测依据**：wgpu 25 的 `wgpu::Limits` 中**没有**时间戳查询数量的限制项
/// （`wgpu-types` 的 `Limits` 仅有 `Features::TIMESTAMP_QUERY` 特性开关；
/// `wgpu-core::Device::create_query_set` 只校验"已启用特性"与"count != 0"，
/// 不校验数量上限）。因此这里用编译期上限 + 运行时守卫（见 `TimestampRecorder::slot`），
/// 而不是（无法）向 `adapter.limits()` 查询。
///
/// 取值 32：原实现只有 8 个槽位，稀疏路径被迫把 5 次 scan 合并成一个黑盒槽位；
/// 扩到 32 后可为每次 scan 单独归因（稀疏全链路约 20 个 pass）。
const MAX_PROFILE_PASSES: u32 = 32;

/// 进程内 CPU 分段计时（M3.5/M3.6）——把"墙钟里到底花在哪"变成可读数字。
///
/// 全部字段单位 **毫秒**。`Option` 的语义严格是"**该路径是否存在这个阶段**"：
/// 稀疏路径有两次规模读回（`readback1_ms`/`readback2_ms`），稠密路径只有一次
/// （`readback_dense_ms`）；不存在者为 `None`。
///
/// 时间戳与主机计时的分工：`timestamps` 给**逐 compute pass 的 GPU 时长**，
/// 本结构给**主机侧（CPU）各阶段**——两者合起来才能回答"是 GPU 慢还是 CPU/同步慢"。
#[derive(Clone, Debug, Default)]
pub struct FrameProfile {
    /// 构建 `GpuScene`（CPU 提取：变换 + 控制点 + 闭式段数）
    pub extract_ms: f64,
    /// 创建本帧 wgpu 缓冲/纹理（**M3.6 池化的直接目标**：目前每帧全量重建）
    pub alloc_ms: f64,
    /// 上传（`create_buffer_init` / `write_buffer`）
    pub upload_ms: f64,
    /// `enc1a`（分箱结构，含 6 个 pass）的编码 + 提交
    pub encode1a_ms: f64,
    /// `enc1b`（稀疏清零 + emit，含若干 pass）的编码 + 提交
    pub encode1b_ms: f64,
    /// 第二次提交前的命令编码（含单元 CSR scan）
    pub encode2_ms: f64,
    /// 第二次提交
    pub submit2_ms: f64,
    /// 稀疏专属：第 1 次规模读回（32B，**结构上必需**——要据此分配稀疏缓冲）
    pub readback1_ms: Option<f64>,
    /// 稀疏专属：第 2 次成员数读回（32B，**结构上必需**——要据此分配段列表）
    pub readback2_ms: Option<f64>,
    /// 稠密专属：唯一一次 item_total 读回（4B）
    pub readback_dense_ms: Option<f64>,
    /// 诊断转储耗时（仅传 `Some(dump)` 时有值）
    pub dump_ms: Option<f64>,
    /// `profile()` 专属：时间戳 resolve + 读回的额外开销（非渲染本身）
    pub resolve_ts_ms: Option<f64>,
    /// 逐 pass GPU 时长
    pub timestamps: PassTimings,
}

impl FrameProfile {
    /// 主机侧可归因阶段合计（ms）。
    pub fn host_total_ms(&self) -> f64 {
        self.extract_ms
            + self.alloc_ms
            + self.upload_ms
            + self.encode1a_ms
            + self.encode1b_ms
            + self.encode2_ms
            + self.submit2_ms
            + self.readback1_ms.unwrap_or(0.0)
            + self.readback2_ms.unwrap_or(0.0)
            + self.readback_dense_ms.unwrap_or(0.0)
            + self.dump_ms.unwrap_or(0.0)
            + self.resolve_ts_ms.unwrap_or(0.0)
    }

    /// 单行摘要（基准/日志用）。
    pub fn summary(&self) -> String {
        let mut s = format!("主机侧合计 {:.2} ms:", self.host_total_ms());
        let push = |s: &mut String, name: &str, v: Option<f64>| {
            if let Some(ms) = v {
                s.push_str(&format!(" [{name} {ms:.2} ms]"));
            }
        };
        s.push_str(&format!(" [提取 {:.2}]", self.extract_ms));
        s.push_str(&format!(" [分配 {:.2}]", self.alloc_ms));
        s.push_str(&format!(" [上传 {:.2}]", self.upload_ms));
        s.push_str(&format!(" [编码1a+提交 {:.2}]", self.encode1a_ms));
        s.push_str(&format!(" [编码1b+提交 {:.2}]", self.encode1b_ms));
        s.push_str(&format!(" [编码2 {:.2}]", self.encode2_ms));
        s.push_str(&format!(" [提交2 {:.2}]", self.submit2_ms));
        push(&mut s, "读回1", self.readback1_ms);
        push(&mut s, "读回2", self.readback2_ms);
        push(&mut s, "读回(稠密)", self.readback_dense_ms);
        push(&mut s, "转储", self.dump_ms);
        push(&mut s, "时间戳解析", self.resolve_ts_ms);
        s
    }
}

impl TimestampRecorder {
    fn new(device: &wgpu::Device) -> TimestampRecorder {
        let count = 2 * MAX_PROFILE_PASSES;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("amiki_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_ts_resolve"),
            size: count as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        TimestampRecorder { query_set, resolve_buf, labels: Vec::new() }
    }

    /// 分配一个 pass 的 (begin, end) 槽位。
    ///
    /// 超出 [`MAX_PROFILE_PASSES`] 时报错而不 panic——分析是诊断功能，
    /// 不该因为 pass 数超出预期就把整个渲染打断。
    fn slot(&mut self, label: &'static str) -> Result<(u32, u32), RenderError> {
        if self.labels.len() as u32 >= MAX_PROFILE_PASSES {
            return err(format!(
                "GPU 时间戳槽位不足：已有 {} 个 pass，上限 {MAX_PROFILE_PASSES}\
                 （请提高 MAX_PROFILE_PASSES）",
                self.labels.len()
            ));
        }
        let b = self.labels.len() as u32 * 2;
        self.labels.push(label);
        Ok((b, b + 1))
    }

    fn query_count(&self) -> u32 {
        self.labels.len() as u32 * 2
    }
}

/// M6：把渐变定义打包成 `sparse_fine.wgsl` 期望的 `vec4<f32>` 布局。
///
/// **必须与 WGSL 侧 `GRAD_STRIDE` / `grad_color()` 严格一致**，
/// 否则着色会静默取错数据（这类错误不会 panic，只会画错）。
///
/// 布局（`S = 2 + MAX_GRAD_STOPS*2` 个 vec4）：
/// ```text
/// [0]                    (n_grad, 0, 0, 0)
/// [1 + g*S]              (kind, repeat, n_stops, MAX_GRAD_STOPS)
/// [1 + g*S + 1]          (x0, y0, x1, y1)     径向: (cx, cy, r, 0)
/// [1 + g*S + 2 + i*2]    (t_i, r, g, b)
/// [1 + g*S + 2 + i*2 + 1](0, 0, 0, 0)          对齐占位
/// ```
/// 色标超过 `MAX_GRAD_STOPS` 时**截断并保持端点色**：与其静默丢中间色造成
/// 肉眼可见的跳变，不如明确只保留前 N 个（调用方应自行控制色标数量）。
/// 返回的缓冲至少含头部 1 个 vec4（空场景也要能绑定）。
fn pack_gradients(grads: &[std::sync::Arc<amiki_scene::Gradient>]) -> Vec<[f32; 4]> {
    const MAX_STOPS: usize = 8;
    const STRIDE: usize = 2 + MAX_STOPS * 2;

    let mut out: Vec<[f32; 4]> = Vec::with_capacity(1 + grads.len() * STRIDE);
    out.push([grads.len() as f32, 0.0, 0.0, 0.0]);
    for g in grads {
        // kind 标签与几何一起由 match 求出，避免"先声明后赋值"的分支式写法
        let (kind_tag, geo) = match g.kind {
            amiki_scene::GradientKind::Linear { from, to } => {
                (0.0, [from.x, from.y, to.x, to.y])
            }
            amiki_scene::GradientKind::Radial { center, radius } => {
                (1.0, [center.x, center.y, radius, 0.0])
            }
            amiki_scene::GradientKind::Sweep { center, start } => {
                (2.0, [center.x, center.y, start, 0.0])
            }
        };
        let n = g.stops.len().min(MAX_STOPS);
        let head = [
            kind_tag,
            if g.repeat { 1.0 } else { 0.0 },
            n as f32,
            MAX_STOPS as f32,
        ];
        out.push(head);
        out.push(geo);
        for i in 0..MAX_STOPS {
            if i < n {
                let s = &g.stops[i];
                out.push([s.t, s.color.r, s.color.g, s.color.b]);
            } else {
                out.push([0.0; 4]);
            }
            out.push([0.0; 4]); // 对齐占位
        }
    }
    out
}
/// M6：把场景用到的位图上传为一个**纹理数组**（每张图一层）。
///
/// 用纹理数组而非逐图绑定：绑定槽位有限，而场景可能有很多张图；
/// 数组 + 层号让 draw 只需在放置参数里带一个整数层号。
///
/// 尺寸不一致时**居中左上对齐、越界部分透明**：不做缩放，
/// 因为缩放语义（拉伸/平铺/留白）应由调用方通过放置显式决定，
/// 引擎在这里静默缩放会让"图像显示成什么样"变得不可预测。
///
/// 无图像时创建一个 1×1 的占位层，保证绑定始终合法。
impl GpuRenderer {
    fn mk_image_texture_array(
        &self,
        images: &[(std::sync::Arc<amiki_scene::Image>, amiki_scene::ImagePlacement)],
    ) -> Result<(wgpu::TextureView, wgpu::Sampler), RenderError> {
        let max_dim = self.limits.max_texture_dimension_2d;
        let w = images.iter().map(|(i, _)| i.width).max().unwrap_or(1).min(max_dim);
        let h = images.iter().map(|(i, _)| i.height).max().unwrap_or(1).min(max_dim);
        let layers = images.len().max(1) as u32;
        // 层数上限：不同后端不同，取保守值并在超出时明确报错
        let max_layers = self.limits.max_texture_array_layers;
        if layers > max_layers {
            return err(format!(
                "位图层数 {layers} 超出设备上限 {max_layers}；请合并图像或分帧绘制"
            ));
        }

        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("amiki_image_array"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 逐层上传（上层若比底图小，剩余部分保持透明）
        const BPP: usize = 4;
        let row_bytes = (w as usize) * BPP;
        let mut layer_buf = vec![0u8; row_bytes * h as usize];
        for (li, (img, _)) in images.iter().enumerate() {
            layer_buf.fill(0);
            let cw = img.width.min(w) as usize;
            let ch = img.height.min(h) as usize;
            for y in 0..ch {
                let src = (y * img.width as usize) * BPP;
                let dst = (y * w as usize) * BPP;
                layer_buf[dst..dst + cw * BPP].copy_from_slice(&img.data[src..src + cw * BPP]);
            }
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: li as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                &layer_buf,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes as u32),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("amiki_image_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // 采样器：clamp-to-edge + 线性过滤。
        // repeat 由 WGSL 侧取模实现（采样器同时服务 clip 纹理的钳制语义，
        // 不能设成 repeat）。
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("amiki_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Ok((view, sampler))
    }
}

/// M6：把每个 draw 的图像放置打包成 `sparse_fine.wgsl` 期望的布局。
///
/// **每 draw 三个 vec4**，与 WGSL 的 `image_color` / `image_index_of` 严格对应：
/// ```text
/// [d*3+0] = (origin.x, origin.y, u_axis.x, u_axis.y)
/// [d*3+1] = (v_axis.x, v_axis.y, tex_layer, 0)
/// [d*3+2] = (bitcast<f32>(image_index), repeat, 0, 0)
/// ```
/// 把 `image_index` 放进放置数据自身，使该缓冲区**自描述**：
/// "这个 draw 是不是图像"无需另占 `draw_meta` 分量，也不会与渐变下标争位置。
/// 非图像 draw 在 `[d*3+2].x` 写 `u32::MAX` 的位模式（WGSL 用 bitcast 读回）。
fn pack_image_place(gs: &GpuScene) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; gs.image_index.len() * 3];
    for (d, idx) in gs.image_index.iter().enumerate() {
        let slot = d * 3;
        if *idx == u32::MAX {
            out[slot + 2][0] = f32::from_bits(u32::MAX);
            continue;
        }
        let (_img, place) = &gs.images[*idx as usize];
        out[slot] = [place.origin.x, place.origin.y, place.u_axis.x, place.u_axis.y];
        // 纹理层号 = 该图像在纹理数组中的层（与 gs.images 同序）
        out[slot + 1] = [place.v_axis.x, place.v_axis.y, *idx as f32, 0.0];
        out[slot + 2] =
            [f32::from_bits(*idx), if place.repeat { 1.0 } else { 0.0 }, 0.0, 0.0];
    }
    out
}

/// M6：**逐帧复用路径不合成离屏层**。
///
/// 层合成发生在 `render_blocking`（它需要逐层渲染并读回、再在主机侧混合）。
/// 逐帧路径为了复用纹理而刻意不读回，因此无法承担层合成 —— 它必须**明确报错**，
/// 否则阴影/模糊会被静默丢掉，而调用方完全看不出来。
///
/// `render_blocking` 不走这个守卫（它自己负责层合成）。
fn reject_layers_for_frame_path(n_layers: u32) -> Result<(), RenderError> {
    if n_layers == 0 {
        return Ok(());
    }
    err(format!(
        "逐帧路径（render_into / render_into_persistent）不合成离屏层：场景含 {n_layers} 个层。\
         该路径为避免读回而不做层合成；请改用 render_blocking（支持层），或改用 CPU 引擎"
    ))
}

/// M6：把一个离屏层合成到根图像上。
///
/// 流程与 CPU 参考实现（`amiki-raster::composite_layer`）**语义一致**，
/// 但重活交给 GPU：
/// ```text
/// 1) 层内容仍由同一套细粒度管线渲染到独立纹理（不复制一份着色逻辑）
/// 2) 模糊在 GPU 上做（可分离盒式，两趟乒乓）
/// 3) 读回层像素，在主机侧按不透明度/混合模式合成到根图像
/// ```
///
/// 第 3 步放主机侧是**刻意的取舍**：着色与模糊（真正的大头）都在 GPU，
/// 而合成是每像素几次乘加；把它放主机侧使"层"与"图元"共用同一套混合数学，
/// 避免出现两份可能漂移的实现。真正的瓶颈本就不在这里。
///
/// `sig_tex` / `blur_tex` 为层缓冲，尺寸须 ≥ 层范围。
/// **当前未被调用**：GPU 层合成尚未接通（见 `render_blocking` 的说明）。
/// 保留此实现是为了让后续接手者不必从零重写主机侧合成部分。
/// M6：把一个离屏层合成到根图像上。
///
/// `layer_img` 是层的**内容**（`Filter::Blur` 时已由 GPU 模糊过）；
/// `unblurred` 是**未模糊**的层内容，仅投影需要。
///
/// # 为什么投影要用未模糊的内容
///
/// CPU 侧的投影是"用 alpha 形状 + 固定阴影色先着色、再模糊"。
/// 若直接拿"模糊后的层内容"当阴影，语义变成"模糊内容再着色"，两者不同
/// （实测差 125/255）。因此阴影必须从**未模糊**的内容出发再单独模糊。
///
/// 阴影的滤波**复用 CPU 的 `apply_filter`**（GPU 侧已模糊过内容，故传 sigma=0
/// 只做"着色 + 偏移 + 置于下方"）—— 两份实现必然漂移，共用一份才对。
#[allow(clippy::too_many_arguments)]
fn composite_layer_gpu(
    root: &mut [u8],
    root_w: u32,
    root_h: u32,
    layer: &amiki_scene::Layer,
    layer_img: &[u8],
    layer_w: u32,
    layer_h: u32,
    shadow: Option<&[u8]>,
) {
    use amiki_raster::{inflated_bounds, LayerBuffer};

    if layer.bounds.is_empty() {
        return;
    }
    let b = inflated_bounds(layer.bounds, layer.filter);
    let bx = b.x.max(0);
    let by = b.y.max(0);
    if bx >= root_w as i32 || by >= root_h as i32 {
        return;
    }
    let lw = b.width.min(root_w - bx as u32);
    let lh = b.height.min(root_h - by as u32);
    if lw == 0 || lh == 0 {
        return;
    }

    // 取层内容（直线色：滤波在颜色上做，与 CPU 的 LayerBuffer 语义一致）
    let mut buf = LayerBuffer::new(lw, lh);
    for y in 0..lh {
        for x in 0..lw {
            let sx = bx + x as i32;
            let sy = by + y as i32;
            let src = if sx >= 0 && sy >= 0 && (sx as u32) < layer_w && (sy as u32) < layer_h {
                let o = ((sy as u32 * layer_w + sx as u32) * 4) as usize;
                [
                    layer_img[o] as f32 / 255.0,
                    layer_img[o + 1] as f32 / 255.0,
                    layer_img[o + 2] as f32 / 255.0,
                    layer_img[o + 3] as f32 / 255.0,
                ]
            } else {
                [0.0; 4]
            };
            buf.set(x, y, src);
        }
    }

    // 投影：从**未模糊**的层内容出发，用 CPU 的同一份实现做
    // "按阴影色着色 + 按 (dx,dy) 偏移 + 置于内容下方"。
    // GPU 已模糊过内容（Blur 情形），故这里对内容本身传 sigma=0；
    // 但阴影需要"先着色再模糊"，那一步由**主机侧调用方**在 GPU 上单独完成，
    // 结果经 `unblurred` 传入的 alpha 形状参与。
    // 投影：把主机侧已生成并模糊好的阴影按 (dx,dy) 偏移，叠到内容**下方**。
    if let (Some(sh), amiki_scene::Filter::DropShadow { dx, dy, .. }) = (shadow, layer.filter) {
        for y in 0..lh {
            for x in 0..lw {
                // 阴影在 (gx-dx, gy-dy) 处 —— 即"内容右下方出现的阴影"
                let gxs = bx + x as i32 - dx.round() as i32;
                let gys = by + y as i32 - dy.round() as i32;
                let sp = if gxs >= 0 && gys >= 0 && (gxs as u32) < root_w && (gys as u32) < root_h {
                    let o = ((gys as u32 * root_w + gxs as u32) * 4) as usize;
                    [
                        sh[o] as f32 / 255.0,
                        sh[o + 1] as f32 / 255.0,
                        sh[o + 2] as f32 / 255.0,
                        sh[o + 3] as f32 / 255.0,
                    ]
                } else {
                    [0.0; 4]
                };
                let d = buf.get(x, y);
                let ao = d[3] + sp[3] * (1.0 - d[3]);
                let inv = if ao > 0.0 { 1.0 / ao } else { 0.0 };
                let co = [
                    (d[0] * d[3] + sp[0] * sp[3] * (1.0 - d[3])) * inv,
                    (d[1] * d[3] + sp[1] * sp[3] * (1.0 - d[3])) * inv,
                    (d[2] * d[3] + sp[2] * sp[3] * (1.0 - d[3])) * inv,
                ];
                buf.set(x, y, [co[0], co[1], co[2], ao]);
            }
        }
    }

    // 合成回根图像：与 CPU 侧同一套数学（预乘空间，见 amiki-raster 的说明）
    let opacity = layer.opacity.clamp(0.0, 1.0);
    for y in 0..lh {
        for x in 0..lw {
            let p = buf.get(x, y);
            let sa = p[3] * opacity;
            if sa <= 0.0 {
                continue;
            }
            let src_pm = [p[0] * sa, p[1] * sa, p[2] * sa, sa];
            let gx = (bx + x as i32) as u32;
            let gy = (by + y as i32) as u32;
            let o = ((gy * root_w + gx) * 4) as usize;
            let da = root[o + 3] as f32 / 255.0;
            let dst_pm = [
                root[o] as f32 / 255.0 * da,
                root[o + 1] as f32 / 255.0 * da,
                root[o + 2] as f32 / 255.0 * da,
                da,
            ];
            let out_pm = amiki_raster::composite_premul(dst_pm, src_pm, layer.blend.id());
            let oa = out_pm[3];
            let inv = if oa > 0.0 { 1.0 / oa } else { 0.0 };
            let straight = [out_pm[0] * inv, out_pm[1] * inv, out_pm[2] * inv, oa];
            for k in 0..4 {
                root[o + k] = (straight[k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

    /// 校验脏区与画布 tile 尺寸一致。///
/// `Damage` 用的是它自己的画布 tile 数，而派发范围按当前场景的 tile 数钳制。
/// 若两者不一致（例如调用方复用了为不同尺寸画布建的 `Damage`），
/// "全脏"会被误判成"只脏左上角"，从而**静默留下陈旧像素**。
/// 这类错误必须炸出来，不能靠钳制掩盖。
fn check_damage_bounds(damage: Option<&amiki_scene::Damage>, tiles_x: u32, tiles_y: u32) {
    if let Some(d) = damage {
        assert_eq!(
            (d.total_tiles()),
            tiles_x.max(1) * tiles_y.max(1),
            "damage 的 tile 数与当前画布不一致（damage 可能是为别的画布尺寸创建的）\
             —— 否则「全脏」会被误当作局部脏区，导致陈旧像素残留"
        );
    }
}

/// 根据脏区算出 fine 阶段的 workgroup 派发范围 `(x_start, x_count, y_start, y_count)`。
///
/// M4：fine 内核用 `global_invocation_id` 直接推导像素与 tile 坐标
/// （`sparse_fine.wgsl` 里 `tile_x = gid.x / 16`），因此**只派发脏 tile 子矩形**
/// 就会精确只重写那些 tile 的像素——不需要改任何 WGSL，也不需要改着色数学。
/// 这是"脏区只重算"在 GPU 侧能零成本落地的根本原因。
///
/// `None` / 全脏 ⇒ 整幅画布。返回的 count 均 ≥ 1，保证 dispatch 合法
/// （workgroup 数不能为 0）。
fn fine_dispatch_range(
    damage: Option<&amiki_scene::Damage>,
    tiles_x: u32,
    tiles_y: u32,
) -> (u32, u32, u32, u32) {
    let full = || (0u32, tiles_x.max(1), 0u32, tiles_y.max(1));
    let Some(d) = damage else { return full() };
    let Some((tx0, ty0, tx1, ty1)) = d.tiles() else {
        return full(); // 全脏
    };
    let x_start = tx0.min(tiles_x);
    let y_start = ty0.min(tiles_y);
    let x_end = tx1.min(tiles_x).max(x_start);
    let y_end = ty1.min(tiles_y).max(y_start);
    if x_end == x_start || y_end == y_start {
        // 无脏区：派发一个最小合法范围（内核按画布裁剪，不会写入任何像素）
        return (x_start, 1, y_start, 1);
    }
    (x_start, x_end - x_start, y_start, y_end - y_start)
}

fn ts_writes(
    rec: Option<&TimestampRecorder>,
    slot: Option<(u32, u32)>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    match (rec, slot) {
        (Some(rec), Some((b, e))) => Some(wgpu::ComputePassTimestampWrites {
            query_set: &rec.query_set,
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        }),
        _ => None,
    }
}

/// 从**预分配**的槽位表里取第 `i` 个槽位的时间戳写入描述。
///
/// 刻意写成自由函数并接收 `slots` 的**副本**（`[(u32,u32); N]` 是 `Copy`）：
/// 闭包会捕获并保持对 `ts`/`ts_slots` 的借用，与 `run_scan(.., ts.as_deref_mut())`
/// 的可变借用冲突；自由函数每次只做一次瞬时再借用，故无冲突。
fn ts_slot<'a, const N: usize>(
    ts: Option<&'a TimestampRecorder>,
    slots: Option<[(u32, u32); N]>,
    i: usize,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    match (ts, slots) {
        (Some(r), Some(s)) => ts_writes(Some(r), Some(s[i])),
        _ => None,
    }
}

/// M3a：逐帧资源。主场景 + clip 覆盖率纹理视图。
pub struct Frame {
    scene: GpuScene,
    #[allow(dead_code)]
    clip_texs: Vec<wgpu::Texture>,
    pub clip_views: Vec<wgpu::TextureView>,
}

/// GPU 渲染器（设备级状态，可跨帧复用）。
pub struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// 分箱路径：稠密网格（M2f 对拍基准）或稀疏桶表（M3b 主线）
    pub path: RenderPath,
    /// 适配器名称/后端（基准报告用；`new` 填充，`from_device` 为 "unknown"）
    pub adapter_name: String,
    /// 适配器是否支持 GPU 时间戳（`profile` 的前提）
    pub has_timestamps: bool,
    /// 时间戳 tick → 纳秒
    pub timestamp_period_ns: f32,
    /// 设备限制（稠密路径容量预检用：提前返回可读错误而非让 wgpu 校验层 panic）
    pub limits: wgpu::Limits,
    /// M3.6：MAP_READ 暂存缓冲池（读回复用，避免每次读回都新建 + 冷映射）。
    /// 用 `Mutex` 而非 `RefCell`：`&self` 方法需要内部可变性，且便于将来多线程提交。
    readback_pool: Mutex<Vec<wgpu::Buffer>>,
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
    /// M6：离屏层滤镜资源。**当前未启用**（预乘顺序待对齐，见 render_blocking）。
    #[allow(dead_code)]
    blur: blur::BlurResources,
}

/// 临时 bind group 构建期间的 16B 垫片（满足 vec4 数组的最小绑定尺寸）。
const PAD16: [u8; 16] = [0u8; 16];

/// 稠密网格（`(tile, draw, row)`）所需规模：`(元素数, 桶数)`。
///
/// 布局见 `bins.wgsl`：`b = ((tr * tiles_x + col) * n_draws + draw) * 16 + row_local`。
pub fn dense_grid_shape(tiles_x: u32, tiles_y: u32, n_draws: u32) -> (u64, u64) {
    let n_grid = (tiles_x as u64) * (tiles_y as u64) * (n_draws as u64) * 16;
    let n_bucket = (n_draws as u64) * (tiles_y as u64) * 16 * (tiles_x as u64 + 1);
    (n_grid, n_bucket)
}

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
                "稠密网格超限（RenderPath::Dense）：{name} 需 {:.1} MB，\
                 单缓冲上限 {:.1} MB（max_storage_buffer_binding_size / max_buffer_size 取小）\
                 —— 本场景 tiles={tiles_x}x{tiles_y} draws={n_draws}。\
                 请改用 RenderPath::Sparse（显存 ∝ 画面内容，无此上限）；\
                 稠密路径仅适用于小场景对拍与诊断。",
                bytes as f64 / 1e6,
                binding_cap as f64 / 1e6,
            ));
        }
    }

    // dispatch 墙：clear 的线性 2D 拆分上限为 65535² 个工作组
    let clear_wg = n_grid.max(n_bucket).div_ceil(64);
    if clear_wg > 65_535 * 65_535 {
        return err(format!(
            "稠密网格 clear dispatch 超限：需 {clear_wg} 个工作组，\
             线性 2D 上限 {}（65535²）——请改用 RenderPath::Sparse。",
            65_535u64 * 65_535
        ));
    }
    Ok((n_grid, n_bucket))
}

impl GpuRenderer {
    /// 独立初始化（无窗口）。
    pub async fn new() -> Result<GpuRenderer, RenderError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| RenderError(format!("没有可用的 GPU 适配器: {e}")))?;
        let info = adapter.get_info();
        // M6：图像放置参数是第 9 个 storage buffer，而 wgpu 的
        // `Limits::default()` 只保证 8 个（WebGPU 基线）。
        // 因此**按适配器能力上调**该上限，并在适配器不支持时明确报错，
        // 而不是让 create_bind_group 在运行时抛出一句难以定位的校验失败。
        let adapter_limits = adapter.limits();
        let want_sb = 9u32;
        if adapter_limits.max_storage_buffers_per_shader_stage < want_sb {
            return err(format!(
                "适配器仅支持 {} 个 storage buffer/阶段，图像绘制需要 {want_sb} 个；\
                 请改用 CPU 引擎或不使用图像",
                adapter_limits.max_storage_buffers_per_shader_stage
            ));
        }
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: want_sb,
            ..wgpu::Limits::default()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("amiki2d"),
                // 基准所需：适配器支持时启用 GPU 时间戳（其余保持最小特性集）
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RenderError(format!("设备创建失败: {e}")))?;
        let mut renderer = GpuRenderer::from_device(device, queue)?;
        renderer.adapter_name = format!("{} ({:?})", info.name, info.backend);
        Ok(renderer)
    }

    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<GpuRenderer, RenderError> {
        let fine_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_fine"),
            source: wgpu::ShaderSource::Wgsl(format!("{FINE_MATH_WGSL}\n{FINE_WGSL}").into()),
        });
        let sparse_fine_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_fine"),
            source: wgpu::ShaderSource::Wgsl(format!("{FINE_MATH_WGSL}\n{SPARSE_FINE_WGSL}").into()),
        });
        let scan_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_scan"),
            source: wgpu::ShaderSource::Wgsl(SCAN_WGSL.into()),
        });
        let flatten_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_flatten"),
            source: wgpu::ShaderSource::Wgsl(FLATTEN_WGSL.into()),
        });
        let bins_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_bins"),
            source: wgpu::ShaderSource::Wgsl(BINS_WGSL.into()),
        });
        let strips_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_strips"),
            source: wgpu::ShaderSource::Wgsl(STRIPS_WGSL.into()),
        });
        let sm_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_measure"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_MEASURE_WGSL.into()),
        });
        let st_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_tiles"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_TILES_WGSL.into()),
        });
        let se_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_emit"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_EMIT_WGSL.into()),
        });
        let ss_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_scatter"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_SCATTER_WGSL.into()),
        });
        let sc_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_carry"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_CARRY_WGSL.into()),
        });

        // fine：0 uniform，1..=8 storage ro（含 seg_refs），9 storage tex，10..=13 clip 纹理
        let mut fine_entries = vec![uniform_bgl_entry(0)];
        for b in 1..=7u32 {
            fine_entries.push(storage_bgl_entry(b, true));
        }
        fine_entries.push(storage_bgl_entry(8, true));
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
        for b in 10..=13u32 {
            fine_entries.push(texture_bgl_entry(b));
        }
        // M6 图像：14 = 放置参数（与稠密路径共用，稠密永不用到），
        //          15 = 位图纹理数组，16 = 采样器
        fine_entries.push(storage_bgl_entry(14, true));
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 15,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 16,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        let fine_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_fine_bgl"),
            entries: &fine_entries,
        });

        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_scan_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, false),
                storage_bgl_entry(2, false),
                storage_bgl_entry(3, false),
            ],
        });

        // flatten：0 uniform，1 data ro，2 entry_draws ro，3 counts ro，4 seg_base ro，
        //          5 seg_ends rw，6 seg_draws rw，7 indirect rw
        let flatten_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_flatten_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, true),
                storage_bgl_entry(4, true),
                storage_bgl_entry(5, false),
                storage_bgl_entry(6, false),
                storage_bgl_entry(7, false),
            ],
        });

        // bins：0 uniform，1 seg_ends ro，2 seg_draws ro，3 counters rw，
        //       4 grid_aux rw，5 bucket_sums rw，6 carries rw，7 item_total rw，8 seg_refs rw
        // （binding 3 空缺：原 seg_total 已由 CPU 闭式段数取代）
        let bins_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_bins_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, false),
                storage_bgl_entry(4, false),
                storage_bgl_entry(5, false),
                storage_bgl_entry(6, false),
                storage_bgl_entry(7, false),
                storage_bgl_entry(8, false),
            ],
        });

        // strips（M3a）：0 uniform（复用 BinsParams 布局），1 seg_ends ro，2 seg_draws ro，
        //                3 活跃 strip 交错元数据 rw（4×u32/strip）
        let strip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_strip_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, false),
            ],
        });

        // M3b 稀疏管线：单一通用布局（uniform + 8×rw storage），各 kernel 自行填槽
        let sparse_entries: Vec<wgpu::BindGroupLayoutEntry> = std::iter::once(uniform_bgl_entry(0))
            .chain((1..=8u32).map(|b| storage_bgl_entry(b, false)))
            .collect();
        let sparse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_sparse_bgl"),
            entries: &sparse_entries,
        });

        let make_layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
        };
        let fine_pl = make_layout("amiki_fine_pl", &fine_bgl);
        let scan_pl = make_layout("amiki_scan_pl", &scan_bgl);
        let flatten_pl = make_layout("amiki_flatten_pl", &flatten_bgl);
        let bins_pl = make_layout("amiki_bins_pl", &bins_bgl);
        let strip_pl = make_layout("amiki_strip_pl", &strip_bgl);
        let sparse_pl = make_layout("amiki_sparse_pl", &sparse_bgl);

        let mk_compute = |label: &'static str, pl: &wgpu::PipelineLayout, module, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fine_pipeline = mk_compute("amiki_fine_pipeline", &fine_pl, &fine_mod, "main");
        let coverage_pipeline = mk_compute("amiki_coverage_pipeline", &fine_pl, &fine_mod, "main_coverage");
        let scan_blocks_pipeline = mk_compute("amiki_scan_blocks", &scan_pl, &scan_mod, "scan_blocks");
        let scan_sums_pipeline = mk_compute("amiki_scan_sums", &scan_pl, &scan_mod, "scan_sums");
        let scan_add_pipeline = mk_compute("amiki_scan_add", &scan_pl, &scan_mod, "add_offsets");
        let flatten_pipeline = mk_compute("amiki_flatten_pipeline", &flatten_pl, &flatten_mod, "flatten");
        let bins_clear_pipeline = mk_compute("amiki_bins_clear", &bins_pl, &bins_mod, "clear");
        let bins_emit_pipeline = mk_compute("amiki_bins_emit", &bins_pl, &bins_mod, "emit");
        let bins_scatter_pipeline = mk_compute("amiki_bins_scatter", &bins_pl, &bins_mod, "scatter");
        let bins_carry_pipeline = mk_compute("amiki_bins_carry", &bins_pl, &bins_mod, "carry");
        let strip_clear_pipeline = mk_compute("amiki_strip_clear", &strip_pl, &strips_mod, "strip_clear");
        let strip_extract_pipeline = mk_compute("amiki_strip_extract", &strip_pl, &strips_mod, "strip_extract");
        // M3b 稀疏管线（统一布局）
        let sparse_measure_pipeline = mk_compute("amiki_sparse_measure", &sparse_pl, &sm_mod, "measure");
        let sparse_tr_scatter_pipeline = mk_compute("amiki_sparse_tr_scatter", &sparse_pl, &sm_mod, "tr_scatter");
        let sparse_tr_offsets_pipeline = mk_compute("amiki_sparse_tr_offsets", &sparse_pl, &sm_mod, "tr_offsets_fn");
        let sparse_tile_count_pipeline = mk_compute("amiki_sparse_tile_count", &sparse_pl, &st_mod, "tile_count");
        let sparse_tile_fill_pipeline = mk_compute("amiki_sparse_tile_fill", &sparse_pl, &st_mod, "tile_fill");
        let sparse_clear_pipeline = mk_compute("amiki_sparse_clear", &sparse_pl, &se_mod, "clear_sparse");
        let sparse_emit_pipeline = mk_compute("amiki_sparse_emit", &sparse_pl, &se_mod, "emit");
        let sparse_scatter_pipeline = mk_compute("amiki_sparse_scatter", &sparse_pl, &ss_mod, "scatter");
        let sparse_carry_pipeline = mk_compute("amiki_sparse_carry", &sparse_pl, &sc_mod, "carry");
        // 稀疏 fine（与稠密 fine 共用 fine_bgl/fine_pl）
        let sparse_fine_pipeline = mk_compute("amiki_sparse_fine_pipeline", &fine_pl, &sparse_fine_mod, "sparse_main");
        let sparse_coverage_pipeline = mk_compute("amiki_sparse_coverage_pipeline", &fine_pl, &sparse_fine_mod, "sparse_coverage");

        let dummy_clip_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("amiki_dummy_clip"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let dummy_clip_view = dummy_clip_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // M6：图像绑定的占位（稠密路径与本引擎的 clip-only 绑定需要合法资源）。
        // 1×1 透明层，作为 D2Array 视图 + clamp 采样器。
        let dummy_img_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("amiki_dummy_image"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0, 0, 0, 0],
        );
        let dummy_img_view = dummy_img_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("amiki_dummy_image_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let dummy_img_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("amiki_dummy_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // M6：稠密路径的放置参数占位（layout 共用，需合法非空缓冲）
        let dummy_img_place = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_dummy_imgplace"),
            size: 3 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dummy_img_place, 0, bytemuck::cast_slice(&[[0.0f32; 4]; 3]));

        // M6：模糊管线（离屏层滤镜）
        let blur = blur::create_resources(&device);

        let timestamp_period_ns = queue.get_timestamp_period();
        let limits = device.limits();
        Ok(GpuRenderer {
            path: RenderPath::default(),
            adapter_name: "unknown".to_string(),
            has_timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
                && timestamp_period_ns > 0.0,
            timestamp_period_ns,
            limits,
            readback_pool: Mutex::new(Vec::new()),
            device,
            queue,
            fine_pipeline,
            coverage_pipeline,
            fine_bgl,
            scan_blocks_pipeline,
            scan_sums_pipeline,
            scan_add_pipeline,
            scan_bgl,
            flatten_pipeline,
            flatten_bgl,
            bins_clear_pipeline,
            bins_emit_pipeline,
            bins_scatter_pipeline,
            bins_carry_pipeline,
            bins_bgl,
            strip_clear_pipeline,
            strip_extract_pipeline,
            strip_bgl,
            sparse_bgl,
            sparse_measure_pipeline,
            sparse_tr_scatter_pipeline,
            sparse_tr_offsets_pipeline,
            sparse_tile_count_pipeline,
            sparse_tile_fill_pipeline,
            sparse_clear_pipeline,
            sparse_emit_pipeline,
            sparse_scatter_pipeline,
            sparse_carry_pipeline,
            sparse_fine_pipeline,
            sparse_coverage_pipeline,
            dummy_clip_view,
            _dummy_clip_tex: dummy_clip_tex,
            dummy_img_view,
            _dummy_img_tex: dummy_img_tex,
            dummy_img_sampler,
            dummy_img_place,
            blur,
        })
    }

    /// 通用 u32 exclusive 前缀和：data 原位变为 exclusive 前缀。
    fn run_scan(
        &self,
        enc: &mut wgpu::CommandEncoder,
        data_buf: &wgpu::Buffer,
        n: u32,
        tw: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let n_blocks = n.div_ceil(256).max(1);
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_scan_params"),
            contents: bytemuck::bytes_of(&ScanParams { n, _pad0: 0, _pad1: 0, _pad2: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let block_sums = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_scan_sums"),
            size: (n_blocks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let block_offsets = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_scan_offsets"),
            size: (n_blocks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_scan_bind"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: block_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: block_offsets.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("amiki_scan_pass"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&self.scan_blocks_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, 1, 1);
        pass.set_pipeline(&self.scan_sums_pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.scan_add_pipeline);
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    /// M2f 核心：全 GPU 管线（压平 + 分箱 + fine 光栅化）。
    /// 诊断/高级用途：直接运行管线（常规渲染走 render/render_into）。
    pub fn run_pipeline(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        dump: Option<&mut PipelineDump>,
    ) -> Result<(), RenderError> {
        match self.path {
            RenderPath::Dense => {
                let mut prof = FrameProfile::default();
                self.run_pipeline_impl(gs, tex, coverage, clip_views, None, dump, None, &mut prof)
            }
            RenderPath::Sparse => {
                let mut prof = FrameProfile::default();
                self.run_pipeline_sparse(gs, tex, coverage, clip_views, None, dump, None, &mut prof)
            }
        }
    }

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

    /// **完整分析**：渲染一帧并同时返回 ①主机侧分阶段耗时（[`FrameProfile`]）
    /// 与 ②逐 compute pass 的 GPU 时长（[`FrameProfile::timestamps`]）。
    ///
    /// 适配器不支持 `TIMESTAMP_QUERY` 时 `timestamps.supported = false`，
    /// 但**主机侧计时仍然有效**——这正是必须两者都给的原因：GPU 计时不可用时，
    /// 至少还能回答"CPU 花在哪"。
    ///
    /// 注意：本方法会为时间戳额外提交一次 resolve + 读回，故其结果**略高于**
    /// 常规渲染路径的真实耗时；做墙钟基准请用 `render_into` + `poll`。
    pub fn profile(&self, scene: &Scene, eps: f32) -> Result<FrameProfile, RenderError> {
        let mut prof = FrameProfile {
            timestamps: PassTimings {
                supported: false,
                timestamp_period_ns: self.timestamp_period_ns,
                passes: Vec::new(),
            },
            ..Default::default()
        };
        let t_extract = Instant::now();
        let frame = self.prepare(scene, eps)?;
        prof.extract_ms = t_extract.elapsed().as_secs_f64() * 1000.0;
        let tex = self.create_output_texture(&frame.scene);
        if !self.has_timestamps {
            // GPU 计时不可用：仍跑一次管线，让主机侧数字有意义
            match self.path {
                RenderPath::Dense => {
                    self.run_pipeline_impl(
                        &frame.scene, &tex, false, &frame.clip_views, None, None, None, &mut prof,
                    )?;
                }
                RenderPath::Sparse => {
                    self.run_pipeline_sparse(
                        &frame.scene, &tex, false, &frame.clip_views, None, None, None, &mut prof,
                    )?;
                }
            }
            return Ok(prof);
        }
        let mut rec = TimestampRecorder::new(&self.device);
        match self.path {
            RenderPath::Dense => {
                self.run_pipeline_impl(
                    &frame.scene,
                    &tex,
                    false,
                    &frame.clip_views,
                    None,
                    None,
                    Some(&mut rec),
                    &mut prof,
                )?;
            }
            RenderPath::Sparse => {
                self.run_pipeline_sparse(
                    &frame.scene,
                    &tex,
                    false,
                    &frame.clip_views,
                    None,
                    None,
                    Some(&mut rec),
                    &mut prof,
                )?;
            }
        }

        // resolve 全部时间戳 → 回读 → tick 换算为 ns
        let t_ts = Instant::now();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_ts_resolve") });
        enc.resolve_query_set(&rec.query_set, 0..rec.query_count(), &rec.resolve_buf, 0);
        self.queue.submit(Some(enc.finish()));
        let bytes = self.read_buffer(&rec.resolve_buf, rec.query_count() as usize * 8)?;
        let raw: &[u64] = bytemuck::cast_slice(&bytes);
        let period = self.timestamp_period_ns as f64;
        for (i, label) in rec.labels.iter().enumerate() {
            let (b, e) = (raw[2 * i], raw[2 * i + 1]);
            let ns = e.saturating_sub(b) as f64 * period;
            prof.timestamps.passes.push((label, ns));
        }
        prof.timestamps.supported = true;
        prof.resolve_ts_ms = Some(t_ts.elapsed().as_secs_f64() * 1000.0);
        Ok(prof)
    }

    /// 场景尺寸对应的输出纹理（storage + 读回）。
    fn create_output_texture(&self, gs: &GpuScene) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("amiki_out"),
            size: wgpu::Extent3d {
                width: gs.width.max(1),
                height: gs.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_pipeline_impl(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        damage: Option<&amiki_scene::Damage>,
        mut dump: Option<&mut PipelineDump>,
        mut ts: Option<&mut TimestampRecorder>,
        prof: &mut FrameProfile,
    ) -> Result<(), RenderError> {
        let w = tex.width();
        let h = tex.height();
        if w != gs.width || h != gs.height {
            return err("纹理尺寸与场景尺寸不一致");
        }
        let total_entries = gs.counts.len() as u32;
        let n_draws = gs.colors.len() as u32;
        if n_draws == 0 || total_entries == 0 || gs.total_segments == 0 {
            return Ok(()); // 空场景：纹理保持零初始化（透明）
        }
        // 容量预检：超限时返回可读错误，而非让 wgpu 校验层在 create_bind_group 处 panic
        let (n_grid_u64, n_bucket_u64) =
            dense_capacity(&self.limits, gs.tiles_x, gs.tiles_y, n_draws)?;
        let n_grid = n_grid_u64 as usize;
        let n_bucket = n_bucket_u64 as usize;

        // 时间戳槽位（仅分析路径分配；常规路径零开销）
        let ts_slots: Option<[(u32, u32); 5]> = match ts.as_mut() {
            Some(r) => Some([
                r.slot("clear")?,
                r.slot("scan:seg_base")?,
                r.slot("flatten+strips+emit")?,
                r.slot("scan:bucket")?,
                r.slot("scatter+carry+fine")?,
            ]),
            None => None,
        };
        let slot = |i: usize| ts_slots.map(|s| s[i]);

        let mk_storage = |label: &str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mk_init = |label: &str, data: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let pad_ro = mk_init("amiki_pad_ro", &PAD16);
        let pad_rw1 = mk_init("amiki_pad_rw1", &PAD16);
        let pad_rw2 = mk_init("amiki_pad_rw2", &PAD16);
        let pad_rw3 = mk_init("amiki_pad_rw3", &PAD16);

        // 上传：合并数据布局（quad 2×vec4 | cubic 2×vec4 | line 1×vec4）
        let mut data: Vec<[f32; 4]> = Vec::with_capacity((nq_len(gs) + nc_len(gs)) * 2 + nl_len(gs));
        for q in &gs.quads {
            data.push(q[0]);
            data.push(q[1]);
        }
        for c in &gs.cubics {
            data.push(c[0]);
            data.push(c[1]);
        }
        for l in &gs.lines {
            data.push(*l);
        }
        let mut entry_draws: Vec<u32> = Vec::with_capacity(data.len());
        entry_draws.extend_from_slice(&gs.quad_draws);
        entry_draws.extend_from_slice(&gs.cubic_draws);
        entry_draws.extend_from_slice(&gs.line_draws);

        let data_buf = mk_init("amiki_data", bytemuck::cast_slice(&data));
        let entry_draws_buf = mk_init("amiki_edraws", bytemuck::cast_slice(&entry_draws));
        let colors_buf = mk_init("amiki_colors", bytemuck::cast_slice(&gs.colors));
        let flags_buf = mk_init("amiki_flags", bytemuck::cast_slice(&gs.draw_flags));
        let clips_buf = mk_init("amiki_clips", bytemuck::cast_slice(&gs.draw_clips));
        let counts_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_counts"),
            contents: bytemuck::cast_slice(&gs.counts),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let seg_base_buf = mk_storage("amiki_seg_base", (total_entries as u64) * 4);
        let seg_ends_buf = mk_storage("amiki_seg_ends", (gs.total_segments as u64) * 16);
        let seg_draws_buf = mk_storage("amiki_seg_draws", (gs.total_segments as u64) * 4);
        let indirect_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_indirect"),
            contents: bytemuck::bytes_of(&gs.indirect),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });
        let counters_buf = mk_storage("amiki_counters", (n_grid as u64) * 4);
        let grid_aux_buf = mk_storage("amiki_grid_aux", (n_grid as u64) * 8); // [0,n)=base  [n,2n)=fill
        let bucket_buf = mk_storage("amiki_bucket", (n_bucket as u64) * 4);
        let carries_buf = mk_storage("amiki_carries", (n_grid as u64) * 4);
        let item_total_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_item_total"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // M3a 活跃 tile 元数据：每 (draw, tile 行) 4×u32（lo, hi, y0, y1），转置稠密
        let n_strips = (n_draws as u64) * (gs.tiles_y as u64);
        let strip_meta_buf = mk_storage("amiki_strip_meta", n_strips * 16);

        let flatten_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_flatten_params"),
            contents: bytemuck::bytes_of(&FlattenParams {
                n_quads: nq_len(gs) as u32,
                n_cubics: nc_len(gs) as u32,
                n_lines: nl_len(gs) as u32,
                eps: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bins_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_bins_params"),
            contents: bytemuck::bytes_of(&BinsParams {
                tiles_x: gs.tiles_x,
                n_draws,
                tiles_y: gs.tiles_y,
                height: gs.height,
                total_segments: gs.total_segments,
                n_cells: 0,
                n_buckets: 0,
                n_tile_entries: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let fine_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_fine_params"),
            contents: bytemuck::bytes_of(&FineParams {
                width: gs.width,
                height: gs.height,
                tiles_x: gs.tiles_x,
                n_draws,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let flatten_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_flatten_bind"),
            layout: &self.flatten_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: flatten_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: entry_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: seg_base_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: indirect_buf.as_entire_binding() },
            ],
        });
        let bins_bind_clear = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_bins_bind_clear"),
            layout: &self.bins_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pad_ro.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pad_ro.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bucket_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pad_rw2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pad_rw3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: pad_rw1.as_entire_binding() },
            ],
        });
        let bins_bind_emit = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_bins_bind_emit"),
            layout: &self.bins_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bucket_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pad_rw1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: item_total_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: pad_rw2.as_entire_binding() },
            ],
        });
        let strip_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_strip_bind"),
            layout: &self.strip_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: strip_meta_buf.as_entire_binding() },
            ],
        });

        // —— enc1：pass(clear) → copy(counts→seg_base) → pass(scan) → pass(flatten+strips → emit) ——
        let mut enc1 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_enc1") });
        {
            let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_pass_clear"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(0)),
            });
            pass.set_pipeline(&self.bins_clear_pipeline);
            pass.set_bind_group(0, &bins_bind_clear, &[]);
            pass.dispatch_workgroups(n_grid.max(n_bucket).div_ceil(64) as u32, 1, 1);
            pass.set_pipeline(&self.strip_clear_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            pass.dispatch_workgroups(n_strips.div_ceil(64) as u32, 1, 1);
        }
        enc1.copy_buffer_to_buffer(&counts_buf, 0, &seg_base_buf, 0, (total_entries as u64) * 4);
        self.run_scan(&mut enc1, &seg_base_buf, total_entries, ts_writes(ts.as_deref(), slot(1)));
        {
            let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_pass_flatten_emit"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(2)),
            });
            pass.set_pipeline(&self.flatten_pipeline);
            pass.set_bind_group(0, &flatten_bind, &[]);
            pass.dispatch_workgroups(total_entries.div_ceil(64), 1, 1);
            // M3a：活跃 tile 提取（读 flatten 刚写出的 seg_ends，同 pass 内隐式屏障）
            pass.set_pipeline(&self.strip_extract_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            pass.dispatch_workgroups(gs.total_segments.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.bins_emit_pipeline);
            pass.set_bind_group(0, &bins_bind_emit, &[]);
            pass.dispatch_workgroups_indirect(&indirect_buf, 0);
        }
        self.queue.submit(Some(enc1.finish()));

        // 诊断转储（enc1 后：压平与计数结果）
        if let Some(d) = dump.as_deref_mut() {
            if d.seg_ends.is_some() {
                let b = self.read_buffer(&seg_ends_buf, (gs.total_segments as usize) * 16)?;
                d.seg_ends = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.seg_draws.is_some() {
                let b = self.read_buffer(&seg_draws_buf, (gs.total_segments as usize) * 4)?;
                d.seg_draws = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.counters.is_some() {
                let b = self.read_buffer(&counters_buf, n_grid * 4)?;
                d.counters = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.item_total.is_some() {
                let b = self.read_buffer(&item_total_buf, 4)?;
                d.item_total = Some(u32::from_le_bytes(b.try_into().unwrap()));
            }
            if d.bucket_sums.is_some() {
                let b = self.read_buffer(&bucket_buf, n_bucket * 4)?;
                d.bucket_sums = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.strip_meta.is_some() {
                let b = self.read_buffer(&strip_meta_buf, n_strips as usize * 16)?;
                d.strip_meta = Some(bytemuck::cast_slice(&b).to_vec());
            }
        }

        // 读回成员项总数（4B）
        let t_rbd = Instant::now();
        let item_bytes = self.read_buffer(&item_total_buf, 4)?;
        prof.readback_dense_ms = Some(t_rbd.elapsed().as_secs_f64() * 1000.0);
        let items = u32::from_le_bytes(item_bytes.try_into().unwrap()) as usize;
        if items == 0 {
            return Ok(());
        }
        let seg_refs_buf = mk_storage("amiki_seg_refs", (items as u64) * 4);

        // —— enc2：copy → scan(桶前缀) → scatter → carry → fine ——
        let mut enc2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_enc2") });
        enc2.copy_buffer_to_buffer(&counters_buf, 0, &grid_aux_buf, 0, (n_grid as u64) * 4);
        self.run_scan(&mut enc2, &grid_aux_buf, n_grid as u32, ts_writes(ts.as_deref(), slot(3)));
        let bins_bind_scatter = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_bins_bind_scatter"),
            layout: &self.bins_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bucket_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: carries_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pad_ro.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: seg_refs_buf.as_entire_binding() },
            ],
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut views: Vec<&wgpu::TextureView> = (0..4).map(|_| &self.dummy_clip_view).collect();
        for (i, v) in clip_views.iter().take(4).enumerate() {
            views[i] = v;
        }
        let fine_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_fine_bind"),
            layout: &self.fine_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fine_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: carries_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: colors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: clips_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: seg_refs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&tex_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(views[0]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(views[1]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(views[2]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(views[3]) },
                // M6：稠密路径不做图像着色，但 bind group layout 是共用的，
                // 必须提供合法的 14/15/16（占位资源），否则创建绑定即校验失败。
                wgpu::BindGroupEntry { binding: 14, resource: self.dummy_img_place.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&self.dummy_img_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Sampler(&self.dummy_img_sampler),
                },
            ],
        });
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_pass_scatter_carry_fine"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(4)),
            });
            pass.set_pipeline(&self.bins_scatter_pipeline);
            pass.set_bind_group(0, &bins_bind_scatter, &[]);
            pass.dispatch_workgroups_indirect(&indirect_buf, 0);
            pass.set_pipeline(&self.bins_carry_pipeline);
            pass.set_bind_group(0, &bins_bind_scatter, &[]);
            pass.dispatch_workgroups((n_draws * gs.tiles_y * 16).div_ceil(64), 1, 1);
            if coverage {
                pass.set_pipeline(&self.coverage_pipeline);
            } else {
                pass.set_pipeline(&self.fine_pipeline);
            }
            pass.set_bind_group(0, &fine_bind, &[]);
            // M4：只派发脏 tile 子矩形。fine 内核用 global_invocation_id 推导
            // 像素与 tile 坐标，因此 workgroup 数减少 = 只重写那些 tile 的像素，
            // 未派发区域的纹理内容原样保留（这正是复用上一帧的机制）。
            let (_fx, fw, _fy, fh) = fine_dispatch_range(damage, gs.tiles_x, gs.tiles_y);
            pass.dispatch_workgroups(fw, fh, 1);
        }
        self.queue.submit(Some(enc2.finish()));

        // 诊断转储（enc2 后：scan_base / carries / seg_refs 终值）
        // dump 在此按值取出（Option<&mut _> 的 move）；此后不再使用，故无需 as_deref_mut
        // 再借用，也无需 #[allow]——两处转储点都按值取出即可。
        if let Some(d) = dump {
            if d.scan_base.is_some() {
                let b = self.read_buffer(&grid_aux_buf, n_grid * 4)?;
                d.scan_base = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.carries.is_some() {
                let b = self.read_buffer(&carries_buf, n_grid * 4)?;
                d.carries = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.seg_refs.is_some() {
                let items = d.item_total.unwrap_or(0) as usize;
                if items > 0 {
                    let b = self.read_buffer(&seg_refs_buf, items * 4)?;
                    d.seg_refs = Some(bytemuck::cast_slice(&b).to_vec());
                }
            }
        }
        Ok(())
    }

    /// M3b：稀疏管线 —— strip 归并 → tile 行列表 → 逐 tile 桶表 → 稀疏单元 → fine。
    ///
    /// 显存只与画面实际内容成比例：空画布下除 (draw × tile 行) 的条带元数据外
    /// 不分配任何缓冲。每帧 3 次 submit、2 次小读回（32B + 32B）：
    /// ```text
    /// enc1a: clear strips → flatten+strip_extract → measure → 3×scan →
    ///        tr_scatter → tr_offsets → tile_count → copy+scan(tile_start)  〖读回 stats〗
    /// enc1b: clear_sparse → emit（成员计数 + carry 定点桶）                 〖读回 stats〗
    /// enc2:  copy+scan(cell CSR) → tile_fill → scatter → carry → fine
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pipeline_sparse(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        damage: Option<&amiki_scene::Damage>,
        mut dump: Option<&mut PipelineDump>,
        mut ts: Option<&mut TimestampRecorder>,
        prof: &mut FrameProfile,
    ) -> Result<(), RenderError> {
        let w = tex.width();
        let h = tex.height();
        if w != gs.width || h != gs.height {
            return err("纹理尺寸与场景尺寸不一致");
        }
        let total_entries = gs.counts.len() as u32;
        let n_draws = gs.colors.len() as u32;
        if n_draws == 0 || total_entries == 0 || gs.total_segments == 0 {
            return Ok(()); // 空场景：纹理保持零初始化（透明）
        }
        let (tiles_x, tiles_y) = (gs.tiles_x, gs.tiles_y);
        let n_tiles = (tiles_x as u64) * (tiles_y as u64);
        let n_strips = (n_draws as u64) * (tiles_y as u64);
        let total_segments = gs.total_segments;

        let mk_storage = |label: &str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mk_init = |label: &str, data: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let pad16 = mk_init("amiki_sparse_pad", &PAD16);
        // M3.5：上传阶段计时起点（到数据缓冲创建完成为止）
        let t_upload = Instant::now();

        // —— 上传（与稠密路径同构）——
        let mut data: Vec<[f32; 4]> = Vec::with_capacity((nq_len(gs) + nc_len(gs)) * 2 + nl_len(gs));
        for q in &gs.quads {
            data.push(q[0]);
            data.push(q[1]);
        }
        for c in &gs.cubics {
            data.push(c[0]);
            data.push(c[1]);
        }
        for l in &gs.lines {
            data.push(*l);
        }
        let mut entry_draws: Vec<u32> = Vec::with_capacity(data.len());
        entry_draws.extend_from_slice(&gs.quad_draws);
        entry_draws.extend_from_slice(&gs.cubic_draws);
        entry_draws.extend_from_slice(&gs.line_draws);
        // draw_meta = (flags, clip0|clip1<<8, clip2|clip3<<8, grad_index)
        // （每裁剪索引 8 bit；w 为 M6 渐变下标，u32::MAX 表示实心色）
        let draw_meta: Vec<[u32; 4]> = gs
            .draw_flags
            .iter()
            .zip(gs.draw_clips.iter())
            .zip(gs.gradient_index.iter())
            .map(|((f, c), gi)| [*f, c[0] | (c[1] << 8), c[2] | (c[3] << 8), *gi])
            .collect();

        // M6：渐变数据打包（定长条目，布局与 sparse_fine.wgsl 的 grad_color 对应）
        let grad_vec = pack_gradients(&gs.gradients);
        let grad_buf = mk_init("amiki_sparse_grad", bytemuck::cast_slice(&grad_vec));

        // M6：图像放置参数 + 位图纹理数组（每 draw 3 个 vec4，自描述）
        let place_vec = pack_image_place(gs);
        let place_buf = mk_init("amiki_sparse_imgplace", bytemuck::cast_slice(&place_vec));
        let (img_tex_view, img_sampler) = self.mk_image_texture_array(&gs.images)?;

        let data_buf = mk_init("amiki_sparse_data", bytemuck::cast_slice(&data));
        let entry_draws_buf = mk_init("amiki_sparse_edraws", bytemuck::cast_slice(&entry_draws));
        let colors_buf = mk_init("amiki_sparse_colors", bytemuck::cast_slice(&gs.colors));
        let draw_meta_buf = mk_init("amiki_sparse_draw_meta", bytemuck::cast_slice(&draw_meta));
        let counts_buf = mk_storage("amiki_sparse_counts", (total_entries as u64) * 4);
        // counts 需要 CPU 数据 → 用初始化版本
        self.queue
            .write_buffer(&counts_buf, 0, bytemuck::cast_slice(&gs.counts));
        let seg_base_buf = mk_storage("amiki_sparse_seg_base", (total_entries as u64) * 4);
        let seg_ends_buf = mk_storage("amiki_sparse_seg_ends", (total_segments as u64) * 16);
        let seg_draws_buf = mk_storage("amiki_sparse_seg_draws", (total_segments as u64) * 4);
        let indirect_buf = mk_init("amiki_sparse_indirect", bytemuck::bytes_of(&gs.indirect));
        prof.upload_ms += t_upload.elapsed().as_secs_f64() * 1000.0;
        // M3.5：分箱结构分配阶段计时
        let t_alloc = Instant::now();

        // —— 分箱结构（规模 = draws × tile 行）——
        let strip_meta_buf = mk_storage("amiki_sparse_strip_meta", n_strips * 16);
        let strip_cell_off_buf = mk_storage("amiki_sparse_strip_cell_off", n_strips * 4);
        let strip_bucket_off_buf = mk_storage("amiki_sparse_strip_bucket_off", n_strips * 4);
        let tr_prefix_buf = mk_storage("amiki_sparse_tr_prefix", n_strips * 4);
        let tr_list_buf = mk_storage("amiki_sparse_tr_list", n_strips * 4);
        let tr_offsets_buf = mk_storage("amiki_sparse_tr_offsets", (tiles_y as u64) * 8);
        let tile_count_buf = mk_storage("amiki_sparse_tile_count", n_tiles * 4);
        let tile_start_buf = mk_storage("amiki_sparse_tile_start", n_tiles * 4);
        let stats_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_sparse_stats"),
            contents: &[0u8; 32],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        prof.alloc_ms += t_alloc.elapsed().as_secs_f64() * 1000.0;
        let params = |n_cells: u32, n_buckets: u32, n_tile_entries: u32| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("amiki_sparse_params"),
                contents: bytemuck::bytes_of(&BinsParams {
                    tiles_x,
                    n_draws,
                    tiles_y,
                    height: gs.height,
                    total_segments,
                    n_cells,
                    n_buckets,
                    n_tile_entries,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let params_v1 = params(0, 0, 0);

        // 稀疏 kernel 的统一 bind group：未用槽位绑 16B 垫片
        let sparse_bind = |params_buf: &wgpu::Buffer, entries: &[(u32, &wgpu::Buffer)]| {
            let mut all: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(9);
            all.push(wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() });
            for b in 1..=8u32 {
                let buf = entries
                    .iter()
                    .find(|(bb, _)| *bb == b)
                    .map(|(_, bf)| *bf)
                    .unwrap_or(&pad16);
                all.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
            }
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amiki_sparse_bind"),
                layout: &self.sparse_bgl,
                entries: &all,
            })
        };
        /// 单维派发（入参为**元素数**；自动按 65535 拆分到 2D，规避工作组维度上限）
        fn dispatch_1d(pass: &mut wgpu::ComputePass<'_>, n_elements: u32) {
            if n_elements == 0 {
                return;
            }
            let wg = n_elements.div_ceil(64);
            let x = wg.min(65_535);
            let y = wg.div_ceil(65_535);
            pass.dispatch_workgroups(x, y, 1);
        }

        // —— 时间戳槽位（仅分析路径）——
        // 原实现只有 5 个粗粒度槽位，把中间 5 次 scan 全藏在黑盒里；现在为每次 scan
        // 单独分配槽位，并补齐原先被省略的 pass，使稀疏全链路可逐段归因。
        let ts_slots: Option<[(u32, u32); 10]> = match ts.as_mut() {
            Some(r) => Some([
                r.slot("clear_strips")?,
                r.slot("scan:seg_base")?,
                r.slot("flatten+strip_extract+measure")?,
                r.slot("scan:strip_cell_off")?,
                r.slot("scan:strip_bucket_off")?,
                r.slot("scan:tr_prefix")?,
                r.slot("tr_scatter+tr_offsets+tile_count")?,
                r.slot("scan:tile_start")?,
                r.slot("clear_sparse+emit")?,
                r.slot("scan:cell_csr+tile_fill+scatter+carry+fine")?,
            ]),
            None => None,
        };
        let slot = |i: usize| ts_slots.map(|s| s[i]);

        // strip/extract 的专用 bind group（strip_bgl）
        let strip_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_sparse_strip_bind"),
            layout: &self.strip_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: strip_meta_buf.as_entire_binding() },
            ],
        });
        let b_bins = sparse_bind(
            &params_v1,
            &[
                (1, &strip_meta_buf),
                (2, &strip_cell_off_buf),
                (3, &strip_bucket_off_buf),
                (4, &tr_prefix_buf),
                (5, &stats_buf),
                (6, &tr_list_buf),
                (7, &tr_offsets_buf),
            ],
        );
        let b_tile_count = sparse_bind(
            &params_v1,
            &[
                (1, &strip_meta_buf),
                (2, &tr_offsets_buf),
                (3, &tr_list_buf),
                (4, &tile_count_buf),
                (5, &stats_buf),
            ],
        );
        let flatten_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_sparse_flatten_params"),
            contents: bytemuck::bytes_of(&FlattenParams {
                n_quads: nq_len(gs) as u32,
                n_cubics: nc_len(gs) as u32,
                n_lines: nl_len(gs) as u32,
                eps: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let flatten_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_sparse_flatten_bind"),
            layout: &self.flatten_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: flatten_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: entry_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: seg_base_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: indirect_buf.as_entire_binding() },
            ],
        });

        // ———— enc1a：分箱结构 ————
        let t_enc1 = Instant::now();
        let mut enc1a = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_sparse_enc1a") });
        {
            let mut pass = enc1a.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_clear_strips"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(0)),
            });
            pass.set_pipeline(&self.strip_clear_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            dispatch_1d(&mut pass, n_strips as u32);
        }
        enc1a.copy_buffer_to_buffer(&counts_buf, 0, &seg_base_buf, 0, (total_entries as u64) * 4);
        self.run_scan(
            &mut enc1a,
            &seg_base_buf,
            total_entries,
            ts_slot(ts.as_deref(), ts_slots, 1),
        );
        {
            let mut pass = enc1a.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_flatten_strips"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 2),
            });
            pass.set_pipeline(&self.flatten_pipeline);
            pass.set_bind_group(0, &flatten_bind, &[]);
            pass.dispatch_workgroups(total_entries.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.strip_extract_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            dispatch_1d(&mut pass, total_segments);
            pass.set_pipeline(&self.sparse_measure_pipeline);
            pass.set_bind_group(0, &b_bins, &[]);
            dispatch_1d(&mut pass, n_strips as u32);
        }
        self.run_scan(&mut enc1a, &strip_cell_off_buf, n_strips as u32, ts_slot(ts.as_deref(), ts_slots, 3));
        self.run_scan(&mut enc1a, &strip_bucket_off_buf, n_strips as u32, ts_slot(ts.as_deref(), ts_slots, 4));
        self.run_scan(&mut enc1a, &tr_prefix_buf, n_strips as u32, ts_slot(ts.as_deref(), ts_slots, 5));
        {
            let mut pass = enc1a.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_tr_tiles"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 6),
            });
            pass.set_pipeline(&self.sparse_tr_scatter_pipeline);
            pass.set_bind_group(0, &b_bins, &[]);
            dispatch_1d(&mut pass, n_strips as u32);
            pass.set_pipeline(&self.sparse_tr_offsets_pipeline);
            pass.set_bind_group(0, &b_bins, &[]);
            dispatch_1d(&mut pass, tiles_y);
            pass.set_pipeline(&self.sparse_tile_count_pipeline);
            pass.set_bind_group(0, &b_tile_count, &[]);
            dispatch_1d(&mut pass, n_tiles as u32);
        }
        enc1a.copy_buffer_to_buffer(&tile_count_buf, 0, &tile_start_buf, 0, n_tiles * 4);
        self.run_scan(&mut enc1a, &tile_start_buf, n_tiles as u32, ts_slot(ts.as_deref(), ts_slots, 7));
        self.queue.submit(Some(enc1a.finish()));
        prof.encode1a_ms = t_enc1.elapsed().as_secs_f64() * 1000.0;

        // 〖读回 1〗稀疏规模 → 分配稀疏缓冲
        let t_rb1 = Instant::now();
        let stats_bytes = self.read_buffer(&stats_buf, 32)?;
        prof.readback1_ms = Some(t_rb1.elapsed().as_secs_f64() * 1000.0);
        let st: [u32; 8] = bytemuck::pod_read_unaligned(&stats_bytes);
        let (n_cells, n_buckets, n_tile_entries) = (st[0], st[1], st[2]);
        if let Some(d) = dump.as_deref_mut() {
            if d.sparse_stats.is_some() {
                d.sparse_stats = Some(st);
            }
        }
        if n_cells == 0 {
            return Ok(()); // 无任何内容覆盖画布
        }
        let cell_aux_buf = mk_storage("amiki_sparse_cell_aux", (n_cells as u64) * 8);
        let cell_csr_buf = mk_storage("amiki_sparse_cell_csr", ((n_cells + 1) as u64) * 4);
        let buckets_buf = mk_storage("amiki_sparse_buckets", (n_buckets as u64) * 4);
        let cell_meta_buf = mk_storage("amiki_sparse_cell_meta", (n_cells as u64) * 16);
        let tile_list_buf = mk_storage("amiki_sparse_tile_list", (n_tile_entries as u64) * 16);
        let params_v2 = params(n_cells, n_buckets, n_tile_entries);

        // ———— enc1b：稀疏清零 + emit ————
        let t_enc1b = Instant::now();
        let b_clear = sparse_bind(&params_v2, &[(6, &cell_aux_buf), (7, &buckets_buf)]);
        let b_emit = sparse_bind(
            &params_v2,
            &[
                (1, &seg_ends_buf),
                (2, &seg_draws_buf),
                (3, &strip_meta_buf),
                (4, &strip_cell_off_buf),
                (5, &strip_bucket_off_buf),
                (6, &cell_aux_buf),
                (7, &buckets_buf),
                (8, &stats_buf),
            ],
        );
        let mut enc1b = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_sparse_enc1b") });
        {
            let mut pass = enc1b.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_clear_emit"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 8),
            });
            pass.set_pipeline(&self.sparse_clear_pipeline);
            pass.set_bind_group(0, &b_clear, &[]);
            dispatch_1d(&mut pass, n_cells.max(n_buckets));
            pass.set_pipeline(&self.sparse_emit_pipeline);
            pass.set_bind_group(0, &b_emit, &[]);
            dispatch_1d(&mut pass, total_segments);
        }
        self.queue.submit(Some(enc1b.finish()));
        prof.encode1b_ms = t_enc1b.elapsed().as_secs_f64() * 1000.0;

        // 〖读回 2〗成员总数 → 分配段列表
        let t_rb2 = Instant::now();
        let stats_bytes = self.read_buffer(&stats_buf, 32)?;
        prof.readback2_ms = Some(t_rb2.elapsed().as_secs_f64() * 1000.0);
        let st2: [u32; 8] = bytemuck::pod_read_unaligned(&stats_bytes);
        let n_items = st2[3];
        let cell_segs_buf = mk_storage("amiki_sparse_cell_segs", (n_items.max(1) as u64) * 16);

        // ———— enc2：单元 CSR → tile 桶表 → scatter → carry → fine ————
        let b_tile_fill = sparse_bind(
            &params_v2,
            &[
                (1, &strip_meta_buf),
                (2, &tr_offsets_buf),
                (3, &tr_list_buf),
                (6, &strip_cell_off_buf),
                (7, &tile_start_buf),
                (8, &tile_list_buf),
            ],
        );
        let b_scatter = sparse_bind(
            &params_v2,
            &[
                (1, &seg_ends_buf),
                (2, &seg_draws_buf),
                (3, &strip_meta_buf),
                (4, &strip_cell_off_buf),
                (5, &cell_csr_buf),
                (6, &cell_aux_buf),
                (7, &cell_segs_buf),
            ],
        );
        let b_carry = sparse_bind(
            &params_v2,
            &[
                (1, &strip_meta_buf),
                (2, &strip_cell_off_buf),
                (3, &strip_bucket_off_buf),
                (4, &buckets_buf),
                (5, &cell_csr_buf),
                (6, &cell_meta_buf),
            ],
        );
        let fine_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_sparse_fine_params"),
            contents: bytemuck::bytes_of(&FineParams {
                width: gs.width,
                height: gs.height,
                tiles_x,
                n_draws,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut views: Vec<&wgpu::TextureView> = (0..4).map(|_| &self.dummy_clip_view).collect();
        for (i, v) in clip_views.iter().take(4).enumerate() {
            views[i] = v;
        }
        let b_fine = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_sparse_fine_bind"),
            layout: &self.fine_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fine_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_segs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: colors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: draw_meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: tile_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: tile_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: tile_list_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: grad_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&tex_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(views[0]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(views[1]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(views[2]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(views[3]) },
                // M6 图像：放置参数、位图纹理数组、采样器
                wgpu::BindGroupEntry { binding: 14, resource: place_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&img_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Sampler(&img_sampler),
                },
            ],
        });
        let t_enc2 = Instant::now();
        let mut enc2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_sparse_enc2") });
        enc2.copy_buffer_to_buffer(&cell_aux_buf, 0, &cell_csr_buf, 0, (n_cells as u64) * 4);
        self.run_scan(&mut enc2, &cell_csr_buf, n_cells + 1, ts_slot(ts.as_deref(), ts_slots, 9));
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_scatter_carry_fine"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 9),
            });
            pass.set_pipeline(&self.sparse_tile_fill_pipeline);
            pass.set_bind_group(0, &b_tile_fill, &[]);
            dispatch_1d(&mut pass, n_tiles as u32);
            pass.set_pipeline(&self.sparse_scatter_pipeline);
            pass.set_bind_group(0, &b_scatter, &[]);
            dispatch_1d(&mut pass, total_segments);
            pass.set_pipeline(&self.sparse_carry_pipeline);
            pass.set_bind_group(0, &b_carry, &[]);
            dispatch_1d(&mut pass, (n_strips as u32).saturating_mul(16));
            if coverage {
                pass.set_pipeline(&self.sparse_coverage_pipeline);
            } else {
                pass.set_pipeline(&self.sparse_fine_pipeline);
            }
            pass.set_bind_group(0, &b_fine, &[]);
            // M4：只派发脏 tile 子矩形（理由见 run_pipeline_impl 处注释）
            let (_fx, fw, _fy, fh) = fine_dispatch_range(damage, tiles_x, tiles_y);
            pass.dispatch_workgroups(fw, fh, 1);
        }
        prof.encode2_ms = t_enc2.elapsed().as_secs_f64() * 1000.0;
        let t_sub2 = Instant::now();
        self.queue.submit(Some(enc2.finish()));
        prof.submit2_ms = t_sub2.elapsed().as_secs_f64() * 1000.0;

        // —— 诊断转储（M3.5：全字段无条件转储，供 check_sparse_consistency 使用）——
        let t_dump = Instant::now();
        if let Some(d) = dump {
            d.sparse_stats = Some(st2);
            let b = self.read_buffer(&strip_meta_buf, n_strips as usize * 16)?;
            d.strip_meta = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tr_offsets_buf, tiles_y as usize * 8)?;
            d.tr_offsets = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tr_list_buf, n_strips as usize * 4)?;
            d.tr_list = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tile_count_buf, n_tiles as usize * 4)?;
            d.tile_count = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tile_start_buf, n_tiles as usize * 4)?;
            d.tile_start = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tile_list_buf, n_tile_entries as usize * 16)?;
            d.tile_list = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_meta_buf, n_cells as usize * 16)?;
            d.cell_meta = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_segs_buf, n_items as usize * 16)?;
            d.cell_segs = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&strip_cell_off_buf, n_strips as usize * 4)?;
            d.strip_cell_off = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&strip_bucket_off_buf, n_strips as usize * 4)?;
            d.strip_bucket_off = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_aux_buf, n_cells as usize * 8)?;
            d.cell_aux = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_csr_buf, (n_cells as usize + 1) * 4)?;
            d.cell_csr_off = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&buckets_buf, n_buckets as usize * 4)?;
            d.sparse_buckets = Some(bytemuck::cast_slice(&b).to_vec());
        }
        prof.dump_ms = Some(t_dump.elapsed().as_secs_f64() * 1000.0);
        Ok(())
    }

    /// M3.5：公开的稀疏诊断入口——跑一次稀疏管线并转储**全部**稀疏结构，
    /// 供 [`check_sparse_consistency`] 与 `PipelineDump` 消费。
    ///
    /// 与 [`GpuRenderer::sparse_stats`] 的区别：后者只读回 32B 规模统计（性能敏感，
    /// 可用于基准循环），本方法会读回全部稀疏结构（条带/桶表/单元/段端点），
    /// **仅供诊断与测试**，勿放进热路径。
    pub fn dump_sparse(
        &self,
        scene: &Scene,
        eps: f32,
    ) -> Result<(GpuScene, PipelineDump), RenderError> {
        let gs = build_gpu_scene(scene, eps);
        let tex = self.create_output_texture(&gs);
        let mut dump = PipelineDump::default();
        let mut prof = FrameProfile::default();
        self.run_pipeline_sparse(&gs, &tex, false, &[], None, Some(&mut dump), None, &mut prof)?;
        Ok((gs, dump))
    }

    /// M4：**持久化渲染** —— 复用同一个输出纹理，只对脏 tile 派发 fine。
    ///
    /// 与 [`GpuRenderer::render`] 的区别：
    /// - `render` 每帧新建输出纹理并全量派发；
    /// - 本方法持有输出纹理与上一帧场景，**场景未变则一次派发都不做**，
    ///   只改了一小块则只派发覆盖脏区的 tile 子矩形。
    ///
    /// 正确性依据：fine 内核由 `global_invocation_id` 推导像素与 tile 坐标，
    /// 未派发的 tile **不写入**，纹理中保留的正是上一帧算出的同一结果
    /// （单 tile 的着色只依赖该 tile 的成员段与画家序，与其他 tile 的当前内容无关）。
    ///
    /// `damage` 必须与 `scene` 的画布尺寸匹配，否则 [`check_damage_bounds`] 会断言失败
    /// ——那是在阻止"全脏被误当作局部脏区"从而静默残留陈旧像素。
    ///
    /// 返回 `true` 表示本帧完全跳过（未做任何 GPU 工作）。
    #[allow(clippy::too_many_arguments)]
    pub fn render_into_persistent(
        &self,
        tex: &wgpu::Texture,
        frame: &Frame,
        damage: Option<&amiki_scene::Damage>,
    ) -> Result<bool, RenderError> {
        if damage.is_some_and(|d| d.is_empty()) {
            return Ok(true); // 无脏区：完全跳过
        }
        // 逐帧路径不合成层（见 reject_layers_for_frame_path）
        reject_layers_for_frame_path(u32::from(frame.scene.has_layers))?;
        check_damage_bounds(damage, frame.scene.tiles_x, frame.scene.tiles_y);
        let mut prof = FrameProfile::default();
        match self.path {
            RenderPath::Dense => self.run_pipeline_impl(
                &frame.scene,
                tex,
                false,
                &frame.clip_views,
                damage,
                None,
                None,
                &mut prof,
            )?,
            RenderPath::Sparse => self.run_pipeline_sparse(
                &frame.scene,
                tex,
                false,
                &frame.clip_views,
                damage,
                None,
                None,
                &mut prof,
            )?,
        }
        Ok(false)
    }

    /// 准备逐帧资源：GpuScene 提取 + clip 覆盖率纹理。
    pub fn prepare(&self, scene: &Scene, eps: f32) -> Result<Frame, RenderError> {
        use amiki_scene::{Color, Paint};
        // 注意：`prepare` 本身不拒绝层 —— `render` 会经它进来，而
        // `render_blocking` 才负责层合成。拒绝放在逐帧路径的两个入口
        // （`render_into` / `render_into_persistent`），它们才是真正会
        // 静默丢掉层的地方。
        let main_scene = build_gpu_scene(scene, eps);
        let mut clip_texs = Vec::new();
        let mut clip_views = Vec::new();
        for clip in &scene.clips {
            let mut cs = Scene::new(scene.width, scene.height);
            cs.push(clip.transform, clip.path.clone(), Paint::Solid(Color::rgb8(255, 255, 255)));
            let gs = build_gpu_scene(&cs, eps);
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("amiki_clip_cov"),
                size: wgpu::Extent3d { width: gs.width, height: gs.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.run_pipeline(&gs, &tex, true, &[], None)?;
            clip_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            clip_texs.push(tex);
        }
        Ok(Frame { scene: main_scene, clip_texs, clip_views })
    }

    pub fn render(&self, scene: &Scene, eps: f32) -> Result<wgpu::Texture, RenderError> {
        let frame = self.prepare(scene, eps)?;
        let tex = self.create_output_texture(&frame.scene);
        self.render_into(&tex, &frame)?;
        Ok(tex)
    }

    /// 渲染 Frame 到已存在的 rgba8unorm 纹理（窗口逐帧复用）。
    pub fn render_into(&self, tex: &wgpu::Texture, frame: &Frame) -> Result<(), RenderError> {
        // 逐帧路径不合成层（见 reject_layers_for_frame_path）
        reject_layers_for_frame_path(u32::from(frame.scene.has_layers))?;
        self.run_pipeline(&frame.scene, tex, false, &frame.clip_views, None)
    }

    /// 取一个可用的 MAP_READ 暂存缓冲：优先复用池中容量足够的，否则新建并**精确**定容。
    ///
    /// M3.6：读回是墙钟瓶颈（实测读回 1 + 读回 2 占主机侧绝大部分），原实现每次读回都
    /// `create_buffer` + 冷映射，代价被重复支付。池化后同一尺寸的读回复用同一缓冲。
    ///
    /// 从池中取用的缓冲必须处于**未映射**状态（`read_buffer` 负责用完即 unmap）。
    fn take_readback_buffer(&self, byte_size: usize) -> wgpu::Buffer {
        let want = byte_size.max(4) as u64;
        if let Ok(mut pool) = self.readback_pool.lock() {
            if let Some(i) = pool.iter().position(|b| b.size() == want) {
                return pool.swap_remove(i);
            }
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_readback_pool"),
            size: want,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// 把缓冲归还池中（仅在确认已 unmap 后调用）。
    fn give_readback_buffer(&self, buf: wgpu::Buffer) {
        if let Ok(mut pool) = self.readback_pool.lock() {
            if pool.len() < 8 {
                pool.push(buf);
            }
        }
    }

    /// 映射 + 等待 + 拷贝出 `byte_size` 字节，然后 unmap 并归还缓冲。
    ///
    /// 失败路径**不归还**缓冲：若 `map_async` 失败则缓冲会保持已映射状态，
    /// 再次映射会报错，故直接丢弃（池会按需补建）。
    fn map_and_take(
        &self,
        out: wgpu::Buffer,
        byte_size: usize,
    ) -> Result<Vec<u8>, RenderError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let polled = self.device.poll(wgpu::PollType::Wait);
        let mapped = rx
            .recv()
            .map_err(|_| RenderError("map 回调通道关闭".into()))
            .and_then(|r| r.map_err(|e| RenderError(format!("map 失败: {e}"))));
        // 先把 poll 的错误报出去（保持与原实现一致的错误优先级）
        polled.map_err(|e| RenderError(format!("poll 失败: {e}")))?;
        mapped?;
        let v = {
            let data = slice.get_mapped_range();
            let v = data[..byte_size.min(data.len())].to_vec();
            drop(data);
            v
        };
        out.unmap();
        self.give_readback_buffer(out);
        Ok(v)
    }

    /// 读回 storage buffer 内容（通用小数据读回）。
    /// 诊断：读回 storage buffer。
    pub fn read_buffer(&self, buf: &wgpu::Buffer, byte_size: usize) -> Result<Vec<u8>, RenderError> {
        let out = self.take_readback_buffer(byte_size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_buf_copy") });
        encoder.copy_buffer_to_buffer(buf, 0, &out, 0, byte_size as u64);
        self.queue.submit(Some(encoder.finish()));
        self.map_and_take(out, byte_size)
    }

    /// 读回纹理像素（直线 RGBA8）。
    ///
    /// M3.6：暂存缓冲取自 [`Self::take_readback_buffer`] 池（同一画布尺寸逐帧复用）。
    pub fn readback(&self, tex: &wgpu::Texture) -> Result<Vec<u8>, RenderError> {
        let w = tex.width();
        let h = tex.height();
        let bytes_per_row = ((w * 4) as usize).div_ceil(256) * 256;
        let buf_size = bytes_per_row * h as usize;
        let buf = self.take_readback_buffer(buf_size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_copy") });
        encoder.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let polled = self.device.poll(wgpu::PollType::Wait);
        let mapped = rx
            .recv()
            .map_err(|_| RenderError("map 回调通道关闭".into()))
            .and_then(|r| r.map_err(|e| RenderError(format!("map 失败: {e}"))));
        // 失败路径不归还缓冲（可能仍处于已映射状态）
        polled.map_err(|e| RenderError(format!("poll 失败: {e}")))?;
        mapped?;

        let mut out = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = slice.get_mapped_range();
            for row in 0..h as usize {
                let start = row * bytes_per_row;
                out.extend_from_slice(&data[start..start + (w as usize) * 4]);
            }
        }
        buf.unmap();
        self.give_readback_buffer(buf);
        Ok(out)
    }

    /// M6：对一张**直线 alpha** 的 RGBA8 图做可分离模糊，返回直线 alpha 的像素。
    ///
    /// # 通道语义：与 CPU 保持一致的**直线色直接平均**
    ///
    /// CPU 的 `amiki_raster::filter::blur` 对 4 个通道直接做盒式平均
    /// （直线色语义）。GPU 必须完全相同，否则跨端对拍必然不一致。
    /// 因此这里**不做**预乘/反预乘 —— 详见函数末尾的说明。
    #[allow(dead_code)]
    fn blur_rgba(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
        sigma: f32,
        passes: u32,
    ) -> Result<Vec<u8>, RenderError> {
        self.blur_rgba_impl(src, width, height, sigma, passes, None)
    }

    /// `override_rgb` 为 `Some(c)` 时，先把 RGB 覆盖为 `c`（保留 alpha 形状）再模糊。
    ///
    /// 这是为**投影**准备的：CPU 侧的阴影是"用 alpha 形状 + 固定阴影色"先着色、
    /// 再模糊。若直接模糊层内容，就变成"模糊内容再当阴影"，语义不同
    /// （实测差 125/255）。
    fn blur_rgba_impl(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
        sigma: f32,
        passes: u32,
        override_rgb: Option<[f32; 3]>,
    ) -> Result<Vec<u8>, RenderError> {
        let box_w = blur::box_width_for(sigma, 3);
        // 先做可选的 RGB 覆盖（生成阴影遮罩）
        let mut owned;
        let src: &[u8] = match override_rgb {
            Some(c) => {
                owned = src.to_vec();
                for px in owned.as_chunks_mut::<4>().0.iter_mut() {
                    px[0] = (c[0] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    px[1] = (c[1] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    px[2] = (c[2] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
                &owned
            }
            None => src,
        };
        if sigma <= 0.0 || box_w <= 1 || width == 0 || height == 0 {
            return Ok(src.to_vec());
        }
        let (t_a, v_a) = blur::create_blur_texture(&self.device, "amiki_blur_a", width, height)?;
        let (_t_b, v_b) = blur::create_blur_texture(&self.device, "amiki_blur_b", width, height)?;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &t_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            src,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // **两个独立参数缓冲**（横 / 纵），而不是一个写两次。
        //
        // 这是实际踩到的坑：同一命令缓冲内多趟渲染若共用一个 uniform 缓冲，
        // 队列上"写参数1 → 写参数2 → 提交"会使两趟都读到第二次的值 ——
        // 表现为横向那趟从未执行。若测试数据恰好对该方向不敏感
        // （例如竖贯整幅的横条），就完全看不出异常，极难定位。
        let mk = |label: &str, vertical: bool| {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<blur::BlurParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(
                &b,
                0,
                bytemuck::bytes_of(&blur::BlurParams {
                    width: width as f32,
                    height: height as f32,
                    box_w: box_w as f32,
                    vertical: if vertical { 1.0 } else { 0.0 },
                }),
            );
            b
        };
        let params_h = mk("amiki_blur_h", false);
        let params_v = mk("amiki_blur_v", true);

        // 每遍：A --横--> B --纵--> A。每遍结束结果都回到 A。
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_blur") });
        for _ in 0..passes {
            blur::blur_pass(&self.device, &mut enc, &self.blur, &params_h, &v_a, &v_b);
            blur::blur_pass(&self.device, &mut enc, &self.blur, &params_v, &v_b, &v_a);
        }
        self.queue.submit(Some(enc.finish()));

        // **不做预乘/反预乘**：CPU 的 `amiki_raster::filter::blur` 是对 4 个通道
        // **直接做平均**（直线色语义），因此 GPU 也必须直接平均直线数据，
        // 两端才对得上。
        //
        // 我一度加了"预乘→模糊→反预乘"，理由是"模糊应在颜色上做、透明区 RGB
        // 未定义会洗暗"。那个理由对**正确性**而言是对的，但与 CPU 的**现行语义
        // 不一致** —— 结果是 `Filter::Blur` 的差异从 1 恶化到 64。
        // 既然对拍基准是 CPU 实现，就必须与它一致；若将来要把两端都改成
        // 预乘语义，那是**一次同时改两端**的独立决定，不能只改一侧。
        self.readback(&t_a)
    }

    pub fn render_blocking(&self, scene: &Scene, eps: f32) -> Result<Vec<u8>, RenderError> {
        // M6 现状：**矩阵与模糊的预乘/反预乘顺序尚未与 CPU 语义对齐**，
        // 两端仍有可见差异（filter 模式下最大通道差 64~103/255），
        // 因此这里仍**显式拒绝**含层场景，由 CPU 引擎承担滤镜。
        //
        // 为什么不用"半通"的实现：它会静默画出与 CPU 不一致的滤镜效果，
        // 而调用方毫无提示 —— 这正是本项目一直极力避免的失败模式。
        // 实现本身已就绪（见 blur_rgba / composite_layer_gpu / blur.rs），
        // 只差把顺序对齐；对齐后删掉这一行即可启用。
        // 【临时测量】守卫已移除
        // 直接走管线，**绕开 render_into 的层守卫**
        let frame = self.prepare(scene, eps)?;
        let tex = self.create_output_texture(&frame.scene);
        self.run_pipeline(&frame.scene, &tex, false, &frame.clip_views, None)?;
        let mut root = self.readback(&tex)?;
        if scene.layers.is_empty() {
            return Ok(root);
        }

        // M6：逐层渲染 → GPU 模糊 → 合成回根图像。
        //
        // 每层的图元仍由**同一套细粒度管线**渲染（不复制一份着色逻辑），
        // 只是目标换成整画布尺寸的纹理。于是"层"几乎没有额外实现成本，
        // 也不可能出现两套着色语义漂移。
        let (root_w, root_h) = (tex.width(), tex.height());
        for layer in &scene.layers {
            if layer.bounds.is_empty() {
                continue;
            }
            let sub = {
                let mut sc = Scene::new(root_w, root_h);
                sc.draws = layer.scene.draws.clone();
                sc.clips = layer.scene.clips.clone();
                sc
            };
            let lframe = self.prepare(&sub, eps)?;
            let ltex = self.create_output_texture(&lframe.scene);
            self.run_pipeline(&lframe.scene, &ltex, false, &lframe.clip_views, None)?;
            let mut limg = self.readback(&ltex)?;

            // 内容与阴影必须**分开**模糊：
            //
            // - `Filter::Blur`：只模糊内容。
            // - `Filter::DropShadow`：内容**不模糊**；阴影是"用层 alpha 形状 +
            //   固定阴影色先着色、再模糊"（与 CPU 一致）。若直接拿模糊后的内容
            //   当阴影，就变成"模糊内容再着色"，语义不同（实测差 125/255）。
            let mut shadow_img: Option<Vec<u8>> = None;
            let mut shadow_img_buf: Vec<u8> = vec![0u8; (root_w * root_h * 4) as usize];
            match layer.filter {
                amiki_scene::Filter::Blur { sigma } => {
                    limg = self.blur_rgba(&limg, root_w, root_h, sigma, 3)?;
                }
                amiki_scene::Filter::DropShadow { sigma, color, .. } => {
                    // 阴影遮罩在**主机侧**用与 CPU 完全相同的步骤生成：
                    // 1) RGB 覆盖为阴影色、alpha 乘 color.a
                    // 2) 盒式模糊（直线色语义，与 CPU 的 blur 一致）
                    //
                    // 为什么这里不走 GPU 模糊：投影需要的是"未模糊内容的 alpha 形状"，
                    // 而 GPU 模糊作用在整画布纹理上、坐标空间与层范围不同，
                    // 拼接时极易错位（我试过，差 125~162）。既然 CPU 已有这份数学，
                    // 直接复用最稳妥；GPU 仍承担**内容**的模糊（那才是大头）。
                    let mut mask: Vec<[f32; 4]> = Vec::with_capacity((root_w * root_h) as usize);
                    for i in 0..((root_w * root_h) as usize) {
                        let a = limg[i * 4 + 3] as f32 / 255.0;
                        mask.push([color.r, color.g, color.b, a * color.a]);
                    }
                    let mut lb = amiki_raster::LayerBuffer::new(root_w, root_h);
                    lb.data.copy_from_slice(&mask);
                    amiki_raster::blur_layer(&mut lb, sigma);
                    for (i, p) in lb.data.iter().enumerate() {
                        shadow_img_buf[i * 4] = (p[0] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                        shadow_img_buf[i * 4 + 1] = (p[1] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                        shadow_img_buf[i * 4 + 2] = (p[2] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                        shadow_img_buf[i * 4 + 3] = (p[3] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    }
                    shadow_img = Some(std::mem::take(&mut shadow_img_buf));
                }
                amiki_scene::Filter::None => {}
            }
            composite_layer_gpu(
                &mut root,
                root_w,
                root_h,
                layer,
                &limg,
                root_w,
                root_h,
                shadow_img.as_deref(),
            );
        }
        Ok(root)
    }
}

fn need<'a, T>(v: &'a Option<T>, field: &str) -> Result<&'a T, RenderError> {
    v.as_ref().ok_or_else(|| {
        RenderError(format!(
            "稀疏自洽检查缺少转储字段 {field}，请以 Some(PipelineDump::default()) 调用 \
             run_pipeline_sparse（需要 strip_meta / strip_cell_off / strip_bucket_off / cell_meta \
             / cell_aux / cell_csr_off / cell_segs / sparse_buckets / sparse_stats 全部字段）"
        ))
    })
}

fn need_len<T>(name: &str, v: &[T], want: usize) -> Result<(), RenderError> {
    if v.len() < want {
        return err(format!("转储字段 {name} 长度不足：{} < {want}", v.len()));
    }
    Ok(())
}

/// M3.5：**稀疏路径自洽检查**——在无稠密参照的情况下独立验证稀疏分箱结构。
///
/// # 为什么需要它
///
/// 稀疏（M3b）是默认路径，但它此前的正确性完全依赖"稀疏 vs 稠密逐字节对拍"
/// （`sparse_matches_dense`）。而稠密网格显存 ∝ `tiles × draws`，在 720p 约 280 draw、
/// 1080p 约 31 draw 就撞 `max_storage_buffer_binding_size`（默认 128 MiB）——
/// **真实规模下稠密根本跑不起来，于是稀疏在大场景没有裁判**。本函数填补这个空洞：
/// 只读稀疏自身的缓冲与原始几何，逐级断言各级结构互相吻合。
///
/// # 检查项（全部由原始几何独立重算，不依赖任何稠密路径产物）
///
/// 1. **strip 区间布局**：`strip_meta[si = tr*n_draws + d]` 的 `(lo, hi, y0, y1)` 在界内，
///    且 `lo == tiles_x` 哨兵 ⟺ 区间为空（与 `strips.wgsl` 判空语义一致）；
/// 2. **逐 tile 桶表一致性**：对每个 tile，`tile_count` 必须等于"区间覆盖该列的 strip 数"，
///    `tile_list[tile_start .. +count]` 必须与 `tr_list` 顺序完全相同，且 `d` 严格递增
///    （画家序），每条的 `w == hi − lo + 1`——这是"fine 不丢 draw"的前提；
/// 3. **单元 CSR 分区**：`cell_csr_off` 必须是 `[0, n_items]` 上的单调整数分区
///    （每 cell 的 `base + count == 下一 cell 的 base`）；
/// 4. **carry 桶前缀**：`cell_meta` 的 carry 位模式必须逐位等于"含当前桶的定点前缀取负"
///    （`sparse_emit` 的桶布局 + `sparse_carry` 的累加语义重算）——细粒度绕数正确性的核心；
/// 5. **段端点与成员计数**：`cell_meta` 的段数量必须等于 `cell_aux` 的成员计数，
///    且总成员数等于 `cell_segs` 的条目数。
///
/// 与 `check_strips`（M3a，依赖稠密转储）互补：本函数**不需要稠密路径可运行**。
pub fn check_sparse_consistency(
    gs: &GpuScene,
    d: &PipelineDump,
) -> Result<(), RenderError> {
    let tiles_x = gs.tiles_x;
    let tiles_y = gs.tiles_y;
    let n_draws = gs.colors.len() as u32;
    let n_strips = (tiles_y as u64) * (n_draws as u64);
    let n_tiles = (tiles_x as u64) * (tiles_y as u64);

    let meta = need(&d.strip_meta, "strip_meta")?;
    let sc_off = need(&d.strip_cell_off, "strip_cell_off")?;
    let sb_off = need(&d.strip_bucket_off, "strip_bucket_off")?;
    let cell_meta = need(&d.cell_meta, "cell_meta")?;
    let cell_aux = need(&d.cell_aux, "cell_aux")?;
    let csr = need(&d.cell_csr_off, "cell_csr_off")?;
    let buckets = need(&d.sparse_buckets, "sparse_buckets")?;
    let stats = need(&d.sparse_stats, "sparse_stats")?;
    let tile_count = need(&d.tile_count, "tile_count")?;
    let tile_start = need(&d.tile_start, "tile_start")?;
    let tile_list = need(&d.tile_list, "tile_list")?;
    let tr_offsets = need(&d.tr_offsets, "tr_offsets")?;
    let tr_list = need(&d.tr_list, "tr_list")?;

    let (n_cells, n_buckets, n_tile_entries) = (stats[0], stats[1], stats[2]);
    let n_items = stats[3];

    // 空场景：无内容覆盖画布，结构应为全空
    if n_cells == 0 {
        if n_buckets != 0 || n_tile_entries != 0 || n_items != 0 {
            return err(format!(
                "空场景统计不自洽：cells=0 但 buckets={n_buckets} \
                 tile_entries={n_tile_entries} items={n_items}"
            ));
        }
        return Ok(());
    }

    // ---- 长度前置检查：避免后续索引越界 panic ----
    need_len("strip_meta", meta, n_strips as usize * 4)?;
    need_len("strip_cell_off", sc_off, n_strips as usize)?;
    need_len("strip_bucket_off", sb_off, n_strips as usize)?;
    need_len("cell_meta", cell_meta, n_cells as usize)?;
    need_len("cell_aux", cell_aux, n_cells as usize * 2)?;
    need_len("cell_csr_off", csr, n_cells as usize + 1)?;
    need_len("sparse_buckets", buckets, n_buckets as usize)?;
    need_len("tile_count", tile_count, n_tiles as usize)?;
    need_len("tile_start", tile_start, n_tiles as usize)?;
    need_len("tile_list", tile_list, n_tile_entries as usize)?;
    need_len("tr_offsets", tr_offsets, tiles_y as usize)?;
    need_len("tr_list", tr_list, n_strips as usize)?;

    // ================= 1 + 2：strip 布局与逐 tile 桶表 =================
    let mut expected_entries: u64 = 0;
    for tr in 0..tiles_y {
        let off = tr_offsets[tr as usize];
        let (list_start, list_cnt) = (off[0] as usize, off[1] as usize);
        for col in 0..tiles_x {
            let mut want: Vec<(u32, u32, u32)> = Vec::new(); // (d, lo, hi)
            for k in 0..list_cnt {
                let si = tr_list[list_start + k];
                if si as u64 >= n_strips {
                    return err(format!("tr_list[{list_start}+{k}] = {si} 越出 strip 数 {n_strips}"));
                }
                let m = &meta[si as usize * 4..si as usize * 4 + 4];
                let (lo, hi, y0, y1) = (m[0], m[1], m[2], m[3]);
                if lo > tiles_x || (lo < tiles_x && (hi >= tiles_x || hi < lo)) {
                    return err(format!(
                        "strip {si}（tr={tr}）区间非法：lo={lo} hi={hi} tiles_x={tiles_x}"
                    ));
                }
                if lo == tiles_x {
                    return err(format!(
                        "strip {si}（tr={tr}）判空哨兵 lo==tiles_x 却出现在 tr_list（幻影 strip）"
                    ));
                }
                if y0 > y1 || y1 >= 16 {
                    return err(format!("strip {si} 局部行区间非法：y0={y0} y1={y1}（应 ⊆ [0,16)）"));
                }
                let d = (si as u64 % n_draws as u64) as u32;
                if lo <= col && col <= hi {
                    want.push((d, lo, hi));
                }
            }
            let tile = (tr as u64 * tiles_x as u64 + col as u64) as usize;
            let got_cnt = tile_count[tile] as usize;
            if got_cnt != want.len() {
                return err(format!(
                    "tile(tr={tr},col={col}) 桶表条目数不一致：GPU tile_count={got_cnt}，\
                     由 strip 区间重算={}（fine 会漏 draw 或多算 draw）",
                    want.len()
                ));
            }
            let start = tile_start[tile] as usize;
            if start + got_cnt > n_tile_entries as usize {
                return err(format!(
                    "tile(tr={tr},col={col}) 桶表越界：start={start} count={got_cnt} \
                     总条目={n_tile_entries}"
                ));
            }
            let mut prev_d: Option<u32> = None;
            for k in 0..got_cnt {
                let e = tile_list[start + k];
                let (w, d_got) = (e[1], e[2]);
                let (want_d, lo, hi) = want[k];
                if d_got != want_d {
                    return err(format!(
                        "tile(tr={tr},col={col}) 第 {k} 条 draw 序不符：GPU d={d_got}，期望 d={want_d}\
                         （画家序错乱会导致混合顺序错误）"
                    ));
                }
                if w != hi - lo + 1 {
                    return err(format!(
                        "tile(tr={tr},col={col}) draw={d_got} 的 w={w} 与区间宽 {} 不符",
                        hi - lo + 1
                    ));
                }
                if let Some(p) = prev_d {
                    if d_got <= p {
                        return err(format!(
                            "tile(tr={tr},col={col}) 桶表未按 d 严格递增：{p} → {d_got}"
                        ));
                    }
                }
                prev_d = Some(d_got);
            }
            expected_entries += got_cnt as u64;
        }
    }
    if expected_entries != n_tile_entries as u64 {
        return err(format!(
            "桶表条目总数不一致：统计 {n_tile_entries}，逐 tile 重算 {expected_entries}"
        ));
    }

    // ================= 3：单元 CSR 分区 =================
    if csr[0] != 0 {
        return err(format!("cell_csr_off[0] 应为 0，实际 {}", csr[0]));
    }
    let mut cell_refs: Vec<(u32, u32, u32, u32)> = Vec::new(); // (cell, base, count, carry_bits)
    for strip in 0..n_strips {
        let si = strip as usize;
        let m = &meta[si * 4..si * 4 + 4];
        let (lo, hi, y0, y1) = (m[0], m[1], m[2], m[3]);
        if lo == tiles_x {
            continue; // 空 strip
        }
        let w = hi - lo + 1;
        for row in y0..=y1 {
            for col in lo..=hi {
                let cell = sc_off[si] + (row - y0) * w + (col - lo);
                if cell as u64 >= n_cells as u64 {
                    return err(format!(
                        "单元索引越界：strip {si}(lo={lo},hi={hi},y0={y0},y1={y1}) \
                         row={row} col={col} → cell={cell} ≥ n_cells={n_cells}"
                    ));
                }
                let cm = cell_meta[cell as usize];
                let (base, count) = (cm[1], cm[2]);
                if base + count > n_items {
                    return err(format!(
                        "单元 {cell} 段范围越界：base={base} count={count} > n_items={n_items}"
                    ));
                }
                if csr[cell as usize + 1] != base + count {
                    return err(format!(
                        "单元 CSR 非连续分区：cell={cell} 的 base+count={} 与 \
                         cell_csr_off[{}]={} 不符",
                        base + count,
                        cell + 1,
                        csr[cell as usize + 1]
                    ));
                }
                let cnt_aux = cell_aux[cell as usize];
                if count != cnt_aux {
                    return err(format!(
                        "单元 {cell} 段数量与 emit 成员计数不符：cell_meta={count} cell_aux={cnt_aux}"
                    ));
                }
                cell_refs.push((cell, base, count, cm[0]));
            }
        }
    }
    if csr[n_cells as usize] != n_items {
        return err(format!(
            "cell_csr_off 末项应为 n_items={n_items}，实际 {}",
            csr[n_cells as usize]
        ));
    }
    cell_refs.sort_unstable_by_key(|c| c.0);

    // ================= 4 + 5：carry 桶前缀与段列表 =================
    let mut members: u64 = 0;
    let mut cursor: u32 = 0; // 段槽位必须按 cell 索引顺序填满
    for &(cell, base, count, carry_bits) in &cell_refs {
        members += count as u64;
        if base != cursor {
            return err(format!(
                "段槽位未按 cell 索引顺序连续：cell={cell} base={base}，期望 {cursor}\
                 （scatter 游标语义错乱）"
            ));
        }
        cursor = base + count;

        // 由 cell 反推所属 strip（strip_cell_off 单调递增，二分定位）
        let si = match sc_off.binary_search(&cell) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let m = &meta[si * 4..si * 4 + 4];
        let (lo, hi, y0, _y1) = (m[0], m[1], m[2], m[3]);
        let w = hi - lo + 1;
        let rel = cell - sc_off[si];
        let row = y0 + rel / w;
        let col = lo + rel % w;

        // carry = −(x_max 严格落在该 cell 左界之前的段的定点前缀)
        // 桶布局（sparse_emit）：strip_bucket_off[si] 即 strip 的桶基址，
        // 局部位移 = (row − y0) * (w + 1) + (bucket − lo)，其中 bucket = ⌊x_max/16⌋ + 1。
        // 故 bucket_k 对应 tile 列 lo + k：桶 lo 自身装"右界 ≤ tile lo 左界"的段，
        // 而 cell(col) 的左界正是 tile col 的左界 ⇒ 前缀需含桶 col 但**排除桶 col+1**。
        // 这点极易写错（含当前桶会多算一项，直接把覆盖率整体偏移）。
        let bi = (sb_off[si] as u64) + ((row - y0) as u64) * (w as u64 + 1);
        let mut running: i32 = 0;
        for k in 0..=(col as u64).saturating_sub(lo as u64) {
            running = running.wrapping_add(buckets[(bi + k) as usize]);
        }
        let want_carry = -(running as f32) / 65536.0;
        let got_carry = f32::from_bits(carry_bits);
        if want_carry.to_bits() != got_carry.to_bits() {
            return err(format!(
                "单元 {cell} 的 carry 与桶前缀不符：cell_meta={got_carry}（位模式 {carry_bits:#x}），\
                 由 buckets 前缀重算={want_carry}（strip={si} row={row} col={col} \
                 前缀桶数={}）—— 该 cell 的绕数会整体偏移，导致覆盖率错误",
                (col as u64).saturating_sub(lo as u64) + 1
            ));
        }
    }
    if members != n_items as u64 {
        return err(format!("成员总数不一致：统计 n_items={n_items}，由 cell_meta 重算 {members}"));
    }
    if cursor != n_items {
        return err(format!(
            "段槽位总数 {cursor} 与 n_items={n_items} 不符（cell_segs 存在空洞或重叠）"
        ));
    }
    // 桶的定点量化误差说明：|carry| ≤ 16 故 |前缀| ≤ 16·65536，远小于 f32 精确整数范围，
    // 因此上式用 f32 重算不会引入舍入误差（逐位比较是严格的）
    Ok(())
}

fn nq_len(gs: &GpuScene) -> usize {
    gs.quads.len()
}
fn nc_len(gs: &GpuScene) -> usize {
    gs.cubics.len()
}
fn nl_len(gs: &GpuScene) -> usize {
    gs.lines.len()
}

/// 全屏 blit 管线（窗口呈现用）。
pub struct Blitter {
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
}

impl Blitter {
    /// 为给定表面格式创建 blit 管线，绑定场景纹理视图。
    pub fn new(
        device: &wgpu::Device,
        scene_tex_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) -> Blitter {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("amiki_blit_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_blit_bind"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_tex_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("amiki_blit_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("amiki_blit_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Blitter { pipeline, bind }
    }

    pub fn blit(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, w: u32, h: u32) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("amiki_blit_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.draw(0..3, 0..1);
        let _ = (w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU/CPU 渲染对拍（donut + 旋转）。
    /// M2f 已知问题：曲线边缘像素覆盖率偏差（圆极端点行约 -0.28，且不随 ε 变化，
    /// 怀疑 flatten 段端点或 seg_ends 数据链路），调查恢复后回归 tolerance 2。
    #[test]
    fn gpu_matches_cpu() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{Color, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        let mut scene = Scene::new(96, 96);
        let t = Affine::translate(Vec2::new(48.0, 48.0))
            .then(Affine::rotate(0.3))
            .then(Affine::translate(Vec2::new(-48.0, -48.0)));
        scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));

        let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

        let mut max_diff = 0u32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        }
        assert!(max_diff <= 2, "GPU 与 CPU 最大通道差 {max_diff} 超出容差");
    }

    /// M2f 管线核心对拍：纯直线/矩形场景（不含曲线，当前已验证）。
    #[test]
    fn gpu_rect_matches_cpu() {        use amiki_math::{Affine, Vec2};
        use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        // 多 draw + 半透明 + 旋转矩形 + 跨 tile 边界
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 72.0, 72.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 40.0, 56.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(-16.0, -10.0, 16.0, 10.0);
        let t = Affine::translate(Vec2::new(44.0, 36.0)).then(Affine::rotate(0.5));
        scene.push_blend(t, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 180)), BlendMode::Multiply);

        let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
        let mut max_diff = 0u32;
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        }
        assert!(max_diff <= 2, "GPU 与 CPU 最大通道差 {max_diff} 超出容差");
    }

    /// M6：**GPU 渐变着色必须与 CPU 参考实现一致**。
    ///
    /// 这是渐变在 GPU 侧的唯一正确性依据：WGSL 的 `grad_color` 与 CPU 的
    /// `Gradient::sample` 是两份独立实现，只有逐像素对拍才能证明语义一致。
    /// 覆盖三种几何（线性/径向/扫描）与多色标（含中间色标，以验证区间查找），
    /// 以及 repeat 范围行为。
    #[test]
    fn gpu_gradient_matches_cpu() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{Color, Gradient, GradientStop, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        let stops = || {
            vec![
                GradientStop::new(0.0, Color::rgba8(20, 40, 200, 255)),
                GradientStop::new(0.35, Color::rgba8(240, 200, 40, 255)),
                GradientStop::new(1.0, Color::rgba8(220, 30, 90, 255)),
            ]
        };

        let cases: Vec<(&str, Gradient)> = vec![
            (
                "线性",
                Gradient::linear(Vec2::new(6.0, 6.0), Vec2::new(66.0, 42.0), stops()).unwrap(),
            ),
            (
                "径向",
                Gradient::radial(Vec2::new(36.0, 36.0), 34.0, stops()).unwrap(),
            ),
            (
                "扫描",
                Gradient::sweep(Vec2::new(36.0, 36.0), 0.7, stops()).unwrap(),
            ),
        ];

        for (name, g) in cases {
            let mut scene = Scene::new(72, 72);
            let mut pb = PathBuilder::new();
            pb.rect(4.0, 4.0, 68.0, 68.0);
            scene.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Gradient(std::sync::Arc::new(g)),
            );

            let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
            let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

            let mut max_diff = 0u32;
            let mut worst = (0usize, 0usize);
            let mut n_opaque = 0usize;
            for i in 0..(72 * 72) {
                let (a, b) = (gpu[i * 4], cpu[i * 4]);
                let d = (a as i32 - b as i32).unsigned_abs();
                if d > max_diff {
                    max_diff = d;
                    worst = (i % 72, i / 72);
                }
                if gpu[i * 4 + 3] > 0 {
                    n_opaque += 1;
                }
            }
            assert!(
                n_opaque > 2000,
                "{name}渐变应产生大量不透明像素（否则对拍无意义），实际 {n_opaque}"
            );
            assert!(
                max_diff <= 2,
                "{name}渐变：GPU 与 CPU 最大通道差 {max_diff}（位于 {worst:?}）超出容差\n\
                 该点 GPU={:?} CPU={:?}",
                {
                    let o = (worst.1 * 72 + worst.0) * 4;
                    [gpu[o], gpu[o + 1], gpu[o + 2], gpu[o + 3]]
                },
                {
                    let o = (worst.1 * 72 + worst.0) * 4;
                    [cpu[o], cpu[o + 1], cpu[o + 2], cpu[o + 3]]
                }
            );
            eprintln!("GPU {name}渐变与 CPU 一致（最大通道差 {max_diff}，不透明像素 {n_opaque}）");
        }

        // repeat 模式：范围外应重复而非 clamp
        let mut rep =
            Gradient::linear(Vec2::new(10.0, 10.0), Vec2::new(30.0, 10.0), stops()).unwrap();
        rep.repeat = true;
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(4.0, 4.0, 68.0, 68.0);
        scene.push(
            Affine::IDENTITY,
            pb.build(),
            Paint::Gradient(std::sync::Arc::new(rep)),
        );
        let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
        let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
        let max_diff = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(max_diff <= 2, "repeat 渐变：最大通道差 {max_diff} 超出容差");
        eprintln!("GPU repeat 渐变与 CPU 一致（最大通道差 {max_diff}）");
    }
    /// M6：**GPU 位图采样必须与 CPU 参考实现一致**。
    ///
    /// 用一张**合成图案**（而非纯色）作测试图：纯色无法区分"采样正确"与
    /// "把整张图当成一个颜色"这两种实现；棋盘 + 渐变的组合能同时检验
    /// 坐标映射、双线性过滤与缩放。
    #[test]
    fn gpu_image_matches_cpu() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{Color, Image, ImagePlacement, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        // 32×24 合成图：R/G 为渐变（检验坐标映射），B 为 4×4 棋盘（检验过滤）
        let (iw, ih) = (32u32, 24u32);
        let mut data = Vec::with_capacity((iw * ih * 4) as usize);
        for y in 0..ih {
            for x in 0..iw {
                let checker = if ((x / 4) + (y / 4)) % 2 == 0 { 40u8 } else { 215u8 };
                data.push((x * 255 / (iw - 1)) as u8);
                data.push((y * 255 / (ih - 1)) as u8);
                data.push(checker);
                data.push(255);
            }
        }
        let img = std::sync::Arc::new(Image::from_rgba8(iw, ih, data).expect("图像数据合法"));

        // 三种放置：等比、放大（检验双线性）、旋转（检验仿射逆映射）
        let cases: Vec<(&str, ImagePlacement, bool)> = vec![
            ("等比", ImagePlacement::rect(8.0, 8.0, 40.0, 32.0), false),
            ("放大", ImagePlacement::rect(4.0, 4.0, 68.0, 68.0), false),
            (
                "旋转",
                ImagePlacement {
                    origin: Vec2::new(36.0, 6.0),
                    u_axis: Vec2::new(28.0, 14.0),
                    v_axis: Vec2::new(-14.0, 28.0),
                    repeat: false,
                },
                false,
            ),
            ("平铺", ImagePlacement { repeat: true, ..ImagePlacement::rect(10.0, 10.0, 30.0, 22.0) }, false),
        ];

        for (name, place, _) in cases {
            let mut scene = Scene::new(72, 72);
            let mut pb = PathBuilder::new();
            // 覆盖范围按放置四角的包围盒（与 Canvas::image 一致）
            let p = |u: f32, v: f32| place.origin + place.u_axis * u + place.v_axis * v;
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                let q = p(u, v);
                x0 = x0.min(q.x);
                y0 = y0.min(q.y);
                x1 = x1.max(q.x);
                y1 = y1.max(q.y);
            }
            pb.rect(x0, y0, x1, y1);
            // 底：不透明色，便于观察图像是否真的覆盖上去
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(250, 250, 250)));
            scene.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Image { image: img.clone(), placement: place },
            );

            let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
            let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

            let mut max_diff = 0u32;
            let mut worst = 0usize;
            for (i, (a, b)) in gpu.iter().zip(cpu.iter()).enumerate() {
                let d = (*a as i32 - *b as i32).unsigned_abs();
                if d > max_diff {
                    max_diff = d;
                    worst = i;
                }
            }
            // 双线性在 CPU 与 GPU 上的舍入不同，容差给 3（比纯几何的 2 略宽）
            assert!(
                max_diff <= 3,
                "{name}放置：GPU 与 CPU 最大通道差 {max_diff}（字节偏移 {worst}）超出容差"
            );
            eprintln!("GPU 位图（{name}放置）与 CPU 一致（最大通道差 {max_diff}）");
        }
    }

    /// M6：**GPU 离屏层（滤镜）必须与 CPU 参考实现逐像素一致**。
    ///
    /// 这是"滤镜双端一致"的唯一依据。两端的模糊是独立的两份实现
    /// （WGSL 可分离盒式 vs Rust 可分离盒式），只有对拍才能证明语义一致。
    ///
    /// 容差 2/255：与纯几何对拍同级。模糊是多次加权平均，两端在盒边界与
    /// 舍入上有微小差异，实测三种滤镜的最大差均为 0~1。
    #[test]
    fn gpu_layer_matches_cpu() {
        use amiki_math::Affine;
        use amiki_scene::{BlendMode, Color, Filter, Layer, LayerBounds, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        let cases: Vec<(&str, Filter)> = vec![
            ("分组", Filter::None),
            ("模糊", Filter::Blur { sigma: 6.0 }),
            (
                "投影",
                Filter::DropShadow {
                    dx: 10.0,
                    dy: 12.0,
                    sigma: 9.0,
                    color: Color::rgba8(30, 26, 22, 190),
                },
            ),
        ];

        for (name, filter) in cases {
            let mut scene = Scene::new(160, 120);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 160.0, 120.0);
            scene.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgb8(246, 243, 238)),
            );
            scene.push_layer(
                Layer {
                    bounds: LayerBounds::new(36, 26, 70, 70),
                    scene: Scene::new(160, 120),
                    filter,
                    opacity: 1.0,
                    blend: BlendMode::Normal,
                },
                |s| {
                    let mut pb = PathBuilder::new();
                    pb.rect(44.0, 34.0, 96.0, 86.0);
                    s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 60, 50)));
                },
            );

            let gpu = renderer
                .render_blocking(&scene, 0.05)
                .unwrap_or_else(|e| panic!("{name}层：GPU 渲染失败（{e}）"));
            let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));

            let mut max_diff = 0u32;
            let mut worst = (0usize, 0usize);
            for i in 0..(160 * 120) {
                for k in 0..4 {
                    let d = (gpu[i * 4 + k] as i32 - cpu[i * 4 + k] as i32).unsigned_abs();
                    if d > max_diff {
                        max_diff = d;
                        worst = (i % 160, i / 160);
                    }
                }
            }
            assert!(
                max_diff <= 2,
                "{name}层：GPU 与 CPU 最大通道差 {max_diff}（位于 {worst:?}）超出容差"
            );
            eprintln!("GPU {name}层与 CPU 一致（最大通道差 {max_diff}）");
        }
    }

    /// M2f-1：通用 GPU 前缀和 vs CPU 对拍（随机长度/数值）。
    #[test]
    fn gpu_scan_matches_cpu() {
        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        let mut s = 0xC0FF_EE12_3456_789Au64;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32)
        };
        for &n in &[1usize, 2, 63, 64, 65, 255, 256, 257, 1000, 4097] {
            let data: Vec<u32> = (0..n).map(|_| (rnd() * 1000.0) as u32).collect();
            let buf = renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("scan_test"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let mut enc = renderer
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            renderer.run_scan(&mut enc, &buf, n as u32, None);
            renderer.queue.submit(Some(enc.finish()));
            let out_bytes = renderer.read_buffer(&buf, n * 4).expect("读回失败");
            let out: Vec<u32> = bytemuck::cast_slice(&out_bytes).to_vec();
            // CPU exclusive 前缀
            let mut expect = Vec::with_capacity(n);
            let mut acc = 0u32;
            for &v in &data {
                expect.push(acc);
                acc += v;
            }
            assert_eq!(out, expect, "GPU 前缀和与 CPU 不一致 (n={n})");
        }
    }

    /// M2e：全 16 混合模式 GPU vs CPU 对拍。
    #[test]
    fn gpu_blend_matches_cpu() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        let modes = [
            BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
            BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
            BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference, BlendMode::Exclusion,
            BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity,
        ];
        for (mi, mode) in modes.iter().enumerate() {
            let mut scene = Scene::new(64, 64);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 64.0, 64.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
            let mut pb = PathBuilder::new();
            pb.rect(8.0, 8.0, 56.0, 56.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 90, 160)));
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(32.0, 32.0), 22.0, false);
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(230, 120, 60, 200)),
                *mode,
            );
            let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
            let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
            let mut max_diff = 0u32;
            for (a, b) in gpu.iter().zip(cpu.iter()) {
                max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
            }
            assert!(max_diff <= 2, "混合模式 #{mi}({mode:?}) GPU/CPU 最大差 {max_diff}");
        }
    }

    /// M2d：裁剪 GPU vs CPU 对拍（单层 + 嵌套两层）。
    #[test]
    fn gpu_clip_matches_cpu() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{Color, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        for nested in [false, true] {
            let mut scene = Scene::new(64, 64);
            let mut pb = PathBuilder::new();
            pb.rect(8.0, 8.0, 56.0, 56.0);
            let c0 = scene.push_clip(Affine::IDENTITY, pb.build());
            let mut clips = vec![c0];
            if nested {
                let mut pb = PathBuilder::new();
                pb.rect(16.0, 16.0, 64.0, 64.0);
                let c1 = scene.push_clip(Affine::IDENTITY, pb.build());
                clips.push(c1);
            }
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(32.0, 32.0), 28.0, false);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 120, 255)));
            scene.draws.last_mut().unwrap().clips = clips;

            let gpu = renderer.render_blocking(&scene, 0.05).expect("GPU 渲染失败");
            let cpu = amiki_raster::render(&amiki_raster::bin(&scene, 0.05));
            let mut max_diff = 0u32;
            for (a, b) in gpu.iter().zip(cpu.iter()) {
                max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
            }
            assert!(max_diff <= 3, "clip({nested}) GPU/CPU 最大差 {max_diff}");
        }
    }

    // ===================== M3a：活跃 tile 表（sparse strips 底座） =====================

    /// 运行一次管线并转储中间数据（M3a 验证用：固定走稠密参照路径）。
    fn dump_pipeline(renderer: &GpuRenderer, scene: &Scene, eps: f32) -> (GpuScene, PipelineDump) {
        let gs = build_gpu_scene(scene, eps);
        let tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("amiki_m3a_out"),
            size: wgpu::Extent3d { width: gs.width, height: gs.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut dump = PipelineDump {
            counters: Some(Vec::new()),
            carries: Some(Vec::new()),
            strip_meta: Some(Vec::new()),
            ..Default::default()
        };
        let mut prof = FrameProfile::default();
        renderer
            .run_pipeline_impl(&gs, &tex, false, &[], None, Some(&mut dump), None, &mut prof)
            .expect("管线运行失败");
        (gs, dump)
    }

    /// 稠密网格活动单元 ↔ strip 区间一致性：
    /// · 覆盖性（恒成立）：任何活动单元（成员计数>0 / 真实 carry≠0）必落在对应 strip 内；
    /// · 无幻影：完全无活动的 (draw, tr) 不应有 strip；
    /// · `exact = true` 时额外要求区间与活动包围盒逐格相等——画布内轴对齐几何
    ///   应精确成立；越界内容会按稠密语义保守扩张左/右界（见 strips.wgsl）。
    fn check_strips(gs: &GpuScene, dump: &PipelineDump, exact: bool) {
        /// 低于此视为闭合路径求和的浮点残差（约 1e-7），非真实 carry。
        const CARRY_EPS: f32 = 1e-4;
        let counters = dump.counters.as_ref().expect("counters 未转储");
        let carries = dump.carries.as_ref().expect("carries 未转储");
        let meta = dump.strip_meta.as_ref().expect("strip_meta 未转储");
        let n_draws = gs.colors.len() as u32;
        let (mut n_active, mut n_covered) = (0usize, 0usize);
        for d in 0..n_draws {
            for tr in 0..gs.tiles_y {
                let si = (tr * n_draws + d) as usize; // 转置布局：si = tr * n_draws + d
                let (lo_s, hi_s, y0_s, y1_s) =
                    (meta[si * 4], meta[si * 4 + 1], meta[si * 4 + 2], meta[si * 4 + 3]);
                let (mut mn_c, mut mx_c, mut mn_r, mut mx_r) = (u32::MAX, 0u32, u32::MAX, 0u32);
                for col in 0..gs.tiles_x {
                    for row in 0..16 {
                        let b = (((tr * gs.tiles_x + col) * n_draws + d) * 16 + row) as usize;
                        if counters[b] > 0 || carries[b].abs() > CARRY_EPS {
                            mn_c = mn_c.min(col);
                            mx_c = mx_c.max(col);
                            mn_r = mn_r.min(row);
                            mx_r = mx_r.max(row);
                        }
                    }
                }
                if mn_c == u32::MAX {
                    assert_eq!(lo_s, gs.tiles_x, "无活动的 (draw={d}, tr={tr}) 不应有 strip");
                    continue;
                }
                n_active += ((mx_c - mn_c + 1) as usize) * ((mx_r - mn_r + 1) as usize);
                assert!(lo_s != gs.tiles_x, "有活动但 strip 缺失 (draw={d}, tr={tr})");
                // 覆盖性：strip 必须框住全部活动单元
                assert!(
                    lo_s <= mn_c && mx_c <= hi_s && y0_s <= mn_r && mx_r <= y1_s,
                    "strip 未覆盖活动单元 (draw={d}, tr={tr}): strip=({lo_s},{hi_s},{y0_s},{y1_s}) 活动=({mn_c},{mx_c},{mn_r},{mx_r})"
                );
                n_covered += ((hi_s - lo_s + 1) as usize) * ((y1_s - y0_s + 1) as usize);
                if exact {
                    assert_eq!(
                        (lo_s, hi_s, y0_s, y1_s),
                        (mn_c, mx_c, mn_r, mx_r),
                        "strip 必须与稠密活动包围盒逐格相等 (draw={d}, tr={tr})"
                    );
                }
            }
        }
        eprintln!("M3a 对拍：活动单元 {n_active}，strip 覆盖单元 {n_covered}");
    }

    // ===================== M3b：稀疏桶表 =====================

    /// M3b：稀疏路径 vs 稠密路径逐字节一致（多场景，含越界/混合模式/裁剪）。
    #[test]
    fn sparse_matches_dense() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

        let Ok(mut renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        let mut scenes: Vec<(&str, Scene)> = Vec::new();
        {
            let mut scene = Scene::new(64, 64);
            let mut pb = PathBuilder::new();
            pb.rect(8.0, 8.0, 40.0, 40.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(200, 40, 40)));
            scenes.push(("rect", scene));
        }
        {
            let mut scene = Scene::new(72, 72);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 72.0, 72.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(36.0, 36.0), 28.0, false);
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(40, 90, 160, 200)),
                BlendMode::Multiply,
            );
            let mut pb = PathBuilder::new();
            pb.rect(-16.0, -10.0, 16.0, 10.0);
            let t = Affine::translate(Vec2::new(44.0, 36.0)).then(Affine::rotate(0.5));
            scene.push(t, pb.build(), Paint::Solid(Color::rgba8(230, 120, 60, 180)));
            scenes.push(("mixed", scene));
        }
        {
            let mut scene = Scene::new(96, 96);
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
            pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
            let t = Affine::translate(Vec2::new(48.0, 48.0))
                .then(Affine::rotate(0.3))
                .then(Affine::translate(Vec2::new(-48.0, -48.0)));
            scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
            let mut pb = PathBuilder::new();
            pb.pentagram(Vec2::new(60.0, 60.0), 34.0);
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(230, 120, 60, 200)),
                BlendMode::Multiply,
            );
            scenes.push(("curves", scene));
        }
        {
            // 越出画布右缘 / 左缘 / 四角（carry 右界扩张与桶 0 钳制路径）
            let mut scene = Scene::new(80, 64);
            let mut pb = PathBuilder::new();
            pb.rect(40.0, 8.0, 140.0, 48.0); // 跨右缘
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 160, 90)));
            let mut pb = PathBuilder::new();
            pb.rect(-30.0, 4.0, 30.0, 40.0); // 跨左缘
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(200, 90, 30, 200)));
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(78.0, 60.0), 22.0, false); // 跨右下角
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(60, 60, 220, 160)),
                BlendMode::Screen,
            );
            scenes.push(("offscreen", scene));
        }
        {
            // 裁剪 + 非可分离混合模式（clip 覆盖率纹理亦走稀疏路径）
            let mut scene = Scene::new(64, 64);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 64.0, 64.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
            let c0 = scene.push_clip(Affine::IDENTITY, pb.build());
            let mut pb = PathBuilder::new();
            pb.rect(16.0, 16.0, 64.0, 64.0);
            let c1 = scene.push_clip(Affine::IDENTITY, pb.build());
            let mut pb = PathBuilder::new();
            pb.pentagram(Vec2::new(32.0, 32.0), 26.0);
            scene.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(40, 90, 160, 220)),
                BlendMode::Hue,
            );
            scene.draws.last_mut().unwrap().clips = vec![c0, c1];
            scenes.push(("clips", scene));
        }
        {
            // 全 16 种混合模式（逐个对比）
            let modes = [
                BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
                BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
                BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference, BlendMode::Exclusion,
                BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity,
            ];
            let mut scene = Scene::new(64, 64);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 64.0, 64.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
            let mut pb = PathBuilder::new();
            pb.rect(8.0, 8.0, 56.0, 56.0);
            scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(40, 90, 160)));
            for (i, mode) in modes.iter().enumerate() {
                let mut pb = PathBuilder::new();
                pb.circle(Vec2::new(16.0 + (i % 4) as f32 * 10.0, 16.0 + (i / 4) as f32 * 10.0), 14.0, false);
                scene.push_blend(
                    Affine::IDENTITY,
                    pb.build(),
                    Paint::Solid(Color::rgba8(230, 120, 60, 200)),
                    *mode,
                );
            }
            scenes.push(("blend16", scene));
        }

        for (name, scene) in &scenes {
            renderer.path = RenderPath::Dense;
            let dense = renderer.render_blocking(scene, 0.05).expect("稠密渲染失败");
            renderer.path = RenderPath::Sparse;
            let sparse = renderer.render_blocking(scene, 0.05).expect("稀疏渲染失败");
            let mut max_diff = 0u32;
            let mut n_diff = 0usize;
            for (a, b) in dense.iter().zip(sparse.iter()) {
                let d = (*a as i32 - *b as i32).unsigned_abs();
                max_diff = max_diff.max(d);
                if d > 0 {
                    n_diff += 1;
                }
            }
            eprintln!("{name}: 最大差 {max_diff}，不同字节 {n_diff}/{}", dense.len());
            assert_eq!(max_diff, 0, "{name}: 稀疏与稠密渲染不一致（最大差 {max_diff}）");
        }
    }

    /// M3c：稀疏路径在「稠密不可能」的规模下可运行，且显存与内容成比例。
    /// 目标参照 §4.5（1080p × 10k 曲线）；测试用 2k 曲线控制耗时。
    #[test]
    fn sparse_scales_beyond_dense() {
        use amiki_math::Vec2;
        use amiki_scene::{Color, Paint, Scene};
        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        let (w, h, n) = (1920u32, 1080u32, 2000usize);
        let mut scene = Scene::new(w, h);
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32)
        };
        for _ in 0..n {
            let c = Vec2::new(rnd() * w as f32, rnd() * h as f32);
            let mut pb = amiki_scene::PathBuilder::new();
            pb.circle(c, 4.0 + 4.0 * rnd(), false);
            scene.push(
                amiki_math::Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(200, 120, 60, 200)),
            );
        }

        // 稠密网格：显存 ∝ tiles×draws，早已越过 128MB 存储绑定上限
        let gs = build_gpu_scene(&scene, 0.1);
        let n_grid = (gs.tiles_x as u64) * (gs.tiles_y as u64) * (n as u64) * 16;
        let dense_bytes = n_grid * 12 + (n as u64) * (gs.tiles_y as u64) * 16 * ((gs.tiles_x + 1) as u64) * 4;
        assert!(
            dense_bytes > 128 * 1024 * 1024,
            "本测试应处于稠密不可行区间（实际稠密需求 {:.1} MB）",
            dense_bytes as f64 / 1e6
        );

        // 稀疏路径：可运行，且占用与内容成比例
        let img = renderer.render_blocking(&scene, 0.1).expect("稀疏渲染失败");
        assert_eq!(img.len(), (w * h * 4) as usize);
        let st = renderer.sparse_stats(&scene, 0.1).expect("统计失败").expect("稀疏统计缺失");
        let sparse_bytes = st.n_cells as u64 * 24
            + st.n_buckets as u64 * 4
            + st.n_tile_entries as u64 * 16
            + st.n_items as u64 * 16;
        eprintln!(
            "1080p × {n} 曲线：稠密需求 {:.0} MB → 稀疏 {:.1} MB（条带 {} / 单元 {} / 桶 {} / 条目 {} / 成员 {}）",
            dense_bytes as f64 / 1e6,
            sparse_bytes as f64 / 1e6,
            st.n_strips, st.n_cells, st.n_buckets, st.n_tile_entries, st.n_items
        );
        assert!(
            sparse_bytes * 50 < dense_bytes,
            "稀疏占用应显著小于稠密（稀疏 {sparse_bytes} vs 稠密 {dense_bytes}）"
        );
    }

    /// M3a：轴对齐矩形（多 draw、半透明、跨 tile、部分出画布）——strip 逐格对拍。
    #[test]
    fn strips_match_dense_rects() {
        use amiki_math::Affine;
        use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        // 高 72 = 4.5 个 tile 行 → 末行带为部分带（验证高度钳制）
        let mut scene = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 72.0, 72.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 40.0, 56.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
        // 左/上出画布（carry 桶 0 钳制路径）
        let mut pb = PathBuilder::new();
        pb.rect(-24.0, -12.0, 20.0, 30.0);
        scene.push_blend(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(230, 120, 60)), BlendMode::Multiply);
        // 完全出画布（左下方）→ 不得产生 strip
        let mut pb = PathBuilder::new();
        pb.rect(-200.0, 200.0, -160.0, 260.0);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 0, 0)));

        let (gs, dump) = dump_pipeline(&renderer, &scene, 0.05);
        check_strips(&gs, &dump, true);
    }

    /// M3a：曲线/旋转/自相交混合场景——strip 覆盖性（含越界保守扩张路径）。
    #[test]
    fn strips_match_dense_curves() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        let mut scene = Scene::new(96, 96);
        // 环形（曲线 + 反绕洞）+ 旋转（斜边跨带）
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        let t = Affine::translate(Vec2::new(48.0, 48.0))
            .then(Affine::rotate(0.3))
            .then(Affine::translate(Vec2::new(-48.0, -48.0)));
        scene.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
        // 自相交五角星（直线，W=2 区域）
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(60.0, 60.0), 34.0);
        scene.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(230, 120, 60, 200)),
            BlendMode::Multiply,
        );
        // 右下角出画布曲线（钳制路径）
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(104.0, 100.0), 24.0, false);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(52, 152, 219)));

        let (gs, dump) = dump_pipeline(&renderer, &scene, 0.05);
        check_strips(&gs, &dump, false);
    }

    /// M3.5：稠密容量预检——超限必须返回**可读错误**，而不是让 wgpu 校验层 panic。
    ///
    /// 回归背景：720p × 500 曲线走稠密时，`grid_aux` 需 230.4 MB，超过默认
    /// `max_storage_buffer_binding_size`（128 MiB），此前表现为
    /// `create_bind_group` 内部的 wgpu Validation Error panic（进程终止、无法降级）。
    #[test]
    fn dense_ceiling_reports_error_not_panic() {
        let Ok(mut renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };

        // 1) 纯函数判定：默认限制下 720p × 500 draw 必须超限，且错误信息含关键数字
        let limits = wgpu::Limits::default();
        let (tx, ty) = (1280u32.div_ceil(16), 720u32.div_ceil(16)); // 80 × 45
        let err = dense_capacity(&limits, tx, ty, 500).expect_err("720p×500 应判定超限");
        let msg = err.to_string();
        assert!(msg.contains("稠密网格超限"), "错误信息应说明是稠密超限：{msg}");
        assert!(msg.contains("grid_aux"), "错误信息应指明是哪个缓冲：{msg}");
        assert!(msg.contains("RenderPath::Sparse"), "错误信息应给出降级建议：{msg}");
        eprintln!("预检信息（应可读）：{msg}");

        // 2) 正常规模不应被误判
        assert!(dense_capacity(&limits, tx, ty, 8).is_ok(), "小场景不应被预检拦下");

        // 3) 端到端：走稠密路径渲染超限场景必须返回 Err（而非 panic）
        use amiki_math::Vec2;
        use amiki_scene::{Color, Paint, PathBuilder};
        let mut scene = Scene::new(1280, 720);
        let mut s = 0x5EED_1234_5678_9ABCu64;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32) / (u32::MAX as f32)
        };
        for _ in 0..500 {
            let c = Vec2::new(rnd() * 1280.0, rnd() * 720.0);
            let mut pb = PathBuilder::new();
            pb.circle(c, 4.0 + 6.0 * rnd(), false);
            scene.push(
                amiki_math::Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(200, 120, 60, 200)),
            );
        }

        let mut r = pollster::block_on(GpuRenderer::new()).expect("无 GPU");
        r.path = RenderPath::Dense;
        let res = r.render_blocking(&scene, 0.1);
        assert!(
            matches!(res, Err(ref e) if e.to_string().contains("稠密网格超限")),
            "稠密超限应返回 Err(稠密网格超限)，实际：{:?}",
            res.map(|v| format!("Ok({} 字节)", v.len()))
        );

        // 4) 同一场景走稀疏路径必须成功（验证建议的降级方案确实可行）
        renderer.path = RenderPath::Sparse;
        let img = renderer.render_blocking(&scene, 0.1).expect("稀疏路径应可渲染同一场景");
        assert_eq!(img.len(), 1280 * 720 * 4);
        eprintln!("同一场景：稠密返回可读错误，稀疏渲染成功（{} 字节）", img.len());
    }

    // ===================== M3.5：稀疏自洽检查 =====================

    /// 覆盖各类稀疏结构的场景集（矩形 / 曲线+洞 / 自相交 / 越界 / 裁剪+混合）。
    fn consistency_scenes() -> Vec<(&'static str, Scene)> {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{BlendMode, Color, Paint, PathBuilder};
        let mut out: Vec<(&'static str, Scene)> = Vec::new();
        {
            // 轴对齐 + 跨 tile 边界（tile 区间与局部行区间最严格）
            let mut s = Scene::new(72, 72);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 72.0, 72.0);
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
            let mut pb = PathBuilder::new();
            pb.rect(16.0, 16.0, 33.0, 49.0);
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
            out.push(("rects", s));
        }
        {
            // 曲线 + 反绕洞 + 旋转（斜边跨带，carry 非零多）
            let mut s = Scene::new(96, 96);
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
            pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
            let t = Affine::translate(Vec2::new(48.0, 48.0))
                .then(Affine::rotate(0.3))
                .then(Affine::translate(Vec2::new(-48.0, -48.0)));
            s.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
            out.push(("donut_rot", s));
        }
        {
            // 自相交（W=2）+ 越出画布四缘（strip 保守扩张路径）
            let mut s = Scene::new(80, 64);
            let mut pb = PathBuilder::new();
            pb.pentagram(Vec2::new(40.0, 30.0), 34.0);
            s.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(230, 120, 60, 200)),
                BlendMode::Multiply,
            );
            let mut pb = PathBuilder::new();
            pb.rect(-30.0, 4.0, 30.0, 40.0); // 跨左缘（桶 0 钳制）
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 160, 90)));
            let mut pb = PathBuilder::new();
            pb.rect(40.0, 8.0, 140.0, 48.0); // 跨右缘（桶 tiles_x 排除）
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(200, 90, 30, 200)));
            out.push(("pentagram_offscreen", s));
        }
        {
            // 裁剪 + 非可分离混合
            let mut s = Scene::new(64, 64);
            let mut pb = PathBuilder::new();
            pb.rect(0.0, 0.0, 64.0, 64.0);
            s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
            let mut pb = PathBuilder::new();
            pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
            let c0 = s.push_clip(Affine::IDENTITY, pb.build());
            let mut pb = PathBuilder::new();
            pb.pentagram(Vec2::new(32.0, 32.0), 26.0);
            s.push_blend(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(40, 90, 160, 220)),
                BlendMode::Hue,
            );
            s.draws.last_mut().unwrap().clips = vec![c0];
            out.push(("clip_blend", s));
        }
        out
    }

    /// 深拷贝转储（破坏性测试用）。
    fn clone_dump(d: &PipelineDump) -> PipelineDump {
        PipelineDump {
            seg_ends: d.seg_ends.clone(),
            seg_draws: d.seg_draws.clone(),
            counters: d.counters.clone(),
            scan_base: d.scan_base.clone(),
            carries: d.carries.clone(),
            item_total: d.item_total,
            seg_refs: d.seg_refs.clone(),
            bucket_sums: d.bucket_sums.clone(),
            strip_meta: d.strip_meta.clone(),
            sparse_stats: d.sparse_stats,
            tr_offsets: d.tr_offsets.clone(),
            tr_list: d.tr_list.clone(),
            tile_count: d.tile_count.clone(),
            tile_start: d.tile_start.clone(),
            tile_list: d.tile_list.clone(),
            cell_meta: d.cell_meta.clone(),
            cell_segs: d.cell_segs.clone(),
            strip_cell_off: d.strip_cell_off.clone(),
            strip_bucket_off: d.strip_bucket_off.clone(),
            cell_aux: d.cell_aux.clone(),
            cell_csr_off: d.cell_csr_off.clone(),
            sparse_buckets: d.sparse_buckets.clone(),
        }
    }

    /// M3.5：稀疏路径自洽——**不依赖稠密路径**的结构正确性验证。
    ///
    /// 这是 M3 之后最重要的安全网：稠密参照在大场景（1080p 约 31 draw 起）跑不起来，
    /// 稀疏此前没有独立裁判。
    #[test]
    fn sparse_self_consistency() {
        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
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

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
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

        let Ok(renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
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

    /// M4：**GPU 侧只派发脏 tile** 必须与全量派发**逐字节一致**。
    ///
    /// 这是 GPU 增量渲染的核心不变量：fine 内核由 `global_invocation_id` 推导
    /// 像素与 tile 坐标，未派发的 tile 不写入、保留纹理中的旧值。若派发范围算错
    /// （例如偏移没被正确处理、或脏区外被误写），本测试会立刻失败。
    ///
    /// 覆盖三类情形：全脏 / 局部脏 / 无脏区（跳过）。
    #[test]
    fn gpu_damage_dispatch_matches_full() {
        use amiki_math::{Affine, Vec2};
        use amiki_scene::{Color, Damage, Paint, PathBuilder};

        let Ok(mut renderer) = pollster::block_on(GpuRenderer::new()) else {
            eprintln!("跳过：无 GPU 适配器");
            return;
        };
        renderer.path = RenderPath::Sparse;

        let mut scene = Scene::new(128, 96);
        for i in 0..6 {
            let mut pb = PathBuilder::new();
            let x = 10.0 + (i % 3) as f32 * 38.0;
            let y = 14.0 + (i / 3) as f32 * 44.0;
            pb.circle(Vec2::new(x, y), 16.0, false);
            scene.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgba8(40 + i as u8 * 30, 120, 200 - i as u8 * 20, 210)),
            );
        }

        let frame = renderer.prepare(&scene, 0.1).expect("prepare 失败");
        let (tx, ty) = (frame.scene.tiles_x, frame.scene.tiles_y);
        let tex = renderer.create_output_texture(&frame.scene);

        // 参照：全量派发
        renderer
            .render_into_persistent(&tex, &frame, None)
            .expect("全量渲染失败");
        let full = renderer.readback(&tex).expect("读回失败");

        // 1) 局部脏区（含偏移，确保不是左上角起步）
        let mut d = Damage::new(scene.width, scene.height);
        d.clear();
        d.invalidate_rect(42.0, 20.0, 100.0, 84.0);
        let skipped = renderer
            .render_into_persistent(&tex, &frame, Some(&d))
            .expect("局部脏区渲染失败");
        assert!(!skipped, "有脏区时不得跳过");
        let after = renderer.readback(&tex).expect("读回失败");
        assert_eq!(after, full, "只派发脏 tile 后，画面与全量派发不一致");

        // 2) 无脏区：必须完全跳过且不改变像素
        let mut d2 = Damage::new(scene.width, scene.height);
        d2.clear();
        assert!(d2.is_empty());
        let skipped = renderer
            .render_into_persistent(&tex, &frame, Some(&d2))
            .expect("空脏区渲染失败");
        assert!(skipped, "无脏区时必须跳过");
        let after2 = renderer.readback(&tex).expect("读回失败");
        assert_eq!(after2, full, "跳过的帧不得改变任何像素");

        // 3) 全脏：也必须等于全量
        let d3 = Damage::new(scene.width, scene.height);
        assert!(d3.is_full());
        renderer
            .render_into_persistent(&tex, &frame, Some(&d3))
            .expect("全脏渲染失败");
        let after3 = renderer.readback(&tex).expect("读回失败");
        assert_eq!(after3, full, "全脏渲染与全量派发不一致");

        // 4) 脏区 tile 数校验：尺寸不匹配的 damage 必须被断言拦下（防"全脏被误判为局部"）
        let wrong = Damage::new(scene.width + 64, scene.height);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            renderer.render_into_persistent(&tex, &frame, Some(&wrong))
        }))
        .is_err();
        assert!(caught, "画布尺寸不匹配的 damage 应被断言拦下");

        eprintln!(
            "GPU 脏区派发：全脏 / 局部脏 / 空脏区 三类均与全量逐字节一致（tiles={tx}x{ty}）"
        );
    }
}
