//! 诊断与容量表：分段计时、转储结构、投机分配的容量表。
//!
//! 从 `lib.rs` 拆出：这一簇是**纯数据结构 + 记账逻辑**，不碰 wgpu 资源，
//! 也不参与渲染决策 —— 与"怎么画"完全正交。
//!
//! - `FrameProfile` / `PassTimings` / `TimestampRecorder`：分段计时
//! - `PipelineDump`：管线中间数据转储（测试用）
//! - `SparseCapacity` / `SceneKey` / `SparseCapacityTable`：投机分配的容量表
//! - `SparseStats`：稀疏结构的内容规模

use crate::{err, GpuScene, RenderError};

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
pub(crate) struct TimestampRecorder {
    pub(crate) query_set: wgpu::QuerySet,
    pub(crate) resolve_buf: wgpu::Buffer,
    pub(crate) labels: Vec<&'static str>,
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
    /// M3.6：本帧因容量不足被作废重做的次数（正常应恒为 0）
    pub capacity_overflows: u32,
    /// M3.6：本帧是否走了投机路径（该场景容量已知）。**运行时报告**，
    /// 诊断据此判断"读回为何仍在计时"，而不是靠注释描述。
    pub speculative: bool,
    /// M3.6 投机路径专属：帧尾读回**实际用量**的耗时（溢出检测与容量更新）。
    /// 测量路径为 `None`（那两次读回自己就给出了用量）。
    pub usage_readback_ms: Option<f64>,
    /// 逐 pass GPU 时长
    pub timestamps: PassTimings,
}

impl FrameProfile {
    /// 主机侧可归因阶段合计（ms）。
    /// **主机侧渲染工作**：提取 + 分配 + 上传 + 编码（**不含任何读回**）。
    ///
    /// # 为什么要单列，而不是用 `host_total_ms()`
    ///
    /// `host_total_ms()` 把**读回**也算进去了，于是"帧路径的主机开销"里
    /// 混进了一笔**正确性开销** —— 那会让人误以为"主机侧还有很大优化空间"。
    /// 这一项才是 P1-1 能优化的对象，也是判据的对象。
    pub fn host_render_ms(&self) -> f64 {
        self.extract_ms
            + self.alloc_ms
            + self.upload_ms
            + self.encode1a_ms
            + self.encode1b_ms
            + self.encode2_ms
    }

    /// **保护性同步读回**：为确认"本帧没有溢出 / 容量足够"而必须做的 GPU 同步。
    ///
    /// 单列的理由：它买的是 **"绝不静默画出残缺帧"**（发布判据第 3 条），
    /// 属于**正确性开销**，不该被当成"没优化好的主机工作"。
    ///
    /// 两种形态：
    /// - `readback1/2`：**首帧**容量未知时测量规模（之后由容量表缓存，不再发生）
    /// - `usage_readback`：**每帧**确认实际用量未超投机分配
    pub fn safety_readback_ms(&self) -> f64 {
        self.readback1_ms.unwrap_or(0.0)
            + self.readback2_ms.unwrap_or(0.0)
            + self.usage_readback_ms.unwrap_or(0.0)
    }

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
            + self.usage_readback_ms.unwrap_or(0.0)
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
        push(&mut s, "用量读回", self.usage_readback_ms);
        push(&mut s, "时间戳解析", self.resolve_ts_ms);
        s.push_str(if self.speculative { " [投机]" } else { " [测量]" });
        if self.capacity_overflows > 0 {
            s.push_str(&format!(" [溢出重做 {}]", self.capacity_overflows));
        }
        s
    }
}

/// M3.6：稀疏结构规模的**跨帧容量**（投机分配的起点）。
///
/// # 为什么需要它
///
/// 稀疏结构的规模只有 GPU 测量之后才知道，于是原实现只能
/// "提交 → 读回规模 → 分配 → 再提交"：两次规模读回各是一次**同步停顿**
/// （实测读回1 ≈ 2.35 ms，其中约 1.4 ms 是纯 fence/map 延迟）。
///
/// 投机分配改成：把**上一帧实测到的用量**当成本帧容量，直接分配、不停顿；
/// 帧尾只读一次实际用量来更新容量。容量不足时超出的写入会被丢弃
/// （naga 对 storage buffer 有边界检查），而用量计数仍如实超过容量 —— 据此
/// 检测溢出、把该帧作废并用实测值重做，**绝不把残缺的画面交出去**。
///
/// 容量只是**元素个数**上限（与画布尺寸、场景内容无关）：它描述"要留多少格"，
/// 因此换场景、换分辨率时它仍然是一个合法（可能偏大或偏小）的上限。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseCapacity {
    /// 稀疏单元数（`cell_aux` / `cell_meta` / `cell_csr` 的规模）
    pub cells: u32,
    /// carry 定点桶数
    pub buckets: u32,
    /// 逐 tile 桶表条目数（`tile_list` 的规模）
    pub tile_entries: u32,
    /// 成员段数（`cell_segs` 的规模）
    pub items: u32,
    /// 连续溢出次数：>0 表示上一帧曾因容量不足被作废重做
    pub overflows: u32,
}

impl SparseCapacity {
    /// 容量是否可用（首帧全 0：此时必须走一次测量路径来建立容量）。
    pub fn is_valid(&self) -> bool {
        self.cells > 0 && self.items > 0
    }

    /// `usage` 是否超出本容量（任一项超出即溢出）。
    ///
    /// 溢出意味着本帧有写入被丢弃、画面残缺 —— **必须作废重做**。
    pub fn exceeds(&self, usage: &SparseCapacity) -> bool {
        usage.cells > self.cells
            || usage.buckets > self.buckets
            || usage.tile_entries > self.tile_entries
            || usage.items > self.items
    }
}

/// M3.6：**场景特征键** —— 用于给不同场景各留一份稀疏容量。
///
/// # 为什么必须按场景分表
///
/// 容量只是"元素个数上限"：换场景时它仍然**合法**，但**大小是否合适**完全
/// 取决于场景 —— 1080p×10k 曲线与 720p×64 曲线所需格数差几个数量级。
/// 共用一个全局容量会二选一地失败，两者都实测到了：
///
/// - **共用 + 直接覆盖**：换场景即溢出 → 作废重做（诊断打印 `[溢出重做 1]`）
/// - **共用 + 只增不减**：变成过度分配 → 小场景每帧按大场景的格数分配大缓冲，
///   实测 720p×64 从 0.67 ms **退化到 2.09 ms**
///
/// 因此按场景特征分表：同 key 的稀疏规模相近，各自收敛、互不影响。
///
/// `seg_bucket` 取 `total_segments` 的**不小于它的 2 的幂**：几何会随动画小幅
/// 变化，量化后 key 才稳定（否则每帧都 miss）。代价是至多 2 倍过度分配，
/// 而稀疏缓冲本身很小（1080p×10k 全量约 34 MB 量级）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SceneKey {
    pub width: u32,
    pub height: u32,
    /// draw 数
    pub draws: u32,
    /// `total_segments` 向上取到 2 的幂
    pub seg_bucket: u32,
}

impl SceneKey {
    /// 从 GPU 场景取廉价特征。只用已在 `GpuScene` 里的计数，不做任何遍历。
    pub fn of(gs: &GpuScene) -> SceneKey {
        SceneKey {
            width: gs.width,
            height: gs.height,
            draws: gs.colors.len() as u32,
            seg_bucket: gs.total_segments.max(1).next_power_of_two(),
        }
    }
}

/// 每个 key 最多容忍多少次连续溢出。超过则**放弃该 key 的投机**，
/// 退回每帧测量 —— 这样即使遇到"容量总也猜不准"的病态场景，
/// 最坏情况也只是回到没有投机时的性能，而不会每帧都作废重做。
const MAX_CONSECUTIVE_OVERFLOWS: u32 = 2;

/// M3.6：**按场景特征分表的稀疏容量**（含 LRU 淘汰与放弃投机的策略）。
#[derive(Debug, Default)]
pub struct SparseCapacityTable {
    pub(crate) entries: std::collections::HashMap<SceneKey, TableEntry>,
    clock: u64,
    /// 累计溢出次数（**只增**）。诊断与测试用：单帧的 `capacity_overflows`
    /// 在重做成功后不易观察（表项里的连续计数会被清零），累计值才能证明
    /// "确实发生过溢出并被处理过"。
    overflow_total: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TableEntry {
    pub(crate) cap: SparseCapacity,
    /// 该 key 连续溢出次数（成功一次即清零）
    overflows: u32,
    /// 最近一次使用的时间戳（LRU 用）
    used: u64,
}

/// 表项上限：超出则淘汰最久未用的 key。设上限是为了让长跑里不断变化场景的
/// 进程不会无限增长内存。
const CAPACITY_TABLE_MAX: usize = 8;

impl SparseCapacityTable {
    /// 取该 key 的可用容量。返回 `None` 表示"这一帧应走测量路径"：
    /// 从未见过该 key，或该 key 已因反复溢出被放弃。
    pub fn get(&mut self, key: SceneKey) -> Option<SparseCapacity> {
        self.clock += 1;
        let e = self.entries.get_mut(&key)?;
        e.used = self.clock;
        if e.overflows >= MAX_CONSECUTIVE_OVERFLOWS {
            return None; // 该 key 已放弃投机
        }
        Some(e.cap)
    }

    /// 记录该 key 的实测用量。`overflowed` 表示这一帧是因容量不足作废的。
    pub fn record(&mut self, key: SceneKey, usage: SparseCapacity, overflowed: bool) {
        self.clock += 1;
        if overflowed {
            self.overflow_total += 1;
        }
        match self.entries.get_mut(&key) {
            Some(e) => {
                // 只增不减：容量是"留多少格"，取历史最大值即可。
                e.cap = SparseCapacity {
                    cells: e.cap.cells.max(usage.cells),
                    buckets: e.cap.buckets.max(usage.buckets),
                    tile_entries: e.cap.tile_entries.max(usage.tile_entries),
                    items: e.cap.items.max(usage.items),
                    overflows: 0,
                };
                e.overflows = if overflowed { e.overflows + 1 } else { 0 };
                e.used = self.clock;
            }
            None => {
                if self.entries.len() >= CAPACITY_TABLE_MAX {
                    self.evict_lru();
                }
                self.entries.insert(
                    key,
                    TableEntry {
                        cap: usage,
                        overflows: u32::from(overflowed),
                        used: self.clock,
                    },
                );
            }
        }
    }

    fn evict_lru(&mut self) {
        if let Some(k) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.used)
            .map(|(k, _)| *k)
        {
            self.entries.remove(&k);
        }
    }

    /// 当前表项数（诊断用）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 表是否为空（诊断用；与 `len` 成对，符合 clippy 的约定）。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 累计溢出次数（只增）。测试用它证明"确实发生过溢出"。
    pub fn overflow_total(&self) -> u64 {
        self.overflow_total
    }

    /// 诊断：该 key 是否已放弃投机。
    pub fn gave_up(&self, key: SceneKey) -> bool {
        self.entries
            .get(&key)
            .is_some_and(|e| e.overflows >= MAX_CONSECUTIVE_OVERFLOWS)
    }
}

impl TimestampRecorder {
    pub(crate) fn new(device: &wgpu::Device) -> TimestampRecorder {
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
    pub(crate) fn slot(&mut self, label: &'static str) -> Result<(u32, u32), RenderError> {
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

    pub(crate) fn query_count(&self) -> u32 {
        self.labels.len() as u32 * 2
    }
}
