//! 场景 → GPU：`GpuScene` 的数据布局与 CPU 侧提取。
//!
//! 从 `lib.rs` 拆出：这一层是**后端无关的数据准备** —— 只做变换、控制点收集
//! 与闭式段数计算，**不压平、不分箱、不碰 wgpu 资源**。放在独立文件里
//! 也让"CPU 提取"这块性能敏感的代码有了明确的边界（P1-1 优化的对象）。
//!
//! # 为什么提取是**串行**的（一次被实测否定的尝试）
//!
//! 逐 draw 的提取互不依赖，看上去是并行的理想对象。我实现过一版
//! "分块 + 按序归并"（顺序固定，结果与串行**逐字节一致**），实测是**净损失**：
//!
//! | 线程数 | 1080p×10k 提取 |
//! |---|---|
//! | **1（串行）** | **0.70–0.91 ms** |
//! | 2 | 1.02–1.68 ms |
//! | 4 | 1.18–1.53 ms |
//! | 8 | 1.22–1.69 ms |
//! | 12 | 1.27–1.62 ms |
//!
//! **从 2 线程起就单调变差**。已排除的原因：
//! - 线程创建开销：直接测 `thread::scope` + 12 个空 spawn = **304 µs**（不是 1.2 ms）
//! - 分配器争用：改成"主线程预分配 12 份缓冲、并行区只 push"后**没有改善**
//!
//! 而同一台机器上**纯计算**负载能加速 **3.45×**（见
//! `tests::measure_actual_parallel_speedup`），说明机器能并行、是这个**负载**
//! 并行不了（剩余原因未定位）。所以按实测选择串行。
//!
//! 真要做的话，正确方向是**常驻线程池**（只建一次、跨帧复用），
//! 而不是每帧 `thread::scope`；本项目刻意不引入 `rayon`，那需要一个基于
//! `std::sync::mpsc` 的小池子 —— 属于后续工作。

use amiki_scene::Scene;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FineParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) tiles_x: u32,
    pub(crate) n_draws: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ScanParams {
    pub(crate) n: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct FlattenParams {
    pub(crate) n_quads: u32,
    pub(crate) n_cubics: u32,
    pub(crate) n_lines: u32,
    pub(crate) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct BinsParams {
    pub(crate) tiles_x: u32,
    pub(crate) n_draws: u32,
    pub(crate) tiles_y: u32,
    pub(crate) height: u32,
    pub(crate) total_segments: u32,
    /// M3b 稀疏管线：单元 / 桶 / tile 条目数（稠密路径恒为 0）
    pub(crate) n_cells: u32,
    pub(crate) n_buckets: u32,
    pub(crate) n_tile_entries: u32,
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
    /// **每个 draw 一个 vec4**：`(flags, clip0|clip1<<8, clip2|clip3<<8, grad_index)`。
    ///
    /// 直接从 per-draw 循环里拼出来，而不是"先各存一份、之后在 `sparse.rs` 里
    /// 再 zip 装配"。P1-1 实测：提取的 82% 时间在 `Vec::push` 上（每帧约 10 万次），
    /// 合并后每 draw 从 4 次 push（3 份分开 + 1 次装配）降到 1 次。
    pub draw_meta: Vec<[u32; 4]>,
    /// M6：每 draw 的渐变下标；`u32::MAX` 表示纯实心色。
    /// 随 `draw_meta.w` 传给 WGSL（该分量此前闲置）。
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
    /// P0-3：场景的离屏层。逐帧路径需要在 GPU 上合成它们，而 `Frame` 只带
    /// `GpuScene` —— 因此层数据必须放在这里（否则帧路径拿不到层）。
    pub layers: Vec<amiki_scene::Layer>,
    /// CPU 闭式段数（每入口：quad/cubic 变长，line=1）
    pub counts: Vec<u32>,
    pub total_segments: u32,
    /// emit/scatter 的 indirect dispatch 参数
    pub indirect: [u32; 3],
}

impl GpuScene {
}

/// 图元提取的累加缓冲。
///
/// # 为什么图元→draw 的映射不逐图元 push
///
/// 每条图元都要记"它属于哪个 draw"。最直接的写法是跟着图元 push 一个 draw 索引，
/// 但那就是**每个图元一次 push**（圆场景是 4 万条三次曲线 → 8 万次 push）。
///
/// 改成：提取时只把**每 draw 的计数**写进三个定长数组（每 draw 一次下标写，
/// 计数在局部变量里累加，不进内存），末尾再用 [`fill_draws`] 按计数
/// **一次性批量填充**。数据完全相同，但少了一半的 push。
#[derive(Default)]
struct DrawPrims {
    quads: Vec<[[f32; 4]; 2]>,
    cubics: Vec<[[f32; 4]; 2]>,
    lines: Vec<[f32; 4]>,
    /// 每 draw 的图元计数（下标 = draw 序号），提取阶段填充。
    n_quads: Vec<u32>,
    n_cubics: Vec<u32>,
    n_lines: Vec<u32>,
    /// 图元 → draw 的映射，末尾由 [`fill_draws`] 批量生成。
    quad_draws: Vec<u32>,
    cubic_draws: Vec<u32>,
    line_draws: Vec<u32>,
}

/// 按每 draw 的计数批量生成"图元 → draw"映射。
///
/// 用 `resize` 而不是逐元素 `push`：一次容量检查 + 一段连续写，
/// 而不是每个元素一次带边界检查的 push。
fn fill_draws(counts: &[u32], total: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(total);
    for (d, &c) in counts.iter().enumerate() {
        if c > 0 {
            out.resize(out.len() + c as usize, d as u32);
        }
    }
    out
}

/// 把一条 draw 的路径提取成图元（变换 + 控制点收集），累加进 `out`。
///
/// 与光栅化无关的部分不在这里：**不压平、不分箱** —— 曲线保持为曲线上传，
/// 压平由 GPU 并行完成（这正是本函数便宜的原因）。
fn extract_prims_into(draw: &amiki_scene::Draw, di: u32, out: &mut DrawPrims) {
    use amiki_math::Vec2;
    use amiki_scene::PathEl;
    let t = &draw.transform;
    let mut cur: Option<Vec2> = None;
    let mut sub_start: Option<Vec2> = None;
    // 计数在**局部变量**里累加，每 draw 只在末尾写一次内存
    let (mut nq, mut nc, mut nl) = (0u32, 0u32, 0u32);
    macro_rules! line {
        ($a:expr, $b:expr) => {{
            let (a, b) = ($a, $b);
            if (a - b).hypot() > 1e-9 {
                out.lines.push([a.x, a.y, b.x, b.y]);
                nl += 1;
            }
        }};
    }
    for el in &draw.path.els {
        match *el {
            PathEl::MoveTo(p) => {
                if let (Some(c), Some(s)) = (cur, sub_start) {
                    line!(c, s);
                }
                let m = t.map_point(p);
                cur = Some(m);
                sub_start = Some(m);
            }
            PathEl::LineTo(p) => {
                if let Some(c) = cur {
                    let m = t.map_point(p);
                    line!(c, m);
                    cur = Some(m);
                }
            }
            PathEl::QuadTo(c, p) => {
                if let Some(c0) = cur {
                    let m = t.map_point(p);
                    let cc = t.map_point(c);
                    out.quads.push([[c0.x, c0.y, cc.x, cc.y], [m.x, m.y, 0.0, 0.0]]);
                    nq += 1;
                    cur = Some(m);
                }
            }
            PathEl::CubicTo(c1, c2, p) => {
                if let Some(c0) = cur {
                    let m = t.map_point(p);
                    let (a, b) = (t.map_point(c1), t.map_point(c2));
                    out.cubics
                        .push([[c0.x, c0.y, a.x, a.y], [b.x, b.y, m.x, m.y]]);
                    nc += 1;
                    cur = Some(m);
                }
            }
            PathEl::Close => {
                if let (Some(c), Some(s)) = (cur, sub_start) {
                    line!(c, s);
                    cur = Some(s);
                }
            }
        }
    }
    if let (Some(c), Some(s)) = (cur, sub_start) {
        line!(c, s);
    }
    out.n_quads[di as usize] = nq;
    out.n_cubics[di as usize] = nc;
    out.n_lines[di as usize] = nl;
}

/// 逐 draw 提取图元，按 draw 顺序累加。
///
/// # 为什么是串行
///
/// 并行的尝试已被**线程数曲线**否定（1080p×10k：1 线程 0.70–0.91 ms，
/// 2 线程起单调变差到 1.27–1.62 ms）。已排除线程创建开销（实测 304 µs）
/// 与分配器争用（主线程预分配无改善），而同一台机器纯计算负载能加速 4.43× ——
/// 也就是**这个负载并行不了**。
///
/// # 容量预留按**真实图元数**而不是 draw 数
///
/// 一条 draw 可能贡献多个图元（一个圆是 4 条三次曲线，一条矩形是 4~5 条线段）。
/// 原先按 `n_draws` 预留，于是实际要 4 万条曲线却只留了 1 万个位置，
/// 仍然反复扩容、反复拷贝。这里按"路径元素数 + 1"求和预留。
fn extract_all_prims(scene: &Scene) -> DrawPrims {
    let cap: usize = scene.draws.iter().map(|d| d.path.els.len() + 1).sum();
    let mut acc = DrawPrims::default();
    acc.quads.reserve(cap);
    acc.quad_draws.reserve(cap);
    acc.cubics.reserve(cap);
    acc.cubic_draws.reserve(cap);
    acc.lines.reserve(cap);
    acc.line_draws.reserve(cap);
    // 计数数组按 draw 数预置（定长，提取阶段按下标写，不 push）
    let n = scene.draws.len();
    acc.n_quads = vec![0; n];
    acc.n_cubics = vec![0; n];
    acc.n_lines = vec![0; n];
    for (d, draw) in scene.draws.iter().enumerate() {
        extract_prims_into(draw, d as u32, &mut acc);
    }
    // 末尾一次性生成"图元 → draw"映射（见 `fill_draws`）
    acc.quad_draws = fill_draws(&acc.n_quads, acc.quads.len());
    acc.cubic_draws = fill_draws(&acc.n_cubics, acc.cubics.len());
    acc.line_draws = fill_draws(&acc.n_lines, acc.lines.len());
    acc
}
/// 场景 → GpuScene（CPU 只做变换、控制点收集与闭式段数计算；不压平、不分箱）。
pub fn build_gpu_scene(scene: &Scene, eps: f32) -> GpuScene {
    use amiki_scene::Paint;
    let width = scene.width.max(1);
    let height = scene.height.max(1);
    let mut gs = GpuScene {
        width,
        height,
        tiles_x: width.div_ceil(16),
        tiles_y: height.div_ceil(16),
        // 只记标志：层的内容由 render_blocking 单独渲染，不进 GpuScene
        has_layers: !scene.layers.is_empty(),
        layers: scene.layers.clone(),
        ..Default::default()
    };
    // per-draw 数组各按 draw 数预留一次 —— 原先完全没有预留，10k 条 draw
    // 会让每个数组从 0 长到 10k（约 14 次扩容 + 拷贝）。纯收益，不改变结果。
    let nd = scene.draws.len();
    gs.colors.reserve(nd);
    gs.draw_meta.reserve(nd);
    gs.image_index.reserve(nd);
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
        gs.image_index.push(ii);
        let _ = di;
        let nclip = draw.clips.len().min(4);
        let mut ci = [0u32; 4];
        ci[..nclip].copy_from_slice(&draw.clips[..nclip]);
        // **直接拼成 draw_meta**（每 draw 1 次 push，而不是 3 次 + 之后 1 次装配）
        gs.draw_meta.push([
            draw.blend.id() | ((nclip as u32) << 8),
            ci[0] | (ci[1] << 8),
            ci[2] | (ci[3] << 8),
            gi,
        ]);

    }

    // P1-1：路径提取是**串行**的 —— 并行的尝试已被线程数曲线否定，
    // 见本文件模块文档与 `extract_all_prims` 的说明。
    {
        let prims = extract_all_prims(scene);
        gs.quads = prims.quads;
        gs.quad_draws = prims.quad_draws;
        gs.cubics = prims.cubics;
        gs.cubic_draws = prims.cubic_draws;
        gs.lines = prims.lines;
        gs.line_draws = prims.line_draws;
    }

    // counts 按【数据布局顺序】（quads | cubics | lines）重算 —— 与 data/entry_draws
    // 的入口顺序严格一致。此前的遍历顺序 counts 与类型分块布局错位，
    // 导致混合场景（线 + 曲线）的段数张冠李戴（M2f 曲线偏差的根因）。
    // counts 按图元数预留：每个 quad/cubic/line 一项，无预留时同样会反复扩容。
    // **批量写而不是逐次 push**：`extend` 能按长度提示一次预留、再紧凑写入；
    // 线段部分只是填 `1`，用 `resize`（一次容量检查 + 连续填）更直接。
    // 圆场景下这一段是 4 万项，push 与批量写的差别是可测的。
    gs.counts.clear();
    let total = gs.quads.len() + gs.cubics.len() + gs.lines.len();
    gs.counts.reserve(total);
    gs.counts.extend(gs.quads.iter().map(|q| {
        amiki_math::QuadBez::new(
            amiki_math::Vec2::new(q[0][0], q[0][1]),
            amiki_math::Vec2::new(q[0][2], q[0][3]),
            amiki_math::Vec2::new(q[1][0], q[1][1]),
        )
        .uniform_segments(eps)
    }));
    gs.counts.extend(gs.cubics.iter().map(|c| {
        amiki_math::CubicBez::new(
            amiki_math::Vec2::new(c[0][0], c[0][1]),
            amiki_math::Vec2::new(c[0][2], c[0][3]),
            amiki_math::Vec2::new(c[1][0], c[1][1]),
            amiki_math::Vec2::new(c[1][2], c[1][3]),
        )
        .uniform_segments(eps)
    }));
    gs.counts.resize(gs.counts.len() + gs.lines.len(), 1);
    gs.total_segments = gs.counts.iter().sum();
    let wg = gs.total_segments.div_ceil(64).max(1);
    gs.indirect = [wg, 1, 1];
    gs
}
