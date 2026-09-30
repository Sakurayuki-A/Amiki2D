//! 场景 → GPU：`GpuScene` 的数据布局与 CPU 侧提取。
//!
//! 从 `lib.rs` 拆出：这一层是**后端无关的数据准备** —— 只做变换、控制点收集
//! 与闭式段数计算，**不压平、不分箱、不碰 wgpu 资源**。放在独立文件里
//! 也让"CPU 提取"这块性能敏感的代码有了明确的边界（P1-1 优化的对象）。
//!
//! 关于代价：**`build_gpu_scene` 是逐帧调用的**。曾试过按 draw 分块并行提取
//! （顺序固定、结果与串行逐字节一致），但每帧创建/回收线程的固定开销约
//! 1.2 ms，反而更慢（10k 场景 942 µs 串行 vs 1346 µs 并行）—— 见
//! `extract_all_prims` 的说明。

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
    /// P0-3：场景的离屏层。逐帧路径需要在 GPU 上合成它们，而 `Frame` 只带
    /// `GpuScene` —— 因此层数据必须放在这里（否则帧路径拿不到层）。
    pub layers: Vec<amiki_scene::Layer>,
    pub draw_flags: Vec<u32>,
    pub draw_clips: Vec<[u32; 4]>,
    /// CPU 闭式段数（每入口：quad/cubic 变长，line=1）
    pub counts: Vec<u32>,
    pub total_segments: u32,
    /// emit/scatter 的 indirect dispatch 参数
    pub indirect: [u32; 3],
}

impl GpuScene {
}

/// 单块 draw 的图元提取结果（P1-1：并行提取的中间产物）。
///
/// 提取是**逐 draw 独立**的，所以可以分块并行、再按序拼接 ——
/// 拼接顺序与串行遍历完全一致，因此结果**逐字节相同**（这是并行化的前提）。
///
/// 按**块**而不是按 draw 收集：10k 条 draw 若各自产出一个结构，
/// 就是 6 万次小分配，开销反而超过串行。
#[derive(Default)]
struct DrawPrims {
    quads: Vec<[[f32; 4]; 2]>,
    quad_draws: Vec<u32>,
    cubics: Vec<[[f32; 4]; 2]>,
    cubic_draws: Vec<u32>,
    lines: Vec<[f32; 4]>,
    line_draws: Vec<u32>,
}

impl DrawPrims {
    fn push_line(&mut self, draw: u32, a: amiki_math::Vec2, b: amiki_math::Vec2) {
        if (a - b).hypot() > 1e-9 {
            self.lines.push([a.x, a.y, b.x, b.y]);
            self.line_draws.push(draw);
        }
    }
    fn push_quad(
        &mut self,
        draw: u32,
        p0: amiki_math::Vec2,
        c: amiki_math::Vec2,
        p2: amiki_math::Vec2,
    ) {
        self.quads.push([[p0.x, p0.y, c.x, c.y], [p2.x, p2.y, 0.0, 0.0]]);
        self.quad_draws.push(draw);
    }
    fn push_cubic(
        &mut self,
        draw: u32,
        p0: amiki_math::Vec2,
        c1: amiki_math::Vec2,
        c2: amiki_math::Vec2,
        p3: amiki_math::Vec2,
    ) {
        self.cubics.push([[p0.x, p0.y, c1.x, c1.y], [c2.x, c2.y, p3.x, p3.y]]);
        self.cubic_draws.push(draw);
    }

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
    for el in &draw.path.els {
        match *el {
            PathEl::MoveTo(p) => {
                if let (Some(c), Some(s)) = (cur, sub_start) {
                    out.push_line(di, c, s);
                }
                let m = t.map_point(p);
                cur = Some(m);
                sub_start = Some(m);
            }
            PathEl::LineTo(p) => {
                if let Some(c) = cur {
                    let m = t.map_point(p);
                    out.push_line(di, c, m);
                    cur = Some(m);
                }
            }
            PathEl::QuadTo(c, p) => {
                if let Some(c0) = cur {
                    let m = t.map_point(p);
                    out.push_quad(di, c0, t.map_point(c), m);
                    cur = Some(m);
                }
            }
            PathEl::CubicTo(c1, c2, p) => {
                if let Some(c0) = cur {
                    let m = t.map_point(p);
                    out.push_cubic(di, c0, t.map_point(c1), t.map_point(c2), m);
                    cur = Some(m);
                }
            }
            PathEl::Close => {
                if let (Some(c), Some(s)) = (cur, sub_start) {
                    out.push_line(di, c, s);
                    cur = Some(s);
                }
            }
        }
    }
    if let (Some(c), Some(s)) = (cur, sub_start) {
        out.push_line(di, c, s);
    }
}

/// 提取所有 draw 的图元。
///
/// # 为什么这里是**串行**的（一次被实测否定的尝试）
///
/// 逐 draw 的提取互不依赖，看上去是并行的理想对象。我按"分块 + 按序合并"实现过
/// 一版（顺序固定，结果与串行逐字节一致），但**实测是净损失**：
///
/// | 场景 | 串行 | 每帧起线程的并行 |
/// |---|---|---|
/// | 1080p×10k 曲线 | **942 µs** | 1346 µs |
/// | 1080p×1000 曲线 | **60 µs** | 370 µs |
///
/// 固定开销约 **1.2 ms** —— `std::thread::scope` 每帧创建/回收线程的成本，
/// 远超提取本身。`build_gpu_scene` 是**逐帧**调用的，所以这个开销每帧都要付。
///
/// 正确的做法是**常驻线程池**（只建一次，跨帧复用），而不是每帧起线程。
/// 本项目刻意不引入 `rayon`（见 `Cargo.toml` 的依赖注释），所以那需要一个
/// 基于 `std::sync::mpsc` 的小池子 —— 属于后续工作。
///
/// 在那之前，**按实测选择串行**：942 µs 已经接近发布计划给的 800 µs 目标，
/// 而每帧起线程会把它推到 1.3 ms 以上。
fn extract_all_prims(scene: &Scene) -> DrawPrims {
    let n = scene.draws.len();
    let mut acc = DrawPrims::default();
    // 预留容量：避免 10k 条 draw 反复扩容（纯收益，不改变结果）
    acc.quads.reserve(n);
    acc.cubics.reserve(n);
    acc.lines.reserve(n);
    for (d, draw) in scene.draws.iter().enumerate() {
        extract_prims_into(draw, d as u32, &mut acc);
    }
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

    }

    // P1-1：路径提取**并行**做（逐 draw 独立），再按序合并。
    //
    // 逐 draw 的图元提取互不依赖，因此可以分块并行、再按顺序拼接 ——
    // 拼接顺序与串行遍历一致，所以结果**逐字节相同**（这是并行化的前提）。
    // 低于 `EXTRACT_PARALLEL_MIN_DRAWS` 时保持串行：起线程的开销（几十微秒）
    // 会超过小场景（如 720p×64，提取仅 0.01 ms）的全部收益。
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
