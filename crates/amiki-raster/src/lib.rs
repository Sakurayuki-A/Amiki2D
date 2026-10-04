//! amiki-raster —— CPU 引擎：压平 → tile 分箱 → 解析式覆盖率光栅化。
//!
//! 本 crate 的算法与 `amiki-gpu` 的 WGSL 计算着色器**逐式对应**，是 GPU 的
//! 正确性基准与无 GPU 环境的兜底引擎。
//!
//! # 核心算法（解析式覆盖率，线积分）
//!
//! 像素 P（中心 c，水平区间 [x_l, x_r]，行带 [c.y−½, c.y+½]）的覆盖率：
//!
//! ```text
//! G(P) = ∫_{x_l}^{x_r} W(x, y) dx = Σ_seg s · ∫_行带 (clamp(x(y), x_l, x_r) − x_l) dy
//! coverage = clamp(|G|, 0, 1)
//! ```
//!
//! 其中 s = sign(dy)，x(y) 为线段在扫描线上的横坐标，W 为绕数。
//!
//! # M2b 分箱结构：逐行精确成员 + 纯前缀和 carry
//!
//! 线段存放在**全局池**（零复制），tile 通过 `seg_refs` 引用。成员关系按
//! **行**精确划分：线段 s 进入 (tile, 行 j) 列表 ⟺ 该行带内线段 x 范围与
//! tile 的 x 范围相交。逐行分区满足恒等式 Σ_all s·len_j = 0（按行、按闭合
//! 路径逐点成立），于是每 (tile, draw, 行) 只需一个标量进位：
//!
//! ```text
//! G(P) = carry[行] + Σ_{(tile,行) 列表内线段} s · ( ∫clamp(x)dy − x_l·len − len )
//! carry[行] = −Σ_{x_max_行 < T_l} s·len   （= −桶前缀和，GPU 化零障碍）
//! ```
//!
//! 对比 M1：无需 tile 行聚合成员、无需「列表内排除」、无需延迟结算 ——
//! 分区按行精确，恒等式逐行直接成立。GPU 化路径：按 (行, ⌊x_max/T⌋) 桶
//! 原子计数 → 桶前缀和 → carry = −前缀，全并行。

#![forbid(unsafe_code)]

use amiki_math::{CubicBez, QuadBez};
use amiki_scene::{rgba8_from_f32_with, PathEl, Paint, Scene, ToneMap};

mod persist;
pub use persist::{FrameStats, PersistentRenderer};

mod filter;
pub use filter::{apply as apply_filter, blur as blur_layer, inflated_bounds, LayerBuffer};

mod even_odd;
pub use even_odd::even_odd_spans;

/// tile 边长（像素）。
pub const TILE: u32 = 16;
const TILE_USIZE: usize = TILE as usize;

/// 画布纵向裁剪用的微小扩张，吸收整型边界处的浮点误差。
const BAND_EPS: f32 = 1e-3;

/// 压平后的一段线段（设备空间，[x0, y0, x1, y1]）。
pub type Seg = [f32; 4];

/// 分箱结果：CPU / GPU 共享的渲染数据布局（M2b：全局线段池 + 逐行列表）。
#[derive(Clone, Debug, Default)]
pub struct BinnedScene {
    pub width: u32,
    pub height: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    /// 全局线段池（零复制）。
    pub segments: Vec<Seg>,
    /// 每 tile 在 `tile_draw_meta` 中的 (start, count)。
    pub tile_draw_offsets: Vec<[u32; 2]>,
    /// 每 (tile, draw) 条目：(draw_id, row_list_base, carry_base, flags)。
    pub tile_draw_meta: Vec<[u32; 4]>,
    /// 每 (tile, draw, 行)：(seg_refs 起点, 数量)，位于 row_list_base + 行内 row。
    pub row_lists: Vec<[u32; 2]>,
    /// 线段引用（全局池索引）。
    pub seg_refs: Vec<u32>,
    /// 每 (tile, draw) 16 行的进位绕数量。
    pub carries: Vec<f32>,
    /// 每 draw 的直线 RGBA。
    ///
    /// M6：当该 draw 使用渐变时，此处的含义变为
    /// `[gradient_index as f32, _, _, 1.0]`（着色阶段据此查 `gradients`）；
    /// `gradient_index` 为 `0xFFFF_FFFF`（即 f32 位模式）时表示纯实心色。
    pub colors: Vec<[f32; 4]>,
    /// M6：本场景用到的渐变定义（`gradient_index[i]` 为其下标）。
    pub gradients: Vec<std::sync::Arc<amiki_scene::Gradient>>,
    /// M6：每 draw 的渐变下标；`u32::MAX` 表示纯实心色。
    /// 与 `colors` 平行；用独立表而非在颜色里编码下标，避免依赖浮点精确相等。
    pub gradient_index: Vec<u32>,
    /// M6：每 draw 的图像下标；`u32::MAX` 表示非图像绘制。
    pub image_index: Vec<u32>,
    /// M6：本场景用到的位图及其放置方式。
    pub images: Vec<(std::sync::Arc<amiki_scene::Image>, amiki_scene::ImagePlacement)>,
    /// M6：离屏层（滤镜）。分箱只处理图元，层在**所有图元之后**合成，
    /// 因此这里保留原始场景与压平误差，供合成阶段递归渲染层内容。
    pub scene_layers: Vec<amiki_scene::Layer>,
    /// M6：源场景（层的递归渲染需要它，且能避免调用方另存一份）
    pub source_scene: amiki_scene::Scene,
    /// M6：压平误差界（层的递归渲染沿用同一误差）
    pub eps: f32,
    /// 每 draw 的裁剪层索引（≤ MAX_DRAW_CLIPS，未用槽位为 0）。
    pub draw_clips: Vec<[u32; 4]>,
    /// 每 scene clip 的 CPU 覆盖率 alpha 图（GPU 侧为纹理，M2d）。
    pub clip_maps: Vec<Vec<u8>>,
    /// B4：mip 采样偏移。不是 `bin()` 的参数 —— 现有调用方（大量测试）
    /// 不该被波及；C ABI 画布在 bin 后设置。
    pub mip_bias: f32,
}

/// (tile, 行) 成员项。
#[derive(Clone, Copy, Debug)]
struct RowItem {
    tile: u32,
    row: u32,
    seg: u32,
}

/// 把一个 draw 的路径变换到设备空间并压平为线段（自动闭合子路径）。
pub fn flatten_draw(transform: &amiki_math::Affine, path: &[PathEl], eps: f32) -> Vec<Seg> {
    let mut segs: Vec<Seg> = Vec::new();
    let mut cur: Option<amiki_math::Vec2> = None;
    let mut sub_start: Option<amiki_math::Vec2> = None;

    fn push_seg(a: amiki_math::Vec2, b: amiki_math::Vec2, out: &mut Vec<Seg>) {
        if (a - b).hypot() > 1e-9 {
            out.push([a.x, a.y, b.x, b.y]);
        }
    }

    for el in path {
        match *el {
            PathEl::MoveTo(p) => {
                if let (Some(c), Some(s)) = (cur, sub_start) {
                    push_seg(c, s, &mut segs);
                }
                let m = transform.map_point(p);
                cur = Some(m);
                sub_start = Some(m);
            }
            PathEl::LineTo(p) => {
                if let Some(c) = cur {
                    let m = transform.map_point(p);
                    push_seg(c, m, &mut segs);
                    cur = Some(m);
                }
            }
            PathEl::QuadTo(c, p) => {
                if let Some(c0) = cur {
                    let m = transform.map_point(p);
                    let q = QuadBez::new(c0, transform.map_point(c), m);
                    let mut pts = Vec::new();
                    // 与 GPU 引擎同一均匀压平（闭式可证明段数界），保证双端段级一致
                    q.flatten_uniform(eps, &mut pts);
                    let mut prev = c0;
                    for p in pts {
                        push_seg(prev, p, &mut segs);
                        prev = p;
                    }
                    cur = Some(m);
                }
            }
            PathEl::CubicTo(c1, c2, p) => {
                if let Some(c0) = cur {
                    let m = transform.map_point(p);
                    let cb = CubicBez::new(c0, transform.map_point(c1), transform.map_point(c2), m);
                    let mut pts = Vec::new();
                    // 与 GPU 引擎同一均匀压平（闭式可证明段数界），保证双端段级一致
                    cb.flatten_uniform(eps, &mut pts);
                    let mut prev = c0;
                    for p in pts {
                        push_seg(prev, p, &mut segs);
                        prev = p;
                    }
                    cur = Some(m);
                }
            }
            PathEl::Close => {
                if let (Some(c), Some(s)) = (cur, sub_start) {
                    push_seg(c, s, &mut segs);
                    cur = Some(s);
                }
            }
        }
    }
    if let (Some(c), Some(s)) = (cur, sub_start) {
        push_seg(c, s, &mut segs);
    }
    segs
}

/// 场景分箱：压平 + 逐行精确成员 + 前缀和 carry。
pub fn bin(scene: &Scene, flatten_eps: f32) -> BinnedScene {
    let width = scene.width.max(1);
    let height = scene.height.max(1);
    let tiles_x = width.div_ceil(TILE);
    let tiles_y = height.div_ceil(TILE);
    let n_tiles = (tiles_x * tiles_y) as usize;
    let tf = TILE as f32;

    let mut out = BinnedScene {
        width,
        height,
        tiles_x,
        tiles_y,
        segments: Vec::new(),
        tile_draw_offsets: vec![[0, 0]; n_tiles],
        tile_draw_meta: Vec::new(),
        row_lists: Vec::new(),
        seg_refs: Vec::new(),
        carries: Vec::new(),
        colors: Vec::new(),
        gradients: Vec::new(),
        gradient_index: Vec::new(),
        image_index: Vec::new(),
        images: Vec::new(),
        scene_layers: Vec::new(),
        source_scene: amiki_scene::Scene::default(),
        eps: flatten_eps,
        // mip 采样偏移（B4）。`bin()` 固定为 0；要设 bias 的调用方
        // （C ABI 画布）在 bin 之后、render 之前改这个字段。
        mip_bias: 0.0,
        draw_clips: Vec::new(),
        clip_maps: Vec::new(),
    };

    let mut tile_carry: Vec<[f32; TILE_USIZE]> = vec![[0.0; TILE_USIZE]; n_tiles];
    let mut member_items: Vec<RowItem> = Vec::new();
    let mut carry_items: Vec<(u32, f32, f32)> = Vec::new(); // (row, x_max, s·len)
    // 每 tile 是否有本 draw 的 carry 非零（避免全 tile 扫描）
    let mut carry_tiles: Vec<u32> = Vec::new();
    // 跨 draw 的 (tile, meta) 收集，最后统一建 CSR
    let mut all_meta: Vec<(u32, [u32; 4])> = Vec::new();

    // M2d：CPU 侧裁剪覆盖率图（GPU 侧渲染为纹理）
    use amiki_scene::Color;
    for clip in &scene.clips {
        let mut cs = Scene::new(scene.width, scene.height);
        cs.push(clip.transform, clip.path.clone(), Paint::Solid(Color::rgb8(255, 255, 255)));
        let cb = bin(&cs, flatten_eps);
        let img = render(&cb);
        out.clip_maps.push(img.as_chunks::<4>().0.iter().map(|px| px[3]).collect());
    }

    for (draw_id, draw) in scene.draws.iter().enumerate() {
        for v in tile_carry.iter_mut() {
            *v = [0.0; TILE_USIZE];
        }
        member_items.clear();
        carry_items.clear();
        carry_tiles.clear();

        let seg_base = out.segments.len() as u32;
        let mut segs = flatten_draw(&draw.transform, &draw.path.els, flatten_eps);
        // even-odd（0.8.0）：把奇偶区域分解成**互不重叠**的梯形（只输出竖边），
        // 之后走的仍是同一条 NonZero 路径 —— 覆盖率数学与 WGSL 一行未改。
        // 为什么要这么做（而不是改累加符号）见 `even_odd.rs` 的模块文档。
        if draw.fill_rule.is_even_odd() {
            segs = even_odd::even_odd_spans(&segs);
        }
        if segs.is_empty() {
            continue;
        }
        for s in &segs {
            out.segments.push(*s);
        }

        // 1) 逐行：成员项（每 (segment, row, tile)）+ carry 项（每 (segment, row)）
        for (si, seg) in segs.iter().enumerate() {
            let [x0, y0, x1, y1] = *seg;
            if (y1 - y0).abs() < 1e-9 {
                continue; // 水平段：无绕数变化、无面积贡献
            }
            let (sy0, sy1) = if y0 < y1 { (y0, y1) } else { (y1, y0) };
            let j_lo = ((sy0 - BAND_EPS).floor() as i64).clamp(0, height as i64 - 1);
            let j_hi = ((sy1 + BAND_EPS).floor() as i64).clamp(0, height as i64 - 1);
            for j in j_lo..=j_hi {
                let band_lo = j as f32;
                let band_hi = j as f32 + 1.0;
                let ya = sy0.max(band_lo);
                let yb = sy1.min(band_hi);
                if ya >= yb {
                    continue;
                }
                let dxdy = (x1 - x0) / (y1 - y0);
                let xa = x0 + dxdy * (ya - y0);
                let xb = x0 + dxdy * (yb - y0);
                let (x_min, x_max) = if xa <= xb { (xa, xb) } else { (xb, xa) };
                let s = if y1 > y0 { 1.0 } else { -1.0 };
                let signed_len = s * (yb - ya);
                carry_items.push((j as u32, x_max, signed_len));
                let t_lo = ((x_min / tf).floor() as i64).clamp(0, tiles_x as i64 - 1);
                let t_hi = ((x_max / tf).floor() as i64).clamp(0, tiles_x as i64 - 1);
                let tr = (j as u32) / TILE; // tile 行
                for t in t_lo..=t_hi {
                    member_items.push(RowItem {
                        tile: tr * tiles_x + t as u32, // 完整 tile 索引
                        row: (j as u32) % TILE,        // tile 内局部行
                        seg: seg_base + si as u32,
                    });
                }
            }
        }

        // 2) carry 前缀游走：逐行按 x_max 排序，tile 指针单调推进
        carry_items.sort_by(|a, b| {
            a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap())
        });
        let mut ci = 0usize;
        while ci < carry_items.len() {
            let row = carry_items[ci].0;
            let row_start = ci;
            let mut row_x_max = f32::NEG_INFINITY;
            while ci < carry_items.len() && carry_items[ci].0 == row {
                row_x_max = row_x_max.max(carry_items[ci].1);
                ci += 1;
            }
            let row_items = &carry_items[row_start..ci];
            let t_lo = ((row_items[0].1 / tf).floor() as i64).clamp(0, tiles_x as i64 - 1);
            let t_hi = ((row_x_max / tf).floor() as i64).clamp(0, tiles_x as i64 - 1);
            let mut p = 0usize;
            let mut acc = 0f32;
            for t in t_lo..=t_hi {
                let t_left = t as f32 * tf;
                while p < row_items.len() && row_items[p].1 < t_left {
                    acc += row_items[p].2;
                    p += 1;
                }
                let carry = -acc;
                if carry != 0.0 {
                    let tr = row / TILE;
                    let t_idx = (tr * tiles_x + t as u32) as usize;
                    tile_carry[t_idx][(row % TILE) as usize] = carry;
                    carry_tiles.push(tr * tiles_x + t as u32);
                }
            }
        }
        // carry 影响的 tile 去重（行序交错推入，需排序后去重）
        carry_tiles.sort_unstable();
        carry_tiles.dedup();

        // 3) 生成 (tile, draw) 条目与逐行列表
        let mut grad_idx = u32::MAX;
        let mut img_idx = u32::MAX;
        let color = match &draw.paint {
            Paint::Solid(c) => [c.r, c.g, c.b, c.a],
            Paint::Gradient(g) => {
                // 去重复用同一定义：文本/图形常对多个 draw 用同一渐变
                grad_idx = out
                    .gradients
                    .iter()
                    .position(|x| std::sync::Arc::ptr_eq(x, g))
                    .unwrap_or_else(|| {
                        out.gradients.push(g.clone());
                        out.gradients.len() - 1
                    }) as u32;
                // colors 仍写入一个占位值（着色走 gradient_index，不读这里）
                [0.0, 0.0, 0.0, 0.0]
            }
            Paint::Image { image, placement } => {
                // 同一位图 + 同一放置可复用；放置不同则各自成项（采样映射不同）
                img_idx = out
                    .images
                    .iter()
                    .position(|(x, p)| std::sync::Arc::ptr_eq(x, image) && p == placement)
                    .unwrap_or_else(|| {
                        out.images.push((image.clone(), *placement));
                        out.images.len() - 1
                    }) as u32;
                [0.0, 0.0, 0.0, 0.0]
            }
        };
        out.colors.push(color);
        out.gradient_index.push(grad_idx);
        out.image_index.push(img_idx);
        let mut clip_idx = [0u32; 4];
        let nclip = draw.clips.len().min(4);
        clip_idx[..nclip].copy_from_slice(&draw.clips[..nclip]);
        out.draw_clips.push(clip_idx);
        member_items.sort_by(|a, b| {
            a.tile.cmp(&b.tile).then(a.row.cmp(&b.row)).then(a.seg.cmp(&b.seg))
        });

        // 本 draw 触及的 tile = 成员 tile ∪ carry tile（各自有序 → 归并去重）
        let mut tiles_touched: Vec<u32> = Vec::new();
        {
            let mut mi = 0usize;
            let mut ci2 = 0usize;
            while mi < member_items.len() || ci2 < carry_tiles.len() {
                let t_item = member_items.get(mi).map(|it| it.tile);
                let t_carry = carry_tiles.get(ci2).copied();
                let t = match (t_item, t_carry) {
                    (Some(a), Some(b)) => a.min(b),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => break,
                };
                tiles_touched.push(t);
                while member_items.get(mi).map(|it| it.tile) == Some(t) {
                    mi += 1;
                }
                if t_carry == Some(t) {
                    ci2 += 1;
                }
            }
        }

        let mut it = 0usize; // member_items 游标（已按 tile 排序）
        for &t in &tiles_touched {
            let row_list_base = out.row_lists.len() as u32;
            let carry_base = out.carries.len() as u32;
            for row in 0..TILE_USIZE as u32 {
                let seg_start = out.seg_refs.len() as u32;
                while it < member_items.len()
                    && member_items[it].tile == t
                    && member_items[it].row == row
                {
                    out.seg_refs.push(member_items[it].seg);
                    it += 1;
                }
                out.row_lists.push([seg_start, (out.seg_refs.len() as u32) - seg_start]);
            }
            out.carries.extend_from_slice(&tile_carry[t as usize]);
            let flags = draw.blend.id() | ((nclip as u32) << 8);
            all_meta.push((
                t,
                [draw_id as u32, row_list_base, carry_base, flags],
            ));
        }
    }

    // 收尾：跨 draw 统一构建 CSR —— tile 内条目按 draw_id（场景顺序）排列。
    all_meta.sort_by(|a, b| a.0.cmp(&b.0).then(a.1[0].cmp(&b.1[0])));
    let mut counts = vec![0u32; n_tiles];
    for &(t, _) in &all_meta {
        counts[t as usize] += 1;
    }
    let mut starts = vec![0u32; n_tiles];
    let mut acc = 0u32;
    for t in 0..n_tiles {
        starts[t] = acc;
        acc += counts[t];
    }
    for &(_t, m) in &all_meta {
        out.tile_draw_meta.push(m);
    }
    for t in 0..n_tiles {
        out.tile_draw_offsets[t] = [starts[t], counts[t]];
    }
    // M6：把层与画布尺寸带出去 —— 分箱只处理图元，层在所有图元之后合成，
    // 那时需要递归渲染层内容。这里**只保留画布尺寸**（层内容已在
    // `scene_layers` 中），避免把整棵层树再克隆一份。
    out.scene_layers = scene.layers.clone();
    out.source_scene = Scene::new(scene.width, scene.height);
    out
}

// ============================================================================
// 核心内核：与 WGSL 逐式对应
// ============================================================================

/// ∫_ya^yb clamp(x(y), xl, xr) dy，其中 x(y) 为线性（xa@ya → xb@yb）。
/// 拐点（x(y)=xl、x(y)=xr）把区间切成至多 3 段，每段闭式梯形。
pub fn int_clamped_linear(xa: f32, xb: f32, ya: f32, yb: f32, xl: f32, xr: f32) -> f32 {
    let len = yb - ya;
    let slope = (xb - xa) / len;
    if slope == 0.0 {
        return xa.clamp(xl, xr) * len;
    }
    let mut ts = [ya, yb, f32::INFINITY, f32::INFINITY];
    let t1 = ya + (xl - xa) / slope;
    if t1 > ya && t1 < yb {
        ts[2] = t1;
    }
    let t2 = ya + (xr - xa) / slope;
    if t2 > ya && t2 < yb {
        ts[3] = t2;
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut sum = 0.0;
    for i in 0..3 {
        let (t1, t2) = (ts[i], ts[i + 1]);
        if t2.is_finite() {
            let f1 = (xa + slope * (t1 - ya)).clamp(xl, xr);
            let f2 = (xa + slope * (t2 - ya)).clamp(xl, xr);
            sum += 0.5 * (f1 + f2) * (t2 - t1);
        }
    }
    sum
}

/// 单线段对单个像素（中心 c）的有符号覆盖率贡献：
/// `s · ( ∫clamp(x)dy − x_l·len − len )`。
/// `−len` 项即 (x_r − x_l)·len，对应恒等式折叠。仅与 carry 联合使用有意义。
pub fn seg_area(x0: f32, y0: f32, x1: f32, y1: f32, cx: f32, cy: f32) -> f32 {
    let dy = y1 - y0;
    if dy.abs() < 1e-9 {
        return 0.0;
    }
    let (s, a, b) = if dy > 0.0 { (1.0, (x0, y0), (x1, y1)) } else { (-1.0, (x1, y1), (x0, y0)) };
    let band_lo = cy - 0.5;
    let band_hi = cy + 0.5;
    let ya = a.1.max(band_lo);
    let yb = b.1.min(band_hi);
    if ya >= yb {
        return 0.0;
    }
    let len = yb - ya;
    let xl = cx - 0.5;
    let xr = cx + 0.5;
    let dxdy = (b.0 - a.0) / (b.1 - a.1);
    let xa = a.0 + dxdy * (ya - a.1);
    let xb = a.0 + dxdy * (yb - a.1);
    let int_clamped = int_clamped_linear(xa, xb, ya, yb, xl, xr);
    s * (int_clamped - xl * len - len)
}

// ============================================================================
// 混合模式（W3C compositing；与 WGSL blend_color/composite 逐式对应）
// ============================================================================

fn hard_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 { b * (2.0 * s) } else { b + (2.0 * s - 1.0) - b * (2.0 * s - 1.0) }
}

fn soft_light_d(x: f32) -> f32 {
    if x <= 0.25 { ((16.0 * x - 12.0) * x + 4.0) * x } else { x.sqrt() }
}

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(mut c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    if n < 0.0 {
        let f = l / (l - n);
        for v in &mut c {
            *v = l + (*v - l) * f;
        }
    }
    if x > 1.0 {
        let f = (1.0 - l) / (x - l);
        for v in &mut c {
            *v = l + (*v - l) * f;
        }
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let dl = l - lum(c);
    clip_color([c[0] + dl, c[1] + dl, c[2] + dl])
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    if mx <= mn {
        return [0.0; 3];
    }
    let mid_v = ((c[0] + c[1] + c[2] - mx - mn) - mn) * s / (mx - mn);
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        out[i] = if c[i] == mx {
            s
        } else if c[i] == mn {
            0.0
        } else {
            mid_v
        };
    }
    out
}

/// B(Cb, Cs)：W3C 混合模式函数。
pub fn blend_color(mode: u32, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    match mode {
        1 => [cb[0] * cs[0], cb[1] * cs[1], cb[2] * cs[2]], // multiply
        2 => {                                              // screen
            let mut o = [0.0f32; 3];
            for i in 0..3 { o[i] = cb[i] + cs[i] - cb[i] * cs[i]; }
            o
        }
        3 => [hard_light(cs[0], cb[0]), hard_light(cs[1], cb[1]), hard_light(cs[2], cb[2])], // overlay
        4 => [cb[0].min(cs[0]), cb[1].min(cs[1]), cb[2].min(cs[2])],                        // darken
        5 => [cb[0].max(cs[0]), cb[1].max(cs[1]), cb[2].max(cs[2])],                        // lighten
        6 => {                                                                              // color-dodge
            let mut o = [0.0f32; 3];
            for i in 0..3 {
                o[i] = if cb[i] <= 0.0 { 0.0 } else if cs[i] >= 1.0 { 1.0 }
                       else { (cb[i] / (1.0 - cs[i])).min(1.0) };
            }
            o
        }
        7 => {                                                                              // color-burn
            let mut o = [0.0f32; 3];
            for i in 0..3 {
                o[i] = if cb[i] >= 1.0 { 1.0 } else if cs[i] <= 0.0 { 0.0 }
                       else { 1.0 - ((1.0 - cb[i]) / cs[i]).min(1.0) };
            }
            o
        }
        8 => [hard_light(cb[0], cs[0]), hard_light(cb[1], cs[1]), hard_light(cb[2], cs[2])], // hard-light
        9 => {                                                                              // soft-light
            let mut o = [0.0f32; 3];
            for i in 0..3 {
                o[i] = if cs[i] <= 0.5 {
                    cb[i] - (1.0 - 2.0 * cs[i]) * cb[i] * (1.0 - cb[i])
                } else {
                    cb[i] + (2.0 * cs[i] - 1.0) * (soft_light_d(cb[i]) - cb[i])
                };
            }
            o
        }
        10 => [ (cb[0] - cs[0]).abs(), (cb[1] - cs[1]).abs(), (cb[2] - cs[2]).abs() ],      // difference
        11 => {                                                                             // exclusion
            let mut o = [0.0f32; 3];
            for i in 0..3 { o[i] = cb[i] + cs[i] - 2.0 * cb[i] * cs[i]; }
            o
        }
        12 => set_lum(set_sat(cs, sat(cb)), lum(cb)),   // hue
        13 => set_lum(set_sat(cb, sat(cs)), lum(cb)),   // saturation
        14 => set_lum(cs, lum(cb)),                     // color
        15 => set_lum(cb, lum(cs)),                     // luminosity
        _ => cs, // normal
    }
}

/// 直线 α 下的 src-over + 混合（与 WGSL composite 逐式对应）。
/// `src` 为直线 RGBA，`src_a` = 源 α × 覆盖率。
pub fn composite(dst: [f32; 4], src: [f32; 4], src_a: f32, mode: u32) -> [f32; 4] {
    if mode == 0 {
        let inv = 1.0 - src_a;
        return [
            src[0] * src_a + dst[0] * inv,
            src[1] * src_a + dst[1] * inv,
            src[2] * src_a + dst[2] * inv,
            src_a + dst[3] * inv,
        ];
    }
    let ab = dst[3];
    let cb = [dst[0], dst[1], dst[2]];
    let cs = [src[0], src[1], src[2]];
    let b = blend_color(mode, cb, cs);
    let ao = src_a + ab * (1.0 - src_a);
    let mut co = [0.0f32; 3];
    for i in 0..3 {
        co[i] = src_a * (1.0 - ab) * cs[i] + src_a * ab * b[i] + (1.0 - src_a) * ab * cb[i];
    }
    let out_rgb = if ao > 0.0 { [co[0] / ao, co[1] / ao, co[2] / ao] } else { [0.0; 3] };
    [out_rgb[0], out_rgb[1], out_rgb[2], ao]
}

/// 把某个 tile 清零为透明（未预乘 RGBA 全 0）。
///
/// **为什么必须清零**：脏 tile 可能"从有内容变成无内容"（例如删掉一条 draw）。
/// `shade_tile` 在 tile 无绘制时是 early-return（保留原像素）——那对**未脏**的 tile
/// 是复用的正确语义；但**已脏**的 tile 必须从零开始重算，否则旧内容永久残留。
fn clear_tile(binned: &BinnedScene, img: &mut [u8], tx: usize, ty: usize) {
    let w = binned.width as usize;
    let h = binned.height as usize;
    let x0 = tx * TILE_USIZE;
    let x1 = (x0 + TILE_USIZE).min(w);
    let y0 = ty * TILE_USIZE;
    let y1 = (y0 + TILE_USIZE).min(h);
    for py in y0..y1 {
        let o = (py * w + x0) * 4;
        let n = (x1 - x0) * 4;
        img[o..o + n].fill(0);
    }
}

/// M6：求某 draw 在设备坐标 `(x, y)` 处的颜色。
///
/// 实心色直接返回；渐变按几何求参数 `t` 并在色标间**解析插值**。
/// 刻意不查表：查表会把着色量化到 1/255，而本项目对解析覆盖率有质量承诺，
/// 不该在着色阶段引入额外近似。
fn resolve_color(binned: &BinnedScene, draw: usize, x: f32, y: f32) -> [f32; 4] {
    if let Some(g) = gradient_of(binned, draw) {
        let c = g.sample(g.param_at(amiki_math::Vec2::new(x, y)));
        return [c.r, c.g, c.b, c.a];
    }
    if let Some((img, place)) = image_of(binned, draw) {
        // 设备坐标 → 图像空间 uv → 采样。
        // 仿射退化（行列式≈0）时返回透明：宁可该像素不画，也不要除零产出 NaN。
        return match place.uv_at(amiki_math::Vec2::new(x, y)) {
            Some(uv) => {
                // B4：有 mip 链走三线性。LOD 用与 GPU 打包完全相同的
                // `compute_lod`（同一份代码），且与放置一样是 draw 常量。
                if img.mip.is_some() {
                    let lod = amiki_scene::compute_lod(
                        place.u_axis,
                        place.v_axis,
                        img.width,
                        img.height,
                        binned.mip_bias,
                    );
                    img.sample_mipmap(uv, place.repeat, lod)
                } else {
                    img.sample_bilinear(uv, place.repeat)
                }
            }
            None => [0.0; 4],
        };
    }
    binned.colors[draw]
}

/// 若该 draw 使用位图，返回位图与其放置方式。
#[inline]
fn image_of(
    binned: &BinnedScene,
    draw: usize,
) -> Option<&(std::sync::Arc<amiki_scene::Image>, amiki_scene::ImagePlacement)> {
    let idx = binned.image_index.get(draw).copied().unwrap_or(u32::MAX);
    if idx == u32::MAX {
        None
    } else {
        binned.images.get(idx as usize)
    }
}

/// 若该 draw 使用渐变，返回其定义。
///
/// 用**独立的索引表**而非从颜色值里编码下标：后者要依赖浮点精确相等这种脆弱约定，
/// 一旦将来有颜色恰好碰撞就会静默出错。
#[inline]
fn gradient_of(binned: &BinnedScene, draw: usize) -> Option<&std::sync::Arc<amiki_scene::Gradient>> {
    let idx = binned.gradient_index.get(draw).copied().unwrap_or(u32::MAX);
    if idx == u32::MAX {
        None
    } else {
        binned.gradients.get(idx as usize)
    }
}

/// 渲染单个 tile 到 `img`（tile 内逐像素解析覆盖率 + 画家序合成）。
///
/// M4：全量 `render` 与增量 `render_damage` **共用**此函数——增量路径绝不复制
/// 一份着色数学，否则两条路径会漂移（这正是本项目历史上踩过的坑）。
///
/// 注意：本函数在 tile 无绘制时 **early-return（保留原像素）**。
/// 这对"未脏、复用上一帧"是正确语义；对"已脏"的 tile 必须先 `clear_tile`。
fn shade_tile(binned: &BinnedScene, img: &mut [u8], tx: usize, ty: usize) {
    let w = binned.width as usize;
    let h = binned.height as usize;
    let tiles_x = binned.tiles_x as usize;
    let off = binned.tile_draw_offsets[ty * tiles_x + tx];
    if off[1] == 0 {
        return; // 本 tile 无任何绘制：保留 img 原值（增量路径即"不触碰"）
    }
    let x0 = tx * TILE_USIZE;
    let x1 = (x0 + TILE_USIZE).min(w);
    let y0 = ty * TILE_USIZE;
    let y1 = (y0 + TILE_USIZE).min(h);
    for py in y0..y1 {
        let row = py % TILE_USIZE;
        for px in x0..x1 {
            let col = shade_pixel(binned, w, off, row, px, py);
            let o = (py * w + px) * 4;
            for k in 0..4 {
                img[o + k] = (col[k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// 单个像素的着色结果，**不量化**（直线 RGBA，f32，可超过 1.0）。
///
/// 从 [`shade_tile`] 里抽出来是为了让 f32 路径（[`shade_tile_f32`]）复用同一份
/// 几何与合成数学 —— 两条路径若各写一份，迟早会在某个边界上漂移。
///
/// 返回值**不裁剪**：这是 HDR 的前提。量化到 8 位发生在调用方，也就是
/// 最终输出那一步；层内容走 f32 路径，因此 >1.0 的高光源能穿过层边界。
fn shade_pixel(
    binned: &BinnedScene,
    w: usize,
    off: [u32; 2],
    row: usize,
    px: usize,
    py: usize,
) -> [f32; 4] {
    let cx = px as f32 + 0.5;
    let cy = py as f32 + 0.5;
    let mut col = [0.0f32; 4];
    for mi in off[0]..off[0] + off[1] {
        let m = binned.tile_draw_meta[mi as usize];
        let rl = binned.row_lists[m[1] as usize + row];
        let mut area = binned.carries[m[2] as usize + row];
        for k in 0..rl[1] {
            let seg = binned.segments[binned.seg_refs[(rl[0] + k) as usize] as usize];
            area += seg_area(seg[0], seg[1], seg[2], seg[3], cx, cy);
        }
        let mut a = area.abs().min(1.0);
        let nclip = ((m[3] >> 8) & 0xF) as usize;
        if nclip > 0 && a > 0.0 {
            let pix = py * w + px;
            let clips = &binned.draw_clips[m[0] as usize];
            for &clip_idx in &clips[..nclip] {
                a *= binned.clip_maps[clip_idx as usize][pix] as f32 / 255.0;
            }
        }
        if a > 0.0 {
            let c = resolve_color(binned, m[0] as usize, cx, cy);
            col = composite(col, c, c[3] * a, m[3] & 0xFF);
        }
    }
    col
}

/// 与 [`shade_tile`] 同数学，但写进 **f32** 缓冲，**不量化、不裁剪**。
///
/// 这是 HDR 的关键一步：层内容必须先以 f32 存在，滤镜（模糊）才能在
/// **未裁剪**的数据上工作。此前层内容先渲染成 8 位再拷进 f32 的
/// `LayerBuffer` —— 裁剪在那一步就已经发生了，>1.0 的高光源直接消失。
fn shade_tile_f32(binned: &BinnedScene, img: &mut [[f32; 4]], w: usize, h: usize, tx: usize, ty: usize) {
    let tiles_x = binned.tiles_x as usize;
    let off = binned.tile_draw_offsets[ty * tiles_x + tx];
    if off[1] == 0 {
        return;
    }
    let x0 = tx * TILE_USIZE;
    let x1 = (x0 + TILE_USIZE).min(w);
    let y0 = ty * TILE_USIZE;
    let y1 = (y0 + TILE_USIZE).min(h);
    for py in y0..y1 {
        let row = py % TILE_USIZE;
        for px in x0..x1 {
            img[py * w + px] = shade_pixel(binned, w, off, row, px, py);
        }
    }
}


/// 把**已滤波**的层缓冲按不透明度/混合合成到根图像上，**只处理指定设备矩形**。
///
/// `rect` 为 `(rx0, ry0, rx1, ry1)`（半开区间，设备像素）。全量合成时传整个画布。
///
/// # 为什么必须与 `layer_buffer_from_content` 分开
///
/// 增量渲染要跨帧复用层缓冲，只在脏 tile 上重新合成。若把"渲染+滤波+合成"
/// 写成一个整体，增量路径就得另写一份合成逻辑 —— 两份实现必然漂移，
/// 表现为"全量与增量在某些边界上不同"。拆开之后两条路径**共用**这两步。
///
/// 合成在**预乘空间**进行：层缓冲是直线色，直接交给要求直线输入的 `composite`
/// 会在小 alpha 处放大模糊量化噪声（实测边缘色偏 86/255）。故把目标也转成预乘，
/// 算完再转回直线存储。
/// 与 [`layer_buffer_from_content`] 同逻辑，但源是**未裁剪的 f32**。
///
/// 为什么另开一个而不是让原函数泛型化：原函数做的是 `u8 / 255.0` 的转换，
/// 那一步正是 HDR 信息的丢失点。f32 版本**完全不做转换**，直接搬
/// `[f32; 4]` —— 语义差别值得一个独立的函数名。
///
/// 边界计算与原函数逐字一致（含 `inflated_bounds` 的滤镜扩散余量），
/// 否则两条路径会在边界上给出不同结果。
fn layer_buffer_from_content_f32(
    content: &[[f32; 4]],
    root_w: u32,
    root_h: u32,
    layer: &amiki_scene::Layer,
) -> Option<(i32, i32, LayerBuffer)> {
    if layer.bounds.is_empty() {
        return None;
    }
    let b = inflated_bounds(layer.bounds, layer.filter);
    let bx = b.x.max(0);
    let by = b.y.max(0);
    if bx >= root_w as i32 || by >= root_h as i32 {
        return None;
    }
    let lw = b.width.min(root_w - bx as u32);
    let lh = b.height.min(root_h - by as u32);
    if lw == 0 || lh == 0 {
        return None;
    }

    let mut buf = LayerBuffer::new(lw, lh);
    for y in 0..lh {
        for x in 0..lw {
            let sx = bx + x as i32;
            let sy = by + y as i32;
            let src = if sx >= 0 && sy >= 0 && (sx as u32) < root_w && (sy as u32) < root_h {
                content[(sy as u32 * root_w + sx as u32) as usize]
            } else {
                [0.0; 4]
            };
            buf.set(x, y, src);
        }
    }
    apply_filter(&mut buf, layer.filter);
    Some((bx, by, buf))
}

/// 与 [`blend_layer_rect`] 同一份数学，但目标是**未裁剪的 f32**。
///
/// # 为什么必须存在
///
/// `render_blocking`（GPU 参考实现）把渲染结果读回成 f32，再在主机侧按层合成。
/// 若用 u8 版，每一层的中间结果都会被裁到 1.0 —— HDR 高光源跨层就丢了。
///
/// # 为什么不是把 u8 版泛型化
///
/// 那样要引入像素存储的抽象 trait，改动面比"多一个函数"大；而这里真正需要
/// 共享的**数学**（[`composite_premul`]）本来就是共享的，两个函数只是存储类型
/// 不同。防漂移靠测试：`blend_layer_rect_f32` 与 u8 版在 `[0,1]` 范围内的结果
/// **必须一致**（见 `raster` 的 `layer_blend_f32_matches_u8`）。
///
/// 唯一的语义差别：本函数**不做 clamp** —— 那是最终输出才该做的事。
#[allow(clippy::too_many_arguments)]
pub fn blend_layer_rect_f32(
    root: &mut [[f32; 4]],
    root_w: u32,
    root_h: u32,
    buf: &LayerBuffer,
    x: i32,
    y: i32,
    opacity: f32,
    blend: u32,
    rect: (u32, u32, u32, u32),
) {
    if buf.is_empty() {
        return;
    }
    let opacity = opacity.clamp(0.0, 1.0);
    let (rx0, ry0, rx1, ry1) = rect;
    let lx0 = (rx0 as i32 - x).max(0).min(buf.width as i32);
    let ly0 = (ry0 as i32 - y).max(0).min(buf.height as i32);
    let lx1 = (rx1 as i32 - x).max(0).min(buf.width as i32);
    let ly1 = (ry1 as i32 - y).max(0).min(buf.height as i32);
    if lx0 >= lx1 || ly0 >= ly1 {
        return;
    }
    for ly in ly0..ly1 {
        for lx in lx0..lx1 {
            let gx = x + lx;
            let gy = y + ly;
            if gx < 0 || gy < 0 || gx >= root_w as i32 || gy >= root_h as i32 {
                continue;
            }
            let p = buf.get(lx as u32, ly as u32);
            let sa = p[3] * opacity;
            if sa <= 0.0 {
                continue;
            }
            let src_pm = [p[0] * sa, p[1] * sa, p[2] * sa, sa];
            let idx = (gy as u32 * root_w + gx as u32) as usize;
            let d = root[idx];
            let da = d[3];
            let dst_pm = [d[0] * da, d[1] * da, d[2] * da, da];
            let out_pm = composite_premul(dst_pm, src_pm, blend);
            let oa = out_pm[3];
            let inv = if oa > 0.0 { 1.0 / oa } else { 0.0 };
            // **不 clamp** —— >1.0 要留给后面的层与最终的色调映射
            root[idx] = [out_pm[0] * inv, out_pm[1] * inv, out_pm[2] * inv, oa];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn blend_layer_rect(
    root: &mut [u8],
    root_w: u32,
    root_h: u32,
    buf: &LayerBuffer,
    x: i32,
    y: i32,
    opacity: f32,
    blend: u32,
    rect: (u32, u32, u32, u32),
) {
    if buf.is_empty() {
        return;
    }
    let opacity = opacity.clamp(0.0, 1.0);
    let (rx0, ry0, rx1, ry1) = rect;
    // 缓冲内与请求矩形相交的范围
    let lx0 = (rx0 as i32 - x).max(0).min(buf.width as i32);
    let ly0 = (ry0 as i32 - y).max(0).min(buf.height as i32);
    let lx1 = (rx1 as i32 - x).max(0).min(buf.width as i32);
    let ly1 = (ry1 as i32 - y).max(0).min(buf.height as i32);
    if lx0 >= lx1 || ly0 >= ly1 {
        return;
    }
    for ly in ly0..ly1 {
        for lx in lx0..lx1 {
            let gx = x + lx;
            let gy = y + ly;
            if gx < 0 || gy < 0 || gx >= root_w as i32 || gy >= root_h as i32 {
                continue;
            }
            let p = buf.get(lx as u32, ly as u32);
            let sa = p[3] * opacity;
            if sa <= 0.0 {
                continue;
            }
            // 源：直线 → 预乘
            let src_pm = [p[0] * sa, p[1] * sa, p[2] * sa, sa];
            let o = ((gy as u32 * root_w + gx as u32) * 4) as usize;
            let da = root[o + 3] as f32 / 255.0;
            // 目标：直线 → 预乘
            let dst_pm = [
                root[o] as f32 / 255.0 * da,
                root[o + 1] as f32 / 255.0 * da,
                root[o + 2] as f32 / 255.0 * da,
                da,
            ];
            let out_pm = composite_premul(dst_pm, src_pm, blend);
            let oa = out_pm[3];
            let inv = if oa > 0.0 { 1.0 / oa } else { 0.0 };
            let straight = [out_pm[0] * inv, out_pm[1] * inv, out_pm[2] * inv, oa];
            for k in 0..4 {
                root[o + k] = (straight[k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// M6：合成一个离屏层到父图像（**全量路径**）。
///
/// 步骤与 `filter` 模块的说明一致：
/// ```text
/// 1) 把层内容渲进独立缓冲（含它自己的嵌套层 —— 层可嵌套）
/// 2) 按 bounds 裁出需要的范围，并向外扩张以容纳滤镜扩散
/// 3) 对整层滤波
/// 4) 按不透明度 + 混合模式合成回父图像
/// ```
///
/// 本函数只是把 1~4 串起来；真正的工作在 [`layer_buffer_from_content`]
/// 与 [`blend_layer_rect`] 里，好让增量路径复用同两步。
fn composite_layer(
    root: &mut [u8],
    scene: &amiki_scene::Scene,
    layer: &amiki_scene::Layer,
    eps: f32,
    mip_bias: f32,
) {
    if layer.bounds.is_empty() {
        return;
    }
    // 1) 层内容（同一画布尺寸，故层内坐标即设备坐标）
    //
    // **走 f32 路径**：层内容必须先以未裁剪的 f32 存在，滤镜才能在 HDR 数据上
    // 工作。此前这里用的是 8 位的 `render_inner`，>1.0 的高光源在模糊**之前**
    // 就被裁到 1.0 —— 发光效果直接消失（`hdr_values_survive_a_blurred_layer`
    // 测到的两个 level 能量完全相同，就是这个原因）。
    let layer_img = render_inner_f32(scene.width, scene.height, &layer.scene, eps, mip_bias);
    let Some((bx, by, buf)) = layer_buffer_from_content_f32(&layer_img, scene.width, scene.height, layer)
    else {
        return;
    };
    // 4) 合成回父图像（整幅）
    blend_layer_rect(
        root,
        scene.width,
        scene.height,
        &buf,
        bx,
        by,
        layer.opacity,
        layer.blend.id(),
        (0, 0, scene.width, scene.height),
    );
}

/// 预乘空间下的 src-over + 混合（用于离屏层合成）。
///
/// 与 [`composite`] 的数学等价，但输入输出都是**预乘**——层合成必须如此，
/// 否则模糊后小 alpha 处的反预乘会放大噪声。
///
/// 公开给 `amiki-gpu`：GPU 的层合成在主机侧做，必须与这里**同一份数学**，
/// 否则两端会漂移（这正是本函数被 `pub` 的原因，而不是复制一份）。
pub fn composite_premul(dst: [f32; 4], src: [f32; 4], mode: u32) -> [f32; 4] {
    let sa = src[3];
    let da = dst[3];
    let out_a = sa + da * (1.0 - sa);
    // 混合函数按定义作用在**直线色**上，故先反预乘取直线色
    let inv_d = if da > 0.0 { 1.0 / da } else { 0.0 };
    let inv_s = if sa > 0.0 { 1.0 / sa } else { 0.0 };
    let cb = [dst[0] * inv_d, dst[1] * inv_d, dst[2] * inv_d];
    let cs = [src[0] * inv_s, src[1] * inv_s, src[2] * inv_s];
    let b = if mode == 0 { cs } else { blend_color(mode, cb, cs) };
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        // 预乘空间的 W3C 合成公式
        out[i] = sa * (1.0 - da) * cs[i] + sa * da * b[i] + (1.0 - sa) * da * cb[i];
    }
    out[3] = out_a;
    out
}


/// 与 [`render_inner`] 同数学，但输出 **f32 且不裁剪** —— 供层内容使用。
///
/// 只用于层：主画布仍走 8 位（见 [`render`]），因为主画布的最终输出本来
/// 就要落到 8 位显示范围。层不一样 —— 它的内容要先经过滤镜，
/// 那一步之前必须保持 HDR。
fn render_inner_f32(
    width: u32,
    height: u32,
    scene: &amiki_scene::Scene,
    eps: f32,
    mip_bias: f32,
) -> Vec<[f32; 4]> {
    let mut binned = bin(scene, eps);
    binned.mip_bias = mip_bias;
    let w = width as usize;
    let h = height as usize;
    // 新建即全零，等价于逐 tile clear（这里没有增量复用，不需要 clear_tile）。
    let mut img = vec![[0.0f32; 4]; w * h];
    for ty in 0..binned.tiles_y as usize {
        for tx in 0..binned.tiles_x as usize {
            shade_tile_f32(&binned, &mut img, w, h, tx, ty);
        }
    }
    img
}

/// 从零渲染，但**只清/着色指定的 tile 列表**（`None` = 全部 tile）。
///
/// M4 测试用：全量 `render` 与"只重算部分 tile"之间需要一条**可控**的中间路径，
/// 以便区分"damage 覆盖不全"与"着色本身不一致"这两类失败。
/// 注意 `clear_tile` 先行使 `None`（全部 tile）与本函数的增量形式语义一致。
pub fn render_tiles(binned: &BinnedScene, tiles: Option<&[(u32, u32)]>) -> Vec<u8> {
    render_tiles_flat(binned, binned.width, binned.height, tiles)
}

/// 渲染（含层）的核心：先画图元，再按顺序合成离屏层。
///
/// 层在**所有图元之后**合成：这样"层内绘制先完成再整层滤波"的语义成立，
/// 也避免层内内容被父场景后续绘制意外穿插。
fn render_tiles_flat(
    binned: &BinnedScene,
    width: u32,
    height: u32,
    tiles: Option<&[(u32, u32)]>,
) -> Vec<u8> {
    assert_eq!(
        TILE,
        amiki_scene::DAMAGE_TILE,
        "分箱 tile 尺寸与 damage 的 tile 尺寸必须一致，否则增量渲染会与全量不一致"
    );
    let w = width as usize;
    let h = height as usize;
    let mut img = vec![0u8; w * h * 4];
    match tiles {
        None => {
            for ty in 0..binned.tiles_y as usize {
                for tx in 0..binned.tiles_x as usize {
                    clear_tile(binned, &mut img, tx, ty);
                    shade_tile(binned, &mut img, tx, ty);
                }
            }
        }
        Some(list) => {
            for &(tx, ty) in list {
                let (tx, ty) = (tx as usize, ty as usize);
                if tx >= binned.tiles_x as usize || ty >= binned.tiles_y as usize {
                    continue;
                }
                clear_tile(binned, &mut img, tx, ty);
                shade_tile(binned, &mut img, tx, ty);
            }
        }
    }
    img
}

/// CPU 光栅化：输出直线 RGBA8（未预乘）。
///
/// M6：图元画完后按顺序合成离屏层（滤镜）。层在**所有图元之后**合成，
/// 保证"层内先画完、再整层滤波"的语义。
/// 与 [`render`] 同结果，但**保持 f32 且不裁剪**（含层，层合成也在 f32 里做）。
///
/// # 为什么需要它
///
/// [`render`] 的输出是 8 位 —— 那对"最终显示"是对的，但 HDR 值在降到 8 位
/// **之前**必须先经过色调映射。要在映射前拿到未裁剪的数据，就需要这条路径。
///
/// 它比 [`render`] 贵：整幅画布用 `[f32; 4]`（每像素 16 字节，是 u8 的 4 倍）。
/// 所以只有**真的要用非 Raw 色调映射**时才走这里（见 [`render_with_tone`]）。
pub fn render_f32(binned: &BinnedScene) -> Vec<[f32; 4]> {
    let w = binned.width as usize;
    let h = binned.height as usize;
    let mut img = vec![[0.0f32; 4]; w * h];
    for ty in 0..binned.tiles_y as usize {
        for tx in 0..binned.tiles_x as usize {
            shade_tile_f32(binned, &mut img, w, h, tx, ty);
        }
    }
    // 层在 f32 上合成：>1.0 的高光源能穿过层边界存活
    for layer in &binned.scene_layers {
        composite_layer_f32(&mut img, &binned.source_scene, layer, binned.eps, binned.mip_bias);
    }
    img
}

/// 与 [`render`] 同样的出口（8 位直线 RGBA），但按 `tone` 把 HDR 压到显示范围。
///
/// `ToneMap::Raw`（默认）走**快路径**，直接调用 [`render`] —— 不分配那块 f32
/// 缓冲。也就是说"不用色调映射"的调用方一分钱都不多花。
///
/// 完整说明见 `docs/COLOR.md`。
pub fn render_with_tone(binned: &BinnedScene, tone: ToneMap) -> Vec<u8> {
    if tone == ToneMap::Raw {
        return render(binned);
    }
    rgba8_from_f32_with(&render_f32(binned), tone)
}

/// [`composite_layer`] 的 f32 版本：全量、未裁剪。
fn composite_layer_f32(
    root: &mut [[f32; 4]],
    scene: &amiki_scene::Scene,
    layer: &amiki_scene::Layer,
    eps: f32,
    mip_bias: f32,
) {
    if layer.bounds.is_empty() {
        return;
    }
    // 层内容走 f32（>1.0 存活），滤镜在未裁剪的数据上做，合成也不裁剪
    let content = render_inner_f32(scene.width, scene.height, &layer.scene, eps, mip_bias);
    let Some((bx, by, buf)) =
        layer_buffer_from_content_f32(&content, scene.width, scene.height, layer)
    else {
        return;
    };
    blend_layer_rect_f32(
        root,
        scene.width,
        scene.height,
        &buf,
        bx,
        by,
        layer.opacity,
        layer.blend.id(),
        (0, 0, scene.width, scene.height),
    );
}

pub fn render(binned: &BinnedScene) -> Vec<u8> {
    let mut img = render_tiles(binned, None);
    for layer in &binned.scene_layers {
        composite_layer(&mut img, &binned.source_scene, layer, binned.eps, binned.mip_bias);
    }
    img
}

/// M4 **增量渲染**：只重算与脏区相交的 tile，其余保留 `img` 原有像素。
///
/// 语义契约（这是 damage region 的正确性铁律）：
///
/// ```text
/// 任意 invalidate 序列下，render_damage(.., damage) 的结果
/// 必须与 render(..) 全量渲染【逐字节一致】
/// ```
///
/// 成立的理由：
/// - 脏区经 `Damage::tiles()` 已**向外扩到 tile 边界**，故任何被改动的像素都在
///   被重算的 tile 内；
/// - 单个 tile 的着色只依赖该 tile 的成员段与画家序（`shade_tile`），
///   与其他 tile 的**当前**内容无关，也与"这是第几帧"无关；
/// - 未被重算的 tile 保留的正是上一帧（或全量）算出的同一结果。
///
/// `img` 尺寸必须是 `width * height * 4`；`damage` 的 tile 尺寸必须与分箱一致。
pub fn render_damage(binned: &BinnedScene, img: &mut [u8], damage: &amiki_scene::Damage) {
    assert_eq!(
        TILE,
        amiki_scene::DAMAGE_TILE,
        "分箱 tile 尺寸与 damage 的 tile 尺寸必须一致，否则增量渲染会与全量不一致"
    );
    // M6：**含层的场景回落到全量渲染**。
    //
    // 原因：层合成是"读层范围 → 滤波 → 与父像素混合"，而滤波需要整层的完整内容。
    // 只重算部分 tile 会让层内容残缺（模糊读到的是缺了块的数据），
    // 结果既不正确也难以察觉。因此这里选择"慢但正确"。
    //
    // 这与 damage 的设计并不冲突：damage 的收益场景是"静态界面 + 少量图元变动"，
    // 而层（滤镜）本身在静态界面里也是静态的 —— 真正需要的优化是
    // **把层的结果也缓存下来**，只重算与层相交的脏区。那是后续工作。
    if !binned.scene_layers.is_empty() {
        let full = render(binned);
        img.copy_from_slice(&full);
        return;
    }
    let tiles_x = binned.tiles_x;
    let tiles_y = binned.tiles_y;
    let (tx0, ty0, tx1, ty1) = match damage.tiles() {
        Some(t) => t,
        // 全脏：等价于全量渲染（调用方也可直接走 render）
        None => {
            for ty in 0..tiles_y as usize {
                for tx in 0..tiles_x as usize {
                    clear_tile(binned, img, tx, ty);
                    shade_tile(binned, img, tx, ty);
                }
            }
            return;
        }
    };
    // 钳到画布 tile 范围（damage 的 tiles() 已按 damage 自己的画布钳过，
    // 但 binned 的画布可能不同，故此处再钳一次）
    for ty in ty0.min(tiles_y)..ty1.min(tiles_y) {
        for tx in tx0.min(tiles_x)..tx1.min(tiles_x) {
            let (tx, ty) = (tx as usize, ty as usize);
            // 先清零再着色：脏 tile 必须从零重算，否则"由有内容变无内容"会残留
            clear_tile(binned, img, tx, ty);
            shade_tile(binned, img, tx, ty);
        }
    }
}
