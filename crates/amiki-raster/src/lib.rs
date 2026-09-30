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
use amiki_scene::{PathEl, Paint, Scene};

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
    pub colors: Vec<[f32; 4]>,
    /// 每 draw 的裁剪层索引（≤ MAX_DRAW_CLIPS，未用槽位为 0）。
    pub draw_clips: Vec<[u32; 4]>,
    /// 每 scene clip 的 CPU 覆盖率 alpha 图（GPU 侧为纹理，M2d）。
    pub clip_maps: Vec<Vec<u8>>,
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
        let segs = flatten_draw(&draw.transform, &draw.path.els, flatten_eps);
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
        let color = match draw.paint {
            Paint::Solid(c) => [c.r, c.g, c.b, c.a],
        };
        out.colors.push(color);
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
                    let c = binned.colors[m[0] as usize];
                    col = composite(col, c, c[3] * a, m[3] & 0xFF);
                }
            }
            let o = (py * w + px) * 4;
            for k in 0..4 {
                img[o + k] = (col[k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// 从零渲染，但**只清/着色指定的 tile 列表**（`None` = 全部 tile）。
///
/// M4 测试用：全量 `render` 与"只重算部分 tile"之间需要一条**可控**的中间路径，
/// 以便区分"damage 覆盖不全"与"着色本身不一致"这两类失败。
/// 注意 `clear_tile` 先行使 `None`（全部 tile）与本函数的增量形式语义一致。
pub fn render_tiles(binned: &BinnedScene, tiles: Option<&[(u32, u32)]>) -> Vec<u8> {
    assert_eq!(
        TILE,
        amiki_scene::DAMAGE_TILE,
        "分箱 tile 尺寸与 damage 的 tile 尺寸必须一致，否则增量渲染会与全量不一致"
    );
    let w = binned.width as usize;
    let h = binned.height as usize;
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
pub fn render(binned: &BinnedScene) -> Vec<u8> {
    render_tiles(binned, None)
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
