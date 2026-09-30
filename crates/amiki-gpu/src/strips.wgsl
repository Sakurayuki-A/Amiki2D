// M3a：活跃 tile 提取（sparse strips）——draw × tile 行 的归并区间。
//
// 每个 (draw, tile 行 tr) 输出该 draw 在 tr 内全部内容（成员段 ∪ carry 影响）
// 覆盖的 tile 列区间 [lo, hi] 与 tile 内局部行区间 [y0, y1]。
//
// 语义必须与稠密管线（emit / carry）**逐式一致**，否则稀疏化会丢内容：
//   · 成员 tile 范围 = [max(⌊x_min/T⌋,0), min(⌊x_max/T⌋, tiles_x−1)]；
//     越出右缘的段不钳入末列而是整体丢弃（emit 的 t_lo 没有上钳）。
//   · carry 桶 = clamp(⌊x_max/T⌋+1, 0, tiles_x)，前缀只覆盖 col < tiles_x。
//     于是：carry[t] = −Σ{s·len : x_max < t·T}（桶 0 吸收了画布左外段；
//     画布右外段落入桶 tiles_x 被排除，永不与左侧抵消）。
//   ⇒ carry 支撑 ⊆ [行内成员最小 tile, min(⌊x_max/T⌋, tiles_x−1)]：
//     左外段 → 左界 0；右外段 → 右界 tiles_x−1（即使它自身无成员）。
//   ⇒ 区间需含这些越界段的行：一行可能只有「左外 + 右外」两段而 carry≠0。
// 行区间 = [⌊ys0⌋, ⌈ys1⌉−1] ∩ 带（与 emit 的 ya<yb 有效行判定精确一致）。
// x(y) 在带内线性 ⇒ 带内各行的 x 极值必在带端点 x 之间（无保守扩张）。
//
// 输出布局：**转置稠密的交错元数据** ——
//   si = tr * n_draws + d，strip_meta[si * 4 + {0,1,2,3}] = (lo, hi, y0, y1)
// 转置的意义（M3b）：同一 tile 行的活跃 strip 落在连续区间，一次全局前缀和即可
// 构建「tile 行 → 活跃 strip 列表」（d 递增 = 画家序）。
// 判空哨兵：lo = tiles_x（读侧以 vec4<u32> 视图索引）。

struct StripParams {
    tiles_x: u32,
    n_draws: u32,
    tiles_y: u32,
    height: u32,
    total_segments: u32,
    n_cells: u32,
    n_buckets: u32,
    n_tile_entries: u32,
}

@group(0) @binding(0) var<uniform> sp: StripParams;
@group(0) @binding(1) var<storage, read> seg_ends: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> seg_draws: array<u32>;
@group(0) @binding(3) var<storage, read_write> strip_meta: array<atomic<u32>>; // 4/strip

const TILE: f32 = 16.0;

@compute @workgroup_size(64)
fn strip_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let si = gid.x;
    if (si >= sp.tiles_y * sp.n_draws) {
        return;
    }
    atomicStore(&strip_meta[si * 4u], sp.tiles_x);  // lo 哨兵
    atomicStore(&strip_meta[si * 4u + 1u], 0u);     // hi
    atomicStore(&strip_meta[si * 4u + 2u], 0xFFFFu); // y0 哨兵
    atomicStore(&strip_meta[si * 4u + 3u], 0u);     // y1
}

@compute @workgroup_size(64)
fn strip_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sp.total_segments) {
        return;
    }
    let seg = seg_ends[i];
    let dy = seg.w - seg.y;
    if (abs(dy) < 1e-9) {
        return; // 水平段：无成员、无 carry
    }
    let ys0 = min(seg.y, seg.w);
    let ys1 = max(seg.y, seg.w);
    // 有效行（与 emit 的 ya<yb 判定精确一致）：⌊ys0⌋ … ⌈ys1⌉−1
    let j_lo = i32(clamp(floor(ys0), 0.0, f32(sp.height) - 1.0));
    let j_hi = i32(clamp(ceil(ys1) - 1.0, 0.0, f32(sp.height) - 1.0));
    if (j_hi < j_lo) {
        return;
    }
    let d = seg_draws[i];
    let dxdy = (seg.z - seg.x) / dy;
    let tr_lo = u32(j_lo) / 16u;
    let tr_hi = u32(j_hi) / 16u;
    for (var tr = tr_lo; tr <= tr_hi; tr = tr + 1u) {
        let band_lo = f32(tr * 16u);
        let band_hi = band_lo + 16.0;
        let ya = max(ys0, band_lo);
        let yb = min(ys1, band_hi);
        if (ya >= yb) {
            continue; // 该带与线段无交集（与 emit 的有效行判定一致，避免幻影 strip）
        }
        // 带端点（线性插值 ⇒ 带内各行的 x 极值必在其间）
        let xa = seg.x + dxdy * (ya - seg.y);
        let xb = seg.x + dxdy * (yb - seg.y);
        let xmin = min(xa, xb);
        let xmax = max(xa, xb);
        let tl = i32(max(floor(xmin / TILE), 0.0));
        let th = i32(min(floor(xmax / TILE), f32(sp.tiles_x - 1u)));
        let si = tr * sp.n_draws + d;
        let base = si * 4u;
        if (tl < i32(sp.tiles_x) && th >= 0) {
            // 有画布内成员：与 emit 的成员 tile 范围一致
            atomicMin(&strip_meta[base], u32(tl));
            atomicMax(&strip_meta[base + 1u], u32(th));
        } else if (th >= 0) {
            // 全在画布右外：其桶被钳到 tiles_x 排除在前缀外 ⇒ 右界须撑到 tiles_x−1
            atomicMax(&strip_meta[base + 1u], u32(th));
        } else {
            // 全在画布左外：桶钳到 0 ⇒ carry 可能自 tile 0 起
            atomicMin(&strip_meta[base], 0u);
        }
        let r0 = u32(max(f32(j_lo), band_lo) - band_lo);      // tile 内局部行
        let r1 = u32(min(f32(j_hi), band_hi - 1.0) - band_lo);
        atomicMin(&strip_meta[base + 2u], r0);
        atomicMax(&strip_meta[base + 3u], r1);
    }
}