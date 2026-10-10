// M3b（3/5）：稀疏单元计数 + carry 定点桶（与稠密 emit 逐式一致）。
//
// 单元 cell = (strip, 行, 列)，条带内布局 (row - y0) * w + (col - lo)。
// 成员计数与桶贡献的判定条件、行/列范围、定点换算与 bins.wgsl 的 emit 完全相同，
// 因此 carry 前缀值与稠密路径逐位一致（M3c 逐字节对拍）。
//
// 槽位：
//   1 = seg_ends        2 = seg_draws       3 = strip_meta
//   4 = strip_cell_off  5 = strip_bucket_off
//   6 = cell_aux（[0,n_cells)=成员计数，[n_cells,2n)=scatter 游标）
//   7 = buckets（i32 定点，×65536）
//   8 = stats（[3] += 成员总数）

struct SparseParams {
    tiles_x: u32,
    n_draws: u32,
    tiles_y: u32,
    height: u32,
    total_segments: u32,
    n_cells: u32,
    n_buckets: u32,
    n_tile_entries: u32,
}

@group(0) @binding(0) var<uniform> sp: SparseParams;
@group(0) @binding(1) var<storage, read_write> seg_ends: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> seg_draws: array<u32>;
@group(0) @binding(3) var<storage, read_write> strip_meta: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> strip_cell_off: array<u32>;
@group(0) @binding(5) var<storage, read_write> strip_bucket_off: array<u32>;
@group(0) @binding(6) var<storage, read_write> cell_aux: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> buckets: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>>;

const TILE: f32 = 16.0;

// 线性 2D 索引：宿主按 `(1024, ⌈n/65536⌉)` 派发（见 `sparse.rs` 的 `dispatch_1d`）。
// 行距 = 65536 元素 = 1024 工作组 × 64 线程，**必须与宿主一致**（理由见该函数）。
fn lin(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65536u;
}

@compute @workgroup_size(64)
fn clear_sparse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = lin(gid);
    let nc = max(sp.n_cells, sp.n_buckets);
    if (i < nc) {
        if (i < sp.n_cells) {
            atomicStore(&cell_aux[i], 0u);              // 成员计数
            atomicStore(&cell_aux[sp.n_cells + i], 0u); // scatter 游标
        }
        if (i < sp.n_buckets) {
            atomicStore(&buckets[i], 0);
        }
    }
}

@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = lin(gid);
    if (i >= sp.total_segments) {
        return;
    }
    let seg = seg_ends[i];
    let dy = seg.w - seg.y;
    if (abs(dy) < 1e-9) {
        return;
    }
    let d = seg_draws[i];
    let ys0 = min(seg.y, seg.w);
    let ys1 = max(seg.y, seg.w);
    // 有效行 = [⌊ys0⌋, ⌈ys1⌉−1]（与稠密 emit 的 ya<yb 判定精确等价）
    let j_lo = i32(clamp(floor(ys0), 0.0, f32(sp.height) - 1.0));
    let j_hi = i32(clamp(ceil(ys1) - 1.0, 0.0, f32(sp.height) - 1.0));
    if (j_hi < j_lo) {
        return;
    }
    let dxdy = (seg.z - seg.x) / dy;
    for (var j = j_lo; j <= j_hi; j = j + 1) {
        let tr = u32(j) / 16u;
        let row = u32(j) % 16u;
        let si = tr * sp.n_draws + d;
        let m = strip_meta[si];
        if (m.x == sp.tiles_x) {
            continue; // 非空 strip（与 strips.wgsl 判空一致）
        }
        if (i32(row) < i32(m.z) || i32(row) > i32(m.w)) {
            continue; // 局部行落在条带行区间外（该行无成员也无桶）
        }
        let band_lo = f32(j);
        let ya = max(ys0, band_lo);
        let yb = min(ys1, band_lo + 1.0);
        if (ya >= yb) {
            continue;
        }
        let xa = seg.x + dxdy * (ya - seg.y);
        let xb = seg.x + dxdy * (yb - seg.y);
        let x_min = min(xa, xb);
        let x_max = max(xa, xb);
        let w = m.y - m.x + 1u;
        let cell_row = strip_cell_off[si] + (row - m.z) * w;
        let t_lo = i32(max(floor(x_min / TILE), 0.0));
        let t_hi = i32(min(floor(x_max / TILE), f32(sp.tiles_x) - 1.0));
        var t = t_lo;
        loop {
            if (t > t_hi) {
                break;
            }
            let cell = cell_row + (u32(t) - m.x);
            atomicAdd(&cell_aux[cell], 1u);
            atomicAdd(&stats[3], 1u);
            t = t + 1;
        }
        // carry 桶（与稠密一致：移位 +1、钳到 [0, tiles_x]；右外段落入尾桶被前缀排除）
        let s = select(-1.0, 1.0, dy > 0.0);
        let bucket = min(max(i32(floor(x_max / TILE)) + 1, 0), i32(sp.tiles_x));
        let kb = min(max(bucket, i32(m.x)), i32(m.y) + 1) - i32(m.x);
        let bi = strip_bucket_off[si] + (row - m.z) * (w + 1u) + u32(kb);
        atomicAdd(&buckets[bi], i32(s * (yb - ya) * 65536.0));
    }
}