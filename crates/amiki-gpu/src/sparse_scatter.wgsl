// M3b（4/5）：段 scatter —— 把段端点直接写入单元列表（cell_segs），fine 无需全局段池引用。
//
// 遍历条件与 sparse_emit.wgsl 的 emit 完全一致（同一 (行, 列) 判定），
// 故每个单元的段列表 = 该单元成员计数的精确对应。
//
// 槽位：
//   1 = seg_ends       2 = seg_draws      3 = strip_meta
//   4 = strip_cell_off 5 = cell_csr_off   6 = cell_aux（游标半区）
//   7 = cell_segs（vec4 段端点，按 cell_csr_off 基址顺序排列）

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
@group(0) @binding(5) var<storage, read_write> cell_csr_off: array<u32>;
@group(0) @binding(6) var<storage, read_write> cell_aux: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> cell_segs: array<vec4<f32>>;

const TILE: f32 = 16.0;

// 线性 2D 索引：宿主以 (≤65535, ⌈n/65535⌉) 派发
fn lin(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65535u;
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
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
            continue;
        }
        if (i32(row) < i32(m.z) || i32(row) > i32(m.w)) {
            continue;
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
            let pos = cell_csr_off[cell] + atomicAdd(&cell_aux[sp.n_cells + cell], 1u);
            cell_segs[pos] = seg;
            t = t + 1;
        }
    }
}