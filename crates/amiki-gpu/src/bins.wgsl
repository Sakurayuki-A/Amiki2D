// M2f：GPU 分箱 —— emit（桶计数 + carry 定点项）/ clear / scatter / carry。
//
// 稠密网格布局（M3 sparse strips 稀疏化的对象）：
//   b = ((tr * tiles_x + col) * n_draws + draw) * 16 + row_local
// carry 项用 i32 定点（×65536）原子加（WGSL 无 f32 原子）。
// carry 桶按 x_max 移位 +1：floor(x_max/16)+1，钳到 [0, tiles_x]；
// 桶 tiles_x（画布右侧之外）被 carry 循环排除 —— 画布外线段零贡献。

struct BinsParams {
    tiles_x: u32,
    n_draws: u32,
    tiles_y: u32,
    height: u32,
    total_segments: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> bp: BinsParams;
@group(0) @binding(1) var<storage, read> seg_ends: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> seg_draws: array<u32>;

@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;   // (tile,draw,row) 成员计数
@group(0) @binding(4) var<storage, read_write> grid_aux: array<atomic<u32>>;   // [0,n_grid)=scan_base  [n_grid,2n_grid)=fill
@group(0) @binding(5) var<storage, read_write> bucket_sums: array<atomic<i32>>;// (draw,tile-row,row) 定点 carry 项
@group(0) @binding(6) var<storage, read_write> carries: array<f32>;            // 稠密，与 counters 同布局
@group(0) @binding(7) var<storage, read_write> item_total: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> seg_refs: array<u32>;

const TILE: f32 = 16.0;

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nc = arrayLength(&counters);
    if (gid.x < nc) {
        atomicStore(&counters[gid.x], 0u);
        atomicStore(&grid_aux[gid.x], 0u);              // scan_base 区
        atomicStore(&grid_aux[gid.x + nc], 0u);         // fill 区
    }
    let nb = arrayLength(&bucket_sums);
    if (gid.x < nb) {
        atomicStore(&bucket_sums[gid.x], 0);
    }
}

// 行遍历公共体：返回该行带的 (x_min, x_max, signed_len)；空行带返回 w 分量 < 0 的哨兵
struct RowExtent {
    x_min: f32,
    x_max: f32,
    signed_len: f32,
    valid: u32,
}

fn row_extent(seg: vec4<f32>, j: i32) -> RowExtent {
    let dy = seg.w - seg.y;
    var out: RowExtent;
    out.valid = 0u;
    out.x_min = 0.0;
    out.x_max = 0.0;
    out.signed_len = 0.0;
    if (abs(dy) < 1e-9) {
        return out;
    }
    var sy0 = seg.y;
    var sy1 = seg.w;
    if (seg.y > seg.w) {
        sy0 = seg.w;
        sy1 = seg.y;
    }
    let band_lo = f32(j);
    let ya = max(sy0, band_lo);
    let yb = min(sy1, band_lo + 1.0);
    if (ya >= yb) {
        return out;
    }
    let dxdy = (seg.z - seg.x) / dy;
    let xa = seg.x + dxdy * (ya - seg.y);
    let xb = seg.x + dxdy * (yb - seg.y);
    out.x_min = min(xa, xb);
    out.x_max = max(xa, xb);
    out.signed_len = select(-1.0, 1.0, dy > 0.0) * (yb - ya);
    out.valid = 1u;
    return out;
}

// emit + scatter 共用的行/tile 遍历
@compute @workgroup_size(64)
fn emit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bp.total_segments) {
        return;
    }
    let seg = seg_ends[i];
    let d = seg_draws[i];
    let j_lo = i32(max(floor(min(seg.y, seg.w) - 1e-3), 0.0));
    let j_hi = i32(min(floor(max(seg.y, seg.w) + 1e-3), f32(bp.height) - 1.0));
    var j = j_lo;
    loop {
        if (j > j_hi) {
            break;
        }
        let ext = row_extent(seg, j);
        if (ext.valid == 1u) {
            let tr = u32(j) / 16u;
            let row_local = u32(j) % 16u;
            let t_lo = i32(max(floor(ext.x_min / TILE), 0.0));
            let t_hi = i32(min(floor(ext.x_max / TILE), f32(bp.tiles_x) - 1.0));
            var t = t_lo;
            loop {
                if (t > t_hi) {
                    break;
                }
                let b = ((tr * bp.tiles_x + u32(t)) * bp.n_draws + d) * 16u + row_local;
                atomicAdd(&counters[b], 1u);
                atomicAdd(&item_total[0], 1u);
                t = t + 1;
            }
            // carry 桶（移位 +1，钳到 [0, tiles_x]；画布右外 → 桶 tiles_x，被排除）
            let bucket = min(max(i32(floor(ext.x_max / TILE)) + 1, 0), i32(bp.tiles_x));
            let bi = ((d * bp.tiles_y + tr) * 16u + row_local) * (bp.tiles_x + 1u) + u32(bucket);
            atomicAdd(&bucket_sums[bi], i32(ext.signed_len * 65536.0));
        }
        j = j + 1;
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bp.total_segments) {
        return;
    }
    let seg = seg_ends[i];
    let d = seg_draws[i];
    let j_lo = i32(max(floor(min(seg.y, seg.w) - 1e-3), 0.0));
    let j_hi = i32(min(floor(max(seg.y, seg.w) + 1e-3), f32(bp.height) - 1.0));
    var j = j_lo;
    loop {
        if (j > j_hi) {
            break;
        }
        let ext = row_extent(seg, j);
        if (ext.valid == 1u) {
            let tr = u32(j) / 16u;
            let row_local = u32(j) % 16u;
            let t_lo = i32(max(floor(ext.x_min / TILE), 0.0));
            let t_hi = i32(min(floor(ext.x_max / TILE), f32(bp.tiles_x) - 1.0));
            var t = t_lo;
            loop {
                if (t > t_hi) {
                    break;
                }
                let b = ((tr * bp.tiles_x + u32(t)) * bp.n_draws + d) * 16u + row_local;
                let pos = grid_aux[b] + atomicAdd(&grid_aux[bp.tiles_y * bp.tiles_x * bp.n_draws * 16u + b], 1u);
                seg_refs[pos] = i;
                t = t + 1;
            }
        }
        j = j + 1;
    }
}

// 每 (draw, row) 一线程：沿桶轴顺序累计 → carry = −(含当前桶的定点前缀)
@compute @workgroup_size(64)
fn carry(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= bp.n_draws * bp.tiles_y * 16u) {
        return;
    }
    let d = idx / (bp.tiles_y * 16u);
    let rem = idx % (bp.tiles_y * 16u);
    let tr = rem / 16u;
    let row = rem % 16u;
    var running: i32 = 0;
    for (var col = 0u; col < bp.tiles_x; col++) {
        running = running + atomicLoad(&bucket_sums[((d * bp.tiles_y + tr) * 16u + row) * (bp.tiles_x + 1u) + col]);
        let b = ((tr * bp.tiles_x + col) * bp.n_draws + d) * 16u + row;
        carries[b] = -f32(running) / 65536.0;
    }
}
