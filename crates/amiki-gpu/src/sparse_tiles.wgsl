// M3b（2/5）：逐 tile 桶表 —— 「活跃 tile 列表 + 偏移」，fine 只遍历真正覆盖本 tile 的 draw。
//
// 桶 = tile（tile 行 × 列），条目 = 覆盖该 tile 的 strip（即 draw），按 d 递增 = 画家序。
// tile_count → 扫描 → tile_start → tile_fill（顺序写入，无原子、确定序）。
//
// 槽位：
//   1 = strip_meta（lo/hi/y0/y1）
//   2 = tr_offsets（每 tile 行的 [start, count]）
//   3 = tr_list
//   4 = tile_count（→ 后再扫描出 tile_start）
//   5 = stats（[2] += 条目总数）
//   6 = strip_cell_off（条带单元基址，用于预计算单元索引基）
//   7 = tile_start
//   8 = tile_list（vec4：base'(i32 位模式), w, d, row_lo | row_hi<<16）

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
@group(0) @binding(1) var<storage, read_write> strip_meta: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tr_offsets: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> tr_list: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_cnt: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> strip_cell_off: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_start: array<u32>;
@group(0) @binding(8) var<storage, read_write> tile_list: array<vec4<u32>>;

// 线性 2D 索引：宿主按 `(1024, ⌈n/65536⌉)` 派发（见 `sparse.rs` 的 `dispatch_1d`）。
// 行距 = 65536 元素 = 1024 工作组 × 64 线程，**必须与宿主一致**（理由见该函数）。
fn lin(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65536u;
}

@compute @workgroup_size(64)
fn tile_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile = lin(gid);
    if (tile >= bp.tiles_y * bp.tiles_x) {
        return;
    }
    let tr = tile / bp.tiles_x;
    let c = tile % bp.tiles_x;
    let off = tr_offsets[tr];
    var cnt = 0u;
    for (var k = 0u; k < off.y; k = k + 1u) {
        let m = strip_meta[tr_list[off.x + k]];
        if (m.x <= c && c <= m.y) {
            cnt = cnt + 1u;
        }
    }
    tile_cnt[tile] = cnt;
    atomicAdd(&stats[2], cnt);
}

@compute @workgroup_size(64)
fn tile_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile = lin(gid);
    if (tile >= bp.tiles_y * bp.tiles_x) {
        return;
    }
    let tr = tile / bp.tiles_x;
    let c = tile % bp.tiles_x;
    let off = tr_offsets[tr];
    var pos = tile_start[tile];
    for (var k = 0u; k < off.y; k = k + 1u) {
        let si = tr_list[off.x + k];
        let m = strip_meta[si];
        if (m.x <= c && c <= m.y) {
            let d = si % bp.n_draws;
            let w = m.y - m.x + 1u;
            // cell = base' + row * w + col：把 (y0, lo) 偏移折进基址（i32 位模式存放）
            let base = i32(strip_cell_off[si]) - i32(m.z) * i32(w) - i32(m.x);
            tile_list[pos] = vec4<u32>(bitcast<u32>(base), w, d, m.z | (m.w << 16u));
            pos = pos + 1u;
        }
    }
}