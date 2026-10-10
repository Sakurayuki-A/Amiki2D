// M3b（5/5）：稀疏 carry —— 逐 (strip, 行) 顺序前缀，产出 cell_meta。
//
// cell_meta[cell] = (carry 的 f32 位模式, 段基址, 段数量, 保留)。
// carry 值与稠密 bins.wgsl 的 carry 内核逐位一致：同样的 i32 定点求和、
// 同样的 ×65536 换算，且条带左端 carry 恒为 0（见 strips.wgsl 的支撑性证明）。
// 段基址/数量来自单元 CSR 前缀（cell_csr_off，长度 n_cells+1）。
//
// 槽位：
//   1 = strip_meta      2 = strip_cell_off  3 = strip_bucket_off
//   4 = buckets（原子读）5 = cell_csr_off   6 = cell_meta

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
@group(0) @binding(1) var<storage, read_write> strip_meta: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> strip_cell_off: array<u32>;
@group(0) @binding(3) var<storage, read_write> strip_bucket_off: array<u32>;
@group(0) @binding(4) var<storage, read_write> buckets: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> cell_csr_off: array<u32>;
@group(0) @binding(6) var<storage, read_write> cell_meta: array<vec4<u32>>;

// 线性 2D 索引：宿主按 `(1024, ⌈n/65536⌉)` 派发（见 `sparse.rs` 的 `dispatch_1d`）。
//
// **行距 = 65536 个元素 = 1024 个工作组 × 64 线程**，与宿主的分批一一对应。
// 两侧步长必须一致：早先这里是 `65535`（元素），而宿主按 65535 **工作组**推进，
// 于是行与行重叠 95%，**尾部的元素永远没人处理** —— 4K/2000 矩形时正是最下面
// 两行瓦片整片空白（0.10.9 修掉的那个静默错误）。
fn lin(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65536u;
}

@compute @workgroup_size(64)
fn carry(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = lin(gid);
    if (idx >= sp.tiles_y * sp.n_draws * 16u) {
        return;
    }
    let si = idx / 16u;
    let row = idx % 16u;
    let m = strip_meta[si];
    if (m.x == sp.tiles_x) {
        return;
    }
    if (row < m.z || row > m.w) {
        return;
    }
    let w = m.y - m.x + 1u;
    let cell_row = strip_cell_off[si] + (row - m.z) * w;
    let bi = strip_bucket_off[si] + (row - m.z) * (w + 1u);
    var running: i32 = 0;
    for (var c = 0u; c < w; c = c + 1u) {
        running = running + atomicLoad(&buckets[bi + c]);
        let cell = cell_row + c;
        let base = cell_csr_off[cell];
        cell_meta[cell] = vec4<u32>(
            bitcast<u32>(-f32(running) / 65536.0),
            base,
            cell_csr_off[cell + 1u] - base,
            0u,
        );
    }
}