// M3b（1/5）：稀疏分箱 —— strip 汇总 → tile 行活跃列表（d 递增 = 画家序）。
//
// 全部输入是 M3a 的 strip_meta（转置稠密：si = tr * n_draws + d，
// 每 strip 4 个 u32 = lo / hi / y0 / y1，lo = tiles_x 表示空）。
// 三个 kernel 共用一套绑定槽位（每模块 binding 唯一）：
//   1 = strip_meta（4 u32/strip，以 vec4 视图索引）
//   2 = strip_cell_off   （先存单元计数 → 扫描后变条带单元基址）
//   3 = strip_bucket_off （先存桶计数   → 扫描后变条带桶基址）
//   4 = tr_prefix        （活跃标记 → 扫描后 exclusive 前缀）
//   5 = stats            （[0]=n_cells [1]=n_buckets [2]=n_tile_entries [3]=n_items [4]=活跃 strip 数）
//   6 = tr_list          （每 tile 行的活跃 strip，d 递增）
//   7 = tr_offsets       （每 tile 行：vec2(start, count)）

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
@group(0) @binding(2) var<storage, read_write> strip_cell_off: array<u32>;
@group(0) @binding(3) var<storage, read_write> strip_bucket_off: array<u32>;
@group(0) @binding(4) var<storage, read_write> tr_prefix: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> tr_list: array<u32>;
@group(0) @binding(7) var<storage, read_write> tr_offsets: array<vec2<u32>>;

// 线性 2D 索引：宿主按 `(1024, ⌈n/65536⌉)` 派发（见 `sparse.rs` 的 `dispatch_1d`）。
// 行距 = 65536 元素 = 1024 工作组 × 64 线程，**必须与宿主一致**（理由见该函数）。
fn lin(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * 65536u;
}

// 活跃 strip 的单元/桶计数（随后被前缀和扫描替换为基址）
@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let si = lin(gid);
    if (si >= bp.tiles_y * bp.n_draws) {
        return;
    }
    let m = strip_meta[si];
    var cells = 0u;
    var buckets = 0u;
    var is_active = 0u;
    if (m.x != bp.tiles_x) { // 非空 strip
        let w = m.y - m.x + 1u;
        let h = m.w - m.z + 1u;
        cells = w * h;
        buckets = (w + 1u) * h;
        is_active = 1u;
        atomicAdd(&stats[0], cells);
        atomicAdd(&stats[1], buckets);
        atomicAdd(&stats[4], 1u);
    }
    strip_cell_off[si] = cells;
    strip_bucket_off[si] = buckets;
    tr_prefix[si] = is_active;
}

// tr_prefix 扫描后：把活跃 strip 按布局序（tr 主、d 次）收集进 tr_list
@compute @workgroup_size(64)
fn tr_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let si = lin(gid);
    if (si >= bp.tiles_y * bp.n_draws) {
        return;
    }
    if (strip_meta[si].x != bp.tiles_x) {
        tr_list[tr_prefix[si]] = si; // exclusive 前缀 ⇒ 位置唯一、顺序即 d 递增
    }
}

// 每 tile 行的 [start, count]（靠前缀和在 n_draws 边界处的取值界定）
@compute @workgroup_size(64)
fn tr_offsets_fn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tr = lin(gid);
    if (tr >= bp.tiles_y) {
        return;
    }
    let n = bp.tiles_y * bp.n_draws;
    let first = tr * bp.n_draws;
    let last = min(first + bp.n_draws, n) - 1u;
    let start = tr_prefix[first];
    var end = tr_prefix[last];
    if (strip_meta[last].x != bp.tiles_x) {
        end = end + 1u;
    }
    tr_offsets[tr] = vec2<u32>(start, end - start);
}