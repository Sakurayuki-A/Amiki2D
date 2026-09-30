// M2f：通用 u32 exclusive 前缀和（三级：块内 Hillis-Steele / 块和扫描 / 偏移回加）。
//
// data[i] ← Σ_{j<i} data[j]（exclusive）。支持任意 n：
//   K1 scan_blocks   每 256 元素一块，块内扫描，块和写入 block_sums
//   K2 scan_sums     单工作组顺序分块扫描 block_sums → block_offsets（exclusive）
//   K3 add_offsets   data[i] += block_offsets[i/256]

struct ScanParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> sp: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_offsets: array<u32>;

var<workgroup> tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_blocks(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = wid.x * 256u + lid.x;
    var v = 0u;
    if (i < sp.n) {
        v = data[i];
    }
    // Hillis-Steele inclusive 扫描（双 barrier 防读写竞争）
    tmp[lid.x] = v;
    workgroupBarrier();
    var offset = 1u;
    loop {
        if (offset >= 256u) {
            break;
        }
        var add = 0u;
        if (lid.x >= offset) {
            add = tmp[lid.x - offset];
        }
        workgroupBarrier();
        tmp[lid.x] = tmp[lid.x] + add;
        workgroupBarrier();
        offset = offset * 2u;
    }
    let inclusive = tmp[lid.x];
    workgroupBarrier();
    tmp[lid.x] = inclusive - v; // exclusive
    workgroupBarrier();
    if (i < sp.n) {
        data[i] = tmp[lid.x];
    }
    if (lid.x == 255u) {
        block_sums[wid.x] = inclusive; // 块总和 = 末元素 inclusive
    }
}

@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = arrayLength(&block_sums);
    let chunk = (n + 255u) / 256u;
    let begin = lid.x * chunk;
    var running = 0u;
    for (var i = begin; i < begin + chunk; i++) {
        if (i < n) {
            block_offsets[i] = running;
            running = running + block_sums[i];
        }
    }
    tmp[lid.x] = running;
    workgroupBarrier();
    var offset = 1u;
    loop {
        if (offset >= 256u) {
            break;
        }
        var add = 0u;
        if (lid.x >= offset) {
            add = tmp[lid.x - offset];
        }
        workgroupBarrier();
        tmp[lid.x] = tmp[lid.x] + add;
        workgroupBarrier();
        offset = offset * 2u;
    }
    // 本线程 chunk 的起始偏移 = exclusive 前缀
    let thread_off = tmp[lid.x] - running;
    for (var i = begin; i < begin + chunk; i++) {
        if (i < n) {
            block_offsets[i] = block_offsets[i] + thread_off;
        }
    }
}

@compute @workgroup_size(256)
fn add_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < sp.n) {
        data[gid.x] = data[gid.x] + block_offsets[gid.x / 256u];
    }
}
