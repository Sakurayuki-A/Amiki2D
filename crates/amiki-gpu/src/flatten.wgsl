// M2f：GPU 压平（统一版）——count → scan(外部通用前缀和) → prep → flatten。
//
// 输入合并为单一 vec4 数组（段内偏移解析）以控制在存储缓冲上限内：
//   元素布局：[quad 2×vec4 × nq][cubic 2×vec4 × nc][line 1×vec4 × nl]
// 段全局索引空间：曲线段 [0, total_curve_segs) + 直线段 [total, total+n_lines)。

struct FlattenParams {
    n_quads: u32,
    n_cubics: u32,
    n_lines: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> fp: FlattenParams;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;      // 合并控制点/直线
@group(0) @binding(2) var<storage, read> entry_draws: array<u32>;     // 每入口的 draw id
@group(0) @binding(3) var<storage, read> counts: array<u32>;          // CPU 闭式段数（上传）
@group(0) @binding(4) var<storage, read> seg_base: array<u32>;        // counts 的 exclusive 前缀（scan 输出）
@group(0) @binding(5) var<storage, read_write> seg_ends: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> seg_draws: array<u32>;
@group(0) @binding(7) var<storage, read_write> indirect: array<u32>;

// 入口 i 的数据元素起点（元素 = vec4）
fn elem_base(i: u32) -> u32 {
    let nq = fp.n_quads;
    if (i < nq) {
        return i * 2u;
    }
    if (i < nq + fp.n_cubics) {
        return nq * 2u + (i - nq) * 2u;
    }
    return (nq + fp.n_cubics) * 2u + (i - nq - fp.n_cubics);
}

@compute @workgroup_size(64)
fn flatten(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < fp.n_quads) {
        let n = counts[i];
        let base = seg_base[i];
        let e = i * 2u;
        let p0 = data[e].xy;
        let p1 = data[e].zw;
        let p2 = data[e + 1u].xy;
        for (var k = 0u; k < n; k++) {
            let t0 = f32(k) / f32(n);
            let t1 = f32(k + 1u) / f32(n);
            let u0 = 1.0 - t0;
            let u1 = 1.0 - t1;
            let a = u0 * u0 * p0 + 2.0 * u0 * t0 * p1 + t0 * t0 * p2;
            let b = u1 * u1 * p0 + 2.0 * u1 * t1 * p1 + t1 * t1 * p2;
            seg_ends[base + k] = vec4<f32>(a, b);
            seg_draws[base + k] = entry_draws[i];
        }
    } else if (i < fp.n_quads + fp.n_cubics) {
        let n = counts[i];
        let base = seg_base[i];
        let e = elem_base(i);
        let p0 = data[e].xy;
        let p1 = data[e].zw;
        let p2 = data[e + 1u].xy;
        let p3 = data[e + 1u].zw;
        for (var k = 0u; k < n; k++) {
            let t0 = f32(k) / f32(n);
            let t1 = f32(k + 1u) / f32(n);
            let u0 = 1.0 - t0;
            let u1 = 1.0 - t1;
            let a = u0 * u0 * u0 * p0 + 3.0 * u0 * u0 * t0 * p1 + 3.0 * u0 * t0 * t0 * p2 + t0 * t0 * t0 * p3;
            let b = u1 * u1 * u1 * p0 + 3.0 * u1 * u1 * t1 * p1 + 3.0 * u1 * t1 * t1 * p2 + t1 * t1 * t1 * p3;
            seg_ends[base + k] = vec4<f32>(a, b);
            seg_draws[base + k] = entry_draws[i];
        }
    } else if (i < fp.n_quads + fp.n_cubics + fp.n_lines) {
        let base = seg_base[i];
        seg_ends[base] = data[elem_base(i)];
        seg_draws[base] = entry_draws[i];
    }
}
