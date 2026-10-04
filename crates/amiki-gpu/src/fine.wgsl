// Amiki2D fine rasterizer —— 解析式覆盖率计算着色器（稠密网格入口）。
//
// 每 workgroup 一 tile、每线程一像素；逐 draw 稠密迭代（M3b 起主线走
// sparse_fine.wgsl 的稀疏桶表；本入口保留为对拍基准与诊断路径）。
// 数学部分在 fine_math.wgsl（宿主拼接），与 amiki-raster 逐式对应。
//
// 0.6.1：派发改成 **1D 脏 tile 表**（见 fine_math.wgsl 的 `tile_hit`）——
// 要写哪些 tile 由 `dirty_tiles` 查表决定，不再从 `global_invocation_id`
// 反推（那条路隐含"从 (0,0) 起步"，会漏掉偏移）。

struct FineParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    n_draws: u32,
    /// 本次派发要重算的 tile 数（脏 tile 表长度）；末尾批次的尾巴靠它退出
    n_tiles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: FineParams;
@group(0) @binding(1) var<storage, read> seg_ends: array<vec4<f32>>;   // (x0,y0,x1,y1)
@group(0) @binding(2) var<storage, read> scan_base: array<u32>;        // 桶起点（exclusive 前缀）
@group(0) @binding(3) var<storage, read> counters: array<u32>;         // 桶计数
@group(0) @binding(4) var<storage, read> carries: array<f32>;          // 每桶标量进位
@group(0) @binding(5) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> draw_flags: array<u32>;       // blend | clip_count<<8
@group(0) @binding(7) var<storage, read> draw_clips: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> seg_refs: array<u32>;         // 桶 → 段索引
// **rgba16float 而不是 rgba8unorm**：HDR 要求 >1.0 的颜色能存下来。
// 8 位格式在写入时就裁到 1.0，高光源会在这一步直接消失。
// 降到显示范围是最后一步（主机侧回读 + 量化）的事。
@group(0) @binding(9) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var clip_tex0: texture_2d<f32>;
@group(0) @binding(11) var clip_tex1: texture_2d<f32>;
@group(0) @binding(12) var clip_tex2: texture_2d<f32>;
@group(0) @binding(13) var clip_tex3: texture_2d<f32>;
/// 0.6.1：要重算的 tile 表（`tx | ty << 16`）
@group(0) @binding(18) var<storage, read> dirty_tiles: array<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = tile_index_of(gid);
    if (ti >= params.n_tiles) {
        return; // z 维分批的尾巴
    }
    let h = tile_hit(dirty_tiles[ti], gid);
    if (h.px >= params.width || h.py >= params.height) {
        return; // 画布边缘的 tile 有一部分落在画布外
    }
    let row = h.py % 16u;
    let center = vec2<f32>(f32(h.px) + 0.5, f32(h.py) + 0.5);
    let tile_base = (h.tile_y * params.tiles_x + h.tile_x) * params.n_draws;
    let pix = vec2<u32>(h.px, h.py);

    var col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var d = 0u; d < params.n_draws; d++) {
        let b = (tile_base + d) * 16u + row;
        let start = scan_base[b];
        let count = counters[b];
        var area = carries[b];
        for (var k = 0u; k < count; k++) {
            let seg = seg_ends[seg_refs[start + k]];
            area += seg_area(seg.xy, seg.zw, center);
        }
        var a = min(abs(area), 1.0);
        let flags = draw_flags[d];
        let nclip = (flags >> 8u) & 0xFu;
        if (nclip > 0u && a > 0.0) {
            let ci = draw_clips[d];
            var cov = 1.0;
            cov *= clip_sample(ci.x, pix, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
            if (nclip > 1u) {
                cov *= clip_sample(ci.y, pix, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
            }
            if (nclip > 2u) {
                cov *= clip_sample(ci.z, pix, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
            }
            if (nclip > 3u) {
                cov *= clip_sample(ci.w, pix, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
            }
            a *= cov;
        }
        if (a > 0.0) {
            col = composite(col, colors[d], colors[d].a * a, flags & 0xFFu);
        }
    }
    textureStore(out_tex, pix, col);
}

/// 覆盖率入口：单 draw（clip 路径）解析覆盖率 → R 通道。
@compute @workgroup_size(16, 16, 1)
fn main_coverage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = tile_index_of(gid);
    if (ti >= params.n_tiles) {
        return;
    }
    let h = tile_hit(dirty_tiles[ti], gid);
    if (h.px >= params.width || h.py >= params.height) {
        return;
    }
    let row = h.py % 16u;
    let center = vec2<f32>(f32(h.px) + 0.5, f32(h.py) + 0.5);
    let tile_base = (h.tile_y * params.tiles_x + h.tile_x) * params.n_draws;

    var a = 0.0;
    for (var d = 0u; d < params.n_draws; d++) {
        let b = (tile_base + d) * 16u + row;
        let start = scan_base[b];
        let count = counters[b];
        var area = carries[b];
        for (var k = 0u; k < count; k++) {
            let seg = seg_ends[seg_refs[start + k]];
            area += seg_area(seg.xy, seg.zw, center);
        }
        a = max(a, min(abs(area), 1.0));
    }
    textureStore(out_tex, vec2<u32>(h.px, h.py), vec4<f32>(a, 0.0, 0.0, 1.0));
}