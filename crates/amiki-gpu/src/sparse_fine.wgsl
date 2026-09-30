// Amiki2D fine rasterizer —— 稀疏桶表入口（M3b 主线）。
//
// 每 workgroup 一 tile、每线程一像素；逐像素**只遍历覆盖本 tile 的 draw**
// （tile_list，按 d 递增 = 画家序），桶索引由 (strip, 行, 列) 直接算出，
// 不再扫描全 draw 网格 —— 显存与逐像素工作量都与画面实际内容成比例。
//
// 单元 meta = (carry 位模式, 段基址, 段数量, _)；段端点已 scatter 进 cell_segs，
// 因此无需全局段池引用。数学部分在 fine_math.wgsl（宿主拼接）。
//
// 槽位（均为只读 storage，与稠密入口共用 fine_bgl 的 1..=8 槽）：
//   1 = cell_segs（vec4<f32> 段端点）
//   2 = cell_meta（vec4<u32>：carry | seg_base | seg_count | _）
//   3 = colors
//   4 = draw_meta（vec4<u32>：flags | clips01 | clips23 | _）
//   5 = tile_count        6 = tile_start
//   7 = tile_list（vec4<u32>：base'(i32 位模式) | w | d | row_lo|row_hi<<16）
//   8 = （保留，绑垫片）

struct FineParams {
    width: u32,
    height: u32,
    tiles_x: u32,
    n_draws: u32,
}

@group(0) @binding(0) var<uniform> params: FineParams;
@group(0) @binding(1) var<storage, read> cell_segs: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cell_meta: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> draw_meta: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> tile_count: array<u32>;
@group(0) @binding(6) var<storage, read> tile_start: array<u32>;
@group(0) @binding(7) var<storage, read> tile_list: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> pad_unused: array<u32>;
@group(0) @binding(9) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var clip_tex0: texture_2d<f32>;
@group(0) @binding(11) var clip_tex1: texture_2d<f32>;
@group(0) @binding(12) var clip_tex2: texture_2d<f32>;
@group(0) @binding(13) var clip_tex3: texture_2d<f32>;

fn shade(gid: vec3<u32>, coverage_only: bool) -> vec4<f32> {
    let tile_x = gid.x / 16u;
    let tile = (gid.y / 16u) * params.tiles_x + tile_x;
    let row = gid.y % 16u;
    let center = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let start = tile_start[tile];
    let cnt = tile_count[tile];

    var col = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var cov_max = 0.0;
    for (var k = 0u; k < cnt; k = k + 1u) {
        let e = tile_list[start + k];
        let rows = e.w;
        if (row < (rows & 0xFFFFu) || row > (rows >> 16u)) {
            continue; // 该 draw 在本行带无内容
        }
        // cell = base' + row * w + tile_x（base' 已把 y0/w 与 lo 折入）
        let cell = u32(i32(e.x) + i32(row) * i32(e.y) + i32(tile_x));
        let cm = cell_meta[cell];
        let base = cm.y;
        let n = cm.z;
        var area = bitcast<f32>(cm.x);
        for (var i = 0u; i < n; i = i + 1u) {
            let seg = cell_segs[base + i];
            area += seg_area(seg.xy, seg.zw, center);
        }
        var a = min(abs(area), 1.0);
        let d = e.z;
        let dm = draw_meta[d];
        let flags = dm.x;
        let nclip = (flags >> 8u) & 0xFu;
        if (coverage_only) {
            cov_max = max(cov_max, a);
            continue;
        }
        if (nclip > 0u && a > 0.0) {
            a *= clip_mul(dm.y, nclip, dm, gid.xy, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
        }
        if (a > 0.0) {
            let c = colors[d];
            col = composite(col, c, c.a * a, flags & 0xFFu);
        }
    }
    if (coverage_only) {
        return vec4<f32>(cov_max, 0.0, 0.0, 1.0);
    }
    return col;
}

@compute @workgroup_size(16, 16, 1)
fn sparse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    textureStore(out_tex, vec2<u32>(gid.x, gid.y), shade(gid, false));
}

@compute @workgroup_size(16, 16, 1)
fn sparse_coverage(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    textureStore(out_tex, vec2<u32>(gid.x, gid.y), shade(gid, true));
}