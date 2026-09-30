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
//   4 = draw_meta（vec4<u32>：flags | clips01 | clips23 | grad_index）
//   5 = tile_count        6 = tile_start
//   7 = tile_list（vec4<u32>：base'(i32 位模式) | w | d | row_lo|row_hi<<16）
//   8 = grad_data（M6 渐变；定长条目，见下）
//
// M6 渐变数据布局（定长 = 4 + MAX_GRAD_STOPS*2 个 vec4<f32>，定长便于直接寻址）：
//   s+0 : (kind, repeat, n_stops, max_stops)     kind: 0=线性 1=径向 2=扫描
//   s+1 : (x0, y0, x1, y1)                       线性: from,to；径向: center,(radius,0)
//   s+2 + i*2     : (t, r, g, b)                 第 i 个色标
//   s+2 + i*2 + 1 : (_, _, _, _)                 占位，保持 vec4 对齐
// 空渐变（无渐变的场景）也必须绑定，故 buffer 至少要有 1 个条目。

const MAX_GRAD_STOPS: u32 = 8u;
// 一个渐变条目占用的 vec4 数：2 头部 + MAX_GRAD_STOPS*2 色标
const GRAD_STRIDE: u32 = 2u + MAX_GRAD_STOPS * 2u;

// M6：按渐变下标求某设备像素处的颜色。
//
// 布局见文件头。**索引与定义共用同一个 buffer**（槽位只有 8 一个）：
//   grad_data[0] = (n_grad, 0, 0, 0)                     头部
//   grad_data[1 + g*GRAD_STRIDE ...]                     第 g 个渐变
//
// 色标按 t 升序（宿主侧保证）。语义与 CPU 参考实现逐条对应：
//   1) 首个 t_i >= t 的区间 → 在 (t_{i-1}, c_{i-1}) 与 (t_i, c_i) 之间线性插值
//   2) t 落在所有色标之前 → 取首色
//   3) t 落在所有色标之后 → 取末色
// （这正是 clamp 语义，故这里不再单独做 clamp；repeat 已在前面取模。）
fn grad_color(g: u32, p: vec2<f32>) -> vec4<f32> {
    let s = 1u + g * GRAD_STRIDE;
    let h = grad_data[s];
    let kind = u32(h.x + 0.5);
    let rep = h.y > 0.5;
    let n = u32(h.z + 0.5);
    let geo = grad_data[s + 1u];

    var t = 0.0;
    if (kind == 0u) {
        // 线性：把 p 投影到 from→to 轴上
        let a = geo.xy;
        let d = geo.zw - a;
        let len2 = dot(d, d);
        if (len2 > 1e-12) {
            t = dot(p - a, d) / len2;
        }
    } else if (kind == 1u) {
        // 径向：到心的距离比半径
        t = length(p - geo.xy) / max(geo.z, 1e-9);
    } else {
        // 扫描：以 start 为起点的角度参数，归一到 [0,1)
        let v = p - geo.xy;
        t = (atan2(v.y, v.x) - geo.z) / 6.283185307179586;
        t = t - floor(t);
    }

    if (rep) {
        t = t - floor(t);
    } else {
        // **必须显式 clamp**：下面的区间查找从 i=1 开始，若 t 为负，
        // 第一次比较 t < stop[1].t 就会成立，于是以负的 k 在 stop[0] 与 stop[1]
        // 之间**外插**，得到比首色标更极端的颜色（曾因此与 CPU 差 19/255）。
        // 两色标场景恰好掩盖了这一点——首色标之外的颜色本来就等于首色标。
        t = clamp(t, 0.0, 1.0);
    }

    if (n == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // c_0 与区间查找并行推进，避免二次遍历
    let first = grad_data[s + 2u];
    var prev_t = first.x;
    var prev_c = first.yzw;
    for (var i = 1u; i < n; i = i + 1u) {
        let st = grad_data[s + 2u + i * 2u];
        if (t < st.x) {
            let span = st.x - prev_t;
            var k = 0.0;
            if (span > 1e-9) {
                k = (t - prev_t) / span;
            }
            return vec4<f32>(mix(prev_c, st.yzw, k), 1.0);
        }
        prev_t = st.x;
        prev_c = st.yzw;
    }
    // t 不小于任何色标 → 取末色（clamp 语义）
    return vec4<f32>(prev_c, 1.0);
}

/// M6：按 draw 求位图颜色。
///
/// 布局（**每 draw 三个 vec4**，与宿主侧 `pack_image_place` 对应）：
/// ```text
/// image_place[d*3+0] = (origin.x, origin.y, u_axis.x, u_axis.y)
/// image_place[d*3+1] = (v_axis.x, v_axis.y, tex_layer, 0)
/// image_place[d*3+2] = (image_index, repeat, 0, 0)
/// ```
/// 把 `image_index` 与 `repeat` 放进**放置数据自身**，于是"这个 draw 是不是图像"
/// 无需另占 draw_meta 分量 —— 放置缓冲区自描述，也不会与渐变下标争位置。
/// （`0xFFFF_FFFF` 表示该 draw 不是图像。）
///
/// 放置是仿射映射：解 `p = origin + u*u_axis + v*v_axis` 得 uv。
/// 退化（行列式≈0）时返回透明 —— 宁可该像素不画，也不要产出 NaN 像素。
///
/// 用 `textureSampleLevel` 而非 `textureSample`：后者要求一致控制流，
/// 而这里在"逐 draw 循环 + 条件"内调用。本引擎位图无 mip 链，故 level 固定 0。
fn image_color(d: u32, p: vec2<f32>) -> vec4<f32> {
    let e0 = image_place[d * 3u];
    let e1 = image_place[d * 3u + 1u];
    let e2 = image_place[d * 3u + 2u];
    let o = e0.xy;
    let ua = e0.zw;
    let va = e1.xy;
    let layer = u32(e1.z + 0.5);

    let det = ua.x * va.y - va.x * ua.y;
    if (abs(det) < 1e-12) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let r = p - o;
    var uv = vec2<f32>(
        (va.y * r.x - va.x * r.y) / det,
        (-ua.y * r.x + ua.x * r.y) / det,
    );
    if (e2.y > 0.5) {
        uv = uv - floor(uv);
    }
    return textureSampleLevel(image_tex, image_smp, uv, i32(layer), 0.0);
}

/// 该 draw 是否为图像（`image_place` 自描述）。
fn image_index_of(d: u32) -> u32 {
    return bitcast<u32>(image_place[d * 3u + 2u].x);
}

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
@group(0) @binding(8) var<storage, read> grad_data: array<vec4<f32>>;
@group(0) @binding(9) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var clip_tex0: texture_2d<f32>;
@group(0) @binding(11) var clip_tex1: texture_2d<f32>;
@group(0) @binding(12) var clip_tex2: texture_2d<f32>;
@group(0) @binding(13) var clip_tex3: texture_2d<f32>;
// M6 图像：放置参数（每 draw 一个 vec4）、位图纹理数组、采样器
@group(0) @binding(14) var<storage, read> image_place: array<vec4<f32>>;
@group(0) @binding(15) var image_tex: texture_2d_array<f32>;
@group(0) @binding(16) var image_smp: sampler;

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
            // M6：draw_meta 的 .w 是渐变下标，.z（见宿主）是图像下标，
            // 0xFFFF_FFFF 均表示"不适用"。索引存在**独立字段**而非从颜色里编码
            // —— 后者依赖浮点精确相等，一旦有颜色恰好碰撞就会静默出错。
            var c = colors[d];
            let gi = dm.w;
            if (gi != 0xFFFFFFFFu) {
                c = grad_color(gi, center);
            } else if (image_index_of(d) != 0xFFFFFFFFu) {
                c = image_color(d, center);
            }
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