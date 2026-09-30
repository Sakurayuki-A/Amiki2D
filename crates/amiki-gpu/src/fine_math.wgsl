// fine 光栅化的**共享数学**（分析式覆盖率 + W3C 混合模式）。
//
// 本文件不包含入口点与资源绑定，由宿主拼接进具体模块：
//   fine.wgsl        = 本文件 + 稠密网格入口
//   sparse_fine.wgsl = 本文件 + 稀疏桶表入口
// 与 amiki-raster 的 CPU 实现逐式对应（三端共享同一份数学）。

/// ∫_ya^yb clamp(x(y), xl, xr) dy，x(y) 线性（xa@ya → xb@yb）。
fn int_clamped_linear(xa: f32, xb: f32, ya: f32, yb: f32, xl: f32, xr: f32) -> f32 {
    let len = yb - ya;
    let slope = (xb - xa) / len;
    if (slope == 0.0) {
        return clamp(xa, xl, xr) * len;
    }
    var ts = array<f32, 4>(ya, yb, 3.4e38, 3.4e38);
    let t1 = ya + (xl - xa) / slope;
    if (t1 > ya && t1 < yb) {
        ts[2] = t1;
    }
    let t2 = ya + (xr - xa) / slope;
    if (t2 > ya && t2 < yb) {
        ts[3] = t2;
    }
    for (var st = 0; st < 5; st++) {
        let ij = array<vec2<u32>, 5>(
            vec2<u32>(0u, 1u), vec2<u32>(2u, 3u), vec2<u32>(0u, 2u),
            vec2<u32>(1u, 3u), vec2<u32>(1u, 2u));
        let p = ij[st];
        let lo = min(ts[p.x], ts[p.y]);
        let hi = max(ts[p.x], ts[p.y]);
        ts[p.x] = lo;
        ts[p.y] = hi;
    }
    var sum = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let t1 = ts[i];
        let t2 = ts[i + 1u];
        if (t2 < 3.0e38) {
            let f1 = clamp(xa + slope * (t1 - ya), xl, xr);
            let f2 = clamp(xa + slope * (t2 - ya), xl, xr);
            sum += 0.5 * (f1 + f2) * (t2 - t1);
        }
    }
    return sum;
}

/// 单线段对像素（中心 c）的有符号覆盖率贡献。
fn seg_area(p0: vec2<f32>, p1: vec2<f32>, c: vec2<f32>) -> f32 {
    let dy = p1.y - p0.y;
    if (abs(dy) < 1e-9) {
        return 0.0;
    }
    var s: f32;
    var a: vec2<f32>;
    var b: vec2<f32>;
    if (dy > 0.0) {
        s = 1.0; a = p0; b = p1;
    } else {
        s = -1.0; a = p1; b = p0;
    }
    let band_lo = c.y - 0.5;
    let band_hi = c.y + 0.5;
    let ya = max(a.y, band_lo);
    let yb = min(b.y, band_hi);
    if (ya >= yb) {
        return 0.0;
    }
    let len = yb - ya;
    let xl = c.x - 0.5;
    let xr = c.x + 0.5;
    let dxdy = (b.x - a.x) / (b.y - a.y);
    let xa = a.x + dxdy * (ya - a.y);
    let xb = a.x + dxdy * (yb - a.y);
    let ic = int_clamped_linear(xa, xb, ya, yb, xl, xr);
    return s * (ic - xl * len - len);
}

// ============ 混合模式（W3C compositing）============

fn hard_light(b: f32, s: f32) -> f32 {
    if (s <= 0.5) { return b * (2.0 * s); }
    return b + (2.0 * s - 1.0) - b * (2.0 * s - 1.0);
}

fn soft_light_d(x: f32) -> f32 {
    if (x <= 0.25) { return ((16.0 * x - 12.0) * x + 4.0) * x; }
    return sqrt(x);
}

fn lum3(c: vec3<f32>) -> f32 {
    return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum3(c);
    let n = min(c.r, min(c.g, c.b));
    let x = max(c.r, max(c.g, c.b));
    if (n < 0.0) {
        let f = l / (l - n);
        c = l + (c - l) * f;
    }
    if (x > 1.0) {
        let f = (1.0 - l) / (x - l);
        c = l + (c - l) * f;
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_color(c + vec3<f32>(l - lum3(c)));
}

fn sat3(c: vec3<f32>) -> f32 {
    return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    if (mx <= mn) {
        return vec3<f32>(0.0);
    }
    let mid_v = ((c.r + c.g + c.b - mx - mn) - mn) * s / (mx - mn);
    var out = vec3<f32>(0.0);
    if (c.r == mx) { out.r = s; } else if (c.r == mn) { out.r = 0.0; } else { out.r = mid_v; }
    if (c.g == mx) { out.g = s; } else if (c.g == mn) { out.g = 0.0; } else { out.g = mid_v; }
    if (c.b == mx) { out.b = s; } else if (c.b == mn) { out.b = 0.0; } else { out.b = mid_v; }
    return out;
}

fn blend_color(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch (mode) {
        case 1u: { return cb * cs; }
        case 2u: { return cb + cs - cb * cs; }
        case 3u: { return vec3<f32>(hard_light(cs.r, cb.r), hard_light(cs.g, cb.g), hard_light(cs.b, cb.b)); }
        case 4u: { return min(cb, cs); }
        case 5u: { return max(cb, cs); }
        case 6u: {
            var o = vec3<f32>(0.0);
            for (var i = 0u; i < 3u; i++) {
                o[i] = select(select(min(cb[i] / (1.0 - cs[i]), 1.0), 1.0, cs[i] >= 1.0), 0.0, cb[i] <= 0.0);
            }
            return o;
        }
        case 7u: {
            var o = vec3<f32>(0.0);
            for (var i = 0u; i < 3u; i++) {
                o[i] = select(select(1.0 - min((1.0 - cb[i]) / cs[i], 1.0), 0.0, cs[i] <= 0.0), 1.0, cb[i] >= 1.0);
            }
            return o;
        }
        case 8u: { return vec3<f32>(hard_light(cb.r, cs.r), hard_light(cb.g, cs.g), hard_light(cb.b, cs.b)); }
        case 9u: {
            var o = vec3<f32>(0.0);
            for (var i = 0u; i < 3u; i++) {
                let flat = cb[i] - (1.0 - 2.0 * cs[i]) * cb[i] * (1.0 - cb[i]);
                let soft = cb[i] + (2.0 * cs[i] - 1.0) * (soft_light_d(cb[i]) - cb[i]);
                o[i] = select(soft, flat, cs[i] <= 0.5);
            }
            return o;
        }
        case 10u: { return abs(cb - cs); }
        case 11u: { return cb + cs - 2.0 * cb * cs; }
        case 12u: { return set_lum(set_sat(cs, sat3(cb)), lum3(cb)); }
        case 13u: { return set_lum(set_sat(cb, sat3(cs)), lum3(cb)); }
        case 14u: { return set_lum(cs, lum3(cb)); }
        case 15u: { return set_lum(cb, lum3(cs)); }
        default: { return cs; }
    }
}

fn composite(dst: vec4<f32>, src: vec4<f32>, src_a: f32, mode: u32) -> vec4<f32> {
    if (mode == 0u) {
        let inv = 1.0 - src_a;
        return vec4<f32>(src.rgb * src_a + dst.rgb * inv, src_a + dst.a * inv);
    }
    let ab = dst.a;
    let b = blend_color(mode, dst.rgb, src.rgb);
    let ao = src_a + ab * (1.0 - src_a);
    let co = src_a * (1.0 - ab) * src.rgb
        + src_a * ab * b
        + (1.0 - src_a) * ab * dst.rgb;
    let out_rgb = select(vec3<f32>(0.0), co / ao, ao > 0.0);
    return vec4<f32>(out_rgb, ao);
}

/// 裁剪采样展开（≤4 层，逐像素乘入）。
fn clip_sample(ci: u32, gid: vec2<u32>, clip_tex0: texture_2d<f32>, clip_tex1: texture_2d<f32>,
               clip_tex2: texture_2d<f32>, clip_tex3: texture_2d<f32>) -> f32 {
    if (ci == 0u) { return textureLoad(clip_tex0, gid, 0u).r; }
    if (ci == 1u) { return textureLoad(clip_tex1, gid, 0u).r; }
    if (ci == 2u) { return textureLoad(clip_tex2, gid, 0u).r; }
    if (ci == 3u) { return textureLoad(clip_tex3, gid, 0u).r; }
    return 1.0;
}

/// clip 索引打包：每层 8 bit（见 draw_meta 说明）。
fn clip_mul(ci: u32, nclip: u32, dm: vec4<u32>, gid: vec2<u32>, clip_tex0: texture_2d<f32>,
            clip_tex1: texture_2d<f32>, clip_tex2: texture_2d<f32>, clip_tex3: texture_2d<f32>) -> f32 {
    var cov = 1.0;
    cov *= clip_sample(ci & 0xFFu, gid, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
    if (nclip > 1u) {
        cov *= clip_sample((ci >> 8u) & 0xFFu, gid, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
    }
    let ci2 = dm.z;
    if (nclip > 2u) {
        cov *= clip_sample(ci2 & 0xFFu, gid, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
    }
    if (nclip > 3u) {
        cov *= clip_sample((ci2 >> 8u) & 0xFFu, gid, clip_tex0, clip_tex1, clip_tex2, clip_tex3);
    }
    return cov;
}