// ============================================================================
// Amiki2D —— WASM 加载器（不依赖 wasm-bindgen）
// ============================================================================
//
// ## 为什么不用 wasm-bindgen
//
// wasm-bindgen 会在 Rust 与 JS 之间再插一层自己的 ABI（`__wbindgen_*` 一堆
// 胶水函数），而 **C ABI 已经是一份设计好、测试过、三种语言验证过的边界** ——
// 再套一层只是把边界复杂化，还多一个构建期依赖（需要 `wasm-bindgen-cli`）。
//
// 这里直接用 **原生 WebAssembly API**，边界就是 C ABI 本身。
//
// ## 结构体怎么过边界（关键）
//
// WebAssembly **没有"结构体按值"这个概念**，所以 C ABI 里的
//
//     AmikiPaint amiki_paint_solid(AmikiColor c);
//     amiki_result amiki_canvas_fill_path(amiki_canvas*, const amiki_path*, amiki_paint);
//
// 会被编译器**降级**成（实测，见 tools/wasm_sig.py）：
//
//     大结构体返回 → sret：调用方传一个隐藏的**第一个参数**作返回空间
//     结构体参数   → 一律**按指针**传
//
// 例如实测签名：
//
//     amiki_paint_solid    (i32, i32)             // sret, color*
//     amiki_canvas_fill_path (i32, i32, i32) -> i32   // canvas, path, paint*
//
// 本加载器把这两条规则集中在 `_sret` / `_struct` 两个辅助里，别处看不到。
//
// ## 用法
//
//     const amiki = await require('./amiki.js').load('amiki2d.wasm');
//     const ctx = amiki.contextCreate(amiki.Backend.Cpu);
//     const cv = amiki.canvasCreate(ctx, 640, 400);
//     amiki.drawRect(cv, 0, 0, 640, 400, amiki.paintSolid(amiki.colorRgb8(24, 26, 32)));
//     const px = amiki.renderRgba8(cv);      // Uint8Array，RGBA8 非预乘
// ============================================================================

'use strict';

// ---------------------------------------------------------------------------
// 枚举（与 include/amiki.h 逐一对应）
// ---------------------------------------------------------------------------
const Result = {
    Ok: 0, InvalidArgument: 1, OutOfMemory: 2, BackendUnavailable: 3,
    NotSupported: 4, Panic: 5, Unknown: 6,
};
const BlendMode = {
    Normal: 0, Multiply: 1, Screen: 2, Overlay: 3, Darken: 4, Lighten: 5,
    ColorDodge: 6, ColorBurn: 7, HardLight: 8, SoftLight: 9, Difference: 10,
    Exclusion: 11, Hue: 12, Saturation: 13, Color: 14, Luminosity: 15,
};
const LineCap = { Butt: 0, Round: 1, Square: 2 };
const LineJoin = { Bevel: 0, Miter: 1, Round: 2 };
const PaintKind = { Solid: 0, Linear: 1, Radial: 2, Sweep: 3 };
const Backend = { Auto: 0, Gpu: 1, Cpu: 2 };

// ---------------------------------------------------------------------------
// 结构体布局（字段顺序与 header 逐字段一致）
// ---------------------------------------------------------------------------
const LAYOUTS = {
    Vec2: [['x', 'f32'], ['y', 'f32']],
    Affine: [['a', 'f32'], ['b', 'f32'], ['c', 'f32'], ['d', 'f32'], ['e', 'f32'], ['f', 'f32']],
    Color: [['r', 'f32'], ['g', 'f32'], ['b', 'f32'], ['a', 'f32']],
    LayerBounds: [['x', 'i32'], ['y', 'i32'], ['width', 'u32'], ['height', 'u32']],
    GradientStop: [['t', 'f32'], ['color', 'Color']],
    StrokeStyle: [['width', 'f32'], ['cap', 'i32'], ['join', 'i32'], ['miter_limit', 'f32']],
    Filter: [['kind', 'i32'], ['sigma', 'f32'], ['dx', 'f32'], ['dy', 'f32'], ['color', 'Color']],
    Paint: [['kind', 'i32'], ['color', 'Color'], ['gradient', 'u32']],
};

const SIZES = { i32: 4, u32: 4, f32: 4 };

function sizeOf(name) {
    let n = 0;
    for (const [f, t] of LAYOUTS[name]) n += LAYOUTS[t] ? sizeOf(t) : SIZES[t];
    return n;
}
// 结构体的对齐 = 成员最大对齐。这里全是 4 字节。
function alignOf(name) {
    return LAYOUTS[name] ? 4 : 4;
}

const STRUCT_SIZES = Object.fromEntries(Object.keys(LAYOUTS).map((k) => [k, sizeOf(k)]));
const MAX_STRUCT = Math.max(...Object.values(STRUCT_SIZES));

// ---------------------------------------------------------------------------
// 加载
// ---------------------------------------------------------------------------
async function load(wasmPath, options = {}) {
    const fs = require('fs');
    const bytes = fs.readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(bytes, options.imports || {});
    return new Amiki(instance.exports);
}

class AmikiError extends Error {
    constructor(code, call, messages) {
        super(`${call} 失败：${messages ? messages[code] : code}（${code}）`);
        this.code = code;
        this.call = call;
    }
}

class Amiki {
    constructor(exports) {
        this.ex = exports;
        this.memory = exports.memory;
        this._u8 = null;
        this._dv = null;
        this._refresh();

        // 结构体收发的常驻暂存区：避免每次调用都分配。
        // 三个槽位够用（一个 sret 返回 + 两个参数），嵌套调用时不会冲突
        // （参数在调用前写入，返回在调用后读出，且调用是同步的）。
        this._scratchBase = exports.amiki_alloc(MAX_STRUCT * 4);
        if (this._scratchBase === 0) throw new Error('无法分配结构体暂存区');
        this._scratchNext = 0;
        this._resultMessages = null;
    }

    /** wasm 的 memory 会随增长换掉 ArrayBuffer，所以每次用之前刷新视图。 */
    _refresh() {
        if (this._u8 === null || this._u8.buffer !== this.memory.buffer) {
            this._u8 = new Uint8Array(this.memory.buffer);
            this._dv = new DataView(this.memory.buffer);
        }
    }

    // ---- 分配 ----
    alloc(size) {
        const p = this.ex.amiki_alloc(size);
        if (p === 0 && size !== 0) throw new Error(`amiki_alloc(${size}) 失败`);
        return p;
    }
    free(ptr, size) {
        this.ex.amiki_free(ptr, size);
    }

    /** 把 JS 数据写进 wasm 内存，返回指针。调用方负责 free。 */
    write(bytes) {
        const p = this.alloc(bytes.length);
        this._refresh();
        this._u8.set(bytes, p);
        return p;
    }

    /** 读一段 wasm 内存成新的 Uint8Array（拷贝出来，不持有视图）。 */
    read(ptr, len) {
        this._refresh();
        return this._u8.slice(ptr, ptr + len);
    }

    readCStr(ptr) {
        if (ptr === 0) return null;
        this._refresh();
        const u8 = this._u8;
        let end = ptr;
        while (end < u8.length && u8[end] !== 0) end++;
        let s = '';
        for (let i = ptr; i < end; i++) s += String.fromCharCode(u8[i]);
        // UTF-8 解码
        return Buffer.from(u8.slice(ptr, end)).toString('utf8');
    }

    // ---- 结构体 ----
    _packInto(ptr, name, values) {
        this._refresh();
        let off = 0;
        for (const [field, type] of LAYOUTS[name]) {
            const v = values[field];
            if (LAYOUTS[type]) {
                off += this._packInto(ptr + off, type, v || {});
            } else {
                if (type === 'f32') this._dv.setFloat32(ptr + off, v || 0, true);
                else this._dv.setUint32(ptr + off, (v || 0) >>> 0, true);
                off += SIZES[type];
            }
        }
        return off;
    }

    /** 把一个结构体值放进暂存区，返回指针。**只在一次调用内有效。** */
    _pack(name, values) {
        const ptr = this._scratchBase + this._scratchNext * MAX_STRUCT;
        this._scratchNext = (this._scratchNext + 1) % 4;
        this._packInto(ptr, name, values);
        return ptr;
    }

    /** 分配一块暂存区用作 sret 返回空间。 */
    _sretSlot() {
        const ptr = this._scratchBase + this._scratchNext * MAX_STRUCT;
        this._scratchNext = (this._scratchNext + 1) % 4;
        return ptr;
    }

    /** 调用一个**返回结构体**的导出（wasm 下走 sret：第一个参数是返回指针）。 */
    _sret(name, structName, ...args) {
        const ret = this._sretSlot();
        this.ex[name](ret, ...args);
        this._refresh();
        return this._readStruct(ret, structName);
    }

    _readStruct(ptr, name) {
        const out = {};
        let off = 0;
        for (const [field, type] of LAYOUTS[name]) {
            if (LAYOUTS[type]) {
                out[field] = this._readStruct(ptr + off, type);
                off += sizeOf(type);
            } else {
                if (type === 'f32') out[field] = this._dv.getFloat32(ptr + off, true);
                else out[field] = this._dv.getUint32(ptr + off, true);
                off += SIZES[type];
            }
        }
        return out;
    }

    /** 所有 `amiki_result` 返回值的统一检查。 */
    _check(rc, call) {
        if (rc !== Result.Ok) {
            throw new AmikiError(rc, call, this._resultMessages);
        }
        return rc;
    }

    /** 错误码 → 说明文字（来自引擎，静态字符串）。 */
    resultString(code) {
        return this.readCStr(this.ex.amiki_result_string(code)) || 'unknown';
    }

    // =======================================================================
    // 版本
    // =======================================================================
    version() { return this.ex.amiki_version(); }
    versionString() { return this.readCStr(this.ex.amiki_version_string()); }

    // =======================================================================
    // 值构造（都返回结构体 → 走 sret）
    // =======================================================================
    affineIdentity() { return this._sret('amiki_affine_identity', 'Affine'); }
    affineTranslate(dx, dy) { return this._sret('amiki_affine_translate', 'Affine', dx, dy); }
    affineRotate(r) { return this._sret('amiki_affine_rotate', 'Affine', r); }
    affineScale(s) { return this._sret('amiki_affine_scale', 'Affine', s); }
    affineConcat(a, b) {
        return this._sret('amiki_affine_concat', 'Affine', this._pack('Affine', a), this._pack('Affine', b));
    }

    colorRgba(r, g, b, a) { return this._sret('amiki_color_rgba', 'Color', r, g, b, a); }
    colorRgb8(r, g, b) { return this._sret('amiki_color_rgb8', 'Color', r, g, b); }
    colorRgba8(r, g, b, a) { return this._sret('amiki_color_rgba8', 'Color', r, g, b, a); }

    strokeStyleDefault() { return this._sret('amiki_stroke_style_default', 'StrokeStyle'); }
    strokeStyleNew(width, cap, join) {
        return this._sret('amiki_stroke_style_new', 'StrokeStyle', width, cap, join);
    }

    filterNone() { return this._sret('amiki_filter_none', 'Filter'); }
    filterBlur(sigma) { return this._sret('amiki_filter_blur', 'Filter', sigma); }
    filterDropShadow(dx, dy, sigma, color) {
        return this._sret('amiki_filter_drop_shadow', 'Filter', dx, dy, sigma, this._pack('Color', color));
    }

    paintSolid(color) {
        return this._sret('amiki_paint_solid', 'Paint', this._pack('Color', color));
    }
    paintGradient(gradientHandle) {
        return this._sret('amiki_paint_gradient', 'Paint', gradientHandle);
    }

    // =======================================================================
    // 上下文
    // =======================================================================
    contextCreate(backend = Backend.Auto) {
        const out = this._sretSlot();
        this._check(this.ex.amiki_context_create(backend, out), 'amiki_context_create');
        this._refresh();
        return this._dv.getUint32(out, true);
    }
    contextDestroy(ctx) { this.ex.amiki_context_destroy(ctx); }
    contextBackend(ctx) { return this.ex.amiki_context_backend(ctx); }
    contextAdapterName(ctx) { return this.readCStr(this.ex.amiki_context_adapter_name(ctx)); }

    // =======================================================================
    // 画布
    // =======================================================================
    canvasCreate(ctx, w, h) {
        const out = this._sretSlot();
        this._check(this.ex.amiki_canvas_create(ctx, w, h, out), 'amiki_canvas_create');
        this._refresh();
        return this._dv.getUint32(out, true);
    }
    canvasDestroy(cv) { this.ex.amiki_canvas_destroy(cv); }
    canvasClear(cv) { return this._check(this.ex.amiki_canvas_clear(cv), 'amiki_canvas_clear'); }

    /** @returns {{w:number,h:number}} */
    canvasSize(cv) {
        const wo = this._sretSlot();
        const ho = this._sretSlot();
        this._check(this.ex.amiki_canvas_size(cv, wo, ho), 'amiki_canvas_size');
        this._refresh();
        return { w: this._dv.getUint32(wo, true), h: this._dv.getUint32(ho, true) };
    }

    setTransform(cv, t) {
        return this._check(this.ex.amiki_canvas_set_transform(cv, this._pack('Affine', t)), 'set_transform');
    }
    resetTransform(cv) { return this._check(this.ex.amiki_canvas_reset_transform(cv), 'reset_transform'); }
    translate(cv, dx, dy) { return this._check(this.ex.amiki_canvas_translate(cv, dx, dy), 'translate'); }
    rotate(cv, r) { return this._check(this.ex.amiki_canvas_rotate(cv, r), 'rotate'); }
    scale(cv, s) { return this._check(this.ex.amiki_canvas_scale(cv, s), 'scale'); }
    setBlend(cv, b) { return this._check(this.ex.amiki_canvas_set_blend(cv, b), 'set_blend'); }
    pushClip(cv, path) { return this._check(this.ex.amiki_canvas_push_clip(cv, path), 'push_clip'); }
    popClip(cv) { return this._check(this.ex.amiki_canvas_pop_clip(cv), 'pop_clip'); }
    setFlattenTolerance(cv, eps) {
        return this._check(this.ex.amiki_canvas_set_flatten_tolerance(cv, eps), 'set_flatten_tolerance');
    }

    // =======================================================================
    // 绘制
    // =======================================================================
    fillPath(cv, path, paint) {
        return this._check(this.ex.amiki_canvas_fill_path(cv, path, this._pack('Paint', paint)), 'fill_path');
    }
    strokePath(cv, path, style, paint) {
        return this._check(
            this.ex.amiki_canvas_stroke_path(cv, path, this._pack('StrokeStyle', style), this._pack('Paint', paint)),
            'stroke_path');
    }
    drawRect(cv, x0, y0, x1, y1, paint) {
        return this._check(this.ex.amiki_canvas_draw_rect(cv, x0, y0, x1, y1, this._pack('Paint', paint)), 'draw_rect');
    }
    drawCircle(cv, cx, cy, r, paint) {
        return this._check(this.ex.amiki_canvas_draw_circle(cv, cx, cy, r, this._pack('Paint', paint)), 'draw_circle');
    }
    drawPentagram(cv, cx, cy, r, paint) {
        return this._check(this.ex.amiki_canvas_draw_pentagram(cv, cx, cy, r, this._pack('Paint', paint)), 'draw_pentagram');
    }
    drawImage(cv, img, x0, y0, x1, y1) {
        return this._check(this.ex.amiki_canvas_draw_image(cv, img, x0, y0, x1, y1), 'draw_image');
    }
    drawText(cv, font, text, x, y, size, paint) {
        const bytes = Buffer.from(text, 'utf8');
        const p = this.write(bytes);
        try {
            return this._check(
                this.ex.amiki_canvas_draw_text(cv, font, p, bytes.length, x, y, size, this._pack('Paint', paint)),
                'draw_text');
        } finally {
            this.free(p, bytes.length);
        }
    }
    beginLayer(cv, bounds, filter, opacity, blend) {
        return this._check(
            this.ex.amiki_canvas_begin_layer(cv, this._pack('LayerBounds', bounds),
                this._pack('Filter', filter), opacity, blend),
            'begin_layer');
    }
    endLayer(cv) { return this._check(this.ex.amiki_canvas_end_layer(cv), 'end_layer'); }

    /** 渲染成 RGBA8（非预乘、行序自上而下）。返回新的 Uint8Array。 */
    renderRgba8(cv) {
        const { w, h } = this.canvasSize(cv);
        const need = w * h * 4;
        const p = this.alloc(need);
        try {
            this._check(this.ex.amiki_canvas_render_rgba8(cv, p, need), 'render_rgba8');
            return this.read(p, need);
        } finally {
            this.free(p, need);
        }
    }

    // =======================================================================
    // 路径
    // =======================================================================
    pathCreate() {
        const out = this._sretSlot();
        this._check(this.ex.amiki_path_create(out), 'amiki_path_create');
        this._refresh();
        return this._dv.getUint32(out, true);
    }
    pathDestroy(p) { this.ex.amiki_path_destroy(p); }
    pathClear(p) { return this._check(this.ex.amiki_path_clear(p), 'path_clear'); }
    pathMoveTo(p, x, y) { return this._check(this.ex.amiki_path_move_to(p, x, y), 'path_move_to'); }
    pathLineTo(p, x, y) { return this._check(this.ex.amiki_path_line_to(p, x, y), 'path_line_to'); }
    pathQuadTo(p, cx, cy, x, y) { return this._check(this.ex.amiki_path_quad_to(p, cx, cy, x, y), 'path_quad_to'); }
    pathCubicTo(p, a, b, c, d, x, y) {
        return this._check(this.ex.amiki_path_cubic_to(p, a, b, c, d, x, y), 'path_cubic_to');
    }
    pathClose(p) { return this._check(this.ex.amiki_path_close(p), 'path_close'); }
    pathRect(p, x0, y0, x1, y1) { return this._check(this.ex.amiki_path_rect(p, x0, y0, x1, y1), 'path_rect'); }
    pathCircle(p, cx, cy, r) { return this._check(this.ex.amiki_path_circle(p, cx, cy, r), 'path_circle'); }

    // =======================================================================
    // 资源
    // =======================================================================
    imageCreateRgba8(pixels, w, h, stride = 0) {
        const p = this.write(pixels);
        const out = this._sretSlot();
        try {
            this._check(this.ex.amiki_image_create_rgba8(p, w, h, stride, out), 'amiki_image_create_rgba8');
            this._refresh();
            return this._dv.getUint32(out, true);
        } finally {
            this.free(p, pixels.length);
        }
    }
    imageDestroy(img) { this.ex.amiki_image_destroy(img); }

    fontCreateFromMemory(bytes) {
        const p = this.write(bytes);
        const out = this._sretSlot();
        try {
            this._check(this.ex.amiki_font_create_from_memory(p, bytes.length, out), 'amiki_font_create_from_memory');
            this._refresh();
            return this._dv.getUint32(out, true);
        } finally {
            this.free(p, bytes.length);
        }
    }
    fontDestroy(f) { this.ex.amiki_font_destroy(f); }

    gradientCreateLinear(x0, y0, x1, y1, stops) {
        return this._gradient('amiki_gradient_create_linear', [x0, y0, x1, y1], stops);
    }
    gradientCreateRadial(cx, cy, r, stops) {
        return this._gradient('amiki_gradient_create_radial', [cx, cy, r], stops);
    }
    gradientCreateSweep(cx, cy, start, stops) {
        return this._gradient('amiki_gradient_create_sweep', [cx, cy, start], stops);
    }
    _gradient(fnName, geom, stops) {
        const stride = STRUCT_SIZES.GradientStop;
        const buf = new Uint8Array(stride * stops.length);
        const dv = new DataView(buf.buffer);
        stops.forEach((s, i) => {
            dv.setFloat32(i * stride + 0, s.t, true);
            dv.setFloat32(i * stride + 4, s.color.r, true);
            dv.setFloat32(i * stride + 8, s.color.g, true);
            dv.setFloat32(i * stride + 12, s.color.b, true);
            dv.setFloat32(i * stride + 16, s.color.a, true);
        });
        const p = this.write(buf);
        const out = this._sretSlot();
        try {
            this._check(this.ex[fnName](...geom, p, stops.length, out), fnName);
            this._refresh();
            return this._dv.getUint32(out, true);
        } finally {
            this.free(p, buf.length);
        }
    }
    gradientDestroy(g) { this.ex.amiki_gradient_destroy(g); }
}

module.exports = {
    load,
    Amiki,
    AmikiError,
    Result, BlendMode, LineCap, LineJoin, PaintKind, Backend,
    STRUCT_SIZES, LAYOUTS,
};
