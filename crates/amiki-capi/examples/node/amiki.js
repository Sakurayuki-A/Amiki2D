// ============================================================================
// Amiki2D C ABI 的 Node.js 绑定
// ============================================================================
//
// ## 为什么用 koffi 而不是 napi-rs 或 WASM
//
// 三条路都试过权衡，结论写在 `README.md` 里，简短版：
//
//   · **napi-rs** 需要在**构建期下载 Node 头文件**。本项目此前上过一次当
//     （`static-dxc` 的构建期下载直接挂住，最后放弃），不想再引入同类风险。
//   · **WASM** 到不了 GPU 后端 —— 引擎的主力路径是 wgpu，wasm32 上拿不到。
//     用 WASM 演示等于只演示了一半能力。
//   · **koffi** 是纯 FFI 绑定，`npm install` 即用，**不需要改 Rust 侧一行代码**。
//     而第三个示例的目的恰恰是证明"这个 ABI 能被非 C / 非 .NET 的运行时调进来"，
//     koffi 完整地证明了这一点。
//
// 换句话说：这一版要的是**能跑、能验证**，而不是绑定层的技术选型最优解。
//
// ## koffi 的两个注意点
//
//   1. 指针参数要标 `_In_` / `_Out_`，否则 koffi 不知道方向。
//   2. 结构体按值传参时直接写类型名即可；数组用 `_In_ const T *`。
//
// 版本：Amiki2D 0.4.0
// ============================================================================

'use strict';

const path = require('path');
const fs = require('fs');
const koffi = require('koffi');

// ---------------------------------------------------------------------------
// 定位原生库：优先环境变量，其次常见的构建输出位置。
// ---------------------------------------------------------------------------
function findLibrary() {
    const names =
        process.platform === 'win32' ? ['amiki2d.dll'] :
        process.platform === 'darwin' ? ['libamiki2d.dylib'] : ['libamiki2d.so'];

    const dirs = [
        process.env.AMIKI_LIB_DIR,
        path.join(__dirname, '..', '..', '..', '..', 'target', 'release'),
        path.join(__dirname, '..', '..', '..', '..', 'target', 'debug'),
        __dirname,
    ].filter(Boolean);

    for (const dir of dirs) {
        for (const name of names) {
            const p = path.join(dir, name);
            if (fs.existsSync(p)) return p;
        }
    }
    throw new Error(
        '找不到原生库。先运行：cargo build -p amiki-capi --release\n' +
        '或用 AMIKI_LIB_DIR 指定目录。找过：\n  ' + dirs.join('\n  ')
    );
}

const LIB_PATH = findLibrary();
const lib = koffi.load(LIB_PATH);

// ---------------------------------------------------------------------------
// 结构体（字段顺序与 include/amiki.h 逐字段一致）
// ---------------------------------------------------------------------------
const Vec2 = koffi.struct('AmikiVec2', { x: 'float', y: 'float' });
const Affine = koffi.struct('AmikiAffine', {
    a: 'float', b: 'float', c: 'float', d: 'float', e: 'float', f: 'float',
});
const Color = koffi.struct('AmikiColor', { r: 'float', g: 'float', b: 'float', a: 'float' });
const LayerBounds = koffi.struct('AmikiLayerBounds', {
    x: 'int32_t', y: 'int32_t', width: 'uint32_t', height: 'uint32_t',
});
const GradientStop = koffi.struct('AmikiGradientStop', { t: 'float', color: Color });
const Placement = koffi.struct('AmikiPlacement', {
    origin: Vec2, u_axis: Vec2, v_axis: Vec2, repeat: 'int32_t',
});
const StrokeStyle = koffi.struct('AmikiStrokeStyle', {
    width: 'float', cap: 'int32_t', join: 'int32_t', miter_limit: 'float',
});
const Filter = koffi.struct('AmikiFilter', {
    kind: 'int32_t', sigma: 'float', dx: 'float', dy: 'float', color: Color,
});
const Paint = koffi.struct('AmikiPaint', {
    kind: 'int32_t', color: Color, gradient: 'void *',
});

// ---------------------------------------------------------------------------
// 枚举（取值与 header 逐一对应）
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
// 函数签名
// ---------------------------------------------------------------------------
const f = (sig) => lib.func(sig);

const native = {
    version: f('uint32_t amiki_version()'),
    version_string: f('const char *amiki_version_string()'),
    result_string: f('const char *amiki_result_string(int32_t r)'),

    affine_identity: f('AmikiAffine amiki_affine_identity()'),
    affine_translate: f('AmikiAffine amiki_affine_translate(float dx, float dy)'),
    affine_rotate: f('AmikiAffine amiki_affine_rotate(float radians)'),
    affine_scale: f('AmikiAffine amiki_affine_scale(float s)'),

    color_rgba: f('AmikiColor amiki_color_rgba(float r, float g, float b, float a)'),
    color_rgb8: f('AmikiColor amiki_color_rgb8(uint8_t r, uint8_t g, uint8_t b)'),
    color_rgba8: f('AmikiColor amiki_color_rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a)'),

    stroke_style_default: f('AmikiStrokeStyle amiki_stroke_style_default()'),
    stroke_style_new: f('AmikiStrokeStyle amiki_stroke_style_new(float w, int32_t cap, int32_t join)'),

    filter_none: f('AmikiFilter amiki_filter_none()'),
    filter_blur: f('AmikiFilter amiki_filter_blur(float sigma)'),
    filter_drop_shadow: f('AmikiFilter amiki_filter_drop_shadow(float dx, float dy, float sigma, AmikiColor c)'),

    paint_solid: f('AmikiPaint amiki_paint_solid(AmikiColor c)'),
    paint_gradient: f('AmikiPaint amiki_paint_gradient(void *g)'),

    context_create: f('int32_t amiki_context_create(int32_t backend, _Out_ void **out)'),
    context_destroy: f('void amiki_context_destroy(void *ctx)'),
    context_backend: f('int32_t amiki_context_backend(void *ctx)'),
    context_adapter_name: f('const char *amiki_context_adapter_name(void *ctx)'),

    canvas_create: f('int32_t amiki_canvas_create(void *ctx, uint32_t w, uint32_t h, _Out_ void **out)'),
    canvas_destroy: f('void amiki_canvas_destroy(void *cv)'),
    canvas_clear: f('int32_t amiki_canvas_clear(void *cv)'),
    canvas_size: f('int32_t amiki_canvas_size(void *cv, _Out_ uint32_t *w, _Out_ uint32_t *h)'),
    canvas_set_transform: f('int32_t amiki_canvas_set_transform(void *cv, AmikiAffine t)'),
    canvas_set_blend: f('int32_t amiki_canvas_set_blend(void *cv, int32_t blend)'),
    canvas_push_clip: f('int32_t amiki_canvas_push_clip(void *cv, void *path)'),
    canvas_pop_clip: f('int32_t amiki_canvas_pop_clip(void *cv)'),

    canvas_fill_path: f('int32_t amiki_canvas_fill_path(void *cv, void *path, AmikiPaint paint)'),
    canvas_stroke_path: f('int32_t amiki_canvas_stroke_path(void *cv, void *path, AmikiStrokeStyle st, AmikiPaint paint)'),
    canvas_draw_rect: f('int32_t amiki_canvas_draw_rect(void *cv, float x0, float y0, float x1, float y1, AmikiPaint paint)'),
    canvas_draw_circle: f('int32_t amiki_canvas_draw_circle(void *cv, float cx, float cy, float r, AmikiPaint paint)'),
    canvas_draw_pentagram: f('int32_t amiki_canvas_draw_pentagram(void *cv, float cx, float cy, float r, AmikiPaint paint)'),
    canvas_draw_image: f('int32_t amiki_canvas_draw_image(void *cv, void *img, float x0, float y0, float x1, float y1)'),
    canvas_draw_text: f('int32_t amiki_canvas_draw_text(void *cv, void *font, const char *utf8, size_t len, float x, float y, float size, AmikiPaint paint)'),
    canvas_begin_layer: f('int32_t amiki_canvas_begin_layer(void *cv, AmikiLayerBounds b, AmikiFilter flt, float opacity, int32_t blend)'),
    canvas_end_layer: f('int32_t amiki_canvas_end_layer(void *cv)'),
    canvas_render_rgba8: f('int32_t amiki_canvas_render_rgba8(void *cv, _Out_ uint8_t *out, size_t out_len)'),
    canvas_set_flatten_tolerance: f('int32_t amiki_canvas_set_flatten_tolerance(void *cv, float eps)'),

    path_create: f('int32_t amiki_path_create(_Out_ void **out)'),
    path_destroy: f('void amiki_path_destroy(void *p)'),
    path_move_to: f('int32_t amiki_path_move_to(void *p, float x, float y)'),
    path_line_to: f('int32_t amiki_path_line_to(void *p, float x, float y)'),
    path_quad_to: f('int32_t amiki_path_quad_to(void *p, float cx, float cy, float x, float y)'),
    path_cubic_to: f('int32_t amiki_path_cubic_to(void *p, float c1x, float c1y, float c2x, float c2y, float x, float y)'),
    path_close: f('int32_t amiki_path_close(void *p)'),
    path_rect: f('int32_t amiki_path_rect(void *p, float x0, float y0, float x1, float y1)'),
    path_circle: f('int32_t amiki_path_circle(void *p, float cx, float cy, float r)'),

    image_create_rgba8: f('int32_t amiki_image_create_rgba8(_In_ const uint8_t *px, uint32_t w, uint32_t h, size_t stride, _Out_ void **out)'),
    image_destroy: f('void amiki_image_destroy(void *img)'),
    image_size: f('int32_t amiki_image_size(void *img, _Out_ uint32_t *w, _Out_ uint32_t *h)'),

    font_create_from_file: f('int32_t amiki_font_create_from_file(const char *path, _Out_ void **out)'),
    font_destroy: f('void amiki_font_destroy(void *font)'),
    font_metrics: f('int32_t amiki_font_metrics(void *font, _Out_ float *a, _Out_ float *d, _Out_ float *g)'),

    // 字体栈 / 字体回退（0.7.0）。
    // `options` 传 null 即用 header 承诺的默认值，所以不必构造结构体。
    font_stack_create: f('void *amiki_font_stack_create(void)'),
    font_stack_destroy: f('void amiki_font_stack_destroy(void *stack)'),
    font_stack_push: f('int32_t amiki_font_stack_push(void *stack, void *font)'),
    font_stack_resolve: f('int32_t amiki_font_stack_resolve(void *stack, uint32_t codepoint, _Out_ uint32_t *out_index)'),
    canvas_draw_text_with_stack: f('int32_t amiki_canvas_draw_text_with_stack(void *cv, void *stack, const char *utf8, size_t len, float x, float y, float size, AmikiPaint paint, void *options)'),

    gradient_create_linear: f('int32_t amiki_gradient_create_linear(float x0, float y0, float x1, float y1, _In_ const AmikiGradientStop *stops, uint32_t n, _Out_ void **out)'),
    gradient_create_radial: f('int32_t amiki_gradient_create_radial(float cx, float cy, float r, _In_ const AmikiGradientStop *stops, uint32_t n, _Out_ void **out)'),
    gradient_create_sweep: f('int32_t amiki_gradient_create_sweep(float cx, float cy, float s, _In_ const AmikiGradientStop *stops, uint32_t n, _Out_ void **out)'),
    gradient_destroy: f('void amiki_gradient_destroy(void *g)'),
};

// ---------------------------------------------------------------------------
// 把返回码变成异常（与 C# 绑定同样的取舍：JS 里抛异常比查返回码自然）
// ---------------------------------------------------------------------------
class AmikiError extends Error {
    constructor(code, call) {
        super(`${call} 失败：${AmikiError.resultString(code)}（${code}）`);
        this.code = code;
        this.call = call;
    }
    static resultString(code) {
        const s = native.result_string(code);
        return s == null ? '(null)' : s;
    }
}

function check(r, call) {
    if (r !== Result.Ok) throw new AmikiError(r, call);
    return r;
}

/** 把 koffi 的 `void**` 出参读成句柄。koffi 用数组接收出参。 */
function outCall(fn, ...args) {
    const out = [null];
    const r = fn(...args, out);
    return { r, value: out[0] };
}

module.exports = {
    libPath: LIB_PATH,
    native, check, outCall, AmikiError,
    Result, BlendMode, LineCap, LineJoin, PaintKind, Backend,
    Vec2, Affine, Color, LayerBounds, GradientStop, Placement, StrokeStyle, Filter, Paint,

    version: () => native.version(),
    versionString: () => native.version_string(),
};
