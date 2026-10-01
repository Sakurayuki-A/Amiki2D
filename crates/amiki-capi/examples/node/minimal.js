// ============================================================================
// Amiki2D C ABI —— Node.js 示例
// ============================================================================
//
// 与 C / C# 两个示例做**完全相同**的绘制，因此三者的输出应当逐像素一致 ——
// 那正是"ABI 封送正确"的最强证据。
//
// 运行：build.cmd （或 node minimal.js，前提是先 cargo build -p amiki-capi --release）
// ============================================================================

'use strict';

const fs = require('fs');
const amiki = require('./amiki');

const W = 640, H = 400;

console.log(`Amiki2D ${amiki.versionString()}（运行期编码 0x${amiki.version().toString(16).padStart(6, '0')}）`);

// ---- context（CPU 后端也能跑；AUTO 会优先 GPU）----
const ctx = amiki.outCall(amiki.native.context_create, amiki.Backend.Auto);
amiki.check(ctx.r, 'amiki_context_create');

const backendName = amiki.native.context_backend(ctx.value) === amiki.Backend.Gpu ? 'GPU' : 'CPU';
console.log(`后端：${backendName}（${amiki.native.context_adapter_name(ctx.value)}）`);

// ---- canvas ----
const cv = amiki.outCall(amiki.native.canvas_create, ctx.value, W, H);
amiki.check(cv.r, 'amiki_canvas_create');

try {
    // ---- 背景 ----
    amiki.check(
        amiki.native.canvas_draw_rect(cv.value, 0, 0, W, H,
            amiki.native.paint_solid(amiki.native.color_rgb8(24, 26, 32))),
        'draw_rect 背景');

    // ---- 描边折线（圆角 + 圆端帽）----
    const poly = amiki.outCall(amiki.native.path_create);
    amiki.check(poly.r, 'amiki_path_create');
    amiki.native.path_move_to(poly.value, 60, 300);
    amiki.native.path_line_to(poly.value, 160, 220);
    amiki.native.path_line_to(poly.value, 260, 320);
    amiki.native.path_line_to(poly.value, 360, 200);
    const style = amiki.native.stroke_style_new(8.0, amiki.LineCap.Round, amiki.LineJoin.Round);
    amiki.check(
        amiki.native.canvas_stroke_path(cv.value, poly.value, style,
            amiki.native.paint_solid(amiki.native.color_rgb8(90, 200, 250))),
        'canvas_stroke_path');
    amiki.native.path_destroy(poly.value);

    // ---- 径向渐变圆 ----
    const stops = [
        { t: 0.0, color: amiki.native.color_rgb8(255, 214, 102) },
        { t: 0.5, color: amiki.native.color_rgb8(255, 122, 89) },
        { t: 1.0, color: amiki.native.color_rgb8(160, 80, 200) },
    ];
    const grad = amiki.outCall(amiki.native.gradient_create_radial, 470, 150, 110, stops, stops.length);
    amiki.check(grad.r, 'amiki_gradient_create_radial');
    amiki.check(
        amiki.native.canvas_draw_circle(cv.value, 470, 150, 100,
            amiki.native.paint_gradient(grad.value)),
        'canvas_draw_circle 渐变');
    // 引擎内部对渐变持计数引用 —— 这里就可以销毁句柄（header 所有权第 4 条）
    amiki.native.gradient_destroy(grad.value);

    // ---- 位图：8×8 棋盘 ----
    const checker = new Uint8Array(8 * 8 * 4);
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            const on = ((x ^ y) & 1) !== 0;
            const o = (y * 8 + x) * 4;
            checker[o] = checker[o + 1] = checker[o + 2] = on ? 255 : 40;
            checker[o + 3] = 255;
        }
    }
    const img = amiki.outCall(amiki.native.image_create_rgba8, Buffer.from(checker), 8, 8, 0);
    amiki.check(img.r, 'amiki_image_create_rgba8');
    amiki.check(amiki.native.canvas_draw_image(cv.value, img.value, 40, 40, 200, 160),
        'canvas_draw_image');
    amiki.native.image_destroy(img.value); // 同样可以立刻销毁

    // ---- 离屏层 + 模糊 ----
    amiki.check(
        amiki.native.canvas_begin_layer(cv.value,
            { x: 300, y: 240, width: 300, height: 140 },
            amiki.native.filter_blur(6.0), 1.0, amiki.BlendMode.Normal),
        'canvas_begin_layer');
    amiki.check(
        amiki.native.canvas_draw_rect(cv.value, 330, 280, 430, 340,
            amiki.native.paint_solid(amiki.native.color_rgb8(120, 255, 170))),
        'draw_rect 层内');
    amiki.check(
        amiki.native.canvas_draw_circle(cv.value, 480, 310, 40,
            amiki.native.paint_solid(amiki.native.color_rgb8(255, 210, 90))),
        'draw_circle 层内');
    amiki.check(amiki.native.canvas_end_layer(cv.value), 'canvas_end_layer');

    // ---- 文本（字体找不到就跳过）----
    const candidates = [
        '../../../../assets/fonts/Inter-Regular.ttf',
        '../../assets/fonts/Inter-Regular.ttf',
        'assets/fonts/Inter-Regular.ttf',
    ];
    let font = null;
    for (const p of candidates) {
        const r = amiki.outCall(amiki.native.font_create_from_file, p);
        if (r.r === amiki.Result.Ok) {
            font = r.value;
            console.log(`字体：${p}`);
            break;
        }
    }
    if (font) {
        const outA = [0], outD = [0], outG = [0];
        amiki.check(amiki.native.font_metrics(font, outA, outD, outG), 'amiki_font_metrics');
        console.log(`字体度量（每 em）：ascent ${outA[0].toFixed(3)} / descent ${outD[0].toFixed(3)} / line_gap ${outG[0].toFixed(3)}`);
        amiki.check(
            amiki.native.canvas_draw_text(cv.value, font, 'Amiki2D C ABI', 0, 40, 120, 42,
                amiki.native.paint_solid(amiki.native.color_rgb8(240, 244, 255))),
            'canvas_draw_text');
        amiki.native.font_destroy(font);
    } else {
        console.log('字体：未找到，跳过文本');
    }

    // ---- 渲染 ----
    const pixels = Buffer.alloc(W * H * 4);
    amiki.check(amiki.native.canvas_render_rgba8(cv.value, pixels, pixels.length),
        'amiki_canvas_render_rgba8');
    writeBmp('amiki_node_demo.bmp', pixels, W, H);
    console.log(`已写出 amiki_node_demo.bmp（${W}×${H}，${pixels.length} 字节）`);

    // ---- 错误路径 ----
    const r = amiki.native.canvas_set_blend(cv.value, 9999);
    if (r === amiki.Result.Ok) {
        console.error('越界枚举竟然被接受了 —— 这是 bug');
        process.exitCode = 1;
    } else {
        console.log(`越界枚举被正确拒绝：${amiki.AmikiError.resultString(r)}`);
    }
} finally {
    // ---- 释放：用 _destroy，不是 GC ----
    amiki.native.canvas_destroy(cv.value);
    amiki.native.context_destroy(ctx.value);
}

console.log('完成');

// ---------------------------------------------------------------------------
// 写 24 位 BMP（与 C / C# 示例同一份逻辑，便于逐像素对比）
// ---------------------------------------------------------------------------
function writeBmp(file, rgba, w, h) {
    const rowRaw = w * 3;
    const pad = (4 - (rowRaw % 4)) % 4;
    const row = rowRaw + pad;
    const dataSize = row * h;
    const buf = Buffer.alloc(54 + dataSize);

    buf.write('BM', 0, 'ascii');
    buf.writeUInt32LE(54 + dataSize, 2);
    buf.writeUInt32LE(54, 10);
    buf.writeUInt32LE(40, 14);
    buf.writeInt32LE(w, 18);
    buf.writeInt32LE(h, 22);
    buf.writeUInt16LE(1, 26);
    buf.writeUInt16LE(24, 28);
    buf.writeUInt32LE(dataSize, 34);
    buf.writeInt32LE(2835, 38);
    buf.writeInt32LE(2835, 42);

    let o = 54;
    for (let y = h - 1; y >= 0; y--) {
        for (let x = 0; x < w; x++) {
            const s = (y * w + x) * 4;
            buf[o + x * 3 + 0] = rgba[s + 2];
            buf[o + x * 3 + 1] = rgba[s + 1];
            buf[o + x * 3 + 2] = rgba[s + 0];
        }
        o += row;
    }
    fs.writeFileSync(file, buf);
}
