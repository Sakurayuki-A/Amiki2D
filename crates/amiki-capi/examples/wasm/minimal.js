// ============================================================================
// Amiki2D WASM 示例
// ============================================================================
//
// 画**与 C / C# / Node 三个原生示例完全相同**的场景，然后写出 BMP ——
// 目的是做**字节级对比**。
//
// 注意：WASM 构建是**纯 CPU**（不含 wgpu），所以要比的对象是**原生 CPU 后端**，
// 不是原生默认的 GPU 后端。跑法见 build.cmd。
//
//   node minimal.js <amiki2d.wasm>
// ============================================================================

'use strict';

const fs = require('fs');
const path = require('path');
const { load, Backend, LineCap, LineJoin, BlendMode } = require('./amiki');

const W = 640, H = 400;

(async () => {
    const wasmPath = process.argv[2] || path.join(__dirname, 'amiki2d.wasm');
    const amiki = await load(wasmPath);

    console.log(`Amiki2D ${amiki.versionString()}（wasm，${(fs.statSync(wasmPath).size / 1024).toFixed(1)} KB）`);

    // 纯 CPU 构建里 GPU 后端不存在 —— 这里显式要 GPU 应当**明确失败**，
    // 而不是静默给一个 CPU 后端。这是这一版特意验证的行为。
    try {
        amiki.contextCreate(Backend.Gpu);
        console.log('后端：GPU 可用');
    } catch (e) {
        console.log(`后端：GPU 不可用（${e.code === 3 ? 'BackendUnavailable' : e.code}），用 CPU`);
    }

    const ctx = amiki.contextCreate(Backend.Cpu);
    const cv = amiki.canvasCreate(ctx, W, H);
    console.log(`后端：${amiki.contextBackend(ctx) === Backend.Cpu ? 'CPU' : 'GPU'}（${amiki.contextAdapterName(ctx)}）`);

    try {
        // ---- 背景 ----
        amiki.drawRect(cv, 0, 0, W, H, amiki.paintSolid(amiki.colorRgb8(24, 26, 32)));

        // ---- 描边折线（圆角 + 圆端帽）----
        {
            const poly = amiki.pathCreate();
            amiki.pathMoveTo(poly, 60, 300);
            amiki.pathLineTo(poly, 160, 220);
            amiki.pathLineTo(poly, 260, 320);
            amiki.pathLineTo(poly, 360, 200);
            const style = amiki.strokeStyleNew(8.0, LineCap.Round, LineJoin.Round);
            amiki.strokePath(cv, poly, style, amiki.paintSolid(amiki.colorRgb8(90, 200, 250)));
            amiki.pathDestroy(poly);
        }

        // ---- 径向渐变圆 ----
        {
            const grad = amiki.gradientCreateRadial(470, 150, 110, [
                { t: 0.0, color: amiki.colorRgb8(255, 214, 102) },
                { t: 0.5, color: amiki.colorRgb8(255, 122, 89) },
                { t: 1.0, color: amiki.colorRgb8(160, 80, 200) },
            ]);
            amiki.drawCircle(cv, 470, 150, 100, amiki.paintGradient(grad));
            // 引擎内部持计数引用 —— 这里就可以销毁句柄
            amiki.gradientDestroy(grad);
        }

        // ---- 位图：8×8 棋盘 ----
        {
            const checker = Buffer.alloc(8 * 8 * 4);
            for (let y = 0; y < 8; y++) {
                for (let x = 0; x < 8; x++) {
                    const on = ((x ^ y) & 1) !== 0;
                    const o = (y * 8 + x) * 4;
                    checker[o] = checker[o + 1] = checker[o + 2] = on ? 255 : 40;
                    checker[o + 3] = 255;
                }
            }
            const img = amiki.imageCreateRgba8(checker, 8, 8);
            amiki.drawImage(cv, img, 40, 40, 200, 160);
            amiki.imageDestroy(img);
        }

        // ---- 离屏层 + 模糊 ----
        amiki.beginLayer(cv, { x: 300, y: 240, width: 300, height: 140 },
            amiki.filterBlur(6.0), 1.0, BlendMode.Normal);
        amiki.drawRect(cv, 330, 280, 430, 340, amiki.paintSolid(amiki.colorRgb8(120, 255, 170)));
        amiki.drawCircle(cv, 480, 310, 40, amiki.paintSolid(amiki.colorRgb8(255, 210, 90)));
        amiki.endLayer(cv);

        // ---- 文本（字体找不到就跳过）----
        // 与 c/ 、csharp/ 、node/ 三个示例**完全相同的顺序与参数** ——
        // 这条链的一端是"wasm 与原生 CPU 逐字节一致"，参数差一点就断。
        {
            const names = ['Inter-Regular.ttf', 'NotoSansSC-Regular.ttf'];
            const fonts = [];
            for (const name of names) {
                const p = path.join(__dirname, '..', '..', '..', '..', 'assets', 'fonts', name);
                if (!fs.existsSync(p)) continue;
                fonts.push(amiki.fontCreateFromMemory(fs.readFileSync(p)));
                console.log(`字体：${path.relative(process.cwd(), p)}`);
            }
            if (fonts.length > 0) {
                const ink = amiki.paintSolid(amiki.colorRgb8(240, 244, 255));
                const stack = amiki.fontStackCreate();
                for (const font of fonts) {
                    amiki.fontStackPush(stack, font);
                }
                // 单字体路径：拉丁
                amiki.drawText(cv, fonts[0], 'Amiki2D C ABI', 40, 120, 42, ink);
                // 字体回退路径：中英混排（汉字由 Noto 渲染）
                amiki.drawTextWithStack(cv, stack, '你好 Amiki2D 世界', 40, 180, 32, ink, 0);
                amiki.fontStackDestroy(stack);
                for (const font of fonts) {
                    amiki.fontDestroy(font);
                }
            } else {
                console.log('字体：未找到，跳过文本');
            }
        }

        // ---- 渲染 ----
        const px = amiki.renderRgba8(cv);
        const outName = process.argv[3] || 'amiki_wasm_demo.bmp';
        writeBmp(outName, px, W, H);
        console.log(`已写出 ${outName}（${W}×${H}，${px.length} 字节）`);

        // ---- 错误路径 ----
        const rc = amiki.ex.amiki_canvas_set_blend(cv, 9999);
        if (rc === 0) {
            console.error('越界枚举竟然被接受了 —— 这是 bug');
            process.exitCode = 1;
        } else {
            console.log(`越界枚举被正确拒绝：${amiki.resultString(rc)}`);
        }
    } finally {
        amiki.canvasDestroy(cv);
        amiki.contextDestroy(ctx);
    }

    console.log('完成');
})().catch((e) => {
    console.error('失败:', e.stack || e.message);
    process.exit(1);
});

// 与其它示例同一份 BMP 逻辑（含分辨率字段，保证可字节对比）
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
