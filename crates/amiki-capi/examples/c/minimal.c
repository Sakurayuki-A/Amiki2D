/*
 * Amiki2D C ABI —— 最小可运行示例
 * ============================================================================
 *
 * 它演示的是**绑定作者需要知道的全部东西**：
 *   1. 建 context（一次）、建 canvas（逐帧复用）
 *   2. 每次调用都检查 amiki_result —— 失败时 out 参数不被写入
 *   3. 路径、渐变、位图、层（模糊）各画一遍
 *   4. 渲染到 RGBA8 缓冲，写出 BMP 供肉眼确认
 *   5. 按 header 的所有权规则释放（`_destroy`，不是 `free`）
 *
 * 编译（Windows / MSVC）：
 *     build.cmd
 *
 * 编译（手工，任意平台）：
 *     cc -I../../include minimal.c -L<target>/release -lamiki2d -o minimal
 *
 * ⚠️ `amiki.h` 是 UTF-8 无 BOM；MSVC 在非 UTF-8 代码页下需要 `/utf-8`。
 */

#include "amiki.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 640u
#define HEIGHT 400u

/* 把每一步的返回码都查一遍 —— 这是 C 侧使用本库的基本纪律。 */
#define CHECK(expr)                                                              \
    do {                                                                         \
        amiki_result r_ = (expr);                                                \
        if (r_ != AMIKI_OK) {                                                    \
            fprintf(stderr, "%s:%d  %s 失败：%s\n", __FILE__, __LINE__, #expr,   \
                    amiki_result_string(r_));                                    \
            return 1;                                                            \
        }                                                                        \
    } while (0)

/* 写 24 位 BMP。不引入任何图像库 —— 这个示例要能单独编译。 */
static int write_bmp(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    const uint32_t row_raw = w * 3;
    const uint32_t pad = (4 - (row_raw % 4)) % 4;
    const uint32_t row = row_raw + pad;
    const uint32_t data_size = row * h;
    const uint32_t file_size = 54 + data_size;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "打不开输出文件：%s\n", path);
        return 0;
    }

    uint8_t hdr[54];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B';
    hdr[1] = 'M';
    memcpy(&hdr[2], &file_size, 4);
    const uint32_t offset = 54;
    memcpy(&hdr[10], &offset, 4);
    const uint32_t dib = 40;
    memcpy(&hdr[14], &dib, 4);
    memcpy(&hdr[18], &w, 4);
    memcpy(&hdr[22], &h, 4);
    const uint16_t planes = 1, bpp = 24;
    memcpy(&hdr[26], &planes, 2);
    memcpy(&hdr[28], &bpp, 2);
    /* 分辨率写 2835（=72 DPI），与 C# / Node 两个示例一致。
     * 不写就是 0 —— 多数解码器不在意，但会让三份产物**字节不同**，
     * 而"三者字节相同"正是验证脚本最容易检查的一条。 */
    const uint32_t ppm = 2835;
    memcpy(&hdr[38], &ppm, 4);
    memcpy(&hdr[42], &ppm, 4);
    memcpy(&hdr[34], &data_size, 4);
    fwrite(hdr, 1, sizeof(hdr), f);

    /* BMP 行序自下而上，而引擎给的是从上到下 —— 反向遍历。
     * 引擎输出是**非预乘** RGBA，BMP 要的是 BGR。 */
    uint8_t *line = (uint8_t *)calloc(row, 1);
    if (!line) {
        fclose(f);
        return 0;
    }
    for (int y = (int)h - 1; y >= 0; y--) {
        for (uint32_t x = 0; x < w; x++) {
            const uint8_t *p = rgba + ((size_t)y * w + x) * 4;
            line[x * 3 + 0] = p[2];
            line[x * 3 + 1] = p[1];
            line[x * 3 + 2] = p[0];
        }
        fwrite(line, 1, row, f);
    }
    free(line);
    fclose(f);
    return 1;
}

int main(void) {
    /* ---- 版本：头文件与运行期都要查 ---- */
    printf("Amiki2D %s（头文件 %d.%d.%d，运行期编码 0x%06X）\n",
           amiki_version_string(),
           AMIKI_VERSION_MAJOR, AMIKI_VERSION_MINOR, AMIKI_VERSION_PATCH,
           amiki_version());
    if (amiki_version() != AMIKI_VERSION) {
        fprintf(stderr, "警告：头文件与运行期库版本不一致\n");
    }

    /* ---- context：一次，跨帧复用 ---- */
    /*
     * 后端可以用环境变量覆盖。存在的理由：WASM 构建是**纯 CPU** 的，
     * 要和它做字节级对比，就得能让原生也走 CPU。
     *   AMIKI_BACKEND=cpu   → 强制 CPU
     *   AMIKI_BACKEND=gpu   → 强制 GPU（拿不到就失败）
     *   未设置 / auto       → 先试 GPU，失败落 CPU
     */
    amiki_backend want = AMIKI_BACKEND_AUTO;
    {
        const char *env = getenv("AMIKI_BACKEND");
        if (env && strcmp(env, "cpu") == 0) {
            want = AMIKI_BACKEND_CPU;
        } else if (env && strcmp(env, "gpu") == 0) {
            want = AMIKI_BACKEND_GPU;
        }
    }
    amiki_context *ctx = NULL;
    CHECK(amiki_context_create(want, &ctx));
    const char *adapter = amiki_context_adapter_name(ctx);
    printf("后端：%s（%s）\n",
           amiki_context_backend(ctx) == AMIKI_BACKEND_GPU ? "GPU" : "CPU",
           adapter ? adapter : "(null)");

    /* ---- canvas：逐帧复用同一个 ---- */
    amiki_canvas *cv = NULL;
    CHECK(amiki_canvas_create(ctx, WIDTH, HEIGHT, &cv));

    /* ---- 背景 ---- */
    CHECK(amiki_canvas_draw_rect(cv, 0, 0, (float)WIDTH, (float)HEIGHT,
                                 amiki_paint_solid(amiki_color_rgb8(24, 26, 32))));

    /* ---- 描边 + 拐角/端帽 ---- */
    amiki_path *poly = NULL;
    CHECK(amiki_path_create(&poly));
    CHECK(amiki_path_move_to(poly, 60, 300));
    CHECK(amiki_path_line_to(poly, 160, 220));
    CHECK(amiki_path_line_to(poly, 260, 320));
    CHECK(amiki_path_line_to(poly, 360, 200));
    amiki_stroke_style st = amiki_stroke_style_new(8.0f, AMIKI_CAP_ROUND, AMIKI_JOIN_ROUND);
    CHECK(amiki_canvas_stroke_path(cv, poly, st,
                                   amiki_paint_solid(amiki_color_rgb8(90, 200, 250))));

    /* ---- 渐变填充圆 ---- */
    amiki_gradient_stop stops[3];
    stops[0].t = 0.0f;
    stops[0].color = amiki_color_rgb8(255, 214, 102);
    stops[1].t = 0.5f;
    stops[1].color = amiki_color_rgb8(255, 122, 89);
    stops[2].t = 1.0f;
    stops[2].color = amiki_color_rgb8(160, 80, 200);
    amiki_gradient *grad = NULL;
    CHECK(amiki_gradient_create_radial(470.0f, 150.0f, 110.0f, stops, 3, &grad));
    CHECK(amiki_canvas_draw_circle(cv, 470.0f, 150.0f, 100.0f, amiki_paint_gradient(grad)));
    /* 引擎内部对渐变持计数引用 —— 这里就可以销毁句柄（header 所有权第 4 条） */
    amiki_gradient_destroy(grad);
    grad = NULL;

    /* ---- 位图：内嵌一个 8×8 棋盘 ---- */
    uint8_t checker[8 * 8 * 4];
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            const int on = ((x ^ y) & 1) != 0;
            uint8_t *p = &checker[(y * 8 + x) * 4];
            p[0] = on ? 255 : 40;
            p[1] = on ? 255 : 40;
            p[2] = on ? 255 : 40;
            p[3] = 255;
        }
    }
    amiki_image *img = NULL;
    CHECK(amiki_image_create_rgba8(checker, 8, 8, 0, &img));
    CHECK(amiki_canvas_draw_image(cv, img, 40.0f, 40.0f, 200.0f, 160.0f));
    amiki_image_destroy(img); /* 同样可以立刻销毁 */

    /* ---- 离屏层 + 模糊 ---- */
    {
        amiki_layer_bounds b;
        b.x = 300;
        b.y = 240;
        b.width = 300;
        b.height = 140;
        CHECK(amiki_canvas_begin_layer(cv, b, amiki_filter_blur(6.0f), 1.0f, AMIKI_BLEND_NORMAL));
        CHECK(amiki_canvas_draw_rect(cv, 330, 280, 430, 340,
                                     amiki_paint_solid(amiki_color_rgb8(120, 255, 170))));
        CHECK(amiki_canvas_draw_circle(cv, 480.0f, 310.0f, 40.0f,
                                       amiki_paint_solid(amiki_color_rgb8(255, 210, 90))));
        CHECK(amiki_canvas_end_layer(cv));
    }

    /* ---- 文本（字体不存在就跳过，示例不假设机器上有哪个字体） ---- */
    {
        /* 示例在 crates/amiki-capi/examples/c/ 下运行，字体在仓库根的
         * assets/fonts/ —— 相对路径要往上四级。也试几个常见位置，
         * 让从别处运行时不至于直接放弃。 */
        const char *candidates[] = {"../../../../assets/fonts/Inter-Regular.ttf",
                                    "../../assets/fonts/Inter-Regular.ttf",
                                    "assets/fonts/Inter-Regular.ttf",
                                    "../assets/fonts/Inter-Regular.ttf"};
        amiki_font *font = NULL;
        for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
            if (amiki_font_create_from_file(candidates[i], &font) == AMIKI_OK) {
                printf("字体：%s\n", candidates[i]);
                break;
            }
        }
        if (font) {
            float ascent = 0.0f, descent = 0.0f, gap = 0.0f;
            CHECK(amiki_font_metrics(font, &ascent, &descent, &gap));
            printf("字体度量（每 em）：ascent %.3f / descent %.3f / line_gap %.3f\n",
                   ascent, descent, gap);
            CHECK(amiki_canvas_draw_text(cv, font, "Amiki2D C ABI", 0, 40.0f, 120.0f, 42.0f,
                                         amiki_paint_solid(amiki_color_rgb8(240, 244, 255))));
            amiki_font_destroy(font);
        } else {
            printf("字体：未找到，跳过文本（示例不假设机器上有哪个字体）\n");
        }
    }

    /* ---- 渲染 ---- */
    const size_t need = (size_t)WIDTH * HEIGHT * 4;
    uint8_t *pixels = (uint8_t *)malloc(need);
    if (!pixels) {
        fprintf(stderr, "分配输出缓冲失败\n");
        return 1;
    }
    CHECK(amiki_canvas_render_rgba8(cv, pixels, need));

    if (!write_bmp("amiki_c_demo.bmp", pixels, WIDTH, HEIGHT)) {
        free(pixels);
        return 1;
    }
    printf("已写出 amiki_c_demo.bmp（%ux%u）\n", WIDTH, HEIGHT);

    /* ---- 错误路径也该演示一次：越界枚举必须被拒绝，且不崩 ---- */
    {
        const amiki_result r = amiki_canvas_set_blend(cv, 9999);
        if (r == AMIKI_OK) {
            fprintf(stderr, "越界枚举竟然被接受了 —— 这是 bug\n");
            free(pixels);
            return 1;
        }
        printf("越界枚举被正确拒绝：%s\n", amiki_result_string(r));
    }

    /* ---- 释放：用 _destroy，不是 free ---- */
    free(pixels);
    amiki_path_destroy(poly);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    printf("完成\n");
    return 0;
}
