/*
 * 头文件自检：**只 include、不链接**，验证 amiki.h 自身是完整且自洽的。
 *
 * 编译方式（C 与 C++ 各一遍，后者验证 extern "C" 保护）：
 *     cl /nologo /W4 /WX /c test_header.c
 *     cl /nologo /W4 /WX /TP /c test_header.c
 *
 * 它必须**不使用**任何未在 amiki.h 中定义的类型 —— 也就是说，
 * 这个文件能编过，就证明 header 里的每个声明都自洽。
 */
#include "amiki.h"

#include <string.h>

/* 把每个枚举与结构体都实际用一遍，避免"声明了但类型不完整"这类错误。 */
static amiki_stroke_style make_style(void) {
    amiki_stroke_style s = amiki_stroke_style_new(2.0f, AMIKI_CAP_ROUND, AMIKI_JOIN_MITER);
    s.miter_limit = 4.0f;
    return s;
}

static amiki_filter make_filter(void) {
    amiki_filter f = amiki_filter_drop_shadow(2.0f, 3.0f, 4.0f, amiki_color_rgba8(0, 0, 0, 128));
    if (f.kind != AMIKI_FILTER_DROP_SHADOW) {
        f = amiki_filter_blur(1.0f);
    }
    if (f.kind == AMIKI_FILTER_NONE) {
        f = amiki_filter_none();
    }
    return f;
}

static amiki_paint make_paint(const amiki_gradient* g) {
    amiki_paint p = amiki_paint_solid(amiki_color_rgb8(1, 2, 3));
    if (g != NULL) {
        p = amiki_paint_gradient(g);
    }
    p.color = amiki_color_rgba(0.1f, 0.2f, 0.3f, 1.0f);
    return p;
}

/* 一个不链接的"用一遍"函数：把所有声明都摸到，编译器才能全部检查。 */
void amiki_header_selfcheck(void) {
    amiki_context* ctx = NULL;
    amiki_canvas* cv = NULL;
    amiki_path* path = NULL;
    amiki_image* img = NULL;
    amiki_font* font = NULL;
    amiki_gradient* grad = NULL;
    uint8_t out[16];
    uint32_t w = 0, h = 0;
    float ascent = 0.0f, descent = 0.0f, gap = 0.0f;
    amiki_gradient_stop stops[2];

    stops[0].t = 0.0f;
    stops[0].color = amiki_color_rgb8(0, 0, 0);
    stops[1].t = 1.0f;
    stops[1].color = amiki_color_rgb8(255, 255, 255);

    (void)amiki_version();
    (void)amiki_version_string();
    (void)amiki_result_string(AMIKI_ERROR_PANIC);
    (void)AMIKI_VERSION;

    (void)amiki_context_create(AMIKI_BACKEND_AUTO, &ctx);
    (void)amiki_context_backend(ctx);
    (void)amiki_context_adapter_name(ctx);
    amiki_context_destroy(ctx);
    amiki_context_destroy(NULL); /* 允许 */

    (void)amiki_canvas_create(ctx, 8, 8, &cv);
    (void)amiki_canvas_size(cv, &w, &h);
    (void)amiki_canvas_clear(cv);
    (void)amiki_canvas_set_transform(cv, amiki_affine_identity());
    (void)amiki_canvas_set_transform(cv, amiki_affine_translate(1.0f, 2.0f));
    (void)amiki_canvas_set_transform(cv, amiki_affine_rotate(0.5f));
    (void)amiki_canvas_set_transform(cv, amiki_affine_scale(2.0f));
    (void)amiki_canvas_set_transform(cv, amiki_affine_concat(amiki_affine_identity(),
                                                              amiki_affine_scale(1.0f)));
    (void)amiki_canvas_reset_transform(cv);
    (void)amiki_canvas_translate(cv, 1.0f, 1.0f);
    (void)amiki_canvas_rotate(cv, 0.25f);
    (void)amiki_canvas_scale(cv, 1.5f);
    (void)amiki_canvas_set_blend(cv, AMIKI_BLEND_LUMINOSITY);
    (void)amiki_canvas_push_clip(cv, path);
    (void)amiki_canvas_pop_clip(cv);
    (void)amiki_canvas_fill_path(cv, path, make_paint(grad));
    (void)amiki_canvas_stroke_path(cv, path, make_style(), make_paint(grad));
    (void)amiki_canvas_draw_rect(cv, 0.0f, 0.0f, 1.0f, 1.0f, make_paint(grad));
    (void)amiki_canvas_draw_circle(cv, 0.0f, 0.0f, 1.0f, make_paint(grad));
    (void)amiki_canvas_draw_pentagram(cv, 0.0f, 0.0f, 1.0f, make_paint(grad));
    (void)amiki_canvas_draw_image(cv, img, 0.0f, 0.0f, 1.0f, 1.0f);
    {
        /* 用逐字段赋值而不是复合字面量 —— 后者是 C 专有，C++ 下要 /W4 /WX 会因为
         * C4576 报错，而这个自检文件需要同时按 C 与 C++ 编译。 */
        amiki_placement pl;
        pl.origin.x = 0.0f;
        pl.origin.y = 0.0f;
        pl.u_axis.x = 1.0f;
        pl.u_axis.y = 0.0f;
        pl.v_axis.x = 0.0f;
        pl.v_axis.y = 1.0f;
        pl.repeat = 0;
        (void)amiki_canvas_draw_image_placed(cv, img, pl);
        (void)amiki_canvas_fill_path_image(cv, path, img, pl);
    }
    (void)amiki_canvas_draw_text(cv, font, "hi", 2, 0.0f, 0.0f, 12.0f, make_paint(grad));
    (void)amiki_canvas_draw_text(cv, font, "hi", 0, 0.0f, 0.0f, 12.0f, make_paint(grad));
    {
        amiki_layer_bounds b;
        b.x = 0;
        b.y = 0;
        b.width = 4;
        b.height = 4;
        (void)amiki_canvas_begin_layer(cv, b, make_filter(), 0.5f, AMIKI_BLEND_NORMAL);
        (void)amiki_canvas_end_layer(cv);
    }
    (void)amiki_canvas_set_flatten_tolerance(cv, 0.05f);
    (void)amiki_canvas_render_rgba8(cv, out, sizeof(out) / sizeof(out[0]));
    /* 异步三入口（0.10.9）与 AMIKI_ERROR_BUSY 也在自检里摸一遍：
     * 这个文件按 C 与 C++ 各编译一次，且都是 /W4 /WX。 */
    {
        int32_t ready = 0;
        (void)amiki_canvas_begin_render_rgba8(cv);
        (void)amiki_canvas_poll_render_rgba8(cv, &ready);
        if (ready) {
            (void)amiki_canvas_try_read_render_rgba8(cv, out, sizeof(out) / sizeof(out[0]));
        }
        (void)amiki_result_string(AMIKI_ERROR_BUSY);
    }
    amiki_canvas_destroy(cv);
    amiki_canvas_destroy(NULL);

    (void)amiki_path_create(&path);
    (void)amiki_path_clear(path);
    (void)amiki_path_move_to(path, 0.0f, 0.0f);
    (void)amiki_path_line_to(path, 1.0f, 0.0f);
    (void)amiki_path_quad_to(path, 0.5f, 0.5f, 1.0f, 1.0f);
    (void)amiki_path_cubic_to(path, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 0.0f);
    (void)amiki_path_close(path);
    (void)amiki_path_rect(path, 0.0f, 0.0f, 1.0f, 1.0f);
    (void)amiki_path_circle(path, 0.0f, 0.0f, 1.0f);
    amiki_path_destroy(path);
    amiki_path_destroy(NULL);

    (void)amiki_image_create_rgba8(out, 2, 2, 0, &img);
    (void)amiki_image_size(img, &w, &h);
    amiki_image_destroy(img);
    amiki_image_destroy(NULL);

    (void)amiki_font_create_from_file("x.ttf", &font);
    (void)amiki_font_create_from_memory(out, sizeof(out), &font);
    (void)amiki_font_metrics(font, &ascent, &descent, &gap);
    (void)amiki_font_metrics(font, NULL, NULL, NULL);
    amiki_font_destroy(font);
    amiki_font_destroy(NULL);

    (void)amiki_gradient_create_linear(0.0f, 0.0f, 1.0f, 1.0f, stops, 2, &grad);
    (void)amiki_gradient_create_radial(0.0f, 0.0f, 4.0f, stops, 2, &grad);
    (void)amiki_gradient_create_sweep(0.0f, 0.0f, 0.0f, stops, 2, &grad);
    amiki_gradient_destroy(grad);
    amiki_gradient_destroy(NULL);

    (void)amiki_stroke_style_default();
    (void)amiki_color_rgba(0.0f, 0.0f, 0.0f, 0.0f);
}

int main(void) {
    /* 只做编译期检查；main 存在是为了让它成为一个完整 TU。 */
    if (AMIKI_VERSION != amiki_version()) {
        return 1;
    }
    return 0;
}
