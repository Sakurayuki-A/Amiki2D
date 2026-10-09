/*
 * amiki.h —— Amiki2D 的 C ABI（手写契约）
 * ============================================================================
 *
 * 这份 header 是**契约**：Rust 侧（`crates/amiki-capi`）去适配它，
 * 而不是反过来由工具从 Rust 类型生成。理由与 Skia / Blend2D / wgpu-native 相同：
 * 自动生成的 header 会随 Rust 类型变动而漂移，绑定作者拿不到稳定界面。
 *
 * 链接：库名是 `amiki2d`（`amiki2d.dll` / `libamiki2d.so` / `libamiki2d.a`），
 * 所有符号以 `amiki_` 开头。
 *
 *
 * ## MSVC 用户注意：请加 `/utf-8`
 *
 * 本文件是 **UTF-8 无 BOM**。在代码页非 936/UTF-8 的系统上，MSVC 会报
 * `warning C4819`（"该文件包含不能在当前代码页中表示的字符"），
 * 若你的工程开了 `/WX` 就会变成错误。加 `/utf-8` 即可：
 *
 *     cl /utf-8 /W4 your_app.c amiki2d.lib
 *
 * 这不是本文件独有的问题，任何带非 ASCII 注释的源码都需要它。
 *
 *
 * ## 版本与兼容承诺
 *
 * `AMIKI_VERSION_*` 是编译期常量，`amiki_version()` 是运行期值 —— 两者都要查，
 * 因为头文件与二进制可能不是同一版本编出来的。
 *
 * - **0.x**：ABI **可以**破坏（每次破坏都会改 `AMIKI_VERSION_MINOR`）
 * - **1.0 之后**：冻结，只允许在末尾追加函数与枚举值
 *
 *
 * ## 错误模型
 *
 * **所有**可能失败的函数都返回 `amiki_result`。
 * 需要返回指针的函数走 **out 参数**：
 *
 *     amiki_canvas* c = NULL;
 *     amiki_result r = amiki_canvas_create(ctx, 800, 600, &c);
 *     if (r != AMIKI_OK) { ... }        // 失败时 *out 保证不被写入
 *
 * **绝不 panic 跨 FFI 边界**：Rust 侧每次调用都在 `catch_unwind` 之内，
 * 任何 panic 都变成 `AMIKI_ERROR_PANIC`。跨 FFI 的 panic 是未定义行为，
 * 不能让调用方承担。
 *
 * ### 哪些函数「不」返回 `amiki_result`
 *
 * 68 个导出里有 42 个返回 `amiki_result`。其余 26 个**都不可能失败**，
 * 分三类 —— 这条规则是刻意的，不是遗漏：
 *
 * 1. **`_destroy`**（返回 `void`）：销毁空指针是合法空操作，没有失败态。
 * 2. **纯值构造**（`amiki_affine_*` / `amiki_color_*` / `amiki_stroke_style_*` /
 *    `amiki_filter_*` / `amiki_paint_*` / `amiki_version*` / `amiki_result_string`）：
 *    只做算术或返回静态字符串，无分配、无 I/O。
 * 3. **不写 out 参数的查询**（`amiki_context_backend`、`amiki_context_adapter_name`）：
 *    它们只**返回一个标量**，传 NULL 时返回中性值（`AMIKI_BACKEND_AUTO` /
 *    `"(null)"`），方便调用方直接 printf 而不必先判空。
 *
 * 反过来，**任何要写 out 参数的函数都返回 `amiki_result`** —— 因为"该不该写"
 * 本身就是一个必须能报告的决定（例如 `amiki_canvas_size` 拿到 NULL 画布时
 * 必须什么都不写）。
 *
 *
 * ## 内存所有权（最容易出事的地方，请逐条读完）
 *
 * 1. **谁创建谁销毁。** `amiki_context` / `amiki_canvas` / `amiki_path` /
 *    `amiki_image` / `amiki_font` / `amiki_font_stack` / `amiki_gradient`
 *    都是拥有所有权的对象，必须用**同名的** `_destroy` 释放。
 *    用 `free()` 释放它们是错的。
 *
 *    `amiki_font_stack_push` 是**拷贝**：压进栈的字体与你的 `amiki_font`
 *    句柄互不影响（字体字节在内部共享，所以这一步不复制数据）。
 *
 * 2. **`_destroy(NULL)` 是合法操作**（空操作），所以不必在调用点判空。
 *
 * 3. **绘制参数里的指针是「借用」**：`amiki_canvas_fill_path(c, path, paint)` 里的
 *    `path` 只在**这次调用期间**被读；函数返回后调用方可以立刻销毁它。
 *    引擎**不会**保留它。
 *
 * 4. **位图与渐变是引用计数的，绘制后即可销毁你自己的句柄。**
 *    `amiki_canvas_draw_image*` / `amiki_canvas_fill_path_image` 的 `image`、
 *    以及 `amiki_paint_gradient` 的 `gradient`，引擎内部持有它们的一份计数引用，
 *    所以函数返回后你就可以 `_destroy` 掉自己的句柄，画面不受影响。
 *    （场景销毁时那份引用自然释放。）
 *
 *    唯一的例外是**上下文**：`amiki_canvas` 只**借用** `amiki_context`，
 *    所以 context 必须比它创建的所有 canvas 活得久。
 *
 * 5. **输入缓冲一律被拷贝**：`amiki_image_create_rgba8` 的像素、
 *    `amiki_font_create_from_memory` 的字节、`amiki_font_create_from_file` 的文件内容，
 *    函数返回后调用方即可释放。
 *
 * 6. **输出缓冲由调用方提供**：`amiki_canvas_render_rgba8` 写入你给的
 *    `out`/`out_len`（RGBA8、行序从上到下、**非预乘** alpha），
 *    长度不足则返回 `AMIKI_ERROR_INVALID_ARGUMENT` 且不写入任何字节。
 *
 * 7. **字符串**一律 UTF-8。带 `\0` 结尾的用 `const char*`；
 *    文本绘制用 `utf8` + `utf8_len` 显式长度，因此可以含 `\0`。
 *
 * 8. **静态字符串**（`amiki_version_string` / `amiki_result_string` /
 *    `amiki_context_adapter_name`）不可释放、永远有效。
 *
 *
 * ## `AMIKI_ERROR_PANIC` 之后的约定
 *
 * panic 被捕获后，出事的那个对象（context / canvas / path …）会被标记为
 * **已中毒**：对它的后续调用一律立刻返回 `AMIKI_ERROR_PANIC`。
 *
 * 这样做的原因：panic 可能发生在一次修改的中途，对象内部状态未必自洽 ——
 * 与其继续用一个状态不明的对象（可能静默画出错误结果），不如**明确拒绝**。
 * 中毒对象只能 `_destroy`，然后重建。
 *
 *
 * ## 线程模型
 *
 * 引擎**不做内部同步**。一个 `amiki_context`（及其所有 `amiki_canvas`）
 * 必须在**同一个线程**上使用。
 *
 * ### 并行**使用**不同 context：允许
 *
 * 每个 context 由自己所在的线程使用即可，**互不干扰**。
 * 实测：16 线程各自渲染自己的 context，无异常。
 *
 * ### 并行**创建** context：不允许（见 `docs/KNOWN_ISSUES.md` B14）
 *
 * `amiki_context_create` 在 `AMIKI_BACKEND_GPU` / `AMIKI_BACKEND_AUTO` 下
 * **会创建 wgpu 设备**（`AMIKI_BACKEND_CPU` 不创建）。多个线程**同时**创建时，
 * 若 **Vulkan 校验层处于启用状态**（wgpu 在 `debug_assertions` 下**默认**启用），
 * 会崩在 `VkLayer_khronos_validation.dll`：
 *
 * ```text
 * 0xC0000005（访问违例）—— 不是 panic，没有任何可捕获的错误
 * ```
 *
 * **根因不在引擎、也不在显卡驱动，而在校验层本身**（实测 A/B，同一份代码同一负载）：
 *
 * | 条件 | 结果 |
 * |---|---|
 * | 校验层**开**（debug 默认），16 线程并发创建 | **3 / 6 崩** |
 * | 校验层**关**（`InstanceFlags::empty()`） | **0 / 6 崩** |
 * | 校验层开，4 线程并发创建 | 0 / 6 |
 * | 校验层开，串行创建 16 个 → 16 线程**并发使用** | **0 / 5 崩** |
 *
 * **release 构建默认不启用校验层，因此不受这条约束影响。** 但不要依赖构建类型 ——
 * 统一按"串行创建"写：**先在单个线程里依次建好所需数量，再分发给各线程**。
 *
 *     amiki_context* pool[N];
 *     for (int i = 0; i < N; i++) {
 *         if (amiki_context_create(AMIKI_BACKEND_AUTO, &pool[i]) != AMIKI_OK) return 1;
 *     }
 *     // 之后各线程用 pool[i]：并行使用是安全的
 *
 * > 这一条**只是约束，不是错误**：引擎无法可靠地检测"别的线程正在创建"，
 * > 因此它不会返回错误码，而是**可能直接崩掉进程**。把它当成硬性调用规则。
 *
 *
 * ## 典型用法
 *
 *     amiki_context* ctx = NULL;
 *     if (amiki_context_create(AMIKI_BACKEND_AUTO, &ctx) != AMIKI_OK) return 1;
 *
 *     amiki_canvas* cv = NULL;
 *     amiki_canvas_create(ctx, 800, 600, &cv);
 *
 *     amiki_canvas_draw_rect(cv, 40, 40, 400, 300,
 *                            amiki_paint_solid(amiki_color_rgb8(231, 76, 60)));
 *
 *     size_t n = 800 * 600 * 4;
 *     uint8_t* px = malloc(n);
 *     if (amiki_canvas_render_rgba8(cv, px, n) == AMIKI_OK) { ... 写 PNG ... }
 *
 *     free(px);
 *     amiki_canvas_destroy(cv);
 *     amiki_context_destroy(ctx);
 */

#ifndef AMIKI_H
#define AMIKI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* --------------------------------------------------------------------------
 * 导出宏（**必须最先定义**，后面所有声明都用它）
 * ------------------------------------------------------------------------ */

#if !defined(AMIKI_API)
#  if defined(_WIN32)
#    if defined(AMIKI_BUILD_SHARED)
#      define AMIKI_API __declspec(dllexport)
#    elif defined(AMIKI_USE_SHARED)
#      define AMIKI_API __declspec(dllimport)
#    else
#      define AMIKI_API   /* 静态库：无需修饰 */
#    endif
#  else
#    define AMIKI_API __attribute__((visibility("default")))
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * 版本
 * ========================================================================== */

#define AMIKI_VERSION_MAJOR 0
#define AMIKI_VERSION_MINOR 10
#define AMIKI_VERSION_PATCH 2
/** 预发布标签（正式版为空串）。**不参与数值比较**，只用于展示。 */
#define AMIKI_VERSION_PRERELEASE "alpha"

/** 编码为 `(major << 16) | (minor << 8) | patch`。 */
#define AMIKI_VERSION_ENCODE(major, minor, patch) \
    (((uint32_t)(major) << 16) | ((uint32_t)(minor) << 8) | (uint32_t)(patch))

/** 本头文件对应的版本。 */
#define AMIKI_VERSION \
    AMIKI_VERSION_ENCODE(AMIKI_VERSION_MAJOR, AMIKI_VERSION_MINOR, AMIKI_VERSION_PATCH)

/**
 * 运行期库版本。**与 `AMIKI_VERSION` 对照**，不一致说明头文件与二进制不匹配。
 * 兼容判定（0.x）：`major` 相同、且库的 `minor` >= 头文件的 `minor`。
 */
AMIKI_API uint32_t amiki_version(void);

/**
 * 完整版本串，形如 `"0.4.0-alpha"`。**不可释放**，永远有效。
 *
 * 与 `AMIKI_VERSION` 的关系：数值宏只含 major/minor/patch（便于比较），
 * 本字符串额外带预发布标签（便于展示与上报）。两者不一致时以本函数为准 ——
 * 它反映的是**运行期库**的真实版本。
 */
AMIKI_API const char* amiki_version_string(void);

/* ==========================================================================
 * 枚举
 * ========================================================================== */

typedef enum amiki_result {
    AMIKI_OK = 0,
    /** 参数为空、越界、NaN/Inf、长度不符等 */
    AMIKI_ERROR_INVALID_ARGUMENT = 1,
    /** 内存不足 */
    AMIKI_ERROR_OUT_OF_MEMORY = 2,
    /** 请求的后端不可用（例如要求 GPU 但机器上没有可用适配器） */
    AMIKI_ERROR_BACKEND_UNAVAILABLE = 3,
    /** 该后端不支持这个能力（例如 GL 后端不支持位图） */
    AMIKI_ERROR_NOT_SUPPORTED = 4,
    /**
     * **引擎内部 panic 被捕获**。这是引擎的 bug，请连同
     * `amiki_version_string()` 一起上报。它保证不会让进程崩掉。
     */
    AMIKI_ERROR_PANIC = 5,
    AMIKI_ERROR_UNKNOWN = 6,
    /**
     * **0.7.0 新增**：找得到对象类型，但里面**没有要找的东西**。
     * 目前只有一处：`amiki_font_stack_resolve` / `amiki_canvas_draw_text_with_stack`
     * —— 字体栈里没有任何字体含该字符。
     */
    AMIKI_ERROR_NOT_FOUND = 7,
    /**
     * **0.9.0 新增**：状态栈已满。
     * `amiki_canvas_save` 在已存了 32 层时返回它；栈**不变**。
     * 出现它几乎一定是漏了 `amiki_canvas_restore`。
     */
    AMIKI_ERROR_STACK_OVERFLOW = 8,
    /**
     * **0.9.0 新增**：状态栈为空。
     * `amiki_canvas_restore` 在没有配对的 `save` 时返回它 ——
     * **不静默 no-op**，因为多 restore 一次是实实在在的 bug。
     */
    AMIKI_ERROR_STACK_UNDERFLOW = 9
} amiki_result;

/** 人类可读的说明（静态字符串，不可释放）。越界值返回 `"unknown"`。 */
AMIKI_API const char* amiki_result_string(amiki_result r);

/** 混合模式：W3C compositing 全集。取值与引擎内部 id 一一对应。 */
typedef enum amiki_blend_mode {
    AMIKI_BLEND_NORMAL = 0,
    AMIKI_BLEND_MULTIPLY = 1,
    AMIKI_BLEND_SCREEN = 2,
    AMIKI_BLEND_OVERLAY = 3,
    AMIKI_BLEND_DARKEN = 4,
    AMIKI_BLEND_LIGHTEN = 5,
    AMIKI_BLEND_COLOR_DODGE = 6,
    AMIKI_BLEND_COLOR_BURN = 7,
    AMIKI_BLEND_HARD_LIGHT = 8,
    AMIKI_BLEND_SOFT_LIGHT = 9,
    AMIKI_BLEND_DIFFERENCE = 10,
    AMIKI_BLEND_EXCLUSION = 11,
    AMIKI_BLEND_HUE = 12,
    AMIKI_BLEND_SATURATION = 13,
    AMIKI_BLEND_COLOR = 14,
    AMIKI_BLEND_LUMINOSITY = 15
} amiki_blend_mode;

typedef enum amiki_line_cap {
    AMIKI_CAP_BUTT = 0,
    AMIKI_CAP_ROUND = 1,
    AMIKI_CAP_SQUARE = 2
} amiki_line_cap;

typedef enum amiki_line_join {
    /** 斜切（引擎默认） */
    AMIKI_JOIN_BEVEL = 0,
    /** 尖角；`miter_limit` 超过则退化为斜切（与 Canvas/SVG 同义） */
    AMIKI_JOIN_MITER = 1,
    AMIKI_JOIN_ROUND = 2
} amiki_line_join;

typedef enum amiki_filter_kind {
    /** 不做滤波（只做离屏分组 / 不透明度） */
    AMIKI_FILTER_NONE = 0,
    AMIKI_FILTER_BLUR = 1,
    AMIKI_FILTER_DROP_SHADOW = 2
} amiki_filter_kind;

typedef enum amiki_paint_kind {
    AMIKI_PAINT_SOLID = 0,
    AMIKI_PAINT_LINEAR = 1,
    AMIKI_PAINT_RADIAL = 2,
    AMIKI_PAINT_SWEEP = 3
} amiki_paint_kind;

typedef enum amiki_backend {
    /** 先试 GPU，不可用则自动落到 CPU 引擎（推荐默认） */
    AMIKI_BACKEND_AUTO = 0,
    /** 必须 GPU；拿不到适配器则 `AMIKI_ERROR_BACKEND_UNAVAILABLE` */
    AMIKI_BACKEND_GPU = 1,
    /** 强制 CPU 引擎（任何机器都能跑，但没有 GPU 加速） */
    AMIKI_BACKEND_CPU = 2
} amiki_backend;

/* ==========================================================================
 * 基础值类型（POD，按值传递）
 * ========================================================================== */

typedef struct amiki_vec2 {
    float x, y;
} amiki_vec2;

/**
 * 仿射变换：
 *     x' = a*x + c*y + e
 *     y' = b*x + d*y + f
 */
typedef struct amiki_affine {
    float a, b, c, d, e, f;
} amiki_affine;

/** 颜色通道均为 `[0,1]`；**非预乘**。 */
typedef struct amiki_color {
    float r, g, b, a;
} amiki_color;

/** 层范围（设备像素，整数）。 */
typedef struct amiki_layer_bounds {
    int32_t x, y;
    uint32_t width, height;
} amiki_layer_bounds;

/* ==========================================================================
 * 文本 shaping 选项（字体回退，0.7.0）
 * ========================================================================== */

/**
 * **缺字策略**：字体栈里没有任何字体含某个字符时怎么办。
 *
 * 「缺字」指**整个栈**都不含该码点；只要有一个字体含它，就由那个字体渲染。
 */
typedef enum amiki_missing_glyph_policy {
    /**
     * **默认**：`amiki_canvas_draw_text_with_stack` 返回 `AMIKI_ERROR_NOT_FOUND`。
     *
     * 这是本项目"不支持就显式报错，不静默画错"的一贯取舍 —— 方框在不少界面里
     * "看起来像是对的"，所以这个决定必须由调用方做。
     */
    AMIKI_MISSING_GLYPH_ERROR = 0,
    /** 用 `.notdef`（字形 0）替代：画出一个方框，位置与前进量按该字体算。 */
    AMIKI_MISSING_GLYPH_NOTDEF = 1,
    /** 跳过该字符：不产生字形，前进量为 0（后续字形紧接着排）。 */
    AMIKI_MISSING_GLYPH_SKIP = 2
} amiki_missing_glyph_policy;

/**
 * shaping 选项。**按值传**（很小），默认值见 `amiki_text_options_default()`：
 *
 *     missing_glyph = AMIKI_MISSING_GLYPH_ERROR
 *     kerning       = true
 *     ligatures     = true
 *
 * `_pad` 必须为 0（保留字段，用于将来扩展；当前版本忽略它）。
 */
typedef struct amiki_text_options {
    amiki_missing_glyph_policy missing_glyph;
    bool kerning;
    bool ligatures;
    uint32_t _pad;
} amiki_text_options;

/** 上述默认值。纯函数，不分配、不会失败。 */
AMIKI_API amiki_text_options amiki_text_options_default(void);

/* ==========================================================================
 * 不透明句柄（先声明，供下面的结构体引用）
 * ========================================================================== */

typedef struct amiki_context amiki_context;
typedef struct amiki_canvas amiki_canvas;
typedef struct amiki_path amiki_path;
typedef struct amiki_image amiki_image;
typedef struct amiki_font amiki_font;
typedef struct amiki_font_stack amiki_font_stack;
typedef struct amiki_gradient amiki_gradient;

/* ==========================================================================
 * 复合值类型（POD，按值传递）
 * ========================================================================== */

/** 渐变的一个色标：位置 `t ∈ [0,1]` 处的颜色。 */
typedef struct amiki_gradient_stop {
    float t;
    amiki_color color;
} amiki_gradient_stop;

/**
 * 位图放置（**设备空间**）。
 * 图像空间 y 向下，与引擎设备坐标一致，因此不需要翻转。
 */
typedef struct amiki_placement {
    amiki_vec2 origin;
    /** `origin` 到 `(u=1, v=0)` 的向量 */
    amiki_vec2 u_axis;
    /** `origin` 到 `(u=0, v=1)` 的向量 */
    amiki_vec2 v_axis;
    /** 非 0 = 平铺 */
    int32_t repeat;
} amiki_placement;

/** 描边样式。 */
typedef struct amiki_stroke_style {
    /** 线宽（**总宽**，不是半宽），用户单位 */
    float width;
    amiki_line_cap cap;
    amiki_line_join join;
    /** 仅 `AMIKI_JOIN_MITER` 用；`<= 0` 视为 10（Canvas/SVG 默认） */
    float miter_limit;
} amiki_stroke_style;

/** 离屏层滤镜。用哪个字段由 `kind` 决定。 */
typedef struct amiki_filter {
    amiki_filter_kind kind;
    /** `AMIKI_FILTER_BLUR` / `AMIKI_FILTER_DROP_SHADOW`：标准差（设备像素） */
    float sigma;
    /** `AMIKI_FILTER_DROP_SHADOW`：偏移 */
    float dx, dy;
    /** `AMIKI_FILTER_DROP_SHADOW`：阴影颜色 */
    amiki_color color;
} amiki_filter;

/** 着色方式。用哪个字段由 `kind` 决定。 */
typedef struct amiki_paint {
    amiki_paint_kind kind;
    /** `AMIKI_PAINT_SOLID` */
    amiki_color color;
    /** `AMIKI_PAINT_LINEAR` / `RADIAL` / `SWEEP`（**借用**，见所有权第 4 条） */
    const amiki_gradient* gradient;
} amiki_paint;

/* ==========================================================================
 * 色调映射
 * ========================================================================== */

/**
 * 把 HDR 值压到显示范围的方式。
 *
 * ## 为什么需要它
 *
 * 层与滤镜路径是 **HDR** 的：`> 1.0` 的颜色穿过层边界存活，模糊也在未裁剪的
 * 数据上做（那是发光效果的前提）。但**显示设备只有 [0,1]** —— 必须有一个
 * 明确的步骤把它压下来。不指定就等于 `AMIKI_TONEMAP_RAW`（只裁剪），
 * `> 1.0` 会糊成一片白。
 *
 * ## 取值
 *
 * - `RAW`（**默认**）：只裁剪。与 Skia / Cairo / canvas 一致，也是 0.5.0 之前
 *   的行为。
 * - `REINHARD`：`x / (1 + x)`。便宜、单调，高光源保留可分辨的层次。
 * - `ACES`：Narkowicz 的 ACES 近似。观感最好，电影/游戏常用。
 *
 * ## 顺序（引擎内部做的，调用方不用管）
 *
 * ```
 * sRGB 编码值 → 解码到线性 → 施加映射 → 编码回 sRGB → 量化到 8 位
 *      RAW 跳过解码/编码（值本来就是显示空间）
 * ```
 *
 * alpha **不参与映射** —— 它是覆盖率，不是光。
 *
 * ## 取值量级（容易搞错的地方）
 *
 * 引擎的颜色值是 **sRGB 编码**的。sRGB 的 `1.5` 已经对应线性 `2.57`，
 * sRGB 的 `4.0` 对应线性 **25.3** —— 那是天文数字的光量，任何映射都会把它
 * 压到 255。所以"高光源"在 sRGB 空间里是 **1.2 ~ 2.0** 这个量级。
 *
 * 完整说明见仓库的 `docs/COLOR.md`。
 */
typedef enum amiki_tone_map {
    AMIKI_TONEMAP_RAW = 0,      /**< 只裁剪（默认） */
    AMIKI_TONEMAP_REINHARD = 1, /**< x / (1 + x) */
    AMIKI_TONEMAP_ACES = 2,     /**< ACES 近似（Narkowicz 2015） */
} amiki_tone_map;

/* ==========================================================================
 * 堆分配（WASM 必需；原生下也有用）
 * ========================================================================== */

/**
 * 分配 `size` 字节，对齐到 **16 字节**。
 *
 * ## 为什么 C ABI 需要这个
 *
 * 在 **WASM** 上，调用方（JS）没法自己分配引擎线性内存 —— 只能通过导出的
 * 分配器。没有它就无法把像素、字体字节、渐变色标这类数组传进来。
 *
 * 原生场景下也有用：用**同一个分配器**分配与释放，避免跨 CRT / 跨 `malloc`
 * 实现释放造成的未定义行为。
 *
 * ## 契约
 *
 * - 返回指针**对齐到 16 字节**；`size == 0` 或分配失败返回 NULL
 * - 必须用 `amiki_free` 释放，**且传入与分配时相同的 `size`**（重建 layout 需要）
 */
/**
 * 设置画布的色调映射方式。默认 `AMIKI_TONEMAP_RAW`。
 *
 * 只影响 `amiki_canvas_render_rgba8` 的**输出量化**，不影响绘制本身。
 * 越界的枚举值被拒绝（返回 `AMIKI_ERROR_INVALID_ARGUMENT`），画布状态不变。
 */
AMIKI_API amiki_result amiki_canvas_set_tone_map(amiki_canvas* cv, amiki_tone_map tone);

/** 读回当前设置。空画布返回 `AMIKI_TONEMAP_RAW`（不写 out 之外的东西）。 */
AMIKI_API amiki_tone_map amiki_canvas_tone_map(const amiki_canvas* cv);

/**
 * 设置画布的 mip 采样偏移（0.6.0 新增）。默认 0。
 *
 * 正值取更糊的 mip 层，负值取更锐的层；采样层级会被钳到链范围内，
 * 过大的偏移只是失效、不会出错。只影响**缩小**位图的采样，
 * 不影响绘制本身，也不影响 1:1 与放大的绘制。
 * 非有限值（NaN / inf）被拒绝，画布状态不变。
 */
AMIKI_API amiki_result amiki_canvas_set_mip_bias(amiki_canvas* cv, float bias);

AMIKI_API uint8_t* amiki_alloc(size_t size);

/** 释放 `amiki_alloc` 的返回值。传 NULL 是空操作。 */
AMIKI_API void amiki_free(uint8_t* ptr, size_t size);

/* ==========================================================================
 * 上下文（持有渲染后端，**创建一次、跨帧复用**）
 * ========================================================================== */

/**
 * 创建上下文。
 *
 * **请复用同一个 context 渲染所有帧。** 创建 GPU 后端要初始化驱动
 * （实测冷启动约 61 ms），逐帧新建会让它变成主要开销 ——
 * 这也是 `context` 与 `canvas` 分开的原因。
 *
 * 失败时 `*out` 不被写入。
 *
 * ## 后端起不来时返回什么（0.9.1 起明确）
 *
 * 请求了拿不到的后端（`AMIKI_BACKEND_GPU` 而机器上没有可用适配器、
 * 或 `AMIKI_BACKENDS` 点名的后端在这台机器上不可用）返回
 * **`AMIKI_ERROR_BACKEND_UNAVAILABLE`**，并且**绝不返回
 * `AMIKI_ERROR_PANIC`** —— 那是"引擎有 bug"，与"这台机器跑不了这个后端"
 * 是两件必须分清的事。
 *
 * 注意这条覆盖**驱动库内部 panic** 的情形：例如 GL 上下文缺 compute 能力时，
 * wgpu 会在创建 compute pipeline 时 panic（见 `docs/KNOWN_ISSUES.md` 的 **B17**），
 * 引擎把它收敛成 `BACKEND_UNAVAILABLE`，并在
 * `amiki_result_string()` 之外把驱动给的原文一并放进内部错误消息（不进 C 错误码）。
 *
 * `AMIKI_BACKEND_AUTO` 则按承诺落到 CPU（拿不到 GPU 不是错误）。
 */
AMIKI_API amiki_result amiki_context_create(amiki_backend backend, amiki_context** out);

/** 销毁上下文。**其下所有 canvas 必须先销毁。** 传 NULL 是空操作。 */
AMIKI_API void amiki_context_destroy(amiki_context* ctx);

/** 实际生效的后端（`AUTO` 下可用来查看最终选了哪个）。 */
AMIKI_API amiki_backend amiki_context_backend(const amiki_context* ctx);

/** 实际使用的适配器名（CPU 后端返回 `"CPU"`）。静态字符串，不可释放。 */
AMIKI_API const char* amiki_context_adapter_name(const amiki_context* ctx);

/* ==========================================================================
 * 画布
 * ========================================================================== */

/**
 * **填充规则**（0.8.0）。默认 `AMIKI_FILL_NONZERO`。
 *
 * 两条规则的差别只在"区域被覆盖奇数次还是非零次"：
 *
 * - `NONZERO`：绕数非零即填充（同向重叠的环**填实**）
 * - `EVENODD`：绕数为奇数才填充（同向重叠的环**互相抵消成洞**）
 *
 * 对不重叠的简单形状（方块、圆、字形轮廓）两者结果**完全相同**；
 * 差别出现在自相交路径与同向嵌套环上。
 */
typedef enum amiki_fill_rule {
    /** 非零绕数（默认） */
    AMIKI_FILL_NONZERO = 0,
    /** 奇偶（even-odd） */
    AMIKI_FILL_EVENODD = 1
} amiki_fill_rule;

/**
 * 设置**后续填充**的填充规则。状态式、粘性：设置后一直生效，直到再次设置。
 *
 * - 只作用于**填充**；描边是轮廓化，与填充规则无关。
 * - 层次：之后新开的层会继承当前规则。
 * - 越界枚举值被拒绝（`AMIKI_ERROR_INVALID_ARGUMENT`），状态不变。
 * - 不调用本函数时行为与 0.7.0 **逐字节一致**（默认 NonZero）。
 *
 * 代价（写在明处）：even-odd 的填充要在 CPU 侧把路径分解成互不重叠的梯形
 * （每次绘制都做，不缓存），因此比 NonZero 贵 —— 见 `docs/KNOWN_ISSUES.md` 的 B16。
 */
AMIKI_API amiki_result amiki_canvas_set_fill_rule(amiki_canvas* canvas, amiki_fill_rule rule);

/** 当前填充规则（传 NULL 返回 `AMIKI_FILL_NONZERO`，不崩）。 */
AMIKI_API amiki_fill_rule amiki_canvas_get_fill_rule(const amiki_canvas* canvas);

/**
 * 创建画布。`ctx` **必须比画布活得久**（画布只借用它）。
 * 失败时 `*out` 不被写入。
 */
AMIKI_API amiki_result amiki_canvas_create(amiki_context* ctx, uint32_t width, uint32_t height,
                                           amiki_canvas** out);

/** 销毁画布。传 NULL 是空操作。 */
AMIKI_API void amiki_canvas_destroy(amiki_canvas* canvas);

/**
 * 清空所有绘制内容与状态（变换、混合、裁剪栈、层栈），并**复用内部缓冲**。
 * 逐帧渲染请调这个，而不是销毁再创建。
 */
AMIKI_API amiki_result amiki_canvas_clear(amiki_canvas* canvas);

/** 画布尺寸。出参可为 NULL。 */
AMIKI_API amiki_result amiki_canvas_size(const amiki_canvas* canvas, uint32_t* out_w,
                                         uint32_t* out_h);

/* ---- 状态 ---- */

AMIKI_API amiki_result amiki_canvas_set_transform(amiki_canvas* canvas, amiki_affine t);

/** 变换重置为单位阵。 */
AMIKI_API amiki_result amiki_canvas_reset_transform(amiki_canvas* canvas);

/** 在当前变换之上再叠加平移。 */
AMIKI_API amiki_result amiki_canvas_translate(amiki_canvas* canvas, float dx, float dy);

/** 在当前变换之上叠加旋转（弧度）。 */
AMIKI_API amiki_result amiki_canvas_rotate(amiki_canvas* canvas, float radians);

/** 在当前变换之上叠加**等比**缩放。 */
AMIKI_API amiki_result amiki_canvas_scale(amiki_canvas* canvas, float s);

AMIKI_API amiki_result amiki_canvas_set_blend(amiki_canvas* canvas, amiki_blend_mode blend);

/** 压入裁剪（与已有裁剪**取交集**）。`path` 只借用。 */
AMIKI_API amiki_result amiki_canvas_push_clip(amiki_canvas* canvas, const amiki_path* path);

/** 弹出最近压入的裁剪。栈空时返回 `AMIKI_ERROR_INVALID_ARGUMENT`。 */
AMIKI_API amiki_result amiki_canvas_pop_clip(amiki_canvas* canvas);

/**
 * **保存当前状态到状态栈**（0.9.0）。
 *
 * 保存的是**四项**：变换、混合模式、裁剪栈（深度与内容）、填充规则。
 * 与 `amiki_canvas_restore` 严格配对使用，是 UI 框架的基本假设 ——
 * 每个控件在自己的局部状态里画完就收回去，不必手工记账。
 *
 * - **最大深度 32**。已满时返回 `AMIKI_ERROR_STACK_OVERFLOW`，栈**不变**
 *   （不静默丢弃，也不覆盖最旧的那层）。
 * - **与层正交**：`amiki_canvas_begin_layer` 打开的层是**独立画布**，
 *   层内的状态栈从空开始，层内的 save/restore 不影响外层计数，
 *   层被 `end_layer` 合成之后其栈即丢弃。
 * - **不进 AMK1**：栈是录制期的概念。编码时每条 draw 已带完整状态、栈已展开。
 *
 * 为什么是 32：UI 典型嵌套 5–10 层，32 有 3–5 倍余量；超过 32 说明
 * 调用方有 bug（多半漏了 restore），显式报错比静默截断好。
 */
AMIKI_API amiki_result amiki_canvas_save(amiki_canvas* canvas);

/**
 * **从状态栈恢复**（0.9.0）：四项状态**整体覆盖**为最近一次
 * `amiki_canvas_save` 时的值（不是叠加）。
 *
 * - 栈空时返回 `AMIKI_ERROR_STACK_UNDERFLOW`，**不静默 no-op**、状态不变。
 * - 恢复会**连同裁剪深度一起回退**：若 save 之后压过裁剪，restore 会把
 *   裁剪深度退回保存点，之后 `amiki_canvas_pop_clip` 仍可独立使用。
 * - 恢复**不会**撤销已经完成的层合成（层一旦 `end_layer` 就已挂进场景）。
 */
AMIKI_API amiki_result amiki_canvas_restore(amiki_canvas* canvas);

/* ---- 绘制 ---- */

/** 用任意 paint 填充路径。 */
AMIKI_API amiki_result amiki_canvas_fill_path(amiki_canvas* canvas, const amiki_path* path,
                                              amiki_paint paint);

/** 描边路径。线宽是**用户单位**，会被当前变换缩放。 */
AMIKI_API amiki_result amiki_canvas_stroke_path(amiki_canvas* canvas, const amiki_path* path,
                                                amiki_stroke_style style, amiki_paint paint);

AMIKI_API amiki_result amiki_canvas_draw_rect(amiki_canvas* canvas, float x0, float y0, float x1,
                                              float y1, amiki_paint paint);

AMIKI_API amiki_result amiki_canvas_draw_circle(amiki_canvas* canvas, float cx, float cy, float r,
                                                amiki_paint paint);

/** 五角星（内接于半径 `r` 的圆）。 */
AMIKI_API amiki_result amiki_canvas_draw_pentagram(amiki_canvas* canvas, float cx, float cy,
                                                   float r, amiki_paint paint);

/**
 * 绘制位图，放置到设备空间矩形 `[x0,x1] × [y0,y1]`（会经过当前变换）。
 *
 * ⚠️ **`image` 会被保留到渲染时**（所有权第 4 条）。
 */
AMIKI_API amiki_result amiki_canvas_draw_image(amiki_canvas* canvas, const amiki_image* image,
                                               float x0, float y0, float x1, float y1);

/** 绘制位图并指定完整放置（含旋转 / 切变 / 平铺）。同样**会保留** `image`。 */
AMIKI_API amiki_result amiki_canvas_draw_image_placed(amiki_canvas* canvas,
                                                      const amiki_image* image,
                                                      amiki_placement placement);

/** 用位图填充任意路径。同样**会保留** `image`。 */
AMIKI_API amiki_result amiki_canvas_fill_path_image(amiki_canvas* canvas, const amiki_path* path,
                                                    const amiki_image* image,
                                                    amiki_placement placement);

/**
 * 绘制一段 UTF-8 文本。**引擎内部做 shaping**；字形轮廓走与图形同一条管线，
 * 因此任意缩放都不糊。
 *
 * - `utf8` / `utf8_len`：**显式给长度**，因此可以含 `\0`；`utf8_len == 0`
 *   表示让引擎按 `\0` 求长。
 * - `x, y`：**基线原点**（第一个字形的基线起点），不是包围盒左上角。
 * - `size`：字号（像素）。
 * - 不支持双向文本（见 `docs/KNOWN_ISSUES.md` C1）。
 */
AMIKI_API amiki_result amiki_canvas_draw_text(amiki_canvas* canvas, const amiki_font* font,
                                              const char* utf8, size_t utf8_len, float x, float y,
                                              float size, amiki_paint paint);

/**
 * 绘制一段 UTF-8 文本，字形按**字体栈**做回退（0.7.0）—— 一段文本里不同字符
 * 可以由不同字体渲染（例如拉丁用 Inter、汉字用 Noto Sans SC）。
 *
 * 与 `amiki_canvas_draw_text` 的**唯一**区别是字体来源：这里给一个有序的
 * `amiki_font_stack`，引擎按顺序取**第一个**含该字符的字体。
 * 基线原点、显式长度、字形走同一管线（任意缩放不糊）三者完全相同。
 *
 * - `options` 可为 `NULL`，等价于 `amiki_text_options_default()`。
 * - 缺字（栈里没有一个字体含该字符）且策略为 `AMIKI_MISSING_GLYPH_ERROR` 时
 *   返回 `AMIKI_ERROR_NOT_FOUND`，**不画出任何东西**（画布状态不变）。
 * - `stack` 为空返回 `AMIKI_ERROR_NOT_FOUND`（没有任何字体可用）。
 * - 不支持双向文本（见 `docs/KNOWN_ISSUES.md` C1）。
 */
AMIKI_API amiki_result amiki_canvas_draw_text_with_stack(amiki_canvas* canvas,
                                                         const amiki_font_stack* stack,
                                                         const char* utf8, size_t utf8_len,
                                                         float x, float y, float size,
                                                         amiki_paint paint,
                                                         const amiki_text_options* options);

/* ---- 离屏层（模糊 / 投影 / 分组不透明度）---- */

/**
 * 开始一个离屏层。**之后的绘制进入该层**，直到配对的 `amiki_canvas_end_layer`。
 *
 * 层可嵌套；必须严格配对（`end` 多于 `begin` 返回 `AMIKI_ERROR_INVALID_ARGUMENT`）。
 *
 * `bounds` 是**设备空间**的层范围，必须由调用方给出：被模糊的内容会超出图形
 * 包围盒，引擎无法在不切掉边缘的前提下自行推导。
 *
 *     amiki_canvas_begin_layer(cv, b, amiki_filter_blur(8.0f), 1.0f, AMIKI_BLEND_NORMAL);
 *     amiki_canvas_draw_rect(cv, ...);      // 这些进入层
 *     amiki_canvas_end_layer(cv);
 *
 * ## 两条非直观行为（绑定作者最容易踩，来自嵌入式集成反馈 P5）
 *
 * 1. **层是独立画布，层内绘制不继承外层画布的变换。** 层内一律从**单位变换**
 *    开始（`bounds` 本身就是设备空间，层内直接用设备坐标即可 —— 这样少一类
 *    坐标错误）。想让层内容跟随外层的平移/缩放，必须在层内自己设一次。
 *    会继承的只有**填充规则**（`amiki_canvas_set_fill_rule` 设的那个）。
 * 2. **层统一在所有普通绘制之后合成。** 场景里的绘制与层是两批：先按顺序画完
 *    所有 `draw`，再按顺序合成所有层。因此"开一个层、画点别的、再 `end_layer`"
 *    里那些"别的"会被压到**层下面**。做 UI 的正确姿势是：**每个面板整体画进
 *    自己的层**，而不是把层当成"括号"夹住一部分绘制。
 *
 * （第 2 条是**有意**的设计：它保证"层内绘制先完成、再整层滤波"的语义，
 * 也避免层内内容被父场景后续绘制意外穿插。）
 */
AMIKI_API amiki_result amiki_canvas_begin_layer(amiki_canvas* canvas, amiki_layer_bounds bounds,
                                                amiki_filter filter, float opacity,
                                                amiki_blend_mode blend);

/** 结束最近的离屏层，按 `filter` / `opacity` / `blend` 合成回父层。 */
AMIKI_API amiki_result amiki_canvas_end_layer(amiki_canvas* canvas);

/* ---- 输出 ---- */

/**
 * 渲染为 RGBA8：写入调用方提供的 `out`（长度至少 `width * height * 4`），
 * 行序**从上到下**、alpha **非预乘**、未覆盖处为透明。
 *
 * 长度不足返回 `AMIKI_ERROR_INVALID_ARGUMENT` 且**不写入任何字节**。
 *
 * ## 与 OpenGL 宿主共存时的性能特征（嵌入式集成反馈 P3）
 *
 * 这是**同步**调用：GPU 后端会提交、等待、再把结果读回 CPU（保护性读回，
 * 见 `docs/KNOWN_ISSUES.md` 的 **B11**）。在**已经持有 OpenGL 上下文**的宿主
 * 进程里（例如 Minecraft 这类游戏），跨 API 的同步可能是主要成本 ——
 * 实测报告是每帧固定 ~20 ms 停顿，且与画布尺寸无关；独立进程里同一画布只要 ~10 ms。
 *
 * 因此：
 * - **多块小画布 + 只在内容变化时重渲染**（嵌入式 UI 的常见做法）能摊销这笔成本；
 * - 游戏内若不走 GPU 的收益不明显，直接用 **CPU 后端**（`AMIKI_BACKEND_CPU`）
 *   反而是更稳的选择 —— 它不涉及任何跨 API 同步；
 * - 目前**没有**异步接口（`begin` / `poll` / `try_read` 分离）。
 *   需要"提交与读回分到不同帧"的宿主，暂时只能在应用层错帧调度。
 */
AMIKI_API amiki_result amiki_canvas_render_rgba8(amiki_canvas* canvas, uint8_t* out,
                                                 size_t out_len);

/** 压平容差（设备像素，默认 0.1）。越小曲线越精确、越慢。 */
AMIKI_API amiki_result amiki_canvas_set_flatten_tolerance(amiki_canvas* canvas, float eps);

/* ==========================================================================
 * 路径
 * ========================================================================== */

AMIKI_API amiki_result amiki_path_create(amiki_path** out);
AMIKI_API void amiki_path_destroy(amiki_path* path);

/** 清空为一条空路径，复用内部缓冲。 */
AMIKI_API amiki_result amiki_path_clear(amiki_path* path);

AMIKI_API amiki_result amiki_path_move_to(amiki_path* path, float x, float y);
AMIKI_API amiki_result amiki_path_line_to(amiki_path* path, float x, float y);
AMIKI_API amiki_result amiki_path_quad_to(amiki_path* path, float cx, float cy, float x, float y);
AMIKI_API amiki_result amiki_path_cubic_to(amiki_path* path, float c1x, float c1y, float c2x,
                                           float c2y, float x, float y);

/** 闭合当前子路径。 */
AMIKI_API amiki_result amiki_path_close(amiki_path* path);

/** 追加一个矩形子路径。 */
AMIKI_API amiki_result amiki_path_rect(amiki_path* path, float x0, float y0, float x1, float y1);

/** 追加一个圆（四段三次曲线）。 */
AMIKI_API amiki_result amiki_path_circle(amiki_path* path, float cx, float cy, float r);

/* ==========================================================================
 * 资源
 * ========================================================================== */

/**
 * 从 RGBA8 像素创建位图。**像素被拷贝**，返回后调用方即可释放 `pixels`。
 *
 * `stride` 是每行字节数；传 0 表示 `width * 4`。
 * `stride < width * 4` 返回 `AMIKI_ERROR_INVALID_ARGUMENT`。
 */
AMIKI_API amiki_result amiki_image_create_rgba8(const uint8_t* pixels, uint32_t width,
                                                uint32_t height, size_t stride,
                                                amiki_image** out);

/**
 * **从文件加载图像**（0.10.0）。支持 **PNG** 与 **JPEG**。
 *
 * - **按内容判格式**（magic bytes），**不看扩展名** —— 扩展名是提示，不是事实。
 * - PNG 覆盖：灰度 / 灰度+α / RGB / RGBA / **索引色（调色板 + tRNS）**，
 *   位深 1/2/4/8/16（低比特按规范缩放；**16 位取高字节**，见 `docs/KNOWN_ISSUES.md` 的 B18）。
 * - JPEG 覆盖：基线 + 渐进、灰度 / RGB（CMYK 与 12 位精度**显式报错**）。
 * - 与 `amiki_image_create_rgba8` 一致：**默认生成 mip 链**（可用
 *   `amiki_image_set_mipmap_mode` 关掉）。
 * - 错误：文件读不到 ⇒ `AMIKI_ERROR_NOT_FOUND`；格式不认识 / 该格式的此项特性
 *   不支持 ⇒ `AMIKI_ERROR_NOT_SUPPORTED`；数据坏或尺寸非法 ⇒
 *   `AMIKI_ERROR_INVALID_ARGUMENT`。**绝不 panic、绝不静默画错。**
 * - 尺寸上限：总像素 64M（护住"解一个 5 万×5 万的头"这种内存放大）。
 */
AMIKI_API amiki_result amiki_image_create_from_file(const char* utf8_path, amiki_image** out);

/**
 * **从内存加载图像**（0.10.0）。与 `amiki_image_create_from_file` 同语义。
 *
 * **不复制输入缓冲**：本调用返回后 `data` 即可释放。
 */
AMIKI_API amiki_result amiki_image_create_from_bytes(const uint8_t* data, size_t len,
                                                     amiki_image** out);

/**
 * **把图像编码成 PNG**（0.10.0，RGBA8 无损）。
 *
 * - 成功时 `*out_data` 指向引擎分配的缓冲、`*out_len` 是它的长度；
 *   **调用方必须用 `amiki_free(*out_data, *out_len)` 释放**（与其它"引擎分配"
 *   的 API 同一套契约，见文件头的所有权约定）。
 * - 失败时**不写出参**。
 */
AMIKI_API amiki_result amiki_image_encode_png(const amiki_image* image, uint8_t** out_data,
                                              size_t* out_len);

AMIKI_API void amiki_image_destroy(amiki_image* image);

AMIKI_API amiki_result amiki_image_size(const amiki_image* image, uint32_t* out_w,
                                        uint32_t* out_h);

/**
 * 位图的 mipmap 模式（0.6.0 新增）。
 *
 * 大幅**缩小**位图时，只靠双线性采样会走样（摩尔纹）；mip 链在 CPU 侧
 * 预先算好各级的"缩小版"，采样时按缩小倍数取用。
 *
 * - `AMIKI_MIPMAP_BOX`（**默认**）：按 2×2 盒式滤波生成 mip 链（在线性光
 *   空间、预乘 alpha 上平均，避免黑边与偏色）。代价是约 **+1/3** 的内存。
 *   `amiki_image_create_rgba8` 创建时即已生效，无需调用本函数。
 * - `AMIKI_MIPMAP_DISABLED`：丢弃 mip 链，永远双线性采样（0.5.0 的行为），
 *   省回那 1/3 内存。
 *
 * 1:1 与放大的绘制**不受影响**（两种模式逐字节一致）。
 */
typedef enum amiki_mipmap_mode {
    AMIKI_MIPMAP_DISABLED = 0, /**< 不生成 mip 链，永远双线性 */
    AMIKI_MIPMAP_BOX = 1,      /**< 2×2 盒式滤波 mip 链（默认） */
} amiki_mipmap_mode;

/**
 * 设置位图的 mipmap 模式。
 *
 * 重建图像的内部数据：**对后续的绘制生效**，已画进画布的引用保持原样。
 * 越界的枚举值被拒绝（返回 `AMIKI_ERROR_INVALID_ARGUMENT`），状态不变。
 */
AMIKI_API amiki_result amiki_image_set_mipmap_mode(amiki_image* image,
                                                   amiki_mipmap_mode mode);

/**
 * 从文件加载字体。**字节被拷贝**，文件可立刻关闭。
 * 支持 TrueType / OpenType；**不支持** `.ttc` 集合与 `.woff` / `.woff2`。
 */
AMIKI_API amiki_result amiki_font_create_from_file(const char* utf8_path, amiki_font** out);

/** 从内存加载字体。**字节被拷贝**，返回后调用方即可释放 `data`。 */
AMIKI_API amiki_result amiki_font_create_from_memory(const uint8_t* data, size_t len,
                                                     amiki_font** out);

AMIKI_API void amiki_font_destroy(amiki_font* font);

/**
 * 字体度量（**每 em 的相对值**，乘以字号即得像素）。任一出参可为 NULL。
 */
AMIKI_API amiki_result amiki_font_metrics(const amiki_font* font, float* out_ascent,
                                          float* out_descent, float* out_line_gap);

/* --------------------------------------------------------------------------
 * 字体栈（字体回退，0.7.0）
 *
 * 用法：
 *
 *     amiki_font_stack* st = amiki_font_stack_create();
 *     amiki_font_stack_push(st, latin);      // 先压的先赢
 *     amiki_font_stack_push(st, cjk);
 *     amiki_canvas_draw_text_with_stack(cv, st, "你好 world", 0, 40, 90, 32, paint, NULL);
 *     amiki_font_stack_destroy(st);          // latin / cjk 句柄仍归你，另行销毁
 * ------------------------------------------------------------------------ */

/** 创建空字体栈。**失败返回 NULL**（内存不足或内部 panic）。 */
AMIKI_API amiki_font_stack* amiki_font_stack_create(void);

/**
 * 销毁字体栈。传 NULL 是空操作。
 *
 * **不影响**压栈用过的字体：压栈是拷贝一份廉价句柄（字体字节内部共享），
 * `amiki_font` 句柄仍归调用方，该 `amiki_font_destroy` 还得调。
 */
AMIKI_API void amiki_font_stack_destroy(amiki_font_stack* stack);

/**
 * 追加一个字体。**顺序即优先级**：靠前的字体先被选中。
 *
 * 同一个字体可以压多次（会各占一个下标 → 各有一份字形缓存键，
 * 一般没必要）。
 */
AMIKI_API amiki_result amiki_font_stack_push(amiki_font_stack* stack, const amiki_font* font);

/**
 * 查询**第一个**含该码点的字体下标（0 起）。
 *
 * - 都不含 ⇒ `AMIKI_ERROR_NOT_FOUND`（且不写出参）。
 * - `codepoint` 不是合法标量值（代理区、> U+10FFFF）⇒ `AMIKI_ERROR_INVALID_ARGUMENT`。
 * - `out_index` 可为 NULL（那样只能靠返回码判断"有没有"）。
 */
AMIKI_API amiki_result amiki_font_stack_resolve(const amiki_font_stack* stack, uint32_t codepoint,
                                                uint32_t* out_index);

/* ==========================================================================
 * 渐变（可跨多次绘制复用）
 * ========================================================================== */

/** 线性渐变。色标少于 2 个返回 `AMIKI_ERROR_INVALID_ARGUMENT`。 */
AMIKI_API amiki_result amiki_gradient_create_linear(float x0, float y0, float x1, float y1,
                                                    const amiki_gradient_stop* stops,
                                                    uint32_t stop_count,
                                                    amiki_gradient** out);

AMIKI_API amiki_result amiki_gradient_create_radial(float cx, float cy, float radius,
                                                    const amiki_gradient_stop* stops,
                                                    uint32_t stop_count,
                                                    amiki_gradient** out);

AMIKI_API amiki_result amiki_gradient_create_sweep(float cx, float cy, float start_radians,
                                                   const amiki_gradient_stop* stops,
                                                   uint32_t stop_count,
                                                   amiki_gradient** out);

AMIKI_API void amiki_gradient_destroy(amiki_gradient* gradient);

/* ==========================================================================
 * 值构造与辅助（纯函数，不分配、不会失败）
 * ========================================================================== */

AMIKI_API amiki_affine amiki_affine_identity(void);
AMIKI_API amiki_affine amiki_affine_translate(float dx, float dy);
AMIKI_API amiki_affine amiki_affine_rotate(float radians);
AMIKI_API amiki_affine amiki_affine_scale(float s);
/** `result = a ∘ b`（先 `b` 后 `a`）。 */
AMIKI_API amiki_affine amiki_affine_concat(amiki_affine a, amiki_affine b);

AMIKI_API amiki_color amiki_color_rgba(float r, float g, float b, float a);
AMIKI_API amiki_color amiki_color_rgb8(uint8_t r, uint8_t g, uint8_t b);
AMIKI_API amiki_color amiki_color_rgba8(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

AMIKI_API amiki_stroke_style amiki_stroke_style_default(void);
/** `miter_limit` 取 Canvas/SVG 的默认值 10。 */
AMIKI_API amiki_stroke_style amiki_stroke_style_new(float width, amiki_line_cap cap,
                                                    amiki_line_join join);

AMIKI_API amiki_filter amiki_filter_none(void);
AMIKI_API amiki_filter amiki_filter_blur(float sigma);
AMIKI_API amiki_filter amiki_filter_drop_shadow(float dx, float dy, float sigma,
                                                amiki_color color);

AMIKI_API amiki_paint amiki_paint_solid(amiki_color color);
/** 渐变 paint **借用** `gradient`；它必须在渲染前一直活着。 */
AMIKI_API amiki_paint amiki_paint_gradient(const amiki_gradient* gradient);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AMIKI_H */
