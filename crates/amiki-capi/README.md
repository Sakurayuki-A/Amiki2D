# amiki-capi —— Amiki2D 的 C ABI

**契约是手写的 [`include/amiki.h`](include/amiki.h)**，本 crate 去适配它。
不由 `cbindgen` 之类工具生成：那样 header 会随 Rust 类型变动而漂移，
绑定作者拿不到稳定界面。Skia / Blend2D / wgpu-native 都是手写 header。

**完整的使用契约（所有权、错误模型、线程模型、版本承诺）都在 header 顶部的注释里** ——
那是唯一权威来源。本文件只讲"怎么用起来"。

---

## 30 秒上手

```bash
cargo build -p amiki-capi --release
```

产物（`target/release/`）：

| 文件 | 用途 |
|---|---|
| `amiki2d.dll` / `libamiki2d.so` / `libamiki2d.dylib` | 动态库（**推荐**，运行期只需要它） |
| `amiki2d.dll.lib` / `libamiki2d.so` | 对应的导入库 |
| `amiki2d.lib` / `libamiki2d.a` | 静态库（从 C 链接需要 Rust 的全部系统依赖，较麻烦） |
| `libamiki2d.rlib` | 给本仓库的 Rust 测试用 |

```c
#include "amiki.h"

amiki_context *ctx = NULL;
amiki_context_create(AMIKI_BACKEND_AUTO, &ctx);

amiki_canvas *cv = NULL;
amiki_canvas_create(ctx, 800, 600, &cv);
amiki_canvas_draw_rect(cv, 40, 40, 400, 300,
                       amiki_paint_solid(amiki_color_rgb8(231, 76, 60)));

size_t n = 800 * 600 * 4;
uint8_t *px = malloc(n);
amiki_canvas_render_rgba8(cv, px, n);      /* RGBA8、非预乘、行序自上而下 */

free(px);
amiki_canvas_destroy(cv);
amiki_context_destroy(ctx);
```

**字体回退**（0.7.0）：一段文本里不同字符可以由不同字体渲染。
调用方给一个**有序**字体栈，引擎按序取第一个含该字符的字体：

```c
amiki_font *latin = NULL, *cjk = NULL;
amiki_font_create_from_file("assets/fonts/Inter-Regular.ttf", &latin);
amiki_font_create_from_file("assets/fonts/NotoSansSC-Regular.ttf", &cjk);

amiki_font_stack *stack = amiki_font_stack_create();
amiki_font_stack_push(stack, latin);   /* 顺序即优先级：先压的先赢 */
amiki_font_stack_push(stack, cjk);

/* options 传 NULL = 默认：缺字返回 AMIKI_ERROR_NOT_FOUND（不静默画方框） */
amiki_canvas_draw_text_with_stack(cv, stack, "你好 Amiki2D 世界", 0,
                                  40, 120, 32, amiki_paint_solid(amiki_color_rgb8(0,0,0)), NULL);

amiki_font_stack_destroy(stack);   /* 压栈是拷贝，字体句柄仍归你 */
amiki_font_destroy(latin);
amiki_font_destroy(cjk);
```

缺字想画方框或跳过？用 `amiki_text_options` 把 `missing_glyph` 设成
`AMIKI_MISSING_GLYPH_NOTDEF` / `AMIKI_MISSING_GLYPH_SKIP`
（默认值见 `amiki_text_options_default()`）。

**异步渲染**（0.10.9）：把"提交"与"读回"分到不同帧 —— 适合"持有 GL 上下文时
每帧固定 ~20 ms 同步停顿"的宿主（见 header 里 `amiki_canvas_render_rgba8` 一节）：

```c
amiki_canvas_begin_render_rgba8(cv);          /* 本帧：只提交，立即返回 */
/* ... 去做别的事 / 等下一帧 ... */

int32_t ready = 0;
amiki_canvas_poll_render_rgba8(cv, &ready);   /* 非阻塞；0 = 还没好，不是错误 */
if (ready) {
    /* 取走并释放这一帧；未就绪会返回 AMIKI_ERROR_BUSY 且不写任何字节 */
    amiki_canvas_try_read_render_rgba8(cv, px, n);
}
```

产物与同步入口 **逐字节相同**（同一套前置判断 + 同一份 CPU 实现）；
CPU 后端与"超范围回落 CPU"的情形**首帧即就绪**。

编译（Windows）：
```
cl /utf-8 /W4 your_app.c /link amiki2d.dll.lib
```
编译（Linux / macOS）：
```
cc your_app.c -I<repo>/crates/amiki-capi/include -L<repo>/target/release -lamiki2d
```

> ⚠️ **MSVC 用户**：`amiki.h` 是 UTF-8 无 BOM，在非 UTF-8 代码页的系统上必须加
> `/utf-8`，否则报 `warning C4819`（开了 `/WX` 就是错误）。

---

## 三条硬保证

写在这里是因为它们是**设计约束**，不是实现细节：

1. **绝不 panic 跨 FFI 边界。** 每个导出函数都在 `catch_unwind` 之内；
   跨 FFI 的 panic 是未定义行为，不能让 C 调用方承担。

2. **panic 之后对象中毒。** 出事的对象后续调用一律返回 `AMIKI_ERROR_PANIC`。
   因为 panic 可能打断在一次修改的中途，继续用它**可能静默画出错误结果** ——
   而"不静默出错"是本项目最硬的一条取舍。

3. **失败时不写出参。** 每个 `_create` 在失败时都不碰 `*out`。

---

## 性能上的一个关键点：**复用 context**

`amiki_context` **持有一个渲染后端**，创建一次即可跨帧复用。

这不是可选的优化：创建 GPU 后端要初始化驱动（实测冷启动 **61 ms**），
逐帧新建会让它成为主要开销。同理 `amiki_canvas_clear` 复用内部缓冲，
**不要**为了清屏而 `destroy` + `create`。

逐帧渲染的正确形态：

```c
amiki_context *ctx = NULL;
amiki_context_create(AMIKI_BACKEND_AUTO, &ctx);   /* 一次 */
amiki_canvas *cv = NULL;
amiki_canvas_create(ctx, W, H, &cv);              /* 一次 */

for (;;) {
    amiki_canvas_clear(cv);                        /* 复用缓冲，不是重建 */
    /* ... 绘制本帧 ... */
    amiki_canvas_render_rgba8(cv, pixels, n);
}
```

---

## 三个绑定示例

[`examples/`](examples/) 下有 **C / C# / Node** 三个能跑的示例，
它们做**完全相同的绘制**，因此输出**逐像素一致** —— 那是"ABI 封送正确"的最强证据。

```
C     vs C#    最大通道差 =   0  不同像素 = 0
C     vs Node  最大通道差 =   0  不同像素 = 0
C#    vs Node  最大通道差 =   0  不同像素 = 0
```

选型与手工构建见 [`examples/README.md`](examples/README.md)。

---

## 测试

```bash
cargo test -p amiki-capi
```

`src/tests.rs` 是 **ABI 层自测**（29 项），**从 C 侧的角度**验证契约：

- 版本号两处一致（编译期宏 ↔ 运行期函数）
- **每个**接收指针的入口都拒绝 NULL，且不崩（空指针解引用会段错误，所以这项的价值在"跑完不崩"）
- 失败时 out 参数**不被写入**
- 越界枚举 / NaN / Inf 都被拒绝
- 中毒机制真的挡住后续调用
- **位图与渐变可以在绘制后立刻销毁**（header 所有权第 4 条的实测）
- 层没配平就渲染 → 显式拒绝（否则会静默丢掉那些绘制）
- 端到端 CPU 渲染真的画出东西
- **异步三入口**（0.10.9）：与同步入口**逐字节相同**；未就绪 ⇒ `AMIKI_ERROR_BUSY`
  且**不写一个字节、也不消费那一帧**；`out_len` 不足不消费帧；没 `begin` 就
  `poll` / `try_read` ⇒ 显式报错；`clear` 会把在飞帧一起丢掉

`src/types.rs` 还有**布局断言**：每个 `#[repr(C)]` 结构体的 `size_of` / `offset_of`
必须与 header 一致。C 侧的 `sizeof` 只要对不上，绑定就会静默踩内存。

---

## 一条命令验证全部

```
crates\amiki-capi\verify.cmd
```

它跑 **5 项检查**，每一项都对应一种"ABI 坏了"的具体表现：

| # | 检查 | 能抓到什么 |
|---|---|---|
| 1 | `cargo build -p amiki-capi --release` | 构建失败 |
| 2 | `amiki.h` 按 **C 与 C++** 各 `cl /W4 /WX` 编译 | 声明写错、类型缺失、`extern "C"` 保护失效 |
| 3 | header 里声明的每个函数**都真的被导出** | 漏了 `#[no_mangle]`、符号名对不上 |
| 4 | 三个绑定示例构建并运行 | 示例坏了、库链接不上 |
| 5 | 三个示例产出的图片**字节完全相同** | **结构体布局、调用约定、字符串编码、所有权**任何一处封送出错 |

**第 5 项是最强的一条** —— 三种完全不同的运行时做同样的绘制，
只要有一处封送错了，输出就不可能字节相同。

> **这个脚本验证过有牙**：把 C 示例背景色的一个通道从 24 改成 25，
> 第 5 项立刻失败并返回退出码 1（C 的哈希变了，C# / Node 没变）。

它检查的符号数当前是 **header 声明 90 / 导出 90，无缺失**（0.10.9 新增异步三入口
`amiki_canvas_begin/poll/try_read_render_rgba8` 与 `AMIKI_ERROR_BUSY`；0.10.0 新增图像编解码 3 个；0.9.0 新增
`amiki_canvas_save` / `amiki_canvas_restore` 与两个状态栈错误码；
0.8.0 新增填充规则枚举与 2 个符号；0.7.0 新增 6 个字体回退符号）。

手工核对（不用脚本时）：

```bash
# Windows
dumpbin /exports target/release/amiki2d.dll | findstr amiki_
# Linux / macOS
nm -D --defined-only target/release/libamiki2d.so | grep amiki_
```

---

## 已知边界（引用 `docs/KNOWN_ISSUES.md`）

C ABI **不隐藏**引擎的边界，而是把它们照实暴露：

| 编号 | 表现 |
|---|---|
| B1 | GL 后端遇到位图 → 渲染返回 `AMIKI_ERROR_NOT_SUPPORTED`，**不会静默画错** |
| B2 | 字体不支持 `.ttc` / `.woff` / `.woff2` → `AMIKI_ERROR_INVALID_ARGUMENT` |
| ~~B5~~ | ~~只有 nonzero 填充规则~~ —— **0.8.0 已关闭**：`amiki_canvas_set_fill_rule` 支持 `AMIKI_FILL_EVENODD` |
| B18 | 图像编解码另有缺项（GIF/WEBP/AVIF/… 与 JPEG 编码）—— **0.10.0 已能读 PNG/JPEG**：`amiki_image_create_from_file` / `_from_bytes` / `amiki_image_encode_png` |
| **B14** | **并发创建 context**：在**启用 Vulkan 校验层**时（debug 构建默认）会崩在 `VkLayer_khronos_validation.dll`；**release 不受影响**。统一按"**串行创建、并行使用**"写。详见 header 的「线程模型」 |
| C1 | 不支持双向文本 |
| — | 引擎不做内部同步：一个 context 必须在**同一线程**上使用 |
