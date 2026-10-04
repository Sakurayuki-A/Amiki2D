# API 稳定性

本文件是 0.3.0 的 **`pub` 面清单与稳定承诺**。
发布判据里有一条「API 稳定性：未声明哪些是稳定接口」—— 这份文件就是为了让这句话不再成立。

## 一、承诺的三个等级

| 等级 | 含义 | 破坏性改动的代价 |
|---|---|---|
| **稳定（Stable）** | 在 0.3.x 系列内保证不破坏 | 需要主版本号提升 |
| **实验（Experimental）** | 可用，但可能在次版本里改 | 会写进 CHANGELOG |
| **内部（Internal）** | 不承诺；用 `#[doc(hidden)]` 或不在文档里出现 | 随时可改 |

**0.3.0 是 alpha**，所以「稳定」的含义是：**在 0.3.x 内不破坏**，
而不是"再也不变"。1.0 才会给出后者。

## 二、稳定面

### 门面 crate：`amiki`

这是**唯一**推荐给使用者的入口。其余 crate 都通过它再导出。

| 项 | 说明 |
|---|---|
| `Canvas` | 立即模式画布 |
| `Canvas::new(width, height)` | |
| `Canvas::fill_path` / `fill_path_with` | 填充 |
| `Canvas::stroke_path` / `stroke_path_with` | 描边（P0-1） |
| `Canvas::image` / `image_with_placement` | 位图（M6） |
| `Canvas::push_layer_filtered` | 离屏层 + 滤镜（M6/P0-2/P0-3） |
| `Canvas::rect` / `circle` / `pentagram` | 便捷图形 |
| `Canvas::text` | 文字（M5） |
| `Canvas::text_with_stack` | 文字 + **字体回退**（0.7.0） |
| `Canvas::set_transform` / `translate` / `rotate` / `scale` / `reset` | 变换 |
| `Canvas::set_blend` | 混合模式 |
| `Canvas::set_fill_rule` / `fill_rule` | **填充规则**：`NonZero`（默认）/ `EvenOdd`（0.8.0，状态式） |
| `Canvas::push_clip` / `pop_clip` | 裁剪 |
| `Canvas::scene` | 产出场景（字段） |
| `amiki::color` | 调色板常量 |
| `render_to_rgba(&Canvas, f32) -> Result<Vec<u8>, gpu::RenderError>` | CPU 渲染为 RGBA8 |

### 再导出的模块

`amiki::{gpu, math, raster, scene}` 是**再导出**，方便使用 `amiki::scene::Path`
这类路径。它们内部的**稳定子集**：

| 路径 | 稳定项 |
|---|---|
| `amiki::math` | `Vec2`、`Affine`、`QuadBez`、`CubicBez` |
| `amiki::scene` | `Scene`、`Draw`、`Path`、`PathBuilder`、`PathEl`、`Paint`、`Color`、`Gradient`、`Image`、`ImagePlacement`、`BlendMode`、`Layer`、`LayerBounds`、`Filter`、`StrokeStyle`、`LineCap`、`LineJoin`、`TextRun`、`PositionedGlyph` |
| `amiki::raster` | `render`、`bin`、`BinnedScene`（CPU 引擎入口） |
| `amiki::gpu` | `GpuRenderer`、`RenderError`、`RenderPath`、`Frame`、`Blitter`、`GpuScene`、`build_gpu_scene` |

### 文本与排版 crate

`amiki-text` 与 `amiki-layout` 是**可选的**（不依赖 GPU，也不被 `amiki` 强制引入）：

| 项 | 说明 |
|---|---|
| `amiki_text::Font`、`from_file` | 字体加载 |
| `amiki_text::shape`、`shape_clustered` | shaping（含字距与连字） |
| `amiki_text::ShapedGlyph` | 带 `cluster` 的字形（排版对齐用） |
| `amiki_layout::layout`、`LayoutOptions`、`LayoutResult`、`Line`、`Align` | 断行与排版 |
| `amiki_layout::is_no_line_start` / `is_no_line_end` | 禁则判定（文本编辑器等也会用到） |
| **0.7.0：`amiki_text::FontStack`**、`MissingGlyphPolicy`、`TextOptions`、`shape_with_stack`、`Font::id`、`FontId` | **字体回退**：有序字体栈按序解析，一段文本里多字体；缺字策略显式可选（默认报错） |
| **0.7.0：`amiki_layout::layout_with_stack`**、`layout_with_stack_policy` | 混排排版：断行/禁则与单字体路径**完全共用**，行高取栈里第一个字体 |
| **0.7.0：`amiki_scene::expand_text_run_with`**、`TextCache::expand_with`、`TextRun::push_glyph_in`、`GlyphKey::with_font` | 按**字体下标**取字形轮廓；缓存键带字体内部 ID |
| **0.7.0：`amiki::Canvas::text_with_stack`** | 门面侧的字体回退绘制入口 |

## 三、实验面

**能跑、有测试，但不承诺兼容。** 破坏时会写进 CHANGELOG。

| 项 | 为什么是实验 |
|---|---|
| `amiki_gpu::profile`（`FrameProfile` / `PassTimings` / `TimestampRecorder`） | 基准与分析用，字段会随测量方法调整 |
| `amiki_gpu::check_sparse_consistency`、`PipelineDump` | 自洽检查是**诊断设施**，随管线结构变化 |
| `SparseCapacity` / `SceneKey` / `SparseCapacityTable` 及其访问器 | 投机分配的内部机制，可能整体替换 |
| `amiki_gpu::dense_grid_shape` / `dense_capacity` | 稠密网格上限估算，仅基准与诊断用 |
| `amiki::gpu::{blur, layer, dense, sparse, readback, extract, consistency, blit}` | 模块对测试与基准开放；内部结构不承诺 |

> 这些之所以仍然 `pub`，是因为**测试与基准需要它们**。
> 用 `#[doc(hidden)]` 藏起来会让"为什么测试能用而我不能用"变得难以解释；
> 明确标成实验级更诚实。

## 四、内部面

不承诺、也不出现在文档里的部分：

- 所有 `pub(crate)` 项
- `#[doc(hidden)]` 标记的项
- 各 crate 的 `*_WGSL` 常量、`PAD16` 之类的实现细节
- `amiki-gpu/src/*.wgsl` 的具体结构与绑定号

## 五、已知的使用限制（不是 API，但同样要声明）

| 限制 | 出处 |
|---|---|
| **GL 后端不支持位图** | 引擎**显式报错**（该后端读不出 `texture_2d_array`），见 RELEASE_PLAN §5 |
| **DX12 冷启动约 2.3 s** | FXC 运行期编译着色器；可用 `AMIKI_BACKENDS` 指定后端 |
| **并发创建 context 在启用 Vulkan 校验层时会崩** | 校验层在 debug 构建下由 wgpu 默认启用；**release 不受影响**。统一按"串行创建、并行使用"写。见 `KNOWN_ISSUES.md` **B14** 与 `amiki.h` 的「线程模型」 |
| **引擎不做字体回退（无 font stack）** | ~~`shape` / `layout` 每次只吃一个 `Font`：一段文本只能用一种字体~~ —— **0.7.0 已关闭**：新增 `FontStack` / `shape_with_stack` / `layout_with_stack` / `Canvas::text_with_stack`（**新入口**，老入口签名与行为不变）。见下方稳定面与 `KNOWN_ISSUES.md` **B15** |
| **不做的能力** | 双向文本、Knuth-Plass、宽色域、各向异性过滤、动画系统 —— 见 RELEASE_PLAN §6（HDR 与 mipmap 已于 0.5.0 / 0.6.0 落地） |

## 六、环境变量

| 变量 | 作用 |
|---|---|
| `AMIKI_BACKENDS` | 指定 wgpu 后端（`vulkan` / `dx12` / `gl` / `all`）；默认全部，以保证"哪台机器都能起来" |

未来若增加环境变量，会先补进这张表。

## 逐帧调用请用「复用缓冲」的 API

`build_gpu_scene` / `prepare` / `profile` 每次都会**新建约 7 MB 缓冲**。
大块分配走 `mmap` + 缺页，实测同一批数据**慢 5.3 倍**。

逐帧路径应当持有并复用一份缓冲：

| 一次性调用 | 逐帧应当用 |
|---|---|
| `build_gpu_scene(scene, eps)` | `build_gpu_scene_into(scene, eps, &mut gs)` |
| `GpuRenderer::prepare(scene, eps)` | `GpuRenderer::prepare_into(scene, eps, &mut frame)` |
| `GpuRenderer::profile(scene, eps)` | `GpuRenderer::profile_into(scene, eps, &mut frame)` |

```rust
let mut frame = amiki_gpu::Frame::default();
loop {
    renderer.prepare_into(&scene, eps, &mut frame)?;
    renderer.render_into(&tex, &frame)?;
}
```

`prepare` / `profile` 保留为兼容包装 —— 一次性调用或测试里用它们没问题。
