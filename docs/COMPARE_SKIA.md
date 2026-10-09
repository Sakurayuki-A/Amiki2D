# 对比：Skia（他们有什么，我们没有什么）

> **目的**：把 Skia 当作"当前最强的 2D 渲染库"这个基准，找出**能力差距**与**路线差异**，
> 用来回答"下一步开发什么"，而不是比输赢 —— 体量与投入不是一个量级（见 §1）。
>
> **方法（可复核）**：对着官方源码快照，**只从源码布局与公共头文件取证据**：
> `include/**`（公共 API 面）、`src/**`（实现模块）、`modules/**`（可选组件）、
> `include/effects/**`（效果目录）、`BUILD.gn` 的 `skia_enable_*`（他们自己的可选边界）。
> 每一行都给出 Skia 侧源码路径；我们这侧对应 `docs/KNOWN_ISSUES.md` 的编号或源码。
>
> **粒度限制（先说清）**：本版做到"**能力区 / API 面**"级别，**没有**逐 API 细化，
> 也**没有**做性能跑分。逐 API 细化与跑分是后续独立工作。
>
> **出处**：
> - Skia：官方 `main` 快照（zip 解包，**无 `.git` ⇒ 报不出 commit**）；
>   版本据 `RELEASE_NOTES.md` 顶部 = **Milestone 156**。
>   （对比时 pip 上可装的 `skia-python` 是 **M144** —— 源码比可装入的轮子更新。）
> - 我们：**`0.9.2-alpha`**（`git describe --tags`）。

---

## 1. 体量对照（先认清不是一个量级）

| | 文件数 | 体积 | 说明 |
|---|---|---|---|
| **Skia**（全仓） | 12 451 | 263 MB | 其中 `resources/` 46 MB、`site/` 24 MB、`tests/` 18 MB |
| Skia 的代码面（`include/`+`src/`+`modules/`） | 3 624 | 64 MB | 这才是与我们可比的部分 |
| **我们** | 8 个 crate | Rust 26.8k 行 + WGSL 1.6k 行 | 4 个核心 crate **零第三方依赖** |

Skia 的 `src/` 分区（文件数）本身就是一张能力图，值得贴在墙上：

```text
core 444 · gpu 1202 · sksl 280 · codec 94 · utils 73 · ports 70 · effects 58 · pathops 58
pdf 48 · shaders 45 · text 33 · sfnt 32 · image 25 · encode 22 · opts 12 · xml 7 · capture 7
svg 4 · xps 4 · lazy 3 · android 4
```

一眼可见的三块"重量级子系统"我们完全没有：**`gpu`（两代后端 Ganesh + Graphite）**、
**`sksl`（着色器语言 + 编译器）**、**`codec/encode`（图像编解码）**。

---

## 2. 表 A：Skia 有、我们没有

分级：**① 该补**（用户很可能会以为我们有）· **② 明确不做**（与定位冲突/已决策）· **③ 我们有等价机制**（不同选择，不算缺）

### A-1 图像编解码 —— 差距最直白的一块

| 项 | Skia 证据 | 我们 | 分级 | 说明 |
|---|---|---|---|---|
| 图像**解码**（14 种格式：PNG/JPEG/WEBP/GIF/BMP/ICO/WBMP/**AVIF/HEIF/JPEGXL**/DNG/PKM/KTX/ASTC） | `include/codec/SkEncodedImageFormat.h`、`src/codec`（94 文件） | **0 种**（引擎只接受调用方给的 RGBA 像素） | **①** | 现在"把 PNG 放进画面"要靠调用方自己解码（我们的示例就是用 `image` crate 现解的）—— **用户默认会以为引擎能读图** |
| 图像**编码**（写出 PNG/JPEG/WEBP） | `include/encode`（7）、`src/encode`（22） | 0 | **①** | 同上下半段 |
| 惰性解码 / 部分解码 | `src/lazy` | 无 | ② | 大图/流式场景才需要 |

### A-2 效果生态 —— 数量级差距（他们 28 种图像滤镜，我们 2 种）

| 项 | Skia 证据 | 我们 | 分级 |
|---|---|---|---|
| **图像滤镜**（28 个工厂：`Arithmetic Blend Blur ColorFilter Compose Crop Dilate DisplacementMap DistantLit Diffuse/Specular DropShadow DropShadowOnly Empty Erode Image Magnifier MatrixConvolution MatrixTransform Merge Offset Picture PointLit* RuntimeShader Shader SpotLit* Tile`） | `include/effects/SkImageFilters.h`、`src/effects`（58） | **只有 2 个**：模糊、投影（且是**层滤镜**，不是可组合的滤镜图） | **①**（常见的 Crop/Offset/Tile/Merge/Compose/MatrixTransform/ColorFilter/Shader 值得补） |
| **颜色滤镜**（色彩矩阵、HSL、亮度、高对比、Overdraw） | `include/effects/SkColorMatrixFilter.h` 等 4 个头 | **无** | **①** |
| **路径效果**（虚线 `Dash`、圆角 `Corner`、离散 `Discrete`、裁剪 `Trim`、1D/2D 图案） | `include/effects/Sk{1D,2D}PathEffect.h`、`SkDashPathEffect.h` 等 6 个头 | **无**（我们只有"描边 = 轮廓化"） | **①**（**虚线是 UI 高频需求**） |
| **运行时着色器**（SkSL：一门语言 + 编译器 + 280 文件实现） | `include/effects/SkRuntimeEffect.h`、`src/sksl`（280） | 无 | **②**（"语言 + 编译器"级子系统，与"小体量、可审计"冲突；如要做，只做**受限子集**） |
| Perlin 噪声 / 着色器遮罩 / 表格遮罩 | `SkPerlinNoiseShader.h`、`SkShaderMaskFilter.h`、`SkTableMaskFilter.h` | 无 | ③（按需再加） |
| 光照（点/远/聚光、漫反射/镜面反射 6 个） | `SkImageFilters.h` 里的 `*Lit*` | 无 | ② |

### A-3 几何与绘制原语

| 项 | Skia 证据 | 我们 | 分级 |
|---|---|---|---|
| **路径布尔运算**（并/交/差/异或，`src/pathops` 58 文件） | `include/pathops/SkPathOps.h` | 无 | **①**（图标布尔、精确裁剪常用） |
| 路径测量（长度、切线、按长度取点） | `include/core/SkPathMeasure.h` | 无 | **①**（动画/文本沿路径都要它） |
| **顶点/网格/贴片/图集**（`drawVertices` `drawMesh` `drawPatch` `drawAtlas`） | `include/core/SkCanvas.h` 的 draw 入口 | 无 | **①**（粒子、自定义网格、字形图集的入口） |
| **九宫格/格栅拉伸**（`drawImageNine` `drawImageLattice`） | 同上 | 无 | **①**（UI 贴图高频） |
| 绘制入口对照（他们 33 个 `draw*`） | `SkCanvas.h` | 我们：`fill_path` / `stroke_path` / `rect` / `circle` / `pentagram` / `image` / `text` | ③ 大部分我们有等价形式（弧/椭圆=贝塞尔；点集=路径）；`drawRegion`/`drawAnnotation`/`drawDrawable` → ② |

### A-4 文本与字体

| 项 | Skia 证据 | 我们 | 分级 |
|---|---|---|---|
| **系统字体管理**（Win GDI/DirectWrite、fontconfig、Android、custom、empty…共 10+ 个 fontmgr 开关） | `GN` 的 `skia_enable_fontmgr_*`、`include/ports`（21） | **只有"从文件加载"** | **①**（"列系统字体、按名字取字体"是常见期待） |
| 字形 strike cache（线程局部队列） | `skia_enable_threadlocal_strikecache` | 引擎有 `TextCache`（`amiki-scene`），但**门面与 C ABI 没有接入** | **①**（低成本，见 §5） |
| 富文本/段落排版（对齐、bidi、Unicode 断行） | `modules/skparagraph`（56）、`modules/skunicode`（30） | 有断行 + CJK 禁则；**无 bidi** | 部分 ②（bidi 是我们的 `C1`）/ 部分 ①（对齐等排版属性） |
| 字体变体轴（可变字体）、字体子集化 | `include/core/SkFontArguments.h` | 需逐项核实（未验） | 待定 |

### A-5 后端与工程设施

| 项 | Skia 证据 | 我们 | 分级 |
|---|---|---|---|
| **PDF / XPS / SVG 输出后端** | `skia_enable_pdf`/`svg`、`src/pdf`（48）、`src/xps`、`modules/svg`（106） | 无 | **②**（我们是**即时模式渲染器**，不是文档生成器 —— 这是产品定位差异） |
| 两代 GPU 后端 + 平台后端 | `include/gpu/{ganesh,graphite,mtl,vk}`、`src/gpu`（1202） | 单后端抽象（wgpu：Vulkan/DX12/GL/Metal） | ③（我们靠 wgpu 覆盖多平台，代价是受 wgpu 能力约束） |
| 调试层（Vulkan/Metal/D3D12 validation、SPIR-V 校验） | `skia_enable_{vulkan,metal,direct3d}_debug_layer`、`skia_enable_spirv_validation` | 无（我们只在 `B14` 里记录了"并发创建 + Vulkan 校验层"的坑） | ③/② |
| **管线/着色器预编译**（治冷启动） | `skia_enable_precompile` | 无（我们冷启动 61 ms，且已知"无 `PipelineCache`"） | **①**（性能，见 §5） |
| **稀疏条带 + Vello 着色器**（解析覆盖率路线） | `skia_enable_sparse_strips`、`skia_enable_vello_shaders` | 我们本来就是解析覆盖率 + tile 分箱 | **③ 同思路** —— 值得注意：**他们也在往这个方向走**，说明我们的路线不孤立 |
| 色彩管理（ICC/宽色域） | `SkColorSpace.h`、`modules/skcms`（16） | HDR 有（f32 层 + 色调映射），**宽色域无** | ②（= 我们的 `C3`，已决策不做） |
| 平台层（Android/iOS/WinUWP 表面） | `src/android`、`skia_enable_winuwp` | 无 | ② |
| **Lottie 动画 / 场景图 / 富组件** | `modules/skottie`（129）、`sksg`（57）、`skplaintexteditor` | 无 | ②（是产品级附加组件，不是渲染核心） |
| YUV/422 图像、图像惰性解码 | `skia_enable_packed_422_yuv`、`src/lazy` | 无 | ② |

---

## 3. 表 B：我们有、他们没有（或明显不同）

| 项 | 我们 | Skia | 证据 |
|---|---|---|---|
| **跨语言 C ABI** | 手写 `amiki.h`（**84 符号**，所有权/线程模型/错误模型写清），C/C#/Node 三示例产物**逐字节一致** | **`include/c/` 已从 main 移除**（`sk_canvas.h` 不存在）⇒ 跨语言只能走 C++ 或各语言自建绑定 | 在快照里搜 `sk_canvas.h` = 无 |
| **CPU↔GPU 逐像素一致承诺** | ≤2/255（实测多为 0–1）；稀疏↔稠密**逐字节一致** | **不承诺** CPU 与 GPU 输出一致（后端不同即结果不同） | 我们的对拍设施 + `VERIFY.md` |
| **二进制流格式有兼容承诺** | AMK1 v1/v2：v1 只解不编、未知版本**显式报错**、`docs/AMK1.md` 写明每个 bit | `.skp`（SkPicture）**明确不保证跨版本兼容** | 我们 `docs/AMK1.md` |
| **WASM 体量** | **987.9 KB**（纯 CPU，与原生字节一致） | CanvasKit（`modules/canvaskit` 149 文件）是 **MB 级**，且需要 WebGL/WebGPU 才有意义 | 两边构建选项对照 |
| **核心零第三方依赖** | `amiki-math` / `scene` / `raster` / `layout` 四个 crate 零依赖，总量可审计（26.8k 行） | 单体 + `third_party`（51 头 / 82 文件）+ `skcms` 等 | 两边 `DEPS`/`Cargo.toml` |
| **状态栈作为一等 API** | `save`/`restore` 四项状态、深度 32、溢出/下溢**显式报错**（0.9.0） | `save/restore` 是 SkCanvas 的**计数式**栈（无溢出错误码） | `SkCanvas.h` vs 我们 `amiki.h` |

---

## 4. 值得"抄"的三件事（技术层面，不是功能）

1. **`sparse_strips` + `vello_shaders`：他们也在做解析覆盖率 / GPU 路径渲染**
   —— 这直接验证了我们 0.8.0 选"几何预处理 + 解析覆盖率"的路线不是偏门；
   值得读的是他们**怎么把这个搬到 GPU**（我们的 even-odd 预处理目前是 CPU 侧）。
2. **`skia_enable_precompile`：预编译管线/着色器治冷启动**
   —— 我们冷启动 61 ms（21 条管线编译、无 `PipelineCache`），这是他们已解决过的问题。
3. **`threadlocal_strikecache`：字形缓存是"库内默认开启"的**
   —— 我们已经有 `TextCache`，但**没接进公共 API**（见 §5 第 2 条）。

---

## 5. 结论：最值得做的三件事（按"用户会不会以为我们有"× 成本排序）

| 序 | 做什么 | 为什么 | 成本 |
|---|---|---|---|
| **1** | **图像编解码（先 PNG + JPEG 解码）** | 表 A-1：这是唯一"用户 100% 会以为我们有、而且一用就撞墙"的能力；没有它，任何真实应用都得在外部接一个解码器 | 中（引入依赖是**决策点**：核心 crate 目前零依赖，解码应放**可选 crate**，别污染核心） |
| **2** | **把已有的 `TextCache` 接进门面与 C ABI** | 表 A-4：货已经在仓库里，只是没上架；UI 文本逐帧重复展开轮廓是纯浪费 | 小 |
| **3** | **路径效果里的"虚线"** + **颜色滤镜（矩阵）** | 表 A-2：这两项是 UI 高频，且**不需要新架构**（前者在路径展开阶段，后者在颜色解析阶段） | 小–中 |

> 排在更后面的（都真实存在，但不该现在做）：图像滤镜全家桶（28 → 我们 2）、
> 路径布尔运算、顶点/网格原语、系统字体管理、管线预编译。

## 6. 对账本的修正建议（`docs/KNOWN_ISSUES.md`）

| 建议 | 理由 |
|---|---|
| 新增 **B18 图像编解码缺失**（解码/编码都没有） | 表 A-1；"用户会以为有"的典型 |
| 新增 **B19 只能从文件加载字体**（无系统字体管理） | 表 A-4 |
| 新增 **B20 效果生态：只有模糊/投影**（无颜色滤镜、无路径效果、图像滤镜 2/28） | 表 A-2 |
| **`A2`（WASM 的 WebGPU）保持"未验证"，但补一句取舍** | CanvasKit 证明 wasm+GPU **可做**（代价是 MB 级体积）；与"WASM 定位 = 小体量纯 CPU"冲突 ⇒ 写清"要 A2 就得放弃体积优势" |
| `B16`（\|W\|≥2 饱和）**保持** | 与 Skia 无关，是我们自己的精度问题 |

---

## 附：本次用到的提取命令（可复现）

```powershell
# 能力地图
Get-ChildItem <skia>\include -Directory          # 公共 API 分区
Get-ChildItem <skia>\src -Directory              # 实现模块
Get-ChildItem <skia>\modules -Directory          # 可选组件
Get-ChildItem <skia>\include\effects -Filter *.h # 效果目录
# 关键目录
Select-String <skia>\include\codec\SkEncodedImageFormat.h -Pattern '^\s+(k\w+)\s*,'
Select-String <skia>\include\effects\SkImageFilters.h -Pattern '\bsk_sp<SkImageFilter>\s+(\w+)\s*\('
Select-String <skia>\include\core\SkCanvas.h -Pattern '^\s+(?:virtual\s+)?void\s+(draw\w+)\('
Select-String <skia>\BUILD.gn -Pattern '^\s*(skia_enable_\w+)\s*='
Get-Content <skia>\RELEASE_NOTES.md -TotalCount 12   # 版本锚点
```
