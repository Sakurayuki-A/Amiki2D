# 变更记录

本文件记录 Amiki2D 的版本演进。格式参考 Keep a Changelog，版本号用
`主.次.修订[-预发布]`。

---

---

---

---

---

---

---

---

---

## [未发布]

> 0.10.9 之后的整理，**不含行为变更**（可以安全跳过）；发下一版时并进对应小节。

### 结构：把 2 000 行级的文件拆开（5 个）

- `amiki-gpu/src/tests.rs` 2893 行 → `src/tests/` 10 个文件；`amiki-scene/src/lib.rs` 2493 行 →
  8 个类型簇模块 + `tests/`；`amiki-capi/src/api_obj.rs` 1691 行 → `api_obj/` 11 个文件；
  `amiki-capi/src/tests.rs` 2038 行 → `src/tests/` 18 个文件；
  `amiki-gpu/src/lib.rs` 2101 行 → 124 行 + 7 个模块（**1109 行的单个 `impl GpuRenderer` 消失**）
- 交付面零变化：`verify.cmd` 仍 **90 声明 / 90 导出**，三示例产物 SHA256 未变
- 实测代价：wasm 模块 **1 033.2 → 1 037.3 KB**（模块边界改变了内联与死代码删除的判定）

### CI：门禁从"人记得跑"变成"机器必跑"

- `.github/workflows/ci.yml` 两个 job：`test`（workspace 测试 + clippy/rustdoc `-D warnings` +
  capi 特性矩阵 + wasm 构建）与 `c-abi`（`verify.cmd` 五项）；步骤本体在 `tools/ci-*.cmd`，
  **本地跑的就是 CI 跑的那一份**，不会各自漂移
- 首次真跑抓到并修掉四件事：`vcvars64.bat` 的硬编码路径（改用 `vswhere.exe`）、
  runner 上只有软件适配器 WARP（性能类断言跳过、正确性断言按 1 LSB 容忍）、
  包装脚本"退出码 0 但其实没跑"的**假绿**、符号计数依赖仓库路径

### 文档：把计数与版本钉在测试上

- 新增 `docs_sync` 三条测试：符号集合（header ↔ 实现 ↔ 文档三处对齐）、
  header 版本宏 ↔ `Cargo.toml`、许可证声明 ↔ `LICENSE` 文件
- 顺手修掉一批积压漂移（80/72 → **68** 个符号（0.4.0 的真实值）、151 → **363** 测试、
  1 025.7 → **1 037.3** KB、crate 5 → 9，以及两个"当前状态"抬头还停在 0.3.0 / 0.7.0）
- 主页 `README.md` 从 199 行压到 40 行，里面每个数字都重新实测过

### 许可证

- 统一为 **MIT**。此前 `Cargo.toml` 写 `MIT OR Apache-2.0` 而仓库里只有 MIT 文本 ——
  双许可需要两份完整文本加标准声明，只有一份就该声明 MIT。

---

## [0.10.9-alpha] —— 异步渲染接进 C ABI + 4K 静默错误**真修**

### ① 异步三入口进 C ABI（P3 第 ③b 步）：符号 87 → 90

`amiki_canvas_begin_render_rgba8` / `amiki_canvas_poll_render_rgba8` /
`amiki_canvas_try_read_render_rgba8`（+ `AMIKI_ERROR_BUSY = 10`）。

把"提交"与"读回"分到不同帧 —— 嵌入式反馈里那条"持有 GL 上下文时每帧固定 ~20 ms
同步停顿"（B11）终于有了对外的错帧手段。

- 在飞帧挂在**画布**上，不是新句柄 ⇒ **不需要第 4 个 `_destroy`**，也就没有"忘了销毁"
  这种泄漏路径（取走即释放；`amiki_canvas_clear` 会把它一起丢掉）；
- `try_read` 未就绪 ⇒ `AMIKI_ERROR_BUSY`：**一个字节都不写**，且那一帧**留在原地**
  （可以继续 `poll`）；`out_len` 不足 ⇒ 拒绝且**不消费**帧；没 `begin` 就
  `poll` / `try_read` ⇒ 显式报错（**不假装已就绪**）；
- **重复 `begin` 允许**：旧的未读帧被丢弃 ⇒ "不想等了"不需要再来一个 cancel 接口；
- 与同步入口**逐字节相同**（共用前置判断与同一份 CPU 实现），CPU 分支首帧即就绪。

门面侧同步补齐两块：`PendingFrame::take_ready(&mut self)`（与门面的
`PendingRender::take_ready`）—— "取不到不算数"，C ABI 的三入口正是建在它上面；
以及 `PendingRender::Cpu(Option<Vec<u8>>)`：取走后再取得 `None`，
而不是一张**零字节的"合法空图"**。

### ② 修：`PendingFrame::try_read` 会在**未映射**的缓冲上取像素

`finish_read_f32` 直接 `get_mapped_range()`，而 map 回调还没跑过时，那正是 wgpu
明确会 **panic** 的用法。现在就绪判据是 `PendingRead::map_outcome`，并且回调结果
**只读不取**：

> 回调跑完之后 `tx` 会被析构 ⇒ 同一个通道**再** `try_recv` 得到的是 `Disconnected`。
> 第一版把结果 `take` 走了，于是 `poll` 报 Ready 之后 `try_read` 反而报"通道关闭" ——
> 是 C ABI 的测试把它抓出来的。

### ③ 真修 4K 静默错误：真因是 1D 派发拆二维时**宿主与内核的行距不一致**

`0.10.8` 把现象记成"GPU 在 `y ≥ 2048` 会算错"并加了止血。这一版**查明真因并修掉**：

```text
宿主 dispatch_1d：x = min(wg, 65535)，y = ⌈wg/65535⌉     ← 按 65535 个【工作组】推进
内核 lin(gid)   ：gid.x + gid.y * 65535u                 ← 按 65535 个【元素】推进
```

两者差 64 倍（一个工作组 64 线程）⇒ 行与行**重叠 95%**（同一个元素被反复处理：
对 `atomicAdd` 的内核就是**重复计数**），而真正超出的尾部**永远不会被处理**。

复现形状（**与 `y` 无关**）：256×2160 × 2000 个矩形 ⇒ `draws × tile 行 = 270 135
> 262 140`，carry 需要 `135×2001×16 = 4 322 160` 个元素 = 67 534 个工作组 ⇒
拆成 `(65535, 2)`：最大元素下标只到 4 259 774，**尾部 62 385 个元素 = 3899 个 strip
从未执行**。strip 按 **tile 行主序**编号 ⇒ 最下面两行瓦片的 `cell_meta` 一直是 0 ⇒
fine 阶段在那两行**什么都不画**（实测最差像素差 255、`gpu=0`，即整行透明）。
`0.10.8` 的"`y ≥ 2048`"只是 256px 探针分块把这条带子框在了那一块。

**修法**：行距统一为 `1024 个工作组 × 64 线程 = 65536 个元素`（`dispatch_dims_1d`），
并把 `strips.wgsl` / `flatten.wgsl` 里**只读 `gid.x`** 的索引也改成同一个 `lin()`。

> 后半句是修的时候**踩出来**的：只改宿主的话，`strip_clear` 只清了前 65 536 个 strip，
> 其余 `strip_meta` 保持 0 ⇒ `measure` 把 20 万个空 strip 当成"活跃"，
> 画面是对的、结构全错。**凡经 `dispatch_1d` 派发的内核都必须按同一个行距解码**。

**顺带的收益**（同一测量，3840×2160 / 2000 矩形，release，各含提取与读回）：

| | CPU | GPU（修前） | GPU（修后） |
|---|---|---|---|
| 耗时 | 371 ms | 435 ms | **313 ms** |
| 最大通道差 | — | **255**（最下两行空白） | **0** |

### ④ 删除止血：`GPU_MAX_SUPPORTED_SIDE` 与"超 2048 回落 CPU"

真因既已修掉，那条限制就没有存在理由了 —— **删掉**（`0.10.8` 引入的公开常量一并移除，
0.x 允许破坏）。大画布照常走 GPU，与 CPU 的差异回到引擎一贯的承诺 **≤ 2/255**；
C ABI 侧"即使 context 是 GPU 后端也回落 CPU"的行为随之消失。

### ⑤ 回归钉子（两条，都**验过有牙**）

- `dispatch_dims_1d_covers_every_element_exactly_once`（**纯算术，不需要 GPU**）：
  任何元素都必须**恰好被处理一次**，且允许的浪费最多 2× —— 把公式换回旧的立刻红。
- `tall_canvas_with_many_draws_matches_cpu_in_the_bottom_rows`（`amiki-gpu`）与
  `tall_canvas_with_many_draws_is_correct_on_the_gpu_backend`（C ABI）：
  上面那个复现形状，判据是 **≤ 2/255 且最下面一行不是透明的**（旧 bug 时那里全透明）。

### 验证

```text
测试       361 passed / 0 failed（+2 回归钉子）
静态检查   clippy 0 告警 · cargo doc 0 告警 · header 按 C 与 C++ /W4 /WX
C ABI      header 声明 90 / 导出 90，无缺失（verify.cmd 五项全过）
零破坏     三示例字节相同，SHA256 90A8B3B8… 未变
WASM       1 033.2 KB（纯 CPU）；原生 CPU 与 wasm 产物逐字节相同
版本       Cargo.toml / header 宏 / amiki_version() 三处一致 = 0.10.9-alpha
```

---

## [0.10.8-alpha] —— GPU 路径：一个 60× 性能修复 + 一个正确性止血

### ① 修：每次渲染都在重建 GPU 设备（60×）

`render_to_rgba` 原先**每次调用**都执行 `GpuRenderer::new_blocking()` —— 一次完整的
wgpu 设备/队列/适配器初始化。实测（`tests/gpu_fixed_cost.rs`）：

```text
修前 64×64/2 矩形：之后每次 [103.1, 137.7, 98.9, 124.9, 86.6] ms     ≈ 100 ms/次
修后 64×64/2 矩形：首次 173.1（一次性），之后 [2.0, 1.3, 1.5, 1.9, 1.8] ms  ≈ 1.7 ms/次
CPU 同场景 ×6：合计 0.73 ms
```

耗时**与尺寸内容无关** ⇒ 正是嵌入式反馈里"与尺寸无关的 ~20 ms 停顿"的形状 ✓。
现改为进程级共享（`OnceLock<Mutex<GpuRenderer>>`；初始化失败也缓存，避免无 GPU 机器
每次重试 ✓；锁只在一帧渲染期间持有 ✓；中毒锁取回内部值不 panic ✓）。

**交叉点因此被改写**（同一测量，修前 → 修后）：

| 场景 | CPU | 修前 GPU | 修后 GPU |
|---|---|---|---|
| 320×180 / 20 矩形 | 1.49 | 81.8 | 3.68 |
| **320×180 / 2000 矩形** | 5.21 | 76.1 | **4.31（GPU 反超）** |
| 1280×720 / 2000 | 34.5 | 112.7 | 38.3 |
| 3840×2160 / 2000 | 302.6 | 435.7 | （见 ②，已回落 CPU） |

### ② 止血：GPU 在 `y ≥ 2048` 会**静默画错**（红线问题）

新增定位探针（`tests/consistency_probe.rs`）实测 3840×2160 / 2000 矩形：

```text
24,883,200 个通道里，194,344 个差值 > 16；最差像素差 255（cpu=255 / gpu=0）
错误全部聚集在 tile 行 8 ⇒ 设备 y ∈ [2048, 2304)
```

⇒ 2160 高的画布，GPU 会把**最下面 112 行画错** ✗ —— 这正是本项目绝不接受的一类
错误 ✓。在查清并修掉那个上限之前：**超过 2048 的画布明确回落 CPU 并打印原因** ✓，
输出**永远正确** ✓（与既有"GPU 不可用 ⇒ 回落 CPU"同一形状 ✓）。同时公开
`GPU_MAX_SUPPORTED_SIDE` 把"已验证范围"写在调用方看得见的地方 ✓。

**更正**：我一度推测上限来自"容量表 8 项 × 256 行"，读代码后**否定** ✗ ——
`CAPACITY_TABLE_MAX = 8` 是 LRU 里 key 的条数上限，与 2048 无关 ✓。真因仍待定位
（线索：`bins.wgsl` 的 `TILE = 16` ⇒ 2048/16 = 128 行瓦片 ✓，以及稠密网格在 4K 下
达 ~10⁹ 项、会切到稀疏路径 ✓）。

### ③ 新增测量工具（都可重跑）

`gpu_fixed_cost.rs`（每次调用开销）· `consistency_probe.rs`（一致性定位，含 2048 铁证）·
`backend_crossover.rs`（CPU/GPU 交叉点，`#[ignore]`）· `image_uv_convention.rs`
（图像采样约定：**texel 中心在 `(i+0.5)`**，3 条断言）· `showcase.rs`（可视化回归，
产出 `out/showcase.png`）。

---

## [0.10.7-alpha] —— 字体**样式属性**（样式匹配的前置）

**新增**：`amiki_text::FontStyle { weight, italic }` + `Font::style() / weight() / is_italic()`
—— 从 `OS/2` 表读 `usWeightClass` 与 `fsSelection` 斜体位；表缺失时兜底为常规值。
实测随包字体：Inter-Regular 与 NotoSansSC-Regular 都是 `400 / 非斜体`。

**为什么先做这一层**：**读不出样式就无从匹配**。`FontStack` 解决的是**字形回退**
（"这个字形谁有"）；"按**样式**挑字体"是另一件事，前置正是把样式读出来。

**`FontStyle::distance` 的取舍**：`|字重差| + （斜体不匹配 ? 1000 : 0）` ——
字重差上限 800 < 1000 ⇒ "斜体但略重"永远优先于"等重但正体"。3 条测试固定这条排序。

**明确不做**：合成加粗 / 合成倾斜（那要改字形轮廓，且与"任意缩放都不糊"冲突 ✓）。

**零侵入实现**：新代码放独立模块 `style.rs`，用**扩展 impl** 给 `Font` 加方法；
`lib.rs` 只在**末尾**加两行。

## [0.10.6-alpha] —— C ABI 文本绘制走**per-font 轮廓缓存**（D8 关闭）

**行为变化（快，但不改输出）**：`amiki_canvas_draw_text` 现在复用**该字体句柄自己的**
字形轮廓缓存。同一字体重复绘制同一段文本时，不再逐字形重查轮廓 + 缩放 + 分配 `Path`。
画面与之前**逐字节相同**（缓存只影响"怎么得到轮廓"，不影响轮廓本身）。

### 为什么**不需要新 API**（与原计划不同，且更好）

原计划是新增"缓存句柄 + 创建/销毁 + 绘制时传入"。读 C 侧既有约定时发现一条事实：

> 调用方传进来的就是 `const amiki_font*` —— 一个**稳定的字体身份**。

于是把 `TextCache` 放进**字体对象内部**即可：

```
缓存键 = (字体句柄, 字形 id, 字号)   ⇒ 构造上不可能把 A 字体的字形当成 B 字体的
```

这正是 D8 当初**不敢在门面里自动加缓存**的原因（门面拿不到字体身份 ⇒ 只能填
`FontId::UNKNOWN` ⇒ 不同字体同字形 id 同字号会命中同一条 ⇒ 静默画错）。
C ABI 侧本来就有字体身份 ⇒ 缓存放在字体上，风险**从构造上消失**，且零新增 API 面。
（Rust 门面侧仍保留 `Canvas::text_cached` 这个**显式**入口，行为不变。）

### 测试（3 条）

```text
重复绘制：第一轮有缺失（首次取轮廓），第二轮不再缺失且命中增加
per-font 缓存：两个字体句柄互不影响 ⇒ 不会串字体（D8 的核心担心的反面证据）
错误路径回归：font = NULL 仍返回 INVALID_ARGUMENT，未被缓存改变
```

### 账本

**D8 关闭**：C ABI 侧已有安全的 per-font 缓存；Rust 侧有显式 `text_cached`；
门面**故意不自动缓存**这一点保留（理由已写在 D8 条目里，不再是"技术债"而是"设计决定"）。
测试 337 → **340**；clippy / rustdoc 零告警。

## [0.10.5-alpha] —— 颜色矩阵滤镜**接进画布**（颜色解析阶段）

**新 API**：`Canvas::set_color_filter(Option<ColorMatrix>)` + `Canvas::color_filter()`。

变换发生在**提交时**（颜色解析阶段），所以 CPU / GPU 两侧看到的仍是普通 `Paint` ——
**没有动光栅器、没有动 WGSL**（与描边轮廓化、九宫格同一个思路）。这也让 0.10.4 那个
"类型层"真正可用。

### 覆盖范围与代价（**不留静默缺口**）

| 绘制类型 | 处理 | 代价 |
|---|---|---|
| `Paint::Solid` | 提交时变换颜色 | 无 |
| `Paint::Gradient` | 提交时变换**每个色标** | 一份渐变浅拷贝 |
| `Paint::Image` | 提交时**逐像素**变换（**源图不被修改**） | 与像素数成正比 |

图像走逐像素是为了"绝不静默不生效"；大图逐帧过滤是真实开销，请在调用侧权衡
（测试里专门断言了源图对象**没有被就地修改**）。

**有意不进 `save` / `restore` 状态栈**（状态栈仍是四项：变换 / 混合 / 裁剪 / 填充规则）——
需要"局部滤镜"就自己存取 `color_filter()` 的返回值。这一条写在文档里，而不是留着让人猜。

### 测试（5 条，逐像素）

```text
无滤镜：set_color_filter(None) 是无操作，两次渲染逐字节相同（零破坏承诺）
去色纯色块：红/绿/蓝 → 54/182/18（= Rec.709 × 255，可算的确定值）
渐变去色：红→绿的两个色标都变换 ⇒ 左暗灰 → 亮灰
图像去色：2×2 四色分别变灰；**源图 data 逐字节不变**
saturate(0) 与 greyscale 渲染一致；与 IDENTITY 不同（交叉验证）
```

测试 329 → **334**；clippy / rustdoc 零告警。

## [0.10.4-alpha] —— 颜色矩阵（**类型层**，尚未接画布）

**新增**：`amiki_scene::ColorMatrix` —— 4×5 仿射颜色变换，约定与 SVG `feColorMatrix`
/ Skia `SkColorMatrix` 一致（第 5 列是常量偏移）。带 `greyscale()`（Rec.709 权重）、
`saturate(s)`、`sepia()`、`scale()`、`offset()`、`then()`（组合，顺序语义 = 先内后外）、
`apply()`（**钳制**到 `[0,1]`，非有限输入不扩散 NaN）。

### 为什么这一版**只做类型、不接画布**（有意的）

接画布要动**两条**管线（CPU 光栅器 + GPU 着色器）—— 按交接文档第 11 条，那属于
"约定未确认、风险高一档"的改动；而本会话在九宫格上已因跳过"先测量"返工 5 次。
所以先把**可独立验证的纯函数层**落地：无渲染改动、无行为变化、零破坏。

接线时只需决定"在哪一层把滤镜应用到**已解析的颜色**上"：
- **纯色**与**渐变停靠点**可以在**提交时**应用（CPU/GPU 都不需要改）✅
- **图像**要逐像素 ⇒ 需要另说（要么在采样后、要么在解码期）⚠️
- 明确**不做**：色彩管理 / ICC / 线性空间转换（那是账本 `C3` 的范围）

### 测试（6 条）

```text
恒等：颜色与 alpha 都不变
去色：红 0.2126 / 绿 0.7152 / 蓝 0.0722（Rec.709），alpha 不变
饱和度：s=0 等价去色、s=1 等价恒等、s=2 越艳但不越界
then：先提亮后变暗 = 0.30；反序 = 0.40（组合顺序语义正确）
钳制：越界落到 [0,1]；NaN 系数不扩散
sepia 偏暖（中灰输入 r>g>b）；scale 通道乘算正确
```

（sepia 那条第一版用**纯白**当输入 ⇒ 系数饱和、`r == g == 1.0`，断言 `r>g` 自己写红 ✗；
改成中灰并把这个坑写在测试注释里。）

测试 323 → **329**；clippy / rustdoc 零告警。

## [0.10.3-alpha] —— 九宫格（nine-patch）

**新能力**：`amiki_scene::nine_patch` + `Canvas::image_nine` —— 一张小贴图铺满任意
尺寸的矩形，**四角保持源尺寸不缩放**（1 源像素 = 1 设备像素），四边单向拉伸、中间
双向拉伸。语义对齐 Skia `drawImageNine` / Android 9-patch：UI 里"圆角面板贴图适配
任意尺寸"就是靠它 —— 圆角与边框装饰放大时**不变形**。

### 实现：**绕开**未确认的坐标约定，而不是硬猜

第一版我用 `ImagePlacement` 的 "uv 子区间 → 设备矩形" 仿射拼 9 块。实测设备像素
`(1,1)` 取到源图四色交汇的混合（`[128,101,76]` ≈ 等权四色 `[125,102,80]`）。

> **⚠️ 更正（0.10.6）**：我当时把这个现象判断为"uv 约定与采样器不符"，**这是错的**。
> 后续实测（`crates/amiki/tests/image_uv_convention.rs`）证明约定是标准的
> **texel 中心在 `(i+0.5)×texel_px`**：`2 texel 铺 2 设备像素` ⇒ 每 texel 恰好
> 1px ⇒ 设备坐标 1.5 **正好落在 texel 边界**上，因此双线性必然取到四色混合。
> 采样器是对的，是我的**采样点选得巧**，以及**没先量约定就下结论**。
> 子图方案本身仍然更好（复用有"1:1 逐像素不变"保证的矩形放置路径），所以保留；
> 但"绕开未确认的约定"这个理由不成立 —— 真正该做的是**先把约定量清楚**。

⇒ 换成**取子图 + 最普通的矩形放置**：`nine_patch` 只返回「源矩形 + 目标矩形」9 组，
`image_nine` 逐个 `sub_image` 后交给既有的 `Canvas::image(rect)`。
**理由是工程性的**：矩形放置是引擎最常用、且被"1:1 逐像素不变"保证覆盖过的路径，
不涉及任何 uv 子区间约定。代价是每块复制一次像素（九宫格每块很小，可忽略）。

CPU / GPU 两侧与 AMK1 编码都**不需要知道九宫格存在** —— 与描边轮廓化同一个思路。

### 边界（显式，不静默）

| 情形 | 行为 |
|---|---|
| `insets` 超出源尺寸 | 按比例夹断 |
| `dst` 比两角之和还小 | 两角**等比压缩**（不产生负尺寸块） |
| 两侧边距之和 = 源尺寸 | 中间行列宽度为 0 ⇒ 跳过（只剩四角） |
| `dst` 退化 / 源图任一维为 0 | 返回**空**，什么都不画 |
| 子图越界 | `sub_image` 返回 `None` ⇒ 跳过该块 |

### 测试（5 条，判据是**逐区域颜色**）

前几轮我在这里连栽三次（"面积和"算错、"等于纯色"撞半 texel 混色、断言与素材规模不符），
所以这次把源图**9 个区域做成 9 种互不相同的颜色**，让"哪块画到哪、角多大"无歧义：

```text
9 区域全部就位；角保持 2px（dst 放大 12.5× 仍不拉伸）
三种尺寸（100×100 / 40×30 / 300×80）角都是 2px、四边带位置一致
边距 0：1 块整幅拉伸，中心/左上颜色正确
退化输入：源 0 / dst 0 返回空；dst 小于两角之和时压缩；边距占满时只剩四角
```

演示图：`out/nine_patch_demo.png`（同一贴图一宽一高两个面板，四角小方块尺寸不变）。
测试 318 → **323**；clippy / rustdoc 零告警。

## [0.10.2-alpha] —— 虚线（dash path effect）

**新能力**：`stroke_path_dashed` —— 虚线描边。Rust 侧两个层次都给：
`amiki_scene::dash_polyline` / `amiki_scene::stroke_path_dashed`（几何层，返回 `Option`）
与门面 `Canvas::stroke_path_dashed`。**老签名一字未动**（新能力走新入口）。

### 实现：复用既有描边，不引入新数学

```text
flatten_path → dash_polyline（按弧长切）→ stroke_polyline（每段开口）→ 合成一条 Path
```

端帽 / 连接 / miter limit 全部走**同一条**既有描边路径，只是每段是**开口**的，
所以各自带端帽 —— 与 SVG / Skia 语义一致。**没有新的覆盖率数学、没有碰 WGSL。**

### 语义（与 SVG / Skia 对齐）

| 情形 | 行为 |
|---|---|
| 模式元素个数为**奇数** | 视为把列表重复一遍（`[10]` 等价 `[10,10]`） |
| 消耗完一轮 | 从头再来 |
| **闭合路径** | 沿周长切（起点从折线首点算），首尾相接处照常成段 |
| 模式非法（空 / 全 0 / 负数 / NaN / Inf） | **显式返回 `None`、什么都不画** —— 绝不静默按实线画 |

### 实测（`crates/amiki/tests/dash.rs`，6 条测试）

| 判据 | 结果 |
|---|---|
| 直线 `[10,5]` 切 100 长 | **7 段**，首段 0→10、末段 90→100（逐点核对） |
| 奇数模式 `[10]` | 与 `[10,10]` **段数相同**（2 段） |
| 闭合正方形周长 40、`[4,4]` | **5 段** |
| 非法模式 5 种 | 全部 `None`（含门面入口） |
| 描边轮廓 | 实线 14 个元素 → 虚线 `[8,6]` **121 个**（确实断了） |

演示图：`out/dash_demo.png`（蓝实线矩形 + 橙虚线矩形 + 绿虚线竖线）。

### 账本

**B20 的"无路径效果"一项收窄**：虚线已做；仍缺 Skia 那套完整路径效果
（`Corner` / `Discrete` / `Trim` / 1D·2D 图案）与**颜色滤镜**、图像滤镜全家桶。
测试 312 → **318**；clippy / rustdoc 零告警。

## [0.10.1-alpha] —— `Canvas::text_cached`：把字形轮廓缓存做成**安全的**入口

**新 API**（Rust 门面，additive）：`Canvas::text_cached(&mut TextCache, run, origin, paint, outline)`。
UI 每帧重画同一批文字时，不必再逐字形重新查轮廓 + 缩放 + 分配 `Path`。

### 为什么不是"在门面里自动加缓存"（这是本题的**关键**）

门面 `text()` 只拿得到"按字形 id 取轮廓"的闭包，**拿不到字体身份**；
而缓存键是 `GlyphKey { font_id, 字形 id, 字号 }`。若在门面内部自建缓存，就只能把
`font_id` 填成 `FontId::UNKNOWN` ⇒ **字体 A 的字形 5** 与 **字体 B 的字形 5** 在
同字号下会命中**同一条** ⇒ **静默画错**。

所以缓存交给**调用方**持有：**谁知道字体身份，谁持有缓存**（[`KNOWN_ISSUES`](docs/KNOWN_ISSUES.md) 的 D8）。

### 实测（`crates/amiki/tests/text_cache.rs`，3 条测试）

| 判据 | 结果 |
|---|---|
| **渲染一致** | `text_cached` 与 `text` 的**最大通道差 0/255** —— 逐像素相同 |
| 缓存真的在复用 | 第一轮 `hits=17 misses=3`；第二轮 **+20 hits、0 misses** |
| 加速 | 重复展开 60 字形 × 60 次：**1.24 ms → 0.97 ms（1.3×）** |

两条**如实记录**的行为差异（都不是 bug，写在测试注释里）：
1. `TextCache` 返回的是**已按字号缩放**的轮廓 + 单位变换，而 `text()` 用
   **em 轮廓 + 缩放变换** ⇒ 设备空间几何相同，但 `Draw` 的 `path` / `transform`
   两个字段合法地不同 ⇒ **判据必须比像素，不能比 `Draw` 字段**。
   演示图在本地 `out/text_cache_demo.png`（真字体 Inter，走缓存路径）—— `out/` 不入库，所以这里不放链接。
2. 缓存键把字号量化到 **1/16 px**（`GlyphKey.size_q`）⇒ 字号不是 1/16 整数倍时，
   缓存路径的缩放最多差 1/32 的相对量（视觉不可见，但**不是逐位相同**）。

### 仍然没做的（D8 的另一半）

**C ABI 侧还没有缓存入口** —— 它需要"缓存句柄 + 字体身份"两样东西（新不透明类型 +
创建/销毁 + 绘制时传入），是独立一轮的设计。D8 因此**保持开放**。

测试 309 → **312**；clippy / rustdoc 零告警。

## [0.10.0-alpha] —— 图像编解码（可选 crate）+ 生态路径定调

**新能力**：引擎第一次能**自己读图**。新增 **`amiki-codec`**（可选 crate）：
PNG 解码 / PNG 编码 / JPEG 解码；C ABI 加 3 个符号（84 → 87）。

### 为什么是"可选 crate"，以及它在定什么调

四个核心 crate（`math` / `scene` / `raster` / `layout`）**零第三方依赖**是本项目的
立身之本（可审计、26.8k 行）。图像编解码必须带依赖，所以它单独一层：

> **核心保持零依赖；生态能力一律放独立可选 crate，用 feature 门控。**

这条规矩比"多加一个功能"重要得多 —— 后面的 `amiki-svg`（图标子集）、
`amiki-fonts`（系统字体枚举）都有了归处。**不启用时不想要的人体积与依赖树一分不涨**
（wasm 构建走 `--no-default-features`，实测编解码给 wasm 带 **+166 KB（PNG）**
到 **+283 KB（PNG+JPEG）** 量级，见下方"体积"）。

### 支持范围（首版，写死在 crate 文档里）

| 格式 | 解码 | 编码 |
|---|---|---|
| PNG（灰度 / 灰度+α / RGB / RGBA / **索引色 + tRNS**；位深 1/2/4/8/16） | ✅ | ✅（RGBA8） |
| JPEG（基线 + 渐进、灰度 / RGB） | ✅ | — |
| 其它（GIF/WEBP/AVIF/HEIF/…） | ❌ **显式报错** | — |

取舍都写在明处：**索引色必须支持**（UI 图标集里最常见，也是实测里最先暴露的缺口）；
**16 位取高字节**（视觉正确、可预期，不静默画黑）；**JPEG 不可能逐字节一致**
（不同解码器 IDCT/上采样不同，实测与 libjpeg-turbo 最大通道差 1–3）。

### C ABI（符号 84 → 87）

| 新增 | 说明 |
|---|---|
| `amiki_image_create_from_file` | 从文件加载（按 **magic** 判格式，不看扩展名） |
| `amiki_image_create_from_bytes` | 从内存加载（**不复制**输入缓冲） |
| `amiki_image_encode_png` | 编码 PNG；输出用 `amiki_alloc` 分配 ⇒ 调用方 `amiki_free(ptr, len)` |

错误映射：读不到文件 ⇒ `NOT_FOUND`；格式不认识 / 特性不支持 ⇒ `NOT_SUPPORTED`；
数据坏 / 尺寸非法 ⇒ `INVALID_ARGUMENT`。**绝不 panic、绝不静默画错**；
另有总像素 **64M** 上限护住内存放大。

### 判据（实测得出，修正了最初写错的一条）

- **PNG：逐字节一致**（无损格式，不同实现必须给出同样像素）——
  与工作区已有的 `image` crate（**独立实现**）对照。
- **JPEG：容差**（最初我写的是"逐字节"，实测证明那不可能：差 1–3）。
- 素材：自造 10 张 + **Skia 官方仓库自带素材 4 张**（含 512×512 PNG、带 EXIF 的 JPEG）。
- 测试 298 → **309**（codec crate 8 条 + C ABI 3 条）；`clippy` 与 `cargo doc` 零告警。

### 体积

- `amiki-codec` 自身依赖 **17 个包，无 `-sys` / `cc`**（纯 Rust）。
- wasm（探针口径：`opt-level=z` + LTO + `panic=abort`，**未过 `wasm-opt`**）：
  `+png` = **166.2 KB**、`+png+zune-jpeg` = **283.4 KB**。
  **引擎级实测**（`cargo build -p amiki-capi --target wasm32-unknown-unknown
--profile wasm-release --no-default-features`，`amiki2d.wasm`）：

| 配置 | 大小 |
|---|---|
| 不带编解码 | **1 025.7 KB** |
| 带 `codec` | **1 369.5 KB** |
| **增量** | **+343.9 KB（+33.5%）** |

⚠️ 两点如实说明：**`wasm-opt` 未安装**，以上是优化前的数字；**基线已不是 987.9 KB**
（那是 0.7.0 的旧值，此后 even-odd 与状态栈让模块长大到 1 025.7 KB）。
这正是 `codec` **默认关（wasm 侧）** 的原因：要读图就付 +33.5%，不读就一分不涨。

## [0.9.2-alpha] —— 层只渲染自己的区域（嵌入式集成反馈 P2）

**性能补丁**：层的子场景不再按**全画布**渲染。输出**逐字节不变**（有哈希证据），
没有改 API、没有改 AMK1、没有改覆盖率数学。

### 问题：层子场景按全画布渲染，浪费与层大小无关

`composite_layer` / `composite_layer_f32` 原来是：

```text
render_inner_f32(scene.width, scene.height, &layer.scene, …)   ← 全画布渲染整个子场景
  → 裁到 inflated_bounds → 模糊 → 合成
```

也就是说：一个 120×90 的小面板层，也要把层内所有绘制在**整幅画布**上着色一遍，
再把这 99% 的结果丢掉。**浪费 ∝ 画布面积 × 层内容**，与层多大**无关** ——
这正是嵌入式宿主观测到的"投影成本与层大小无关、固定 22–35 ms"。

### 修法：只渲 `inflated_bounds` 覆盖的 tile

新增 `render_inner_f32_region`：同一套着色数学，但只遍历区域覆盖的 tile，
缓冲也只分配区域大小（不再分配全画布 f32）。

**为什么结果必然逐字节相同**：`shade_tile` 只读三样东西 —— 本 tile 的绘制表、
本行在 tile 内的段引用、以及跨 tile 的**前缀**（`carries`）。三者都与
"别的 tile 有没有被渲过"无关（`carries` 在 `bin` 阶段就算好了）。
**这就是 M4 增量渲染能成立的前提**，这里只是把它用到层上。

### 证据

**（a）逐字节不变** —— 改动前后各跑一遍同一个探针（7 个场景，含负坐标层、
超界层、嵌套层、空层、HDR+模糊），每个场景的 FNV-1a 哈希与**合计哈希**
`04da5aabad3a0989` **完全相同**：

```text
小层+投影 e661dcc1d9d7e2f5 | 全画布+投影 8398a096142e24ac | 负坐标+投影 8398a096142e24ac
层超界+模糊 b0cd7e5ce8f64025 | 空层+投影 b0cd7e5ce8f64025 | 嵌套层 b0cd7e5ce8f64025
HDR+模糊 66aa6a50a6d8c935 | 合计 04da5aabad3a0989        （改动前后一致）
```

**（b）收益**（release，304×314，层内**重内容** = 20 个铺满画布的矩形 + 投影）：

| 场景 | 改动前等价（bounds = 全画布） | 改动后（bounds = 120×90） |
|---|---|---|
| 含投影的一层 | **5.16 ms** | **1.88 ms** |

⇒ **2.7× 提速**（每层每帧省 ~3.3 ms）。宿主那种"6 个面板各带投影"的用法，
单帧省下约 20 ms —— 与它报告的 22–35 ms 量级一致。
小层带影子（层内容很轻）时收益小得多（2.56 → 2.44 ms），因为那种情况下
浪费的主要是全画布 f32 缓冲的分配：现在**空层几乎不再有额外开销**
（根的 1.44 → 1.47 ms，改前是 1.44 → 1.61 ms）。

### 顺手结掉一条上一轮留下的疑问

0.9.1 的报告里记过一个反常：**"含层反而比不含层快 3–4 倍"**。
本轮用修正后的探针重测：**不可复现，且方向相反** ——
只有根绘制 1.44–1.50 ms、根 + 空层 1.47–1.61 ms，层只会**更贵**。
上一轮那个 0.30 ms 是**探针自己的 bug**：影子颜色用了 `Default`，
而 `Default` 是**全透明**（`Color{r:0,g:0,b:0,a:0}`）⇒ 等于没画影子。
（同一条"先验证检查器"的教训，这次出现在**测量代码**里。）

### 明确未做

- **增量/持久路径（`PersistentRenderer`）仍按全画布渲染层**：那条路径手里只有
  整幅内容，且它的契约是"增量与全量**逐字节一致**"，改用区域渲染需要它自己
  一轮对拍验证。故 `layer_buffer_from_content_f32` 保留给它用（并写明原因）。
  影响面：只有用 `PersistentRenderer` 的调用方享受不到这次提速。
- **P1**（宿主报告的 ~15 ms 固定开销）仍然开着：我方实测引擎侧不存在，
  需要宿主的复现数据（构建档位 / 空场景耗时）。

---

## [0.9.1-alpha] —— 后端起不来不再报 PANIC（嵌入式集成反馈 P4）

**补丁**：修复一个**错误码误导**的真 bug，并补齐三处绑定作者最容易踩的文档。
没有改渲染数学、没有改 API、没有改 AMK1。

### Fixed：后端初始化失败泄漏成 `AMIKI_ERROR_PANIC`

**现象**（来自嵌入式集成反馈 P4，本机已复现）：`AMIKI_BACKENDS=gl` 且进程无窗口时，
`amiki_context_create` 返回 **5（`AMIKI_ERROR_PANIC`）** —— 而 `amiki.h` 的契约是
"请求的后端不可用 ⇒ `AMIKI_ERROR_BACKEND_UNAVAILABLE`"。

**根因**：panic 不在引擎里，在 **wgpu 内部** —— GL 上下文缺 compute 能力时，
wgpu 25 在**创建 compute pipeline** 那一步 panic
（`wgpu-25.0.2/src/backend/wgpu_core.rs:1410`），而不是返回错误。
引擎的 C ABI 兜底捕获把它一律映射成 `PANIC`，于是调用方以为"引擎有 bug"。

**修复**：新增 `GpuRenderer::new_blocking()` —— 用 `catch_unwind` 把**初始化那一段**
的 panic 收敛成 `Err`（只包初始化：渲染过程中的 panic 照旧冒出，那是真 bug，
不该被这条吞掉）。C ABI 与门面共 3 个调用点改用它，于是：

```text
修复前：amiki_context_create → 5（PANIC）
修复后：amiki_context_create → 3（BACKEND_UNAVAILABLE）
```

**回归测试**：`crates/amiki-capi/tests/gl_backend_unavailable.rs`（独立集成测试进程，
因为要设进程级的 `AMIKI_BACKENDS`）。它钉的契约是：结果是 `AMIKI_OK`
（机器真能起 GL）或 `BACKEND_UNAVAILABLE`/`NOT_SUPPORTED`（起不来但说清了），
**绝不能是 `PANIC`**。修复前该测试**确实失败**（先把测试写成红的，再改代码）。

**登记**：`docs/KNOWN_ISSUES.md` 新增 **B17** —— GL 后端**需要 compute shader**，
受限上下文（无窗口进程 / 老驱动 / 部分 ANGLE）里**根本起不来**，请改用
Vulkan / DX12 / CPU。已按纪律第 9 条同步 `docs/OVERVIEW.md` 的 5.2 与图例计数。

### Docs：三处绑定作者最容易踩的说明（写进 `amiki.h`）

| 处 | 补了什么 |
|---|---|
| `amiki_context_create` | 后端起不来返回什么（含"驱动库内部 panic 也被收敛"），以及**绝不返回 PANIC** 的理由；`AUTO` 落 CPU 的承诺不变 |
| `amiki_canvas_begin_layer` | **P5.1** 层是独立画布、**不继承外层变换**（只继承填充规则）；**P5.2** 层统一在**所有**普通绘制之后合成 ⇒ 把层当"括号"夹住一部分绘制，后画的会被压到层下面；正确姿势是每个面板整体画进自己的层 |
| `amiki_canvas_render_rgba8` | **P3** 与 OpenGL 宿主共存时的同步读回特征（B11）：跨 API 同步可能是主要成本、与画布尺寸无关；多块小画布摊薄 + 游戏内优先 CPU 后端；目前**没有**异步接口 |

### 测试

297 → **298**（+1：上面那条 GL 回归）。既有测试与三示例产物**均未变**
（`verify.cmd` 的 SHA256 与 0.9.0 相同）。

### 已知未做（都留了明确的下一步，不藏）

- **P1**（宿主报告"CPU 渲染有 ~15ms 固定开销"）：我方实测**引擎侧不存在**
  （空场景 release 0.15–0.55 ms / debug 0.23–0.50 ms，且随面积线性）。
  相同内容在 **debug 下 10.3–10.9 ms**、release 下 1.45–1.74 ms ⇒ 优先怀疑
  宿主侧的构建档位或每次调用的固定工作。**需要对方的复现数据**才能继续。
- **P2**（层影子的子场景按**全画布**渲染）：设计缺陷**已确认**
  （`amiki-raster/src/lib.rs` 里先 `render_inner_f32(scene.width, scene.height, …)`
  再裁到 `inflated_bounds`），修复方向也已定（只渲染 bounds 区域，可复用 M4 的
  tile 机制；tile 之间本就独立 ⇒ 结果应逐像素不变）。**本轮未实施** ——
  它是自成一体的改动，值得单独一轮带对拍验证。
- **"含层反而更快"的反常**：实测含层时 CPU 渲染比不含层**快 3–4 倍**
  （release 1.45 ms → 0.30 ms；墨迹像素数相同 ⇒ 不是绘制被丢）。
  两条 CPU 路径的逐像素成本差异可疑，**待查**。

---

## [0.9.0-alpha] —— `save` / `restore` 状态栈

**新能力**：`Canvas::save()` / `Canvas::restore()` —— 保存与恢复
**四项状态**（变换、混合模式、裁剪栈、填充规则）。这是 0.8.0 验收 6 的原文前提，
也是 UI 框架的核心假设（每个控件在局部状态里画完就收回去，不必手工记账）。

不调用它们的代码**逐字节不变**：既有 278 条测试一条没改，`verify.cmd`
三示例 SHA256 与 0.8.0 **完全相同**（`90A8B3B8…`）。

### 语义（全部有测试钉住）

| 行为 | 定义 |
|---|---|
| `save()` | 把当前四项状态压栈。**第 33 层**返回 `StackOverflow`，栈不变（不静默丢弃） |
| `restore()` | 弹出并**整体覆盖**当前状态（不是叠加）。空栈返回 `StackUnderflow`，**不静默 no-op** |
| 深度上限 | `MAX_STACK_DEPTH = 32`（UI 典型嵌套 5–10 层，3–5 倍余量） |
| 裁剪 | `restore` 把裁剪深度退回保存点；`push_clip` / `pop_clip` 仍可**独立**使用（两者正交） |
| 层 | `amiki_canvas_begin_layer` 内的画布是**独立**画布，栈从空开始、独立计数、独立报错 |
| 层初始状态 | 子画布继承父画布的**填充规则**；变换一律从单位变换开始 |
| AMK1 | **不进编码流**。栈是录制期概念，编码时每条 draw 已带完整状态、栈已展开（格式仍 v2） |

### 为什么不做"只恢复某一项"

`restore` 是**整体覆盖**：四项一起回到保存点。支持"只恢复某一项"会让调用方
必须记住"当前哪些项被局部改过"，那正是状态栈要消灭的负担。

### C ABI（符号 82 → 84）

| 新增 | 说明 |
|---|---|
| `amiki_canvas_save` / `amiki_canvas_restore` | 与门面同语义；文档写明 32 层上限、溢出/下溢显式报错、与层正交、不进 AMK1 |
| `AMIKI_ERROR_STACK_OVERFLOW = 8` | 状态栈已满 |
| `AMIKI_ERROR_STACK_UNDERFLOW = 9` | 状态栈为空 |

### 本轮抓到并修掉的两个真 bug（都是 C ABI 的**镜像漂移**）

C 侧独立维护着两份门面状态的镜像 —— `clip_depth`（push 加 / pop 减）与
`fill_rule`（`set_fill_rule` 写）。`restore` 会把门面这两项一起回退，
但**不会**自动同步 C 侧的镜像：

1. `clip_depth` 停在旧值 ⇒ `save → push_clip → restore` 之后
   `amiki_canvas_pop_clip` 会以为栈里还有一层（测试 `canvas_restore_syncs_clip_depth`）。
2. `fill_rule` 停在旧值 ⇒ 恢复后 `amiki_canvas_get_fill_rule` 仍报 `EVENODD`，
   与**实际绘制**不一致（测试 `canvas_save_restore_roundtrip` 抓到的就是这个）。

现在 `amiki_canvas_restore` 成功后会重新读回门面的真实 `clip_depth` 与 `fill_rule`。
这类"同一状态存两份"的问题，正是状态栈该消灭的东西 —— 成因已写在代码注释里。

### 行为变化（Fixed）：门面 `push_layer_filtered` 补继承填充规则

门面的 `push_layer_filtered` **没有继承填充规则**（0.8.0 只在 C ABI 的
`begin_layer` 里继承了）⇒ "门面 + EvenOdd + 层"会静默退回 NonZero。
已补齐，并加测试 `layer_inherits_current_fill_rule`。

> ⚠️ **注意：对 0.8.0 里"门面 + EvenOdd + 层"的调用方，输出会变** ——
> 层内原本被静默画成 NonZero（例如同向嵌套环被填实），现在按调用方设定的
> EvenOdd 画（镂空）。这是**有意识的决定**，不是顺手改动：
>
> - 依据是**契约**而非偏好：`amiki.h` 的 `amiki_canvas_set_fill_rule` 已经写明
>   "层次：之后新开的层会继承当前规则"，而 0.8.0 只让 **C ABI 那侧**兑现了它，
>   门面这侧没有 ⇒ 同一句契约在两条入口上行为不一致。
> - 两个选择都不"免费"：保持静默 NonZero 会让文档继续骗人；补齐则让一小撮
>   0.8.0 调用方的层内输出变化。选择补齐，因为**"契约说 A、实现做 B"是更坏的
>   状态**（调用方无法通过阅读文档推出行为），而输出变化是可见、可解释、
>   且有测试钉住的。
> - 影响面很窄：只在同时满足"走门面 + 设过 EvenOdd + 开了层"时才会不同；
>   只走 C ABI 的调用方不受影响（0.8.0 那边本来就是对的）。
> - 这条变化**没有**进 AMK1（编码流不变），也没有改动 NonZero 的任何路径。

### 门面 API 变化

- 新增：`save()` / `restore()` / `CanvasError` / `CanvasState` / `MAX_STACK_DEPTH`
- 新增只读查询：`save_depth()` / `clip_depth()`（后者同时供 C ABI 同步用）
- **四个 setter 的签名一字未动**，但它们现在改的是 `self.state.*` ——
  `CanvasState` 成为 Canvas 状态的**唯一来源**，`restore` 整体写回它。

### 测试（278 → 297，+19）

| 位置 | 新增 | 覆盖 |
|---|---|---|
| `amiki/tests/state_stack.rs` | 12 | 四项状态各自往返（比 `Draw` 逐字段而非比 getter）、三层嵌套、32/33 溢出、空栈与多 restore 下溢、整体覆盖非叠加、裁剪往返与独立使用、层内栈独立、层继承填充规则、**空配对零破坏** |
| `amiki/tests/state_stack_gpu.rs` | 3 | **GPU↔CPU 最大通道差 1**、空配对在 GPU/CPU 两侧都不改变输出、`restore` 后的绘制与"从未改过状态"逐字节相同 |
| `amiki-capi`（内联） | 4 | ABI 往返（渲染结果逐字节比对）、32/33 与两个错误码、**裁剪深度镜像同步**、层内栈独立 |

### 顺手清理

- `docs/HANDOFF.md` 的"当前状态"数字：**151 → 297**（既有 D1 漂移）
- `docs/KNOWN_ISSUES.md` 新登记 **D5**：even-odd 的跨度分解**没有缓存**
  （0.8.0 引入，逐帧重画同一 even-odd 路径会逐帧重算；只有 even-odd 付这个成本，
  正确性不受影响）。**没顺手做缓存**的理由写在条目里：缓存键与失效条件需要先想清楚。
- `docs/AMK1.md` 补一句"状态栈不进 AMK1"（格式仍 v2，无变化）。

---

## [0.8.0-alpha] —— even-odd 填充规则

**新能力**：`FillRule::EvenOdd`（奇偶填充）与既有的 `NonZero` 并存。
默认仍是 `NonZero` —— **不调用新 API 的代码逐字节不变**（278 测试全绿，
既有 256 条一条没改）。

### ⚠️ 实现方式与计划不同：改的是**几何**，不是累加符号

计划 §2.1 设想"复用 `seg_area`、只改累加符号、取 Σ 的奇偶性"。**这条前提被实验反证**：

引擎的覆盖率是 `min(|Σ seg_area|, 1)`，几何上等于该像素的**绕数加权面积** `∬ W dA`；
而 even-odd 要的是**奇偶区域**，那是绕数的**非线性**函数（`W mod 2`）——
从 `∬ W dA` 无论怎么改符号都取不到。实测（`crates/amiki-raster/tests/fill_rule_semantics.rs`，
参照是独立的 32× 超采样逐点绕数裁判）：

| 形状 | "取 Σ 奇偶"的偏差 | 说明 |
|---|---|---|
| 普通方块 | **0.7207**（≈184/255） | 边界覆盖率被压成 0/1，**解析抗锯齿全丢** |
| 同向双环 | **0.8477**（≈216/255） | 同上 |

**改用的做法**：把 even-odd 的形状**预处理成等价的 NonZero 几何** ——
按 y 扫描成带、带内按 x 排序、**两两配对**成互不重叠的梯形，
每个梯形只输出两条竖边（水平段对 `seg_area` 贡献恒为 0；一对竖边的贡献之和
正好是 `∫(clamp(x_r) − clamp(x_l))dy`，即该梯形的精确裁剪覆盖面积，
`−s·len` 常数项自动相消）。之后走**原有**的 NonZero 分箱/着色路径。

- **覆盖率数学一行未改**，`fine_math.wgsl` / `sparse_fine.wgsl` / `fine.wgsl` **一行未改**，
  绑定号未动 —— 计划里"给 WGSL 加 fill_rule 分支"的阶段**不再需要**
  （顺带消掉计划风险表里的 FXC 展开风险）。
- 同一份预处理在 **CPU 侧**被两端共用：`amiki-raster` 的分箱与 `amiki-gpu` 的提取
  都调用 `amiki_raster::even_odd_spans` ⇒ 双端一致是**构造性**的。
- 代价：even-odd 的 draw 每帧多做一趟 CPU 侧压平 + 分解（不缓存），
  上传线段数也明显多于原曲线数。已写进 header 与 KNOWN_ISSUES 的 B5 关闭说明。

### 另外三处与计划不同（都实测）

1. **判别用例写错了**：计划里的"外圈顺时针 + 内圈**逆时针**"环在两条规则下都是环
   （中心都给 0）—— **没有判别力**。真正能区分的是**同向**双环
   （NonZero 中心 1.000 / EvenOdd 0.000）。
2. **§2.2 的边界约定不能按计划改**：现有分箱是"行表偏包含（`floor` 两端）+ carry 严格
   `<` 排除"这一对**互补**约定；把前缀排除改成 `<=` 会让正好结束在 tile 左边界上的段
   被计两次。已把**现有**约定写成可执行断言（`tests/fill_rule_boundary.rs`），并说明
   为什么不能"顺手统一"。
3. **AMK1 没有 `flags` 字段**：真实布局是 `transform | color | blend u32 | n_refs | refs | path`。
   按计划"零布局变更 + bit 12"的**意图**，落点取 **`blend` 那个 u32 的 bit 12**
   （blend 只占 0..3）—— 布局一字未改，bit 号与计划一致。

### AMK1 v2（只解不编）

- `STREAM_VERSION` 1 → **2**；**v1 与 v2 布局完全相同**，唯一差别是 bit 12 是否定义为
  `fill_rule`。编码器统一写 v2（没有 v1 编码器可删，避免死代码漂移）；
  解码器同时接受 v1 与 v2，**未知版本（≥3）显式报错**。
- 测试：v2 混排往返（4 条 draw 含 2 条 EvenOdd，顺序与 blend 都不变）、
  v1 字节（bit 12 仍为 1）解码必得 `NonZero`、版本 3 被拒绝。

### C ABI（符号 80 → 82）

| 新增 | 说明 |
|---|---|
| `amiki_fill_rule` 枚举 | `AMIKI_FILL_NONZERO = 0`（默认）/ `AMIKI_FILL_EVENODD = 1` |
| `amiki_canvas_set_fill_rule` | 状态式、粘性；只作用于填充；越界值被拒绝；之后新开的层会继承 |
| `amiki_canvas_get_fill_rule` | 传 NULL 返回默认值（不崩） |

`fill_path` / `fill_path_with` / `stroke_path` 的签名**一字未动**。

### 门面与场景

- `amiki::Canvas::set_fill_rule` / `fill_rule`（状态式，与 `set_blend` 一致）；
  三个绘制落点（`fill_path_with`、两处文本）统一 `stamp_fill_rule`，
  避免"某个入口静默失效"。
- `amiki_scene::FillRule` 增加 `EvenOdd` + `Default`（`NonZero`）+ `bit()` / `from_bit()`。

### 测试（256 → 278）

| 位置 | 新增 | 覆盖 |
|---|---|---|
| `amiki-raster/tests/fill_rule_semantics.rs` | 5 | 设计决策**证据**：装置自检、"取奇偶"被反证、判别用例必须是同向双环、等价几何能复现 even-odd |
| `amiki-raster/tests/fill_rule_boundary.rs` | 1 | 段与 tile 的归属约定（6 条断言）+ 为什么不能改 |
| `amiki-raster/tests/fill_rule_even_odd.rs` | 7 | 简单形状/同向双环判别/反向环无判别力/**自相交五角星**/三层嵌套/两条规则一致/退化路径 |
| `amiki-scene`（内联） | 3 | AMK1 v2 往返、v1 字节解成 NonZero、未知版本拒绝 |
| `amiki-capi`（内联） | 3 | 默认值 + 往返、越界拒绝且不改状态、**经 ABI 的端到端镂空** |
| `amiki-gpu`（内联） | 2 | **GPU↔CPU 最大通道差 1**、**稀疏↔稠密逐字节一致** |

### 顺带查到的一条**既有**精度限制（已登记 B16）

追 even-odd 精度时发现：`min(|Σ|, 1)` 在 **`|W| ≥ 2` 区域的边缘会饱和** ——
自相交五角星上 NonZero 与 32× 裁判的偏差达 **0.1387**（≈35/255），
而它**不是本次改动引入的**（nonzero 路径一行未改，且 even-odd 走的是互不重叠的梯形、
不触发饱和：同一张图 even-odd 偏差只有 **0.0027**）。已登记为 **B16**。

---

## [0.7.0-alpha] —— 字体回退（一段文本里多个字体）

**新能力**：一段文本里的不同字符可以由**不同字体**渲染 —— 拉丁走 Inter、
汉字走 Noto Sans SC。此前引擎只接受单个字体，于是"支持 CJK"实际上要求调用方
**自己为每段文本挑对字体**：挑错了（比如只加载 Inter）就整段画成方框，
而这个缺口此前只写在 `docs/API.md` 的限制表里，没有编号、也不在测试覆盖内。

### 新增 API（**新入口，老 API 一字未动**）

| 层 | 新增 |
|---|---|
| `amiki-text` | `FontStack`（有序字体栈）、`MissingGlyphPolicy`（`Error`/`NotDef`/`Skip`）、`TextOptions`、`shape_with_stack`、`Font::id()`（内部稳定 ID）、`FontId` |
| `amiki-scene` | `PositionedGlyph::font_index` / `cluster`、`TextRun::push_glyph_in`、`GlyphKey::with_font`、`expand_text_run_with`、`TextCache::expand_with` |
| `amiki-layout` | `layout_with_stack` / `layout_with_stack_policy` |
| `amiki`（门面） | `Canvas::text_with_stack` |
| C ABI | `amiki_font_stack_create/destroy/push/resolve`、`amiki_canvas_draw_text_with_stack`、`amiki_text_options_default`（符号 **74 → 80**）+ `amiki_missing_glyph_policy` / `amiki_text_options` / `AMIKI_ERROR_NOT_FOUND` |

### 设计与取舍

- **顺序显式**：`FontStack` 按顺序解析，**第一个含该字形的字体胜出**。
  不做任何"按脚本优先级猜"的隐式规则 —— 谁先压谁赢，可预测、可测试
  （有一条测试专门把栈倒过来验证这一点）。
- **缺字默认报错**：`MissingGlyphPolicy::Error` 返回 `Err(MissingGlyph(码点))`，
  由调用方显式选择 `.notdef` 兜底或跳过。方框在很多界面里"看起来像是对的"，
  所以这个决定不能替调用方做（沿用本项目"不支持就显式报错"的取舍）。
- **缓存键用字体内部 ID**：`GlyphKey` 新增 `font_id`，取字体**自己声明的**名字
  （PostScript 名 → Unique ID → Full name → Family+Subfamily → 字节哈希）。
  用路径或栈下标都不行：路径会变、下标会随栈调整而变，而同一个 glyph id
  在不同字体里是**不同字形** —— 键里不带字体就会跨字体误命中（画出串形的字）。
  `FontId` 放在 `amiki-scene`（`GlyphKey` 在那里，而依赖方向是 `text → scene`），
  从 `amiki_text` 再导出。
- **行高取栈里第一个字体**：混合串里拉丁与汉字的高度不同，不做"取最大值"的
  隐式合并 —— 否则行高会随内容变化（同字号不同文本行高不同，界面会抖）。
- **run 边界不做字距**：那是字体内部信息。同字体的整段文本不受影响
  （有一条测试用 4 行混排验证"行宽 = 各 run 之和 ±0.5px"）。

### 零破坏怎么保证（可执行证据）

- `shape` / `shape_clustered` / `layout` / `expand_text_run` / `TextCache::expand` /
  `Canvas::text` 的**签名与实现都没动**；
- `shape_with_stack` 有条**快速路径**：单字体栈 + `TextOptions::LEGACY`（缺字交给
  shaping）时直接调用老的 `shape` ⇒ 结果逐字节一致（测试断言 `f32::to_bits` 相等）；
- `layout_with_stack` 在单字体栈时直接调用老的 `layout`（测试逐位比对行数/宽度/基线）；
- 老入口的**行为**按原样保留，包括单字体路径遇缺字时"交给 shaping 映射成 `.notdef`"
  （栈路径才默认报错）—— 否则老调用方的输出会变。

### 顺带修掉的

- `Font` 的字节改为 `Arc` 共享 ⇒ `Font` 变**廉价可克隆**（C ABI 的
  `amiki_font_stack_push` 依赖它：压栈不复制那 10 MB），并手写 `Debug`
  避免打印几 MB 字节；
- 三处过时注释：`amiki-layout` 说"仓库里没有 CJK 字体"（早已入库）、
  `tests/layout.rs` 的同一个说法；
- `docs/API.md` 里"`assets/fonts/` 没有 CJK 字体"那条**本来就是错的**（0.3.0 起
  就有 Noto），已改成真正的限制（当时是"不做字体回退"），如今随本版关闭。

### 测试（224 → 256）

| 位置 | 新增 | 覆盖 |
|---|---|---|
| `amiki-text/tests/stack.rs` | 13 | 解析/run 切分/缺字三策略/字距开关/单字体栈等价/cluster 指回原文/`FontId` 稳定性 |
| `amiki-scene`（内联） | 3 | 按字体取轮廓、缓存按字体分键、老 `GlyphKey::new` 语义不变 |
| `amiki-layout/tests/stack_layout.rs` | 7 | 混排单行/折行不丢内容/**各 run 之和 ⇄ 行宽 ≤0.5px**/CJK 禁则/行高随首字体/单字体栈等价/缺字策略 |
| `amiki-capi`（内联 + types） | 8 | 空指针、`resolve` 顺序与 `NOT_FOUND`、**压栈后即可销毁字体句柄**、缺字策略端到端、**栈路径与单字体路径逐字节一致**、结构体布局与枚举映射 |
| `amiki-gpu`（内联） | 1 | **CJK 字形 GPU↔CPU 逐像素 ≤2/255**（实测最大通道差 **1**，4127 个非零通道） |

### 人工验证

`window_demo` 现在加载 Inter + Noto 并全程走字体栈：
列表首行 `第 109 行 · Amiki2D 渲染引擎 dogfooding` 的汉字**全部正常显示，无方框**
（截图 `out/cjk_list.png`）。绘制顺序与三语言示例一致，`verify.cmd` 的三份产物
仍然**逐字节相同**。

---

## [0.6.1-alpha] —— GPU 增量渲染的脏区派发修复

**修复一个**从 M4（0.3.0）起就存在**的静默错误**：GPU 帧持久路径只重算
"**错误的那些 tile**"，脏区不在左上角时画面保留上一帧像素（残影）。

### 缺陷：派发丢掉了脏区的**起点**

`fine_dispatch_range` 把脏区换算成 `(x_start, x_count, y_start, y_count)`，
调用方只用了后两个：

```rust
let (_fx, fw, _fy, fh) = fine_dispatch_range(damage, tiles_x, tiles_y);
pass.dispatch_workgroups(fw, fh, 1);      // ← 起点被丢弃
```

fine 内核从 `global_invocation_id` 反推 tile 与像素坐标，于是**隐含**了
"派发从 (0,0) 起步"的假设。wgpu 的 `dispatch_workgroups` **没有起点偏移**，
所以脏区是 `(5,1)..(7,6)` 时，真正被重算的是 `(0,0)..(2,5)` —— 脏 tile 一个都
没重算；而 `render_into_persistent` 的未派发区域按设计**不写入**（那正是复用
上一帧的机制），画面于是停在上一帧。

**为什么藏了这么久**：唯一的回归测试先把纹理渲成"全量正确"，再用脏区重算一批
tile 后断言等于全量 —— **无论重算的是哪一批 tile 都会通过**。它检验的是"重算
tile 的结果与全量一致"，而不是"重算的确实是脏 tile"。唯一会踩到它的是真实
应用：`examples/window_demo.rs` 的滚动（脏区随内容一路下移，起点每帧都不同）。

### 修法：1D 派发 + 脏 tile 索引表

位置不再由 invocation 反推，而是**由宿主的表决定**：

- 新绑定 **binding 18**：`dirty_tiles: array<u32>`，每项打包 `tx | ty << 16`
- **1D 派发**：x 维 = `tile 序号 * 16 + tile 内列`，y 维 = tile 内行；
  z 维分批（每批 65535 个 tile，规避单维 workgroup 上限 —— 8K 画布的
  129 600 个 tile 因此仍在支持范围内）
- `FineParams` 增加 `n_tiles`：末尾批次的尾巴据此提前退出
  （否则会去重算 tile (0,0)，白做功）
- 全量渲染时这张表就是整幅画布的 tile ⇒ **只有一条代码路径**
- `sparse.rs` 与 `dense.rs` 同步；两个 fine 入口都用共享的 `tile_hit`
  （`fine_math.wgsl`），**裁剪采样也改用真实画布像素坐标**而非 `gid.xy`
  （这一处最容易漏：`clip_mul` 原本拿 `gid.xy` 当像素坐标）
- **设备契约变更**：fine 的 storage buffer 从 9 个增到 **10** 个，
  `amiki_gpu::REQUIRED_STORAGE_BUFFERS` 同步 9 → 10。自建 `wgpu::Device` 的
  调用方必须继续用 `required_limits()`（`window_demo` 本来就是）

### 测试：把"没有牙"的测试换成"有牙"的

| 测试 | 之前 | 现在 |
|---|---|---|
| `gpu_damage_dispatch_matches_full` | 纹理先渲成全量正确 → 恒真 | 纹理停在**上一帧 A**，用 A→B 脏区做增量，与 B 全量**逐字节**对拍；并断言脏区**不在原点**（否则测试自身退化） |
| `gpu_persistent_matches_full_across_scroll_like_frames` | 不存在 | **8 帧滚动序列**（脏区每帧移动、起点每帧不同）逐帧与全量对拍 —— 与 `window_demo` 同一条路径 |
| `dirty_tile_list_covers_exactly_the_dirty_rect` | 不存在 | 纯函数：全量 / 局部 / 空脏区 / 越界钳制 |
| `fine_dispatch_dims_batches_beyond_the_1d_workgroup_limit` | 不存在 | 纯函数：65535 边界与 z 维分批（8K 规模） |

**验证有牙**（修复前分别红在这两处）：
`不同字节 = 3872`（单帧 A→B，脏区 tile `(5,1)..(7,6)`）与
`不同字节 = 8136`（滚动第 1 帧，脏区 tile `(2,1)..(9,10)`）。

### 顺带查明：并发**创建 context** 会崩 —— 根因是 **Vulkan 校验层**

见 `docs/RELEASE_PLAN.md` §8.5 与 `docs/KNOWN_ISSUES.md` 的 **B14**。

`amiki_context_create` 在 GPU / AUTO 后端下会创建 `wgpu::Device`；多个线程
**同时**创建时，整个进程异常退出（`0xC0000005`，**无 panic**，也没有任何可捕获的
错误）。三列对照把范围逐层收窄（同一份代码、同一台机器、同一个负载）：

| 场景 | 崩溃 |
|---|---|
| 16 线程**并发创建**（校验层开 = debug 默认） | **3 / 6** |
| 校验层**关**（`InstanceFlags::empty()`），同一负载 | **0 / 6** |
| 串行创建 16 个 → 16 线程**并发使用** | **0 / 5** |
| 4 线程并发创建 / 并发使用 | 0 / 6 · 0 / 6 |

**根因不在引擎、也不在显卡驱动**：wgpu 在 `debug_assertions` 下默认启用
`InstanceFlags::VALIDATION`，Vulkan 后端据此加载校验层；Windows 事件日志把
faulting module 直接写了出来 —— `VkLayer_khronos_validation.dll`，
`Exception code: 0xc0000005`，与上表 3 次一一对应。

**影响范围（关键）**：**release 构建默认不启用校验层 ⇒ 不受影响**；受影响的只有
debug 构建或显式打开 wgpu 校验的运行（含 `cargo test` 默认配置）。这也解释了 §8.5
那次"不可复现"：`docs/VERIFY.md` 的清单一直用 `cargo test --release`（校验层关），
两次不是同一套条件。

**ABI 侧的写法**：原句"不同 context 之间可以并行"说的是**使用**，这一半**成立**；
缺的是"**创建要串行**"。已写进 `amiki.h` 的「线程模型」、
`crates/amiki-capi/README.md` 与 `docs/API.md`（不依赖构建类型，统一按串行创建写）。

> 这是本项目"不支持就**显式报错**"原则的**反例** —— 它既不报错也不 panic，
> 而是直接崩进程。三种后续处置记在 B14：引擎内部串行化创建（推荐）/ ABI 层显式
> 拒绝 / 保持现状，**尚未决策**（因发布构建不受影响，价值已变小）。

**测试侧**：`amiki-gpu` 的 GPU 测试共用一个**串行闸门**（`tests.rs` 的
`GPU_TEST_LOCK` / `gpu()`）—— 默认线程下 **6 / 6 全绿**（此前 **4 / 6** 崩），
整套仍只要 ~7 s；顺带消除了"性能断言在并行环境里失真"这个反复出现的假失败
（`HANDOFF.md` 纪律第 7 条）。

### 其它

- `blur.rs` 的注释仍写"格式固定 `Rgba8Unorm`" —— 实际管线目标与模糊纹理都是
  `Rgba16Float`（0.5.0 HDR 改造漏改的一行注释），已更正
- 端到端人工验证：`window_demo` 切到列表视图后 `脏 tile 0.0% · 0.00 ms/帧`
  （静态界面零 GPU 工作），滚动到 680 时首行是被裁一半的第 22 行、之后 23→44
  连续有序，**无残影**（截图见报告）
- 测试：**224 passed / 0 failed**（`amiki-gpu` 34 → 37）

---

## [0.6.0-alpha] —— mipmap

**第六个 Alpha：大幅缩小的位图不再走样（B4 关闭）。**

UI 里把大图缩成小图标、文档查看器的缩略图、地图缩放 —— 这些场景此前
只有线性采样，会出摩尔纹。现在引擎在 CPU 预先算好 mip 链，
采样时按缩小倍数取用。**AMK1 编码流不变**：mip 链是运行时派生物，
场景文件仍只存原图。

> **边界说明（诚实清单）**
> - 各向异性过滤仍不做（登记为后续里程碑）；GPU 生成 mip 链明确不做（一致性）
> - 对拍覆盖**等尺寸**位图场景；混合尺寸场景 GPU 数组的补齐语义与 CPU
>   各自独立（mipmap 之前就如此，未在本版收窄）

### 设计：一致性优先，三个"只在 CPU"

1. **mip 链只在 CPU 生成**（`amiki_scene::mipmap`）。2×2 盒式滤波，
   在**线性光空间 + 预乘 alpha** 上平均 —— 在编码空间直接平均会偏亮
   （黑白棋盘的平均是 sRGB 188，不是 128），不预乘会在透明边缘出黑边。
   两点都有专项测试钉住。
2. **LOD 只在 CPU 计算**（`compute_lod`）。从放置轴向量的逆雅可比出发，
   纹理宽高不同则逐轴缩放；算好连同 `mip_bias` 打包进 GPU 放置缓冲的
   空闲分量（`image_place[d*3+2].z`），WGSL 只读不算。
   `log2` 在 CPU libm 与 GPU 内建函数间可能差 1 ULP，采样层级对它敏感 ——
   宿主算好打包，这个漂移源就不存在了。
3. **GPU 只上传**：数组纹理按全场景最大尺寸开 mip 层，逐层逐级上传
   CPU 生成的字节；小于数组的图像用补齐版链（同一套滤波数学，
   与直接生成逐字节一致，有测试）。

因为 LOD 是放置的派生量（2D 无透视），它是 **draw 常量**：平铺的 LOD
天然按单个平铺单元算，不会被平铺次数放大（任务书 2.4 的坑自动避开）。

### 顺带修掉的潜伏 bug：平铺采样的接缝带

mipmap 对拍首次跑平铺就红了（差 11/255）。定位后发现与 mip 无关：
**双线性路径一直就有** —— CPU 平铺在取纹素时按 `rem_euclid` 真回绕，
GPU 采样器是 ClampToEdge；uv 落在 [0,1] 边缘约 1.6% 带内、且跨层取纹素
越过边缘时，GPU 钳到边缘、CPU 回绕到另一侧。此前的平铺测试像素中心
恰好够不到这条带，所以一直是绿的。

修复：新增 **Repeat 地址模式采样器**（fine 绑定布局 binding 17，
稠密路径用占位资源），平铺分支用它采样 —— 真回绕发生在采样器上，
每个 mip 层的取纹素都会正确回绕。修复后 GPU↔CPU 最大通道差 1。

### 顺带修掉的第二个坑：增量路径的 `mip_bias`（本版**新引入**的缺陷）

前面那个是潜伏的旧 bug，这一个不是 —— 它是 mipmap 自己带进来的。

`mip_bias` 被加成 `BinnedScene` / `PersistentRenderer` 的 `pub` 字段，但
**增量路径根本没把它用在主画布上**：`PersistentRenderer::update` 在 `bin()`
之后直接渲染，而 `bin()` 把 `mip_bias` 固定为 0；只有**层内容**经
`render_inner_f32` 的入参拿到了调用方的值。后果是"含位图的场景里，增量结果
与全量结果不一致"，而且不报错。

第二个洞在同一条路上：改了 `mip_bias` 而场景未变时，`update` 会命中
"场景未变 ⇒ 零工作"的早退分支，画面**停在旧 bias 上**（同样是静默的旧图）。

修法：

1. 两条路径都显式写 `binned.mip_bias = self.mip_bias`（`apply_damage_for_test`
   也补上 —— 它的全部价值就是"与 `update` 内部路径一致"）；
2. `update` 额外比较上一帧**实际用过**的 bias，不同就整幅重建
   （缓存的 tile 与层缓冲都是按旧 bias 渲染的，只改字段必错）；
3. `compute_lod` 加一道非有限 bias 的护栏：C ABI 会拒绝 NaN/inf，但这两个字段
   在 Rust 侧是 `pub` 的，直接写 NaN 会一路传染成 NaN 像素。

**验证有牙**（分别回退两处修复，各红一条）：
`persistent_renderer_uses_the_same_mip_bias_as_the_full_path`
（raster 层，走真实 bin + render 路径）——
忽略 bias → `首帧的 bias 必须传到主画布`；不重画 → `改 bias 后必须重画`。

### API

- Rust：`amiki_scene::Image` 新增 `mip: Option<MipChain>` 字段与
  `from_rgba8_with_mipmap`；`generate_mip_chain` / `compute_lod` /
  `generate_padded_mip_levels` / `mip_level_count` 公开。
  `from_rgba8` **不变**（不生成链，行为与 0.5.0 完全一致）
- C ABI（72 → **74** 个符号）：
  - `amiki_image_set_mipmap_mode`：BOX（默认，创建时已生效）/ DISABLED（省回
    1/3 内存、锁死双线性）；对**后续绘制**生效，越界枚举显式拒绝且不改状态
  - `amiki_canvas_set_mip_bias`：正负均可，钳在链范围内；NaN/inf 拒绝
  - `amiki.h` 版本常量同步 0.6.0
- GPU：`render_blocking_with` 增加 `mip_bias` 参数；`prepare_with` /
  `prepare_into_with` 对应提取路径

### 兼容性（硬约束逐条兑现）

- **1:1 与放大的绘制逐字节不变**：LOD < 0 钳回 level 0，
  与无 mip 链走同一条双线路径（测试断言相等）
- `AMIKI_MIPMAP_DISABLED` 下回到 0.5.0 行为
- 三个绑定示例（C / C# / Node）输出**字节一致**不变
  （它们都是放大绘制，本就对 mip 不敏感）
- `bin()` / `render()` 签名不变：bias 作为 `BinnedScene` 的字段
  在 bin 之后设置，既有调用方零改动

### 性能：链生成从 56 ms 到 ~2.5 ms（22×）

验收时发现朴素实现生成 1024² 的链要 **56 ms**（每纹素 7 次 `powf`），
不满足任务书"生成 ≤ 2×原图上传"的判据。四步优化后到 **~2.5 ms**：

1. **sRGB 编解码查表**：解码 256 项 f32 表；编码用 16384 项**量化边界对齐**
   表（值 `l` 得到 `v` 当且仅当 `l` 落在四舍五入到 `v` 的线性区间内，与直接
   `powf` 逐字节一致）。16384 是下限：暗部线性段斜率 12.92，每个 u8 量化
   步长占 ~0.0003 线性值，4096 桶只有 1.24 桶/步会丢字节（实测 7 个值往返错）
2. **配对求和** `(a+b)+(c+d)`：每步都是 2 的幂倍，均匀色往返精确到字节
   （顺序加会经过不可精确表示的 `3d`，±1 ULP 掉进相邻量化桶）
3. **行级并行**（原生目标）：输出行彼此独立，`chunks_mut` 切互斥行带喂
   scoped 线程，无 unsafe、分块方式不影响任何字节；wasm 保持串行。
   说明：C5"CPU 并行不做"针对光栅器采样负载（瓶颈在分配），链生成是
   一次性上传期工作，负载形态不同，实测有效
4. **不透明快路径**：alpha 恒 255（UI 贴图、照片的常态）跳过预乘/反预乘/
   除法，字节与慢路径逐位一致

**口径注记**：判据 1 的分母（`write_texture` 主机侧上传）本身在 1.1–1.6 ms
间波动，生成 2.4–3.2 ms，比值 ~1.6–2.9× —— 判据"≤2×"处于门槛上，
多次运行 3 过 2 未过，**计为边际达成**（判据 2 采样回退 −1.9%、
判据 3 内存 33.3% 干净通过）。测量示例：`examples/mipmap_perf`。

### WASM：三浏览器字节验证（门 3）

纯 CPU 构建此前只在 Node 里验证过。本版新增浏览器验证页
`examples/wasm/browser_test.html`（场景与 `minimal.js` 逐调用一致，
SHA-256 直接显示在页面上），在 **Chrome / Edge / Firefox**（均无头）实测：
三者与原生 CPU 输出**逐字节一致**（`d565cc36…`）。加载器垫了
`require`/`module`/`Buffer` 三个浏览器垫片，页面同时验证
"GPU 后端显式拒绝"与"越界枚举拒绝"两个行为契约。

**模块体积 950.7 → 965.9 KB**（`--profile wasm-release`，实测
989 074 字节）：+mipmap 的链生成与 sRGB 查表（表住在 `amiki-scene`，
wasm 构建包含它）。`build.cmd` 的 Node 演示 + 原生 CPU 字节对比同时复跑通过，
期望哈希仍是 `d565cc36…`（与浏览器页里写死的值一致）。

### 测试

- 默认 feature（含 GPU）：**221 测试 / 0 失败**（新增 26 个 mipmap 专项：
  scene 14 · raster 7 · GPU 1 · C ABI 4）
- 纯 CPU 构建全绿（**151 测试**）：`--no-default-features` 下四个示例都按
  `required-features = ["gpu"]` 自动跳过。本轮新增的两个 mipmap 示例原先漏了这条
  声明，会让 `cargo test -p amiki --no-default-features` **直接编不过** ——
  与"纯 CPU 构建全绿"矛盾，已补齐声明（四个示例确实都用 GPU 后端）
- clippy 与 cargo doc 零警告
- `verify.cmd` 五项全过：74/74 符号、header /W4 /WX 编译、
  三示例输出字节一致
- 摩尔纹消除的**量化判定**：8px 棋盘缩小到奈奎斯特之后
  （LOD ≈ 4.5），带链输出整幅均匀灰 188（线性平均的正确值），
  无链是 0/255 椒盐噪声

---

## [0.5.0-alpha] —— WASM + HDR

**第五个 Alpha：引擎能在浏览器与任何 JS 运行时里跑。**

C ABI 解决了"别的语言能不能用"，WASM 解决"Web 上能不能用" —— Web 是
2D 渲染器最大的分发渠道。

> **边界说明（诚实清单）**
> - **WebGPU 构建未验证**：本机 Node 24 的 `navigator.gpu` 是 `undefined`，
>   也没有可用的浏览器环境。所以这条**不写成"已支持"**
> - 跨厂商硬件仍未验证（A1，已知限制）
> - 现有 `examples/node/` 用的是 **koffi FFI 而不是 WASM** —— 两者是不同的东西

### 模块体积：11.7 MB → **950.7 KB**

实测最大的影响因素是仓库根 `[profile.release] debug = true` —— 那是为**原生
profiling** 加的，在 wasm 里是纯浪费（浏览器不需要 DWARF）。同一个模块：

| 构建 | 体积 |
|---|---|
| `--profile release` | 11.7 MB |
| `--profile wasm-release`（`debug = false`） | **950.7 KB** |

另立 `[profile.wasm-release]` 固化这一点。该 profile 里另外三个开关
（`opt-level = "z"` / `lto = "fat"` / `panic = "abort"`）**实测只再省约 5 KB**，
在噪声范围内 —— 保留是因为语义正确，不声称有效。

### GPU 路径变成可选 feature

新增 `gpu` feature（**默认开**，native 行为完全不变）。关掉之后：

- `amiki-capi` / `amiki` 都可以纯 CPU 构建（`--no-default-features`）
- wgpu 不被编进去 —— 这正是 23 MB → 950 KB 的另一半原因
- 行为是**显式的**：`AMIKI_BACKEND_GPU` 返回 `AMIKI_ERROR_BACKEND_UNAVAILABLE`，
  `AMIKI_BACKEND_AUTO` 落到 CPU。**不静默降级**

顺带把 `amiki` 的两个示例与 `stroke_api` 测试声明为 `required-features = ["gpu"]`。

### C ABI 新增 `amiki_alloc` / `amiki_free`

WASM 上调用方（JS）没法自己分配引擎线性内存，只能通过导出的分配器 ——
没有它就无法把像素、字体字节、渐变色标传进来。原生下也有用（同一个分配器
分配与释放，避免跨 CRT 释放的未定义行为）。

### JS 加载器：不依赖 wasm-bindgen

`crates/amiki-capi/examples/wasm/amiki.js` 用**原生 WebAssembly API** 直接调
C ABI。理由：wasm-bindgen 会在 Rust 与 JS 之间再插一层自己的 ABI，而
C ABI 已经是一份设计好、测试过、三种语言验证过的边界。

**这一版最实际的技术点**：WebAssembly 没有"结构体按值"的概念，所以 C ABI 会被
**降级**成：

1. **结构体返回 → sret**（隐藏的第一个参数是返回空间指针）
2. **结构体参数 → 一律按指针传**

实测 `amiki_canvas_stroke_path(i32, i32, i32, i32) -> i32` = canvas, path,
`style*`, `paint*`。加载器把这两条规则集中在两个辅助里。
新增 `tools/wasm_sig.py` 可随时打出真实签名 —— 核对比猜快。

### 强验证：原生 CPU 与 WASM 输出**字节完全相同**

```
原生 CPU:  C1DDAF7DF7DBBCB0A5355B5A00A3EEB56C125E2F…
WASM:      C1DDAF7DF7DBBCB0A5355B5A00A3EEB56C125E2F…
```

同一个场景、**不同目标架构**（wasm32 vs x86_64）、**不同的 ABI 降级方式** ——
输出字节完全相同。这同时证明了 WASM 构建正确、加载器对降级的处理正确、
渲染器跨目标确定。

`examples/wasm/build.cmd` 会自动跑这个对比。

（对比基准是**原生 CPU** 而不是原生默认的 GPU —— WASM 是纯 CPU 构建，
与 GPU 产物本来就该不同。`examples/c/minimal.c` 为此支持了 `AMIKI_BACKEND` 环境变量。）

### 测试

- 默认 feature（含 GPU）：**180 测试 / 0 失败**
- 纯 CPU（`--no-default-features`）：**31 测试 / 0 失败**
- clippy 与 cargo doc 零警告

---

## [0.4.0-alpha] —— C ABI

**第四个 Alpha：让 Amiki2D 能被 Rust 之外的任何语言调用。**

0.3.0 的能力清单放到任何一个 2D 渲染器面前都不丢人，真正卡住它的是**没人能用它** ——
Rust 2D 生态本身很小，而 C# / Java / Node / Python 用户因为没有 C ABI 全都进不来。
本版兑现"通用"这个词的入口。

> **边界说明（诚实清单）**
> - 三个绑定示例（C / C# / Node）做**完全相同的绘制**，输出**逐像素一致** —— 这是
>   "ABI 封送正确"的证据，详见 `crates/amiki-capi/examples/README.md`
> - **跨厂商硬件仍未验证**（A1）：三个后端都在同一块 GPU 上验证过
> - C ABI 不隐藏引擎的边界，而是照实暴露（GL 不支持位图 → `AMIKI_ERROR_NOT_SUPPORTED`）；
>   见 `docs/KNOWN_ISSUES.md`

### 新增：`amiki-capi` + 手写 `amiki.h`

**契约是手写的 `crates/amiki-capi/include/amiki.h`，Rust 侧去适配它。**
之所以不用 `cbindgen` 之类的工具生成：那样 header 会随 Rust 类型变动而**漂移**，
绑定作者拿不到稳定界面。Skia / Blend2D / wgpu-native 都是手写 header。

导出 **68 个符号**，与 header 的声明数**逐一核对无缺失**。

**三条硬保证**（写进了 header，是设计约束不是实现细节）：

1. **绝不 panic 跨 FFI 边界** —— 每个导出函数都在 `catch_unwind` 之内。
   跨 FFI 的 panic 是未定义行为，不能让 C 调用方承担。
2. **panic 之后对象中毒** —— 出事的对象后续调用一律返回 `AMIKI_ERROR_PANIC`。
   因为 panic 可能打断在一次修改的中途，继续用它**可能静默画出错误结果**。
3. **失败时不写出参** —— 每个 `_create` 在失败时都不碰 `*out`。

**错误模型**：所有可能失败的函数返回 `amiki_result`，指针走 out 参数。

**内存所有权**写进 header 注释并逐条编号。其中一条值得单独说：
位图与渐变**是引用计数的**，引擎内部持有一份，所以调用方**可以在绘制后立刻销毁**
自己的句柄 —— 这不是猜测，`crates/amiki-capi/src/tests.rs` 里有实测。

**版本策略**：编译期 `AMIKI_VERSION_*` + 运行期 `amiki_version()` 都要查
（头文件与二进制可能不是同一版编出来的）。数值宏便于比较，`amiki_version_string()`
带预发布标签便于展示。

### 三个绑定示例（都真实编译运行）

| 示例 | 语言 | 绑定方式 |
|---|---|---|
| `examples/c/` | C99 | 直接 `#include "amiki.h"`，`cl /utf-8 /W4 /WX` 零警告 |
| `examples/csharp/` | C# / .NET 10 | 手写 P/Invoke（**不用** `StringMarshalling`，它在 .NET 10 上不可用） |
| `examples/node/` | Node.js 24 | koffi FFI |

三者做同样的绘制，输出**逐像素完全一致**（三对比较的最大通道差都是 **0**）。

**Node 为什么选 koffi 而不是 napi-rs / WASM**：napi-rs 需要**构建期下载 Node 头文件**
（本项目此前上过 `static-dxc` 构建期下载挂住的当，不想再引入同类风险）；
WASM **到不了 GPU 后端**（引擎主力是 wgpu）。koffi 不改 Rust 一行、`npm install` 即用，
而第三个示例的目的正是"证明这个 ABI 能被非 C / 非 .NET 的运行时调进来"。

### 性能上的关键设计：context 持有后端

`amiki_context` 持有渲染后端，**创建一次跨帧复用**。这不是可选优化 ——
创建 GPU 后端要初始化驱动（实测冷启动 **61 ms**），逐帧新建会让它成为主要开销。
`amiki_canvas_clear` 同理（复用内部缓冲），逐帧渲染不该 `destroy` + `create`。

### 测试

`cargo test -p amiki-capi` → **29 项 ABI 层自测**，从 C 侧的角度验证契约：
每个接收指针的入口都拒绝 NULL 且不崩、失败时不写出参、越界枚举/NaN/Inf 被拒绝、
中毒真的挡住后续调用、位图与渐变可在绘制后立刻销毁、
**层没配平就渲染被显式拒绝**（否则会静默丢内容）、端到端 CPU 渲染真的画出东西。

`src/types.rs` 还有**布局断言**（`size_of` / `offset_of`）—— C 侧的 `sizeof`
只要对不上，绑定就会静默踩内存。

### 顺带

- workspace 版本 → `0.4.0-alpha`
- 修掉 `amiki.h` 里一条**不准确的所有权说明**（原写"位图要活到渲染时"，
  实际引擎用 `Arc` 持有，调用方可以立刻销毁）

---

## [0.3.0-alpha] —— 能力门槛全部完成（P0/P1/P2/P3）

第三个 Alpha：补齐**描边**、**含层场景的增量渲染**、**帧路径的层合成**，做完
**CJK 断行与字距**，并把工程成熟度六项（P3）全部收口。发布判据 6 条**全部满足**。

> **边界说明（诚实清单）**
> - **跨厂商硬件未验证**：三个 wgpu 后端（Vulkan / DX12 / GL）全部跑通，但都在
>   **同一块 GPU** 上 —— 单人开发、只有一块显卡，这条已决定收缩范围，记为
>   **「已知未验证项」而非「已通过」**。见 `docs/RELEASE_PLAN.md` 的「范围收缩记录」。
> - **GL 后端不支持位图**：该后端读不出 `texture_2d_array`。引擎**显式报错**，
>   不会静默画错。
> - **DX12 冷启动约 2.3 s**：FXC 在运行期编译着色器，不是引擎代码慢。
> - 仍**不做**：双向文本、Knuth-Plass 全局最优断行、HDR/宽色域、C ABI/WASM
>   （理由见 `docs/RELEASE_PLAN.md` 第六节）。

### P0-1 描边

miter / round / bevel 三种拐角（含 miter limit）、butt / round / square 三种端帽、
开放折线与**闭合环**。闭合环的处理是"两条反向绕行的轮廓" —— 这样即使旋转、
非均匀缩放后绕行方向翻转，也仍然只有重叠处被填充。

用**解析计算的精确面积**验证几何，而不是只看"画出来像不像"。

### P0-2 含层场景的增量渲染

此前带离屏层的场景**完全没有脏区收益**（每帧全画）。现在层缓冲跨帧复用、
层内容未变时**不重新滤波**（模糊是层里最贵的一步），并把层脏区经
`inflated_bounds` 传播到主画布。

**带投影的静态界面省下 96.5% 的 tile 着色**（此前为 0）。60 轮随机编辑与
全量渲染逐字节一致。

### P0-3 帧路径的层合成

逐帧路径此前**显式拒绝**含层场景（否则会静默丢层）。现在在 GPU compute 上
直接合成，**不读回**。分组 / 模糊 / 投影三种滤镜与 CPU 参考实现双端对拍一致
（差 0 / 1 / 1）。

> 过程中发现并修掉一个只有"与 CPU 对拍"才能抓到的 bug：`composite_pass` 的
> 两个纹理参数写反了，导致层边缘出现幽灵阴影（7567 像素不同）。
> 只跑 `render_blocking` 是看不出来的。

### P1-1 性能：提取 2.03 → 0.59 ms（−71%），帧路径判据拆成三项可测口径

**提取达标**：`1080p×10k` 从 **2.03 ms** 降到 **586–617 µs**（判据 0.8 ms）。

真因是**每帧重新分配约 7 MB 缓冲**，不是 push、不是算术、不是并行 ——
大于 128 KB 的分配走 `mmap`、释放走 `munmap`，于是每帧重新缺页。
实测同一批数据：**新建 0.291 ms vs 复用 0.053 ms（差 5.3 倍）**。

调查中**逐条实测否定**了四个方向（全部记入文档，避免重复浪费）：
逐元素 push 比 memcpy 慢（❌ 都是带宽受限）、`sqrt` 是瓶颈（❌ 去掉全部 sqrt
只快 14%）、手工内联（❌ 无差别）、把段数计算移到 GPU（❌ 需改流水线但收益极小）。

修法：新增 `build_gpu_scene_into` / `GpuScene::reset_for_reuse` /
`GpuRenderer::prepare_into` / `profile_into`，逐帧路径复用缓冲。
`build_gpu_scene` / `prepare` / `profile` 保留为兼容包装。

**帧路径判据改写**：原判据「总墙钟 < 6 ms」把三种性质不同的东西混在一个数字里
（主机侧渲染 / GPU 计算 / **保护性同步读回**），既指导不了优化，也会诱导去砍
正确性开销。现拆成三项可测口径，并**写进代码**（`FrameProfile::host_render_ms` /
`safety_readback_ms`）+ 断言锁定：

| 口径 | 实测 |
|---|---|
| 主机侧渲染（提取 + 分配 + 上传 + 编码） | **2.2–2.8 ms**（判据 <3） |
| GPU pass（时间戳） | **2.1 ms**（判据 <5） |
| 保护性同步读回 | 4.3–4.6 ms（只设上限） |

第三项**不作为性能目标**：那 4 ms 买的是"绝不静默画出残缺帧"（投机分配必须
读回实际用量才能确认没溢出），而判据第 3 条正是"没有已知的静默错误路径"。

### P2-1 CJK 断行 + **中文字体入库**

实现**禁则处理**：`is_no_line_start`（`。、，！？：；」』）】》` 等收尾标点不得在行首）、
`is_no_line_end`（`「『（【《“` 等起始括号不得在行尾）。`pick_break` 抽成**纯函数**
（禁则只与字符有关、与字体无关，这样即使没有字体也能测）。

**测试有牙**：停用禁则后，6 条单元测试中 2 条失败、端到端 1 条失败。
（第一版单元测试只有 1 条失败 —— 说明"行尾禁则"那条无论开关都会通过，
已重写成"让被禁断点恰好成为最后一个候选"。）

**`assets/fonts/NotoSansSC-Regular.ttf`（10.07 MB，OFL 1.1）随仓库分发**，
收口了验收标准里的「字体许可文件随仓库分发」。此前仓库里只有 Inter
（2937 个字形，**不含汉字**），端到端测试每次落到系统字体上 ——
**干净检出（无系统 CJK 字体）会跳过这条测试，跳过等于验不了**。

### P2-2 字距 —— 并修掉一个相关真 bug

字距本身已由 shaping（`rustybuzz`）处理。但对拍时发现：**排版用的是
"逐字符 advance_width 相加"，而渲染走 shaping** —— 于是**排版宽度 ≠ 渲染宽度**
（`AV` 一行差 **7%**）。已改为用 `shape_clustered` 并按 cluster 归并。

### P3 工程成熟度六项

| 项 | 结果 |
|---|---|
| **多后端验证** | Vulkan / DX12 / GL **各 26 测试全通**（范围经决策收缩，见上） |
| **启动开销** | **337 → 61 ms**（判据 200）。改为"**先试平台首选、拿不到适配器再回退全部后端**" —— 与之前那次"默认只用 Vulkan"的区别在于**有回退**，可移植性不变 |
| **文件拆分** | `lib.rs` **5719 → 1461 行（−74%）**，分成 10 个模块 |
| **API 稳定性** | `docs/API.md`：稳定 / 实验 / 内部三级清单 + 已知使用限制 + 环境变量表 |
| **dogfooding** | 真实窗口应用可滚动可交互；**第一次真去跑就抓到"示例一直起不来"**（`from_device` 不校验设备上限） |
| **文档同步** | 清掉 5 处过时的代码注释（含两处"说并行、实际已串行"） |

### 测试基础设施：三次"性能断言在测试环境里失真"

`startup_cost` 与 `profile_frame_path_phases` 各假失败过，原因是**测试环境本身**：

| 环境 | 主机侧渲染 |
|---|---|
| 单跑（`--test-threads=1`） | **2.35 ms** |
| 并行跑（`cargo test --workspace` 默认） | **5.6 – 19.7 ms**（同一二进制内争用 GPU） |
| GL 后端单跑 | **20.8 ms** |

处理：性能断言改为**取多次最小值**（真实回归会让每次都慢），
或改成**只做测量 + 数量级兜底**，严格判据交给独立基准；
三种环境的数字写进测试注释，避免反复调阈值。

> `startup_cost` 那次的实际现象：冷启动测出 **1598.8 ms**（当时刚连跑完 10 轮基准），
> 单独复跑是 **348.4 ms**。

### 修复的缺陷（本版全部）

- 层合成两个纹理参数写反 → 层边缘幽灵阴影
- 模糊 ping-pong 第 2/3 轮读了原图而非上一轮结果 → 只生效一次
- `inflate_by_filter` 的 DropShadow 原点被 `dx` 平移而非取并集 → 左上被裁
  （**CPU 与 GPU 共享的真 bug**，只在 `sigma=0` 时暴露）
- 排版宽度用 advance_width 相加而非 shaping → 与渲染宽度不符（`AV` 差 7%）
- `from_device` 不校验设备上限 → 示例启动即 panic
- WGSL 动态下标导致 DX12 的 FXC 循环展开失败（`X3511`）
- `textureLoad(..., 0u)` 在 GL 上类型不匹配（`textureSize(sampler2D, uint)`）

### 跨后端统计

| 后端 | 结果 | 备注 |
|---|---|---|
| 默认（→ Vulkan） | 151 / 0（全工作区） | 冷启动 61 ms |
| Vulkan（显式） | 26 / 0 | |
| DX12 | 26 / 0 | 冷启动 2.3 s（FXC 运行期编译） |
| GL | 26 / 0 | 位图**显式拒绝** |

---

## [0.2.0-alpha] —— M4 增量渲染 + M5 文本 + M6 渐变 / 位图 / 滤镜

第二个 Alpha：从"能画图形"扩展到"能画文字、位图、渐变与滤镜，且只重画改动处"。

> **边界说明（诚实清单）**
> - M4 功能闭环；M5 已能渲染真实字体文本**并具备排版层**（断行/行高/对齐/段落），
>   但尚无双向文本、字距微调与全局最优断行（Knuth-Plass）；
> - M6 的渐变与位图在 CPU/GPU 两侧逐像素一致；
> - **M6 的滤镜（离屏层）已在 CPU 与 GPU 两侧实现并逐像素对拍一致**。


### P3 dogfooding：把示例做成真应用 —— 第一次真去跑就抓到「示例一直跑不起来」

`window_demo` 原本只是展示页（动画 + 混合模式画廊 + 文字），没有滚动与鼠标交互。
按计划的 dogfooding 要求，加了**可滚动、可点击选中的 200 行长列表**（默认进入，
Tab 切回展示页）：滚轮与方向键滚动、左键命中切换选中、每行一段 shaping、
**只画可见行**、固定表头。

**第一次真去跑它，启动即 panic：**

```
In Device::create_bind_group_layout, label = 'amiki_fine_bgl'
  Too many bindings of type StorageBuffers in Stage COMPUTE, limit is 8, count was 9.
```

两处叠加：引擎的 `amiki_fine_bgl` 需要 **9 个** storage buffer（第 9 个是图像
放置参数），而 `wgpu::Limits::default()` 只保证 **8 个**；`GpuRenderer::new()`
会上调并检查，但 `from_device()` **不检查** —— 示例走的正是 `from_device`。
**示例不在测试覆盖里，所以一直没被发现。**

修法（两层防线）：
- 新增 `amiki_gpu::required_limits()` 与 `REQUIRED_STORAGE_BUFFERS`，
  函数文档把那条报错原样写进去（让搜索报错的人能搜到这里）
- `from_device` **先校验设备上限再动任何资源**，不满足就返回明确错误，
  而不是把问题留给 wgpu 深处的校验失败

修后示例正常启动并持续渲染（运行 10 秒不崩、累计 CPU 1.89 s）。
另用 `CopyFromScreen` 截图确认过画面正确（表头文字、行内容、色条、隔行底色都正常，
静态时**脏 tile 0.0%**）。

> 中途有一次看到 `滚动 2120`，怀疑"应用会自己滚"。加无条件事件打印后复跑：
> **stderr 一条事件都没有、滚动为 0** —— 那次是环境真的送来了滚轮事件。
> **不是代码问题。加个打印比继续猜快得多。**

### P3：多后端验证 —— 修好 DX12 与 GL，"一套 WGSL 通吃"原本是假的

发布计划宣称"wgpu 一套 WGSL 通吃 + CPU 兜底"。**实测这句宣称，两个后端都跑不起来。**

**DX12 根本起不来。** `create_shader_module` 直接失败：

```
FXC D3DCompile error: amiki_fine(…): error X3511:
forced to unroll loop, but unrolling failed.
```

根因在**我们自己的 WGSL**：FXC 看到常量次数的循环会尝试展开以把下标静态化，
而下标依赖运行期值。三处这种写法（`int_clamped_linear` 的排序网络 `ij[st]`/`ts[p.x]`、
`blend_color` 的 case 6/7/9 `o[i]`）全部显式展开后通过，且**数值逐位相同**。

**GL 编不过（一个字符）。** naga 生成的 GLSL 是 `textureSize(sampler2D, uint)`，
驱动要 `int` —— 来源是 4 处 `textureLoad(clip_texN, gid, 0u)`。
把 mip 层参数写成 `0`（i32）即可：GL 从 **12 条失败**降到 **1 条**。

**GL 的位图读不出来 → 改为显式拒绝。** 经采样器得 `(0,0,0,1)`
（GL 规范对"不完整纹理"的规定返回值），直接 `textureLoad` 得 `(0,0,0,0)`。
已排除 LOD 钳位、`textureSampleLevel` vs `Grad`、辅助函数内联、
`write_texture` 行距对齐、手工双线性（全部实测无效），判断是 wgpu/naga
的 GL 后端在 `texture_2d_array` 上的问题。按项目一贯约定**显式报错，不静默画错**。

**结果**：默认（→Vulkan）**145 通过 / 0 失败**、DX12 **20/0**、GL **20/0**。

### P3：启动开销 —— 定位结论推翻了计划原先的假设

分段实测：**`Instance` + 请求适配器占 86%**，而**全部 21 条管线只占 12.7 ms**。
所以计划里写的"用 PipelineCache / 惰性创建"**是错的**（最多省 13 ms）。

| 配置 | 冷 | 热 |
|---|---|---|
| 默认（全部后端 → 本机选 Vulkan） | 337 ms | 202 ms |
| `AMIKI_BACKENDS=vulkan` | **154 ms** | 53 ms |
| `AMIKI_BACKENDS=dx12` | 2272 ms | 2093 ms |

**结论：有条件达标** —— 默认未达标，显式指定后端达标。默认保持全部后端，
因为默认值要保证"哪台机器都能起来"（一度改成默认只用 Vulkan，快一倍，已改回）。
DX12 慢是 FXC 运行期编译的固有代价；`static-dxc` 需**构建期联网下载**
，与项目轻依赖取向冲突，不采用。

### P2-2 字距（已完成，并修掉一个相关真 bug）

字距**本来就是生效的**（`AV` 收紧 0.0684 em ≈ 7%），缺的是**验证** —— 已补
`kerning_is_applied` 与对照 `non_kerned_pair_is_not_tightened`。

**但排版压根没用 shaping 的宽度（真 bug）**：`measure_chars` 逐字符累加
`advance_width`，**既没有字距也没有连字**，而同一文件的 `measure_run` 用的是
shaping 结果。于是**排版宽度 ≠ 渲染宽度**。

修法：`amiki-text` 新增 `shape_clustered`（返回字形及 `cluster` 字节偏移），
排版整段 shaping 一次并把推进量映射回字符边界（连字共享 cluster ⇒ 不可拆）。
实测 `AV` 行宽 **131.15**（逐字符累加会得到 137.99），与渲染宽度差 < 0.05 px。

### P2-1 CJK 断行（已完成，字体入库待办）

**实现禁则处理**（此前完全没有）：收尾标点不得在行首、起始括号不得在行尾。
`pick_break` 抽成**纯函数**，理由是**可测** —— 禁则只与字符有关、与字体无关，
而仓库里没有 CJK 字体，端到端排版验证不了它。6 条单元测试 + 1 条端到端
（12 行、禁则通过、内容不丢字）。**两条都验证过有牙**：停用禁则即失败。

> 第一版单元测试其实没牙（只有 1 条失败），因为"行尾禁则"那条无论禁则开不开
> 都会通过。已重写成"让被禁断点恰好成为最后一个候选"。

**未完成**：计划要求 `assets/fonts/` 里有 OFL 中文字体，仓库里没有
（我没有下载二进制字体的手段；系统字体非 OFL 不能入库）。
端到端测试因此**依赖机器上有系统 CJK 字体**，干净环境上会跳过并打印原因。

### P1-1 提取优化（部分完成）

容量预留有效（帧路径 1080p×10k **9.93 → 9.34 ms**）。
**每帧起线程被实测否定**：固定开销约 1.2 ms（`std::thread::scope` 每帧建/销线程），
10k 场景 942 µs（串行）vs 1346 µs（并行）、1000 场景 60 vs 370。
已按实测撤掉并行，并写明正确做法是**常驻线程池**。

> **测量教训**：同一份串行算法在隔离基准里 940 µs，帧内 `提取` 分段却报 2.0–2.4 ms
> （与 GPU 争用带宽）。**隔离基准给的是下界。** 另有一次 1278 µs 是负载离群值
> —— **单次测量不足以下结论**。

> **另一条测量教训**：多后端排查中，我从"最差像素位移 + 最大差 255→250"
> 推断 `textureLoad` 能取到数据，**这是错的** —— 250 是测试场景垫底的背景色。
> **只看差值大小不足以判断，必须看实际值。**

### P0-3 帧路径的层合成（已完成）

GPU 帧路径（`render_into` / `render_into_persistent`）此前对含层场景**整体报错**。
现在它在 **GPU 上直接合成层**（compute pass，不读回）：

| 滤镜 | 与 `render_blocking` | 与 CPU | 帧持久路径 |
|---|---|---|---|
| `None`（分组） | 0 | 0 | 0 |
| `Blur` | 1 | 1 | 1 |
| `DropShadow` | 1 | 1 | 1 |
| `DropShadow`（sigma=0） | 0 | 0 | 0 |

**为什么是 compute 而不是 render pass**：调用方提供的输出纹理只有
`STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC` —— 没有 `RENDER_ATTACHMENT`
也没有 `COPY_DST`，render pass 写不了、结果也拷不回去。

**过程中修掉的缺陷**（全部由对拍暴露）：

1. **`inflate_by_filter` 的 DropShadow 范围算错**（**CPU/GPU 共用的真 bug**）：
   原点也加了 `dx`，那是平移而非并集 → 正偏移时裁掉左边/上边。
   sigma 大时被模糊外扩掩盖，`sigma=0` 才暴露（差 156/255）。
2. **调度原点**：着色器直接拿 `gid` 当像素坐标、没加 `rect.xy` → 合成整体
   平移到左上角（差 179/255）。
3. **乒乓写错**：第二、三轮横向 pass 仍读原始内容 → 等于只模糊一遍（差 45/255）。
4. **`target` 是 WGSL 保留字** → 着色器解析失败。
5. **参数传反**：`composite_pass` 的 `layer_tex` 与 `shadow_tex` 位置写反 ——
   于是"模糊后的内容"被当成图层内容、阴影拿到未模糊的内容。症状是层边缘出现
   一圈不该有的淡影（7567 个像素不同，首个差异点的值恰好等于模糊值）。
   **这个错误只跟 `render_blocking` 对比会被漏掉**（两者都变了），
   必须同时跟 CPU 对比才暴露 —— 所以测试里两个基准缺一不可。

另把模糊从**线性采样改为整数抽头**（与 CPU 滑动窗口逐抽头一致）：线性过滤即使
采样点落在纹素中心也会因浮点误差混入相邻纹素，在真实半径外漏出微小数值，
三遍累积成假阴影。

### P0-2 含层场景的增量渲染（已完成）

修掉两个**静默错误** —— 它们共同导致"有滤镜的界面从第二帧起悄悄丢掉滤镜"：

1. `PersistentRenderer` 的增量分支**从不合成层**（首帧走全量所以正确）
2. `edit_damage` 只遍历 `draws`，**完全忽略 `layers`** → 层内改动产生零脏区

既有的层增量测试覆盖的是 `render_damage`（含层时回落全量，那条路是对的），
**`update` 是另一条路**，洞就在这里。

**修法**：三个缓冲分工 —— `base`（仅图元）/ `image`（最终）/ `layers`（已滤波
缓冲跨帧缓存）；脏 tile 上 `image ← base` 再按原顺序叠层。把 `composite_layer`
拆成 `layer_buffer_from_content`（渲染+滤波，缓存的单位）与 `blend_layer_rect`
（合成到指定矩形），让**全量与增量共用同一份数学**。层缓存按**内容键**配对
而非下标（层增删/重排后下标会错位）。`inflated_bounds` 从 `amiki-raster`
移到 `amiki-scene`，好让 damage 与光栅化共用同一份外扩规则。

**实测**：

```
M4 增量（无层静态界面）：   3.5% tile 重画（省 96.5%）
P0-2 增量（带投影静态界面）：3.5% tile 重画（省 96.5%）   ← 此前是 100%
```

**测试**（+16）：4 条层脏区（`amiki-scene`）+ 11 条层增量
（`amiki-raster`，核心是 60 轮含层随机编辑与全量逐字节对拍）+ 1 条把原先
**侥幸通过**的测试改成真正有判别力的版本（原版只改层外图元、脏 tile 恰好
不与层相交，所以修不修都能过；改成改层内容后失败：5580 像素不同）。

### P0-1 描边（已完成）

新增 `amiki-scene/src/stroke.rs`：把路径转成**闭合轮廓**，复用既有填充管线。

**为什么不是另推一套闭式覆盖率**：填充用有符号面积 + 绕数做解析覆盖率，
对简单闭合轮廓给出精确面积。若把描边拆成「每段一个四边形 + 拐角图形」直接
累加，转角处必然重叠 —— 有符号面积对重叠是**相加**而非**取并**。
走轮廓路线后描边**不需要任何新的覆盖率数学**，抗锯齿精度与填充完全一致，
而且 **CPU 与 GPU 两侧都不需要改动**。

范围：`miter`/`round`/`bevel` 拐角（含 miter limit）、`butt`/`round`/`square`
端帽、开放折线（1 条轮廓）与闭合折线（**环**：外圈 + 反向内圈）、
以及任意 paint（纯色/渐变/位图）。

门面 API：`Canvas::stroke_path_with` / `Canvas::stroke_path`。
线宽是**用户单位** —— 画布变换会同时放大线宽，且**只作用一次**。

**过程中修掉的四个真实缺陷**（都由解析精确面积暴露）：

1. 顶点处漏推「后一段的起点偏移」，bevel 需要的两个点少一个 → 转角被切掉
   （实测面积 632，正确 712）
2. 方端帽只推一个角点 → 退化成三角形（618 vs 636）
3. 闭合路径多推起点的顶点偏移（1264 vs 1280）
4. 闭合路径被拼成单一轮廓 → 从内圈跳到外圈成蝴蝶结（960，正确 1248）；
   改为产出两条**绕向相反**的独立轮廓

**另有两次是我的期望值算错而非代码错**，值得记录：
「两矩形并集」不等于描边区域（并集并不包含转角那一块），所以 bevel 是在并集上
**加**三角形；miter 恰好等于「两矩形面积直接相加」——那不是重复计数，
是方形转角的正确并集。另外我最初的 miter limit 测试取的角其实不尖
（转角 11°，miter 比仅 1.005，没超限）。

**验证**：`amiki-scene` 内 9 条（解析精确面积）、`amiki-raster` 内 4 条
（与超采样绕数参考对拍，含「环心必须为空」）、`amiki` 门面 4 条（含
GPU/CPU 对拍与「变换恰好作用一次」）。

### M3.6：帧路径的投机分配（已完成）

稀疏结构的规模只有 GPU 测量后才知道，原实现必须"提交 → 读回规模 → 分配 →
再提交"，两次读回各是一次**同步停顿**（实测 `读回1` = 2.35 ms，其中约
1.4 ms 是纯 fence/map 延迟）。

改为**投机分配**：用该场景上一帧实测到的规模作为本帧容量，直接分配、不再停顿；
帧尾读一次实际用量做溢出检测与容量更新。容量不足则作废该帧重做。

**实测**：

```
720p×64:   756 µs → 486 µs   (−36%)
1080p×10k: 10.41 ms → 10.00 ms (−4%)
```

**容量必须按场景分表**，不能是单一全局值 —— 两种全局做法都实测失败：
直接覆盖会让换场景即溢出；只增不减会变成过度分配（720p×64 曾退化到 2.09 ms）。

**这一版推翻了此前的两条诊断**（都是实测否定的）：

1. ❌「每帧重建约 26 个缓冲导致缺口」→ 分配实测仅 **0.13 ms**
   （wgpu 缓冲创建惰性，显存由驱动内部池化）
2. ❌「3 次 submit 同步点」→ 三次编码**与提交合计仅 0.56 ms**

真正的大项是 `读回1` 的同步停顿与 CPU 侧的 `提取`（属 M7）。

### 测试方法上的两个坑（值得记住）

1. **测试要有牙**：溢出重做测试第一版检查的是溢出之后**下一帧**的像素，
   那时容量已被修正，即使完全取消重做也照样通过。必须检查**溢出帧本身**，
   并用"临时取消重做"验证测试确实会失败（实测 FAILED，最大通道差 255）。
2. **`Copy-Item -Force` 保留源文件旧修改时间**，导致 `cargo test` 复用旧二进制。
   曾被误判为"测试非确定性"，实际是构建陈旧。还原文件后须显式刷新时间戳。

### M6：滤镜的 GPU 实现（已完成）

GPU 侧离屏层现已实现并与 CPU **逐像素对拍一致**：

| 滤镜 | 最大通道差 |
|---|---|
| `Filter::None`（仅分组） | **0** |
| `Filter::Blur` | **1** |
| `Filter::DropShadow` | **1** |

实现方式：层内容由**同一套细粒度管线**渲染到独立纹理（不复制着色逻辑），
模糊在 GPU 上以**可分离盒式**完成，合成在主机侧共用 `composite_premul`。

### 修复过程中定位的三个真实缺陷

1. **同一命令缓冲内多趟共用 uniform 缓冲**（真正的"看起来没跑"的根因）：
   `blur_rgba` 原本用一个 `params_buf`，在同一个命令缓冲里记录横、纵两趟并
   写了两次参数。队列顺序"写参数1 → 写参数2 → 提交"使**两趟都读到第二次的值**，
   **横向那趟从未执行**。测试数据恰好是"竖贯整幅的横条"（纵向模糊对它无可见
   效果），因此一直表现为"输出与输入逐字节相同"。修法：横纵各用独立缓冲。

2. **模糊前多加了预乘**：我一度按"模糊应在颜色上做"加了预乘/反预乘，但 CPU 的
   `blur` 是对 4 个通道**直接平均**（直线色语义），于是 `Filter::Blur` 的差异
   从 1 **恶化到 64**。既然对拍基准是 CPU 实现，就必须与它一致；
   要改成预乘语义得**同时改两端**，那是独立的决定。

3. **投影的阴影必须从"未模糊内容"生成**：CPU 的投影是"用 alpha 形状 + 固定
   阴影色先着色、再模糊"；若直接拿"模糊后的层内容"当阴影，语义就变成
   "模糊内容再着色"（实测差 125~162）。现在阴影遮罩在主机侧按与 CPU 完全相同的
   步骤生成（着色 → 盒式模糊），GPU 仍承担**内容**模糊。

### 方法论（本次最大的收获）

严格按**最小实验**逐级隔离，每一步只引入一个变量：

| 步骤 | 只引入的变量 | 结果 |
|---|---|---|
| 1 | 渲染纯色到纹理再读回 | ✅ 通路可用 |
| 2 | fragment 采样 | ✅ 最近邻往返逐字节一致 |
| 3 | 直接驱动模糊着色器（box_w=1） | ✅ 恒等拷贝正确 |
| 3b | 直接驱动模糊着色器（box_w=9） | ✅ 确实模糊 → 锁定在 `blur_rgba` 编排 |

对比此前"把通路与卷积混在一起调"，这是本质区别 —— 后者浪费了大量轮次。

### M6：滤镜（层树）—— CPU 参考实现

- **为什么需要层**：模糊读取的是**周围像素**，而周围像素只有整层画完之后才存在，
  因此无法逐图元计算。必须"渲进独立缓冲 → 整层滤波 → 合成回去"。
- **`Layer` + `LayerBounds` + `Filter`**（`amiki-scene`）：层持有子场景 + 滤镜 +
  不透明度 + 混合模式。层在**所有图元之后**合成，保证"层内先画完、再整层滤波"。
- **边界由调用方显式给出**：被模糊的层影响范围会超出内容包围盒，引擎自行推导
  必须算准每个滤镜的外扩量，任何保守不足都会让边缘被切掉且很难察觉。
  引擎另提供 `inflated_bounds`（按 3σ 外扩）作为默认余量。
- **三次盒式模糊近似高斯**（`amiki-raster/filter.rs`）：真二维高斯是 O(r²) 每像素，
  三次盒式在 O(1)（滑动窗口）下即可良好逼近，且可分离。盒宽
  `w ≈ sqrt(12σ²/n + 1)` 并强制为奇数以保证窗口对称。
- **层缓冲存预乘**：模糊是对颜色加权平均，若存直线 alpha，透明区域的
  "未定义颜色"会被平均进来，边缘出现暗边。
- **`DropShadow` 语义是"附加"而非"替换"**：保留原内容，在其**后方**叠加模糊着色的副本。
- 门面 API：`Canvas::push_layer_filtered(bounds, filter, opacity, blend, |canvas| {...})`。

### M6：位图绘制（CPU + GPU 一致）

- **`Image`**：线性（非预乘）RGBA8。引擎只认这一种内存布局，调用方负责解码与
  色彩空间转换 —— 于是核心 crate 都不需要引入任何图像解码库。
- **`sample_bilinear`**：刻意不用最近邻 —— 图像常被缩放，最近邻会明显锯齿抖动，
  而本引擎的输出质量基准是解析覆盖率，采样阶段不该成为质量瓶颈。
- **`ImagePlacement` 用三个点**（origin + u_axis + v_axis）描述放置，而非四个角：
  图像放置本质是仿射映射，三点即完整确定，天然表达旋转/切变。退化（行列式≈0）
  时返回 `None` 由调用方决定，而不是除零产出 NaN 像素。
- GPU 侧把位图上传为**纹理数组**（每张一层），尺寸不一致时左上对齐、越界透明，
  **不静默缩放**（缩放语义应由调用方通过放置显式决定）。
- 示例：`headless_png` 输出 `out/amiki_showcase.png`（位图 + 渐变 + 文字 + 滤镜同场）。

### M6：渐变（CPU + GPU 一致）

- **模型**（`amiki-scene`）：`GradientKind` 支持**线性 / 径向 / 扫描**，
  几何存**设备空间**（着色在逐像素阶段，画布变换已作用完）。
  构造时按 `t` 排序；色标 < 2 或半径非正时返回 `None`（拒绝退化输入）。
- **解析采样，刻意不查表**：常见做法是把渐变烘成 256 项 LUT，
  但那会把着色量化到 **1/255**。本项目对解析覆盖率有质量承诺，
  不该在着色阶段引入额外近似 —— 因此逐像素解析求值。
- **CPU**（`amiki-raster`）：`BinnedScene` 增加 `gradients` 与 `gradient_index`
  两张平行表。用独立索引表而非在颜色值里编码下标 ——
  后者依赖浮点精确相等，一旦有颜色恰好碰撞就会静默出错。
- **GPU**（`amiki-gpu` + `sparse_fine.wgsl`）：`grad_data` 打包为定长条目
  （`MAX_GRAD_STOPS = 8`），下标经 `draw_meta.w` 传入（该分量此前闲置）。
- **编码流**：渐变**完整编码**（类型 + 几何 + repeat + 色标），不降级为实心色。
- 示例：`headless_png` 的文字改用渐变着色，同时验证 M5 与 M6 正交可组合。

### 本阶段修复的缺陷（M6）

- **GPU 渐变在首色标之外外插而非 clamp**：WGSL 的区间查找从 `i = 1` 开始，
  当 `t` 为负时第一次比较即成立，于是以负的 `k` 在 `stop[0]` 与 `stop[1]`
  之间**外插**，得到比首色标更极端的颜色（实测与 CPU 差 19/255）。
  两色标场景**恰好掩盖**了这一点（首色标之外的颜色本来就等于首色标），
  三色标才暴露。现显式 `clamp(t, 0, 1)`。
- **增量路径会丢掉层**：`render_damage` 原本不合成层，含滤镜的场景走增量渲染会
  **静默少掉阴影/模糊**。现含层场景回落到全量渲染（慢但正确），并加测试锁定。
- **GPU 会静默忽略层**：GPU 路径尚未实现离屏层。原状会把阴影/模糊**悄悄丢掉**，
  调用方毫无察觉。现 `reject_layers` **显式报错**，由上层决定回落。
- **`image` crate 未启用 jpeg feature**：JPEG 无法解码，而加载函数用 `.ok()?`
  把"解码失败"与"文件不存在"混为一谈，误报成"未找到 img.jpg"，浪费一轮排查。
  现启用 jpeg 并把解码错误单独报出。
- **`Canvas::text` 位置未随字号缩放**：`TextCache` 把字号折进坐标却仍按 em
  平移位置，而 `expand_text_run` 用 `scale.then(translate)`（平移量也被缩放）⇒
  `size≠1` 时字形摆到错误位置。由"缓存展开 vs 直接展开逐点比较"的测试抓出。
- **`shape()` 忽略 advance**：初版只用 offset，会让所有字形叠在同一处。
  advance 才决定位置，offset 只做微调（kerning/mark），两者都必须用。
- **排版断行退化成硬切**：`greedy_break` 在"本行内没有可断点"时写成
  `cut = end`，那正是**硬切**——会把长 URL 截成两段并破坏内容语义。
  初版甚至没有触发 `overflow`。由"字符不得丢失 + 必须置 overflow"的测试抓出。
- **`render_damage` 陈旧像素**：`shade_tile` 在 tile 无绘制时保留原像素（那是
  "未脏 tile 复用上一帧"的正确语义），但**已脏**的 tile 必须从零重算，
  否则"某 tile 从有内容变无内容"会永久残留。现脏 tile 先清零再着色。
- **`Damage` 三态建模**：初版用 `Option<DamageRect>` 把"全脏"与"无脏"混为一谈，
  导致 4 个测试失败。现用显式 `DamageState { All, Clean, Rect }`。

### 工程质量（0.2.0-alpha）

- 测试 36 → **99**（GPU 17 / CPU 正确性与滤镜 37 / 排版 9 / 场景 12 / 字体 8 /
  门面 2 / 快照 1 / 几何 4 …）
- `cargo clippy --workspace --all-targets`：**零警告**
- 规模：Rust 约 12100 行 / 20 文件；WGSL 约 1510 行 / 13 模块
- crate 分层：`amiki-math` / `amiki-scene` / `amiki-raster` **零第三方依赖**；
  `amiki-gpu` 仅 wgpu/bytemuck/pollster；字体栈仅在 `amiki-text`；排版在 `amiki-layout`
- 引入版本控制（23 提交），`.gitattributes` 统一 LF

### M4：damage region（增量渲染）

- **tile 级脏区模型**（`amiki-scene::Damage`）：三态显式建模
  （`All` 全脏 / `Clean` 无脏 / `Rect` 部分脏）。
  用枚举而非 `Option`——本模型有三态，`Option` 会把"全脏"与"无脏"混为一谈。
- **编辑契约**（`edit_damage`）：从旧场景与新场景**自动推导**脏区，
  把"改动 draw 必须标记旧∪新包围盒"这条易漏的规矩变成 API 默认行为。
  包围盒取所有控制点凸包（贝塞尔必落在其内）⇒ 只多标不漏标。
- **CPU 增量渲染**（`render_damage` / `render_tiles` / `PersistentRenderer`）：
  只重算与脏区相交的 tile，其余复用上一帧像素。
- **GPU 只派发脏 tile**（`render_into_persistent`）：fine 内核由
  `global_invocation_id` 推导坐标，故**只缩小 dispatch 范围**即可精确重写脏 tile
  —— 不需要改任何 WGSL 或着色数学。
- **收益实测**（`headless_png` 输出，可复现）：

  | 场景 | 平均重画 tile | 省下 |
  |---|---|---|
  | 全屏动画 | 91.5% | 8.5% |
  | **静态界面**（背景/侧栏/面板 + 一个滑动点） | **3.5%** | **96.5%** |

  这界定了 damage region 的适用边界：**静态内容为主 + 少量元素变动**收益是量级的。

### M5：文本（字形轮廓即路径）

- **设计决策**：字形轮廓作为**普通 path** 进入既有管线，引擎**无文本专用代码**。
  免费获得：复用压平/分箱/解析覆盖率/混合/裁剪/damage region；
  **任意缩放不糊**（抗锯齿来自真实面积积分，非预渲染字形位图）。
- **`amiki-text`（新 crate）**：本项目**唯一**引入第三方依赖处
  （`ttf-parser` 0.25 + `rustybuzz` 0.20）。核心五 crate 保持零字体依赖，
  字体栈可整体替换。
- **字体资产**：`assets/fonts/Inter-Regular.ttf`（Inter 4.1，411 KB，SIL OFL 1.1），
  附 `LICENSE.txt` 与来源说明。原始 33 MB 发布包不随仓库分发。
- **门面 API** `Canvas::text(&TextRun, origin, paint, outline_lookup)`：
  接收**已 shaping** 的排版结果与取轮廓闭包，故门面不依赖字体栈。
  每字形一条独立 draw（可被 damage region 独立处理）。
- 示例：`headless_png` 输出 `out/amiki_text_demo.png`（真字 "Amiki2D"）；
  `window_demo` 显示随时间缩放的真字，直观展示"放大不糊"。

### 本阶段修复的缺陷

- `render_damage` 曾只调用 `shade_tile`，而后者在 tile 无绘制时保留原像素
  ⇒ "某 tile 从有内容变为无内容"（删 draw）会永久残留陈旧像素。
  现在脏 tile 先清零再着色，未脏 tile 才复用。
- `TextCache` 曾把字号折进坐标却仍按 em 平移位置，而 `expand_text_run` 用
  `scale.then(translate)`（平移量也被缩放）⇒ `size≠1` 时字形摆错位置。
- `shape()` 初版忽略 **advance** 只用了 offset ⇒ 所有字形叠在同一处。
  advance 才决定位置，offset 只做微调（kerning/mark），两者都必须用。
- 演示字形初版是"单线闭合路径"，面积为零 ⇒ 渲染不出像素；
  真实字体轮廓是**填充区域**，已改为有宽度的矩形笔画验证。

### 工程质量

- 测试 36 → **62**（GPU 14 / CPU 正确性 24 / 场景 12 / 字体 5 / 门面文本 2 /
  快照 1 / 几何 4）
- `cargo clippy --workspace --all-targets`：**零警告**
- 规模：Rust 约 8700 行 / 14 文件；WGSL 1371 行 / 13 模块
- 依赖边界：`amiki-math`/`amiki-scene`/`amiki-raster` **零第三方依赖**；
  `amiki-gpu` 仅 wgpu/bytemuck/pollster；字体栈仅在 `amiki-text`

### 正确性不变量（本阶段新增）

- **INV-3**：任意 invalidate 序列下，增量渲染与全量渲染**逐字节一致**
  —— 200 轮随机序列 + 40 轮随机场景编辑（增/删/改 draw）验证
- GPU 侧：全脏 / 局部脏 / 空脏区三类派发均与全量**逐字节一致**
- 文本：逐字形与 **16× 超采样**裁判对拍，容差 0.03（与几何图形同一容差）

### 已知边界：布局不在渲染器职责内

需要在文档里写清楚，因为很容易搞混：

| 层 | 职责 | 归属 |
|---|---|---|
| shaping | 字符 → 字形 id + 字距/连字 | `amiki-text` |
| **布局** | **断行 / 行高 / 对齐 / 段落** | **独立可选层（未建）** |
| 渲染 | 字形轮廓 → 覆盖率 → 像素 | `amiki-raster` / `amiki-gpu` |

业界共识是把两者分开：Skia 的 `SkTextBlob` 只承载"预先定位好的字形"，
不含断行与段落；HarfBuzz 亦明确声明不做断行
（[What HarfBuzz doesn't do](https://harfbuzz.github.io/what-harfbuzz-doesnt-do.html)）。

因此 `amiki-text::shape()` 产出"带位置的字形序列"（`TextRun`）、
门面 `Canvas::text()` 消费它——**渲染器一侧没有排版概念，也不应该有**。
布局层将是独立且可选的：只想画一行字的调用方不必拖进整套排版引擎。

---

## [0.1.0-alpha] —— 首个 Alpha

首个可运行的里程碑版本：**M1–M3.5 完成，M3.6 起步**。

### 核心能力

- **解析式覆盖率光栅化**：逐像素线积分闭式解，精确 AA，无状态并行结构
  （三端共享同一份数学：`amiki-raster` / `fine_math.wgsl` 双入口）
- **双压平器**：CPU 误差界自适应二分 + GPU 均匀压平（闭式可证明段数界，附性质测试）
- **稀疏桶表全 GPU 分箱（M3，默认路径）**：strip 归并 → tile 行活跃列表 →
  逐 tile 桶表 → 稀疏单元（成员 CSR + carry 定点桶 + 段端点直存）。
  **显存与画面实际内容成比例**，1080p × 10k 曲线实测 20.93 GB → 34.5 MB（约 600×）
- **稠密网格分箱（M2f）**：保留为对拍基准与诊断路径
- **16 种混合模式**：W3C compositing 全集（含 HSL 非可分离四模式）
- **裁剪合成**：覆盖率纹理逐像素乘入，≤4 层嵌套取交集
- **AMK1 编码流**：紧凑二进制场景格式（缓存/传输/绑定交换）
- **门面 API**：`Canvas` 立即模式（变换栈/混合/裁剪栈）+ headless PNG + winit 动画示例

### 本版新增（M3.5 工程硬化 + 可诊断性）

- **稀疏自洽检查** `check_sparse_consistency()`：**不依赖稠密参照**，逐级断言
  strip 区间布局 / 逐 tile 桶表一致性（含画家序）/ 单元 CSR 分区 /
  carry 桶前缀逐位 / 段端点与成员计数。填补"大场景稀疏没有裁判"的空洞
  （稠密在 1080p 约 129 draw 即撞 128 MiB 绑定上限）
- **稀疏诊断入口** `GpuRenderer::dump_sparse()`：一次调用转储全部稀疏结构
- **稠密容量预检** `dense_capacity()`：超限返回**可读错误**（含所需/上限 MB 与降级建议），
  取代原先 `create_bind_group` 内部的 wgpu 校验层 panic
- **CPU 分段计时** `FrameProfile`：提取/分配/上传/读回/提交/时间戳解析逐段可读
- **逐 pass GPU 归因**：`PassTimings` 槽位上限 8 → 32，稀疏路径从 5 个粗粒度槽位
  细化为 **10 个逐段可归因** pass
- **渲染输出快照**：8 个锚点场景用 FNV-1a 钉住 CPU 参考输出，挡住"两个实现同时被改错"
  的静默漂移（已验证：改动 `lum()` 系数会被精确指认为 `blend16`）

### 本版新增（M3.6 帧路径性能，进行中）

- **MAP_READ 暂存缓冲池**：读回不再每次新建缓冲 + 冷映射。
  实测 **720p×64 曲线帧路径墙钟 6.59 ms → 1.03 ms（−85%）**；读数 `读回2`
  从 1.37 ms 降至 0.08 ms

### 工程质量

- `cargo test --workspace`：**36 passed / 0 failed**
  （13 GPU 对拍+自洽 / 4 几何 / 16 CPU 正确性 / 1 输出快照 / 2 编码流）
- `cargo clippy --workspace --all-targets`：**零警告**
- GPU/CPU 逐像素对拍 ≤2/255；稀疏 vs 稠密**逐字节**一致（`max_diff == 0`，6 场景）
- 规模：Rust 6425 行 / 10 文件 + WGSL 1371 行 / 13 模块
  （`amiki-gpu` 3722 / `amiki-raster` 1146 / `amiki-scene` 579 / `amiki` 529 / `amiki-math` 449）

### 已知限制

- **无描边**、**仅 nonzero 填充规则**（无 even-odd）
- **无文本、渐变、图像、滤镜**（M4 计划）
- **无 damage region / 保留模式增量**（M5 计划）
- CPU 引擎为单线程参考实现，无 SIMD/JIT（M7 计划）
- `GpuRenderer::new()` 首次创建约 0.5–0.7 s（21 条管线编译，无 `PipelineCache`）；
  离屏一次性出图场景受此支配
- 1080p×10k 帧路径墙钟约 12 ms，与 GPU 侧 2.97 ms 仍有约 9 ms 缺口：
  来自每帧缓冲重建 + 3 次 submit 同步点（M3.6 剩余项）
- 注释与文档为中文，代码标识符为英文

### 后续计划（按依赖顺序）

1. **M3.6 完成**：每帧资源池化 + 合并 submit → 目标把 1080p×10k 墙钟压向 GPU 时长
2. **M4 damage region**：tile 级脏区 + 增量提交 + 层树（提前到文本之前，属跨切面改动）
3. **M5 文本**：rustybuzz shaping + 字形轮廓直入解析覆盖率管线（变换后 + 量化尺度两级缓存）
4. **M6 渐变/图像/滤镜**、**M7 CPU 引擎 SIMD+多线程**、**M8 C ABI/WASM/HDR**

---

## 版本命名说明

`0.1.0-alpha` 的语义：核心渲染管线（M1–M3）已可用且经双端对拍验证，
但**内容能力尚不完整**（无文本/渐变/图像/滤镜），**API 未承诺稳定**。
适合评测渲染质量与性能，不适合承载生产应用。
