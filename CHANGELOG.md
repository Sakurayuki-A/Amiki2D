# 变更记录

本文件记录 Amiki2D 的版本演进。格式参考 Keep a Changelog，版本号用
`主.次.修订[-预发布]`。

---

## [未发布] —— M4 damage region + M5 文本

> 尚未打版本号。M4 功能闭环，M5 已能渲染真实字体文本但**尚无换行排版**。

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
