# Amiki2D

**版本 `0.9.0-alpha`** —— 新增 **状态栈** `Canvas::save()` / `restore()`：保存与
恢复**四项状态**（变换 / 混合模式 / 裁剪栈 / 填充规则），这是 UI 框架"局部状态"的
基本假设。深度上限 32，**溢出与下溢都显式报错**（不静默）；层内栈独立；**不进 AMK1**。
不调用它的代码逐字节不变（三示例 SHA256 与 0.8.0 完全相同）。C ABI 符号 **82 → 84**。
本轮还抓修了 C ABI 的两处**镜像漂移**（`clip_depth` 与 `fill_rule` 在 `restore` 后
没同步，会导致 `pop_clip` 误判、`get_fill_rule` 报出与实际绘制不一致的值）。
详见 [变更记录](CHANGELOG.md)。

`0.8.0-alpha` 新增 **even-odd 填充规则**，与既有 nonzero 并存（默认仍是 nonzero）。
实现方式是**几何预处理**：把奇偶区域分解成互不重叠的梯形再走原有管线 ——
覆盖率数学与 **WGSL 一行未改**；顺带查明了既有的精度限制 B16。
`0.7.0-alpha` 新增 **字体回退**（一段文本里多字体，老 API 零破坏）。
`0.6.1-alpha` 修复了 **GPU 增量渲染的一个静默错误**（脏区不在左上角时被重算的是
错误的 tile）；`0.6.0-alpha` 新增 **mipmap**（大幅缩小的位图不再走样）。
C ABI（0.4.0）让引擎能被 Rust 之外的语言调用（C / C# / Node 三个示例输出**逐字节一致**；
三者均为**原生 FFI**，Node 走 koffi 而非 WASM）；
WASM（0.5.0）让它在**浏览器与任何 JS 运行时**里跑（模块 **1 037.3 KB**（0.10.9 实测 1 033.2 KB，
模块化拆分 +4.1 KB；启用可选 `codec` 会 +343.9 KB）**，
与原生 CPU 后端输出**逐字节一致**）。`0.3.0-alpha` 的发布判据 6 条**全部满足**。
核心渲染管线（M1–M3）、工程硬化（M3.5）、增量渲染（M4）、文本（M5）、
渐变/位图/滤镜（M6），加上 P0 能力门槛（描边、含层增量、帧路径层合成）、
P1-1 性能（提取 **2.03 → 0.59 ms**）与 P3 工程成熟度六项。
详见 [发布报告](docs/RELEASE_0.3.0-alpha.md)、[现状总览](docs/OVERVIEW.md)
与 [变更记录](CHANGELOG.md)。

**已能**：矩形/圆/任意贝塞尔、自相交填充（**nonzero 与 even-odd 两种规则**）、
16 种混合模式、裁剪、**描边**
（miter/round/bevel + 三种端帽 + 闭合环）、真实字体文本（含**断行/行高/对齐/段落**
与 **CJK 禁则**）、**字体回退**（一段文本里多字体，缺字策略显式可选）、
渐变（线性/径向/扫描）、位图（旋转/缩放/平铺）、
模糊与投影（CPU/GPU 两侧逐像素一致），且**只重画改动处** ——
含离屏层的场景也有脏区收益（带投影的静态界面**省 96.5% tile**）。
**尚未**：双向文本、Knuth-Plass 全局最优断行、宽色域、各向异性过滤；
**跨厂商硬件未验证**（三个后端都在同一块 GPU 上验证，见[发布报告](docs/RELEASE_0.3.0-alpha.md)），
**WASM 的 WebGPU 构建也未验证**（当前是纯 CPU 构建，见 `docs/KNOWN_ISSUES.md` 的 A2）。

下一代通用 2D 渲染引擎 —— 目标不是复刻 Skia / Direct2D / Blend2D，而是站在它们肩上，以
**GPU 计算着色器 + 解析式覆盖率** 为主线，构建一个前沿、通用、强大的 2D 渲染底座。

## 核心理念

1. **前沿** —— 主线采用计算着色器路径渲染（Vello 所开创、仍在快速演进的方向）：
   路径压平 → 分箱 → tile 级解析式覆盖率积分，抗锯齿不是"近似"，而是数学上精确的面积积分。
2. **通用** —— 同一份场景编码，多个后端：GPU 引擎（wgpu：Vulkan / D3D12 / Metal / WebGPU）
   与 CPU 引擎（SIMD 软光栅）共享核心算法；无 GPU 的环境自动降级。
3. **强大引领** —— 场景以紧凑编码流表达，渲染与内容复杂度成比例（sparse strips 路线）；
   API 同时服务立即模式（Canvas）与保留模式（UI 场景图 / damage region）。

## 当前状态（0.3.0-alpha：发布判据 6 条全部满足）

**已实现能力**（GPU/CPU 双端逐像素对拍 ≤2/255，**151 个测试全绿**，clippy 与 cargo doc 零警告）：

- **解析式覆盖率光栅化**：逐像素线积分闭式解，精确 AA，无状态并行结构
- **双压平器**：CPU 误差界自适应二分 + GPU 均匀压平（闭式可证明段数界）
- **稀疏桶表分箱（M3）**：strip 归并 → tile 行活跃列表 → 逐 tile 桶表（活跃 tile
  列表 + 偏移）→ 稀疏单元（成员 CSR + carry 定点桶 + 段端点直存）。
  **显存与画面实际内容成比例**：1080p × 10k 曲线自 20.93 GB（稠密网格）降到
  **34.5 MB**（约 600×），空画布零开销；稀疏与稠密渲染**逐字节一致**（对拍断言）
- **稠密网格分箱（M2f）**：保留为对拍基准与诊断路径
- **16 种混合模式**：W3C compositing 全集（含 HSL 非可分离四模式）
- **裁剪合成**：覆盖率纹理 + 逐像素乘入，≤4 层嵌套取交集
- **AMK1 编码流**：紧凑二进制场景格式（缓存/传输/绑定交换）
- **可诊断性（M3.5）**：**稀疏自洽检查**（不依赖稠密参照，逐级断言 strip 区间 /
  tile 桶表 / 单元 CSR / carry 桶前缀 / 段端点）+ 稀疏全链路转储 `dump_sparse` +
  稠密容量预检（超限返回可读错误而非 panic）+ `FrameProfile` 主机侧分段计时 +
  逐 pass GPU 归因（稀疏 10 段）+ **8 锚点输出快照**（挡住两边同时改错的静默漂移）
- **帧路径提速（M3.6 起步）**：MAP_READ 暂存缓冲池 —— 720p×64 曲线帧路径墙钟
  **6.59 ms → 1.03 ms（−85%）**
- **基准套件**：criterion + GPU timestamp query（逐 pass 时长），§4.5 目标场景实测
  GPU 侧 2.55 ms（RTX 3050）
- **增量渲染（M4 闭环）**：tile 级脏区模型 + 编辑契约（`edit_damage` 自动推导脏区）
  + CPU 帧间持久化 + GPU **只派发脏 tile**。实测：静态界面平均只重画 **3.5%** 的 tile
  （省 96.5%），且与全量渲染**逐字节一致**（200 轮随机序列 + 40 轮随机场景编辑验证）
- **文本（M5）**：字形轮廓**作为普通路径**进入既有管线，引擎无文本专用着色代码。
  任意缩放不糊（抗锯齿来自真实面积积分，不依赖预渲染字形位图）。
  `amiki-text` 隔离字体栈（ttf-parser + rustybuzz），核心 crate 仍零字体依赖。
  字体度量（M5.1）已就绪，为排版层铺路
- **渐变 / 位图 / 滤镜（M6）**：线性/径向/扫描渐变与位图在 CPU/GPU 两侧**逐像素一致**；
  模糊与投影经**离屏层**实现，**CPU 与 GPU 双端均已实现并逐像素对拍一致**（差 0~1）。
  渐变**解析采样不查表**，避免 1/255 的着色量化
- **描边（P0-1）**：miter / round / bevel 拐角（含 miter limit）、butt / round / square
  端帽、开放折线与**闭合环**。用**解析精确面积**验证几何
- **含层场景的增量渲染（P0-2）**：层缓冲跨帧复用、层内容未变时不重滤波、
  层脏区经 `inflated_bounds` 传播。**带投影的静态界面省 96.5% tile**（此前为 0）
- **帧路径的层合成（P0-3）**：GPU compute 直接合成（不读回）
- **CJK 断行 + 禁则（P2-1）**：收尾标点不在行首、起始括号不在行尾；
  **OFL 中文字体随仓库分发**（Noto Sans SC）
- Canvas 立即模式 API（变换栈/混合/裁剪栈）+ headless PNG + winit 动画示例
- **C ABI（0.4.0）**：手写 `amiki.h` 契约（**72 个导出符号**）+ 三个绑定示例
  （C / C# / Node，输出逐字节一致）。**三个示例都走原生 FFI**：
  Node 用的是 [koffi](https://koffi.dev/)（纯 FFI，`npm install` 即用，
  **不需要构建期下载任何工具链**，也不改 Rust 侧一行代码）——
  它不是 WASM，与下面这条是两件独立的交付。
  引擎**绝不 panic 跨 FFI 边界**；panic 后对象中毒并显式拒绝后续调用。
  详见 [`crates/amiki-capi/README.md`](crates/amiki-capi/README.md)
- **WASM（0.5.0）**：`crates/amiki-capi/examples/wasm/` 下的独立交付——
  纯 CPU 构建（`--no-default-features`，模块 **1 025.7 KB（0.10.0 实测；启用可选 `codec` 会 +343.9 KB）**；0.5.0 时为
  950.7 KB，0.6.0 加入 mipmap 后 +15 KB，0.7.0 加入字体回退后 +22 KB），JS 加载器用
  **原生 WebAssembly API** 直调同一份 C ABI（不依赖 wasm-bindgen），
  与原生 CPU 后端输出**字节一致**。
  注意：浏览器里走 WebGPU **未验证**（A2）

**尚未实现**（明确的边界）：双向文本（阿拉伯/希伯来）；Knuth-Plass 全局最优断行；
**宽色域**（HDR 已于 0.5.0 落地）；各向异性过滤；WASM 的 **WebGPU** 构建（A2）。
CPU 引擎 SIMD/多线程**实测对这个负载无效**，且提取已达标。
理由见 [`docs/RELEASE_PLAN.md`](docs/RELEASE_PLAN.md) 第六节与
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) 的 C 节。

> ⚠️ **跨厂商硬件未验证**：三个后端全部跑通，但都在**同一块 GPU** 上。
> 已记为「已知未验证项」而非「已通过」，见[发布报告](docs/RELEASE_0.3.0-alpha.md)。

crate 一览：

- `amiki-math` —— 零依赖几何核心：Vec2/Affine、贝塞尔、双压平器
- `amiki-scene` —— 路径/绘制/裁剪/混合模型 + AMK1 编码流
- `amiki-raster` —— CPU 参考光栅器（与 GPU 逐式对应），兼作正确性基准
- `amiki-gpu` —— wgpu 计算着色器渲染器（稀疏/稠密双分箱 + 双 fine 入口 + 基准）
- `amiki` —— 门面 `Canvas` API

## 快速开始

```bash
cargo run --release -p amiki --example headless_png   # 输出 out/amiki_headless.png
cargo run --release -p amiki --example window_demo    # 打开动画窗口
cargo test  --workspace                               # 全量测试（含 GPU/CPU 对拍）
cargo bench -p amiki-gpu                              # 基准（含显存与逐 pass 时间戳）
```

## 文档

- [**现状总览**](docs/OVERVIEW.md) —— 一页看清"这是什么、现在到哪了、接下来做什么"
- [发布快照 0.6.0-alpha](docs/RELEASE_0.6.0-alpha.md) —— mipmap 交付物、验证证据、性能口径、已知问题
- [发布报告 0.3.0-alpha](docs/RELEASE_0.3.0-alpha.md) —— 门禁状态、能力清单、性能数据、已知问题
- [**C ABI**](crates/amiki-capi/README.md) —— 手写 header 契约 + 三个绑定示例（C / C# / Node）
- [**WASM 示例**](crates/amiki-capi/examples/wasm/README.md) —— 原生 WebAssembly API，不依赖 wasm-bindgen
- [**已知问题登记表**](docs/KNOWN_ISSUES.md) —— 未验证项 / 已知限制 / 明确不做 / 技术债（**有稳定编号**）
- [发布计划](docs/RELEASE_PLAN.md) —— 各项验收标准与**已知未验证项**
- [多后端验证清单](docs/VERIFY.md) —— 换机器时照着跑
- [发布报告 0.1.0-alpha](docs/RELEASE_0.1.0-alpha.md)（历史）
- [变更记录](CHANGELOG.md)
- [架构总览](docs/ARCHITECTURE.md)
- [渲染策略与算法](docs/RENDERING_STRATEGY.md) —— 解析式覆盖率推导、路线图（sparse strips、
  全 GPU 化、文本、滤镜、HDR、CPU JIT 兜底引擎、UI damage region）

## 对标

| 能力 | Skia | Direct2D | Blend2D | **Amiki2D** |
|---|---|---|---|---|
| 主线路径填充 | CPU 扫描线 + GPU 传统管线 | GPU 细分 + stencil | CPU JIT 扫描线 | **GPU compute 解析覆盖率** |
| AA 质量 | 解析/超采样混合 | MSAA/解析 | 解析 | **精确面积积分（逐像素解析）** |
| 显存占用 | 与画布成比例 | 与画布成比例 | 内存与画布成比例 | **与画面内容成比例（M3 实测 600× ↓）** |
| 跨平台后端 | 多后端并行维护 | Windows | CPU | **wgpu 一套 WGSL 通吃 + CPU 兜底**（已验证 Vulkan / DX12 / GL 三个后端） |
| 绑定生态 | C++ 核心 + 各语言绑定 | COM | C/C++ | **Rust 核心 + C ABI → 全生态** |

## 文档

| 文件 | 内容 |
|---|---|
| [`docs/OVERVIEW.md`](docs/OVERVIEW.md) | 用大白话讲这是什么、做到哪了 |
| [`docs/API.md`](docs/API.md) | **API 稳定性**：稳定面 / 实验面 / 内部面的清单与承诺 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 分层与边界 |
| [`docs/RENDERING_STRATEGY.md`](docs/RENDERING_STRATEGY.md) | 算法推导与实测数据 |
| [`docs/RELEASE_PLAN.md`](docs/RELEASE_PLAN.md) | 发布计划、验收标准与当前状态 |
| [`docs/VERIFY.md`](docs/VERIFY.md) | 多适配器验证清单（换机器时照着跑） |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | **待你决策的三件事**（实测数据 + 选项 + 建议） |
| [`CHANGELOG.md`](CHANGELOG.md) | 逐项变更与踩过的坑 |

## 已验证的后端与已知限制

后半句"一套 WGSL 通吃"是**实测过**的，不是宣称：同一份 WGSL 在三个 wgpu 后端上
跑通全部测试（`AMIKI_BACKENDS` 可指定后端）。

> **Verified on NVIDIA RTX 3050 / Windows only. Cross-vendor validation is not planned for 0.x.**
>
> ⚠️ **范围说明**：这三个后端是在**同一块 GPU** 上验证的。
> **跨厂商硬件属于已知未验证项** —— 单人开发、手上只有一块显卡，无条件执行。
> 这条验证曾抓到两个"一换后端就起不来"的真 bug，所以未验证意味着
> **可能还存在厂商/驱动相关的未知问题**。见 `docs/RELEASE_PLAN.md` 的范围收缩记录。

| 后端 | 测试 | 备注 |
|---|---|---|
| Vulkan（Windows 默认选中） | **145 通过 / 0 失败** | 全工作区 |
| DX12 | **20 通过 / 0 失败** | 冷启动较慢（FXC 运行期编译着色器，约 2.3 s） |
| GL | **20 通过 / 0 失败** | **位图不受支持**：该后端读不出 `texture_2d_array`（实测采样恒为不透明黑），引擎**显式报错**而不是画出空白 |

`amiki-raster`（CPU 引擎）零第三方依赖，是同一份数学的独立实现 ——
GPU 与 CPU 的每个像素都对拍（纯几何容差 ≤2/255）。
