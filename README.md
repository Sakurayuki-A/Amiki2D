# Amiki2D

**版本 `0.2.0-alpha`** —— 第二个 Alpha：核心渲染管线（M1–M3）、工程硬化（M3.5）、
增量渲染（M4）、文本（M5）、渐变/位图/滤镜（M6）。
详见 [发布报告](docs/RELEASE_0.1.0-alpha.md)、[现状总览](docs/OVERVIEW.md)
与 [变更记录](CHANGELOG.md)。

**已能**：矩形/圆/任意贝塞尔、自相交填充、16 种混合模式、裁剪、真实字体文本
（含**断行/行高/对齐/段落**排版）、渐变（线性/径向/扫描）、位图（旋转/缩放/平铺）、
模糊与投影（CPU/GPU 两侧逐像素一致），且**只重画改动处**。
**尚未**：描边、双向文本、含层场景的增量渲染、全局最优断行。

下一代通用 2D 渲染引擎 —— 目标不是复刻 Skia / Direct2D / Blend2D，而是站在它们肩上，以
**GPU 计算着色器 + 解析式覆盖率** 为主线，构建一个前沿、通用、强大的 2D 渲染底座。

## 核心理念

1. **前沿** —— 主线采用计算着色器路径渲染（Vello 所开创、仍在快速演进的方向）：
   路径压平 → 分箱 → tile 级解析式覆盖率积分，抗锯齿不是"近似"，而是数学上精确的面积积分。
2. **通用** —— 同一份场景编码，多个后端：GPU 引擎（wgpu：Vulkan / D3D12 / Metal / WebGPU）
   与 CPU 引擎（SIMD 软光栅）共享核心算法；无 GPU 的环境自动降级。
3. **强大引领** —— 场景以紧凑编码流表达，渲染与内容复杂度成比例（sparse strips 路线）；
   API 同时服务立即模式（Canvas）与保留模式（UI 场景图 / damage region）。

## 当前状态（M1–M3 · M3.5 · M3.6 部分 · M4 闭环 · M5 文本 · M6 渐变/位图/滤镜）

**已实现能力**（GPU/CPU 双端逐像素对拍 ≤2/255，89 个测试全绿，clippy 零警告）：

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
  模糊与投影经**离屏层**实现（CPU 参考实现，GPU 侧显式报错而非静默丢效果）。
  渐变**解析采样不查表**，避免 1/255 的着色量化
- Canvas 立即模式 API（变换栈/混合/裁剪栈）+ headless PNG + winit 动画示例

**尚未实现**：描边、even-odd 填充、文本、渐变、图像、滤镜、damage region、
CPU 引擎 SIMD/多线程、C ABI/WASM/HDR。

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
- [发布报告 0.1.0-alpha](docs/RELEASE_0.1.0-alpha.md) —— 门禁状态、能力清单、性能数据、已知问题
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
| 跨平台后端 | 多后端并行维护 | Windows | CPU | **wgpu 一套 WGSL 通吃 + CPU 兜底** |
| 绑定生态 | C++ 核心 + 各语言绑定 | COM | C/C++ | **Rust 核心 + C ABI → 全生态** |
