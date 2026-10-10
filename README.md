# Amiki2D

**Rust + wgpu** 写的通用 2D 渲染引擎：GPU 计算着色器 + **解析式覆盖率**抗锯齿（AA 不是近似，
是逐像素的精确面积积分）。同一份场景编码有 GPU 与 CPU 两条实现，**逐像素对拍**（纯几何 ≤2/255），
核心分层零第三方依赖 —— 目标是**小而可审计、可嵌入**。

- **现状 `0.10.9-alpha`**：贝塞尔与自相交填充（**nonzero + even-odd**）、16 种混合、裁剪、描边、
  真字体文本（**CJK 断行与禁则**）、字体回退、渐变、位图、模糊/投影，且**只重画改动处**
  —— 带投影的静态界面平均只重画 **3.5%** 的 tile。
- **多语言**：手写 C ABI，**90 个符号**，C / C# / Node 三个示例输出**逐字节一致**；
  WASM 纯 CPU 模块 **1 037.3 KB**，与原生 CPU 后端**逐字节一致**。
- **门禁**：`cargo test --workspace` **363 通过 / 0 失败**，clippy 与 rustdoc 零告警；
  两条 CI（测试矩阵、C ABI 五项）在 GitHub Actions 上每次推送都跑。

## 快速开始

```bash
cargo run --release -p amiki --example headless_png   # 出图 out/amiki_headless.png
cargo run --release -p amiki --example window_demo    # 打开动画窗口
cargo test  --workspace                               # 全量测试（含 GPU/CPU 对拍）
```

## 接着读

- [**现状总览**](docs/OVERVIEW.md) —— 一页看清"是什么、到哪了、接下来做什么"
- [**已知问题登记表**](docs/KNOWN_ISSUES.md) —— 未验证项 / 已知限制 / 明确不做 / 技术债（有稳定编号）
- [架构分层](docs/ARCHITECTURE.md) · [API 稳定性承诺](docs/API.md) · [渲染策略与算法](docs/RENDERING_STRATEGY.md) · [变更记录](CHANGELOG.md)
- 绑定与交付：[C ABI 三个示例](crates/amiki-capi/README.md) · [WASM 示例](crates/amiki-capi/examples/wasm/README.md)

## 已知边界（诚实起见）

**跨厂商硬件未验证**：Vulkan / DX12 / GL 三个后端都在**同一块 RTX 3050** 上跑通，AMD / Intel /
Apple 未测；**WASM 的 WebGPU 构建**也未验证（当前是纯 CPU 构建）；**GL 后端不支持位图**
（读不出 `texture_2d_array`，引擎**显式报错**而不是画出空白）。
细节见 [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) 的 A / B 节。

## 许可证

**MIT**（[`LICENSE`](LICENSE)）。`assets/fonts/` 里的两份字体是第三方作品，按
**SIL OFL 1.1** 分发，其许可证文本必须随字体保留（见 [`assets/fonts/README.md`](assets/fonts/README.md)）。
