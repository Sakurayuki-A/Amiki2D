# assets/fonts

## 随仓库分发的字体

| 文件 | 来源 | 许可证 |
|---|---|---|
| `Inter-Regular.ttf` | [Inter 4.1](https://github.com/rsms/inter/releases/tag/v4.1)（取自发布包 `extras/ttf/`） | **SIL Open Font License 1.1**（见 `LICENSE.txt`） |
| `LICENSE.txt` | 同上（发布包内 `LICENSE.txt`） | 同上 |

Inter 由 The Inter Project Authors 开发，版权 (c) 2016。

## 为什么把字体放进仓库

引擎的**核心**（`amiki-math` / `amiki-scene` / `amiki-raster` / `amiki-gpu` / `amiki`）
保持零第三方依赖；字体解析与 shaping 隔离在独立 crate `amiki-text`
（`ttf-parser` + `rustybuzz`）。

但**示例与集成测试需要一个真实字体**才能验证字形轮廓能走通既有渲染管线。
把 411 KB 的 Regular 字重随仓库分发，使 `cargo test` 与 `cargo run --example headless_png`
在干净检出后即可运行，无需额外下载步骤。

原始发布包（`Inter-4.1.zip`，约 33 MB）**不**随仓库分发（见根目录 `.gitignore`）。

## 许可证义务

SIL OFL 1.1 允许自由使用、修改与**随软件分发**，条件是保留许可证文本。
因此 `LICENSE.txt` 与本说明一并保留在本目录。OFL 的"保留字体名称"条款要求
**修改后的字体不得沿用原名**——若将来要改动字形，需另起字体名。
