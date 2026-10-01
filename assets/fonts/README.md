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

---

## CJK 字体：**目前没有随仓库分发**（已知缺口）

### 现状

`amiki-layout` 的 **禁则处理**已经实现并有单元测试（不依赖字体）；
但 `crates/amiki-layout/tests/layout.rs::cjk_wraps_and_respects_kinsoku`
这条**端到端**验证需要一份含汉字的字体。

该测试按下面的顺序查找，**都不存在就跳过并打印原因**（不会静默通过）：

1. 本目录下的 `NotoSansSC-Regular.otf` / `NotoSansCJKsc-Regular.otf` /
   `SourceHanSansSC-Regular.otf`
2. 系统字体（Windows 的雅黑/黑体、Linux 的 Noto CJK、macOS 的苹方）
3. 都没有 → 打印 `跳过：未找到 CJK 字体（仓库 assets/fonts 与系统都没有）`

**这台开发机上跑的是第 2 条**（系统字体），所以本地的端到端结果是真实的；
但干净检出（无系统 CJK 字体的机器）上会跳过 —— **这是已知缺口，不是通过。**

### 为什么不随仓库分发

CJK 字体动辄 **10–20 MB、上万字形**。而本仓库把 411 KB 的 Inter 放进来，
只是为了"干净检出后 `cargo test` 直接可跑"。放一份 15 MB 的中文字体会让
克隆体积翻几十倍，而收益只覆盖一条测试。

真要入库，正确的做法是**按需加载 + 字形缓存淘汰**（把字体当成运行期资源而不是
测试夹具）—— 那是独立一块工作，本版不做。

### 如果要补上（验收怎么做）

把任一份 **OFL 许可**的中文字体放到本目录（推荐
[Noto Sans SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC)，OFL 1.1），
并把它的许可证文本一并放在这里（OFL 要求保留）。然后：

```bash
cargo test -p amiki-layout --test layout -- --nocapture
```

应当看到 `CJK 断行：N 行，禁则检查通过`（而不是"跳过"）。

**注意**：系统字体（雅黑/黑体/苹方）**不是 OFL**，不能入库 ——
它们只能用于本地验证，这正是上面第 2 条只作为回退的原因。
