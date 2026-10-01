# assets/fonts

## 随仓库分发的字体

| 文件 | 来源 | 许可证 |
|---|---|---|
| `Inter-Regular.ttf` | [Inter 4.1](https://github.com/rsms/inter/releases/tag/v4.1)（取自发布包 `extras/ttf/`） | **SIL Open Font License 1.1**（见 `LICENSE.txt`） |
| `LICENSE.txt` | 同上（发布包内 `LICENSE.txt`） | 同上 |
| `NotoSansSC-Regular.ttf` | [Noto Sans SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC)（取自发布包 `static/`） | **SIL Open Font License 1.1**（见 `OFL-NotoSansSC.txt`） |
| `OFL-NotoSansSC.txt` | 同上（发布包内 `OFL.txt`） | 同上 |

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

## CJK 字体：**已随仓库分发**

`NotoSansSC-Regular.ttf`（10.07 MB，OFL 1.1）是 `amiki-layout` 的
**禁则处理与 CJK 断行**端到端验证所用的字体。

### 测试怎么用它

`crates/amiki-layout/tests/layout.rs::cjk_wraps_and_respects_kinsoku` 按顺序查找：

1. **本目录**的 `NotoSansSC-Regular.ttf` —— **现在走的就是这一条**
2. 系统 CJK 字体（Windows 雅黑/黑体、Linux Noto CJK、macOS 苹方）—— 回退
3. 都没有 → 打印 `跳过：未找到 CJK 字体…`

跑起来会打印用的是哪一份，两种情况可区分：

```
（P2-1 使用**仓库内**字体做验证：…/assets/fonts/NotoSansSC-Regular.ttf）   ← 现在
（P2-1 使用系统 CJK 字体做验证：C:\Windows\Fonts\msyh.ttc）            ← 回退
```

### 为什么是 10 MB

CJK 字体动辄 10–20 MB、上万字形，而 Inter 只有 411 KB。这一份是**判据要求**
（"字体许可文件随仓库分发"），不是随手加的：

- 它让 **干净检出（无系统 CJK 字体的机器）也能跑通 CJK 端到端验证** ——
  此前那条测试在干净环境上会跳过，等于验不了；
- 更彻底的方案是**按需加载 + 字形缓存淘汰**（把字体当运行期资源而不是测试夹具），
  那是独立一块工作，本版不做。

系统字体（雅黑/黑体/苹方）**不是 OFL**，不能入库 —— 所以它们只作回退。

### 发布包不入库

`Noto_Sans_SC.zip`（约 107 MB，含 9 个字重 + 可变字体）与 `Inter-4.1.zip`
（约 33 MB）**都不随仓库分发**，已在根 `.gitignore` 中排除。
仓库里只保留这两份 Regular 字重与其许可证文本。
