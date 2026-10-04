# 多后端 / 多适配器验证清单（**未来可选用**）

> **当前状态（已决策）**：判据原要求「两个不同厂商的 GPU」，
> 因单人开发、手上只有一块显卡，**已决定放弃这一条**（2026 年，用户决策）。
> 达成的是 **同一块 GPU 上的三个 wgpu 后端**（Vulkan / DX12 / GL）全部跑通。
>
> **跨厂商硬件属于「已知未验证项」，不是「已验证」。** 详见
> `RELEASE_PLAN.md` 的「范围收缩记录」。
>
> 本文件保留下来，供**将来有第二台机器/第二块显卡时**直接照着跑
> —— 清单本身仍然是有效的。

---

## 一、为什么这一条重要（而不是走过场）

"一套 WGSL 通吃"是本项目对外宣称的能力。**没在第二个厂商的硬件上跑过，
这句话就只是宣称。** 事实上正是执行这条验证时，发现了两个真问题：

- **DX12 根本起不来**：我们自己的 WGSL 里有动态下标（`ij[st]` / `o[i]`），
  FXC 试图展开常量循环来静态化下标但失败 →
  `error X3511: forced to unroll loop, but unrolling failed`
- **GL 编不过**：`textureLoad(..., 0u)` 让 naga 生成 `textureSize(sampler2D, uint)`，
  而 GLSL 要 `int`

两个都已修复。所以这条验证**是有牙齿的**，不是仪式。

---

## 二、在目标机器上要跑的（可直接复制）

```bash
# 0) 干净检出 + 编译
cargo build --workspace --release

# 1) 记录实际用的是哪块卡 / 哪个后端
cargo test -p amiki-gpu --release --lib startup_cost -- --nocapture
#    → 会打印 "[启动] 适配器 = <名字> (<后端>)；首次(冷) = …"

# 2) 全工作区测试（默认后端，通常是该平台的首选）
cargo test --workspace --release

# 3) 逐个后端强制跑一遍（能跑几个算几个）
#    Windows: vulkan / dx12 / gl 皆可；Linux: vulkan / gl；macOS: metal（见下）
AMIKI_BACKENDS=vulkan cargo test -p amiki-gpu --release
AMIKI_BACKENDS=dx12   cargo test -p amiki-gpu --release
AMIKI_BACKENDS=gl     cargo test -p amiki-gpu --release

# 4) 静态检查
cargo clippy --workspace --all-targets
cargo doc --workspace --no-deps
```

> macOS 上 wgpu 走 Metal，`AMIKI_BACKENDS` 目前只认 `vulkan` / `dx12` / `gl` / `all` ——
> 在 macOS 上用默认值（`all`，会落到 Metal）即可，第 3 步跳过。

---

## 三、期望结果与例外

| 检查 | 期望 |
|---|---|
| 第 2 步 | **149 passed / 0 failed** |
| 第 3 步（Vulkan / DX12） | **20 passed / 0 failed**（`amiki-gpu`） |
| 第 3 步（GL） | **20 passed / 0 failed** —— 其中位图那条会**打印"按预期显式拒绝"** |
| 第 4 步 | **零警告** |

### 已知例外（不是失败）

- **GL 不支持位图**：该后端读不出 `texture_2d_array`。引擎**显式报错**
  而不是画出空白，测试把它当作已知结果。这是**有意的**设计（与本项目
  "不支持就报错、不静默画错"的一贯取舍一致）。
- **DX12 冷启动约 2.3 s**：FXC 在运行期编译着色器，不是我们的代码慢。
- **`startup_cost` 在显式指定后端时不套用预算断言**（那时测的不是默认配置）。

### 什么算"失败"

- 任何 `test result: FAILED`
- **编译期 panic**：`Validation Error` / `create_shader_module` 失败
  —— 这正是当初 DX12/GL 的样子
- `create_bind_group_layout` 报 `Too many bindings of type StorageBuffers`

---

## 四、结果记录（填这张表，附到发布说明里）

| 项 | 值 |
|---|---|
| 机器 / 操作系统 | |
| GPU 型号与厂商 | |
| 驱动版本 | |
| wgpu 后端（第 1 步打印） | |
| 第 2 步：passed / failed | |
| 第 3 步 Vulkan | |
| 第 3 步 DX12 | |
| 第 3 步 GL | |
| 第 4 步 clippy / doc | |
| 备注（异常、跳过项） | |

**判据达成条件**：本表与**另一块不同厂商 GPU**（或独显 + 核显）上的同类表格
一起，才算证明同一份 WGSL 在两套硬件上都跑通。

> ⚠️ 这一条**当前未达成且已决定放弃**。所以本表单独填完**只能证明多后端**，
> 不能证明跨厂商 —— 不要在发布说明里把它写成后者。

---

## 五、如果失败了，怎么定位

1. **先看是不是编译期失败**（`create_shader_module` / `Validation Error`）。
   若是，问题在**我们的 WGSL**，而不是硬件 —— 当初两次都是这样。
   - `X3511: forced to unroll loop` → 找常量次数的循环 + 动态下标，显式展开
   - `textureSize(sampler2D, uint)` → 把 `textureLoad` 的 mip 参数从 `0u` 写成 `0`
2. **再看是不是 limits 不够**（`Too many bindings`）。
   引擎需要 **10 个 storage buffer/阶段**（0.6.1 起，fine 新增脏 tile 索引表）；
   自行创建 `wgpu::Device` 必须用
   `amiki_gpu::required_limits()`。
3. **若是结果不一致**（测试跑完但数值对不上），记录**具体的后端 + 驱动版本 +
   哪个测试 + 差多少**，那属于真正的跨厂商差异，需要单独调查。

---

## 六、当前已有的证据

| 后端 | 结果 | 环境 |
|---|---|---|
| 默认（→ Vulkan） | **145 / 0**（全工作区，当时用例数） | RTX 3050 / Windows |
| DX12 | **20 / 0** | 同上 |
| GL | **20 / 0**（位图显式拒绝） | 同上 |

⚠️ **这三行都来自同一块 GPU。** 判据要求的是**两个不同厂商** ——
本文件的作用就是把"另一块"的验证变得可执行。

---

## 七、浏览器验证记录（WASM，纯 CPU）

WASM 交付的强验证是"**浏览器里的输出与原生 CPU 参考逐字节相同**"。
验证页 `crates/amiki-capi/examples/wasm/browser_test_standalone.html`
（单文件、`file://` 直开）在**页内**自校验 SHA-256，所以只要拿得到
`document.title`（`amiki-wasm-pass` / `amiki-wasm-fail`）或 `#result` 的文本，
就等于拿到了结论 —— 不需要人去比像素。

| 浏览器 | 版本 | 结果 | 期望哈希 | 记录时间 |
|---|---|---|---|---|
| Chrome | 无头模式，版本未记录 | ✅ **PASS** | `fc9482ba…` | 0.7.0-alpha |
| **Firefox** | **157.0** | ✅ **PASS** | `fc9482ba…` | 0.7.0-alpha |
| Edge | 未复验 | — | — | — |

**Firefox 157.0 的完整证据**：

```text
browserVersion = 157.0        platformName = windows
TITLE          = amiki-wasm-pass
RESULT_TEXT    = PASS
PAGE_LOG:
  Amiki2D 0.7.0-alpha（wasm 987.9 KB）      ← 模块加载成功
  后端：GPU 不可用（BackendUnavailable），用 CPU
  后端：CPU（CPU）
  字体：Inter-Regular.ttf（内嵌）            ← 两个字体都来自页内 base64
  字体：NotoSansSC-Regular.ttf（内嵌）
  越界枚举被正确拒绝：rc=1
  BMP SHA-256 = fc9482ba147ffd075b06055c95141eee06bec710ce31767c65ce7de58adccb0d
  期望         = fc9482ba147ffd075b06055c95141eee06bec710ce31767c65ce7de58adccb0d
  字节一致 ✅
PASS
```

画面（由 `WebDriver:TakeScreenshot` 取回）同时渲染了 `Amiki2D C ABI`（拉丁）
与 `你好 Amiki2D 世界`（汉字走 Noto），**中文无方框**；
截图存于 `out/wasm_firefox_pass.png`。

### 怎么复跑（不用装 Playwright）

Firefox **自带 Marionette**（WebDriver 协议），所以**零下载**即可做"等待式"验证：

```bash
# 1) 起一个独立 profile 的 Firefox：headless + 打开 Marionette 端口 2828
"…/firefox.exe" --headless --marionette -no-remote -profile <临时目录> about:blank
# 2) 连 127.0.0.1:2828：轮询 document.title，出现 amiki-wasm-pass / amiki-wasm-fail
#    之后再取 #result / #log 文本与 WebDriver:TakeScreenshot
node out/ff_marionette.js 2828 "file:///…/browser_test_standalone.html" out/wasm_firefox_pass.png
```

**四个坑（都实测踩过，照抄可省时间）**：

1. **不要用 `--screenshot`**：Firefox 的截图发生在 `load` 事件附近，
   而本页是"wasm 实例化 + 渲染"的异步流程 —— 定时/自动截图会抓到
   `loading…/RUNNING` 的半成品。**那不是失败，是验证未完成。**
2. **Marionette 协议 3 用数组帧**：命令 `[0, id, name, params]`、
   响应 `[1, id, error, result]`；按对象帧（协议 2 的样子）发过去会**没有任何响应**
   （表现为"连接建立了但命令永远超时"）。连接后还要先握手：
   `{"applicationType":"gecko","marionetteProtocol":3}`（服务端通常会主动发）。
3. **CDP 的 `/json/list` 在 Firefox 上不存在**（`/json`、`/json/version` 都是 404，
   端口只在听裸 WebSocket），所以不要照搬 Chrome 的发现流程。
4. **抓 Firefox 窗口不要用 `PrintWindow` / 屏幕拷贝**：Firefox 走 GPU 合成，
   前者内容区是空白或启动画面，后者依赖 z 序（实测抓到过别的窗口）。
   用 `WebDriver:TakeScreenshot` 才拿得到真实内容。
