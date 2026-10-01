# 多适配器验证清单

发布判据「P3 多适配器验证」的要求是：
**至少在两个不同厂商的 GPU（或独显 + 核显）上跑通全部测试。**

当前状态：**同一块 GPU（RTX 3050）上三个 wgpu 后端全部跑通**，
但**换厂商**这一条需要另一台机器或另一块显卡 —— 本文件就是为了把这件事
变成"照着敲几条命令、把结果填进表里"。

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
一起，证明同一份 WGSL 在两套硬件上都跑通。

---

## 五、如果失败了，怎么定位

1. **先看是不是编译期失败**（`create_shader_module` / `Validation Error`）。
   若是，问题在**我们的 WGSL**，而不是硬件 —— 当初两次都是这样。
   - `X3511: forced to unroll loop` → 找常量次数的循环 + 动态下标，显式展开
   - `textureSize(sampler2D, uint)` → 把 `textureLoad` 的 mip 参数从 `0u` 写成 `0`
2. **再看是不是 limits 不够**（`Too many bindings`）。
   引擎需要 **9 个 storage buffer/阶段**；自行创建 `wgpu::Device` 必须用
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
