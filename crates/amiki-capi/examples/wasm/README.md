# Amiki2D WASM 示例

用**原生 WebAssembly API** 调用 Amiki2D —— **不依赖 wasm-bindgen**。
边界就是那份已经设计好、测试过、三种语言验证过的 C ABI。

## 跑起来

```
build.cmd
```

它做三件事：装 `wasm32-unknown-unknown` 目标（若缺）、构建 wasm 模块、
跑 Node 演示 —— 然后**自动做字节级对比**：

```
[3/3] running the Node demo ...

Amiki2D 0.7.0-alpha（wasm，987.9 KB）
后端：GPU 不可用（BackendUnavailable），用 CPU
后端：CPU（CPU）
字体：..\..\..\..\assets\fonts\Inter-Regular.ttf
字体：..\..\..\..\assets\fonts\NotoSansSC-Regular.ttf
已写出 amiki_wasm_demo.bmp（640×400，1024000 字节）
越界枚举被正确拒绝：invalid argument
完成

[verify] building a native CPU reference and byte-comparing ...
[OK]   native CPU and WASM outputs are BYTE-IDENTICAL
```

手工构建：

```bash
cargo build -p amiki-capi --profile wasm-release \
    --target wasm32-unknown-unknown --no-default-features
node minimal.js target/wasm32-unknown-unknown/wasm-release/amiki2d.wasm
```

## 体积：**1 037.3 KB**

这是这一版最需要注意的数字。同一个模块：

| 构建 | 体积 |
|---|---|
| `--profile release`（`debug = true`） | **11.7 MB** |
| `--profile wasm-release`（`debug = false`） | **1 037.3 KB** |

> **历史**：0.5.0 首次落地时是 **950.7 KB**；0.6.0 把 mipmap 的链生成与
> sRGB 查表加进 `amiki-scene`（wasm 构建包含它），模块 **+15 KB**；
> 0.7.0 加字体回退（`FontStack` 与配套解析/缓存），模块 **+22 KB**；
> 0.10.9 之后把 2 000 行级的 `lib.rs` 拆成多文件，模块 **+4.1 KB**
> （模块边界改变了内联与死代码删除的判定 —— 那是结构整理的实测代价）。
> 几个数字都对，只是属于不同版本 —— 引用时请带上版本号。

**主导因素是 `debug = false`**，差 12 倍。仓库根 `Cargo.toml` 里的
`[profile.release] debug = true` 是为**原生 profiling** 加的，在 wasm 里
纯属浪费（浏览器不需要 DWARF）。所以另立了 `wasm-release` profile。

> 该 profile 里另外三个开关（`opt-level = "z"` / `lto = "fat"` /
> `panic = "abort"`）**实测只再省约 5 KB，在噪声范围内**。保留它们是因为
> 语义正确（`panic = "abort"` 对 wasm 是该有的），但**不会声称它们有效**。

## 为什么不用 wasm-bindgen

wasm-bindgen 会在 Rust 与 JS 之间再插一层自己的 ABI（`__wbindgen_*` 一堆胶水），
而 **C ABI 已经是一份设计好、测试过、三种语言验证过的边界** —— 再套一层只是
把边界复杂化，还多一个构建期依赖（需要 `wasm-bindgen-cli`）。

## 结构体怎么过边界（这一版最实际的技术点）

WebAssembly **没有"结构体按值"这个概念**，所以 C ABI 里的

```c
AmikiPaint amiki_paint_solid(AmikiColor c);
amiki_result amiki_canvas_fill_path(amiki_canvas*, const amiki_path*, amiki_paint);
```

会被编译器**降级**。实测签名（`tools/wasm_sig.py` 能打出来）：

```
amiki_paint_solid            (i32, i32)              // sret, color*
amiki_canvas_fill_path       (i32, i32, i32) -> i32  // canvas, path, paint*
amiki_canvas_stroke_path     (i32, i32, i32, i32) -> i32  // canvas, path, style*, paint*
```

即两条规则：

1. **结构体返回 → sret**：调用方传一个隐藏的**第一个参数**作返回空间
2. **结构体参数 → 一律按指针传**

`amiki.js` 把这两条集中在 `_sret` / `_pack` 两个辅助里，别处看不到。

> `tools/wasm_sig.py <wasm> [过滤词]` 可以随时打出真实签名 ——
> 换编译器版本或改 ABI 之后，用它核对比猜快。

## 纯 CPU 构建（`--no-default-features`）

WASM 构建**关掉了 `gpu` feature**（否则 wgpu 会把模块推到 23 MB）。
关掉之后的行为是**显式的**：

| 请求 | 结果 |
|---|---|
| `AMIKI_BACKEND_CPU` | 正常 |
| `AMIKI_BACKEND_AUTO` | 落到 CPU（这是 `AUTO` 的语义） |
| `AMIKI_BACKEND_GPU` | **`AMIKI_ERROR_BACKEND_UNAVAILABLE`** —— 明确拒绝，不静默降级 |

演示里特意验证了第三条。

## 为什么对比的对象是「原生 CPU」而不是「原生默认」

原生默认走 GPU，而 WASM 构建是纯 CPU 的 —— 两者产物**本来就该不同**。
所以对比基准是 `AMIKI_BACKEND=cpu` 的原生 C 示例（`examples/c/minimal.c`
支持这个环境变量，就是为了这个对比）。

**同一个场景、不同目标架构（wasm32 vs x86_64）、不同的 ABI 降级方式，
输出字节完全相同** —— 这同时证明了：

1. WASM 构建是正确的
2. `amiki.js` 对 sret / 按引用降级的处理是正确的
3. 渲染器在跨目标时是确定的

## ⚠️ 未验证：WebGPU 构建

浏览器里 wgpu 可以走 **WebGPU** 后端，但**这一版没有验证过**：
本机 **Node 24 的 `navigator.gpu` 是 `undefined`**（无 WebGPU），也没有可用的
浏览器环境。所以「WebGPU 构建」是**已知未验证项**，不写成"已支持"。

**体积不是障碍**（0.6.0 实测）：带 `--features gpu` 的 wasm 模块 **2.5 MB**
—— 远低于早先 23 MB 的预估 —— 经 wasm-bindgen CLI 胶水后 **1.43 MB**，
C ABI 导出完整保留，CPU 路径可用。

真正的阻塞点是**架构的**：`amiki_context_create` 是同步的，而 WebGPU 初始化
是异步的 —— 引擎用 `pollster::block_on` 同步等待，在无 WebGPU 的环境直接
panic-abort（wasm 构建为 `panic = "abort"`），在真实浏览器主线程则会**死锁**
（被阻塞的事件循环永远等不到异步请求完成）。解法需要 **C ABI 层的异步重构**
（GPU 上下文异步创建或就绪轮询），是设计变更而非稳定性修复。
详见 [`docs/KNOWN_ISSUES.md`](../../../../docs/KNOWN_ISSUES.md) 的 **A2**。

## 演示覆盖了什么

与 C / C# / Node 三个原生示例**完全相同**的绘制：背景、描边折线（圆角 + 圆端帽）、
径向渐变圆、8×8 棋盘位图（绘制后立刻销毁句柄）、模糊层、真字体文本、
以及越界枚举的错误路径。
