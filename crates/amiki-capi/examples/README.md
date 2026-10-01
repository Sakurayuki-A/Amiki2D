# Amiki2D C ABI —— 三个绑定示例

这三个示例是 **C ABI 不是纸上谈兵**的证据。它们用**同一个头文件**、调**同一份
`amiki2d` 库**、做**完全相同的绘制**，因此三者的输出**逐像素一致** ——
这比"能跑起来"强得多：它证明结构体布局、调用约定、字符串编码、所有权规则
在三种完全不同的运行时里**都被正确地封送了**。

```
$ crates\amiki-capi\verify.cmd

[5] the three examples produce byte-identical images
    OK  all three identical -- SHA256 92B430853D1C675C24189F18D56DF941...
ALL CHECKS PASSED
```

三者不只是"逐像素一致"，而是**字节完全相同** —— 所以那条检查只要比 SHA256
就够了，不需要逐像素遍历。

| 示例 | 语言 | 构建 | 绑定方式 |
|---|---|---|---|
| [`c/`](c/) | C99 | `build.cmd`（MSVC） | 直接 `#include "amiki.h"` |
| [`csharp/`](csharp/) | C# / .NET 10 | `build.cmd`（dotnet） | 手写 P/Invoke |
| [`node/`](node/) | Node.js 24 | `build.cmd`（npm） | [koffi](https://koffi.dev/) FFI |

**前置条件**：三者都先 `cargo build -p amiki-capi --release`（脚本会自己做）。

---

## 每个示例演示了什么

三个示例的绘制顺序与参数完全相同，覆盖了 ABI 的主要面：

1. **背景矩形** —— 最基础的 `fill_path`
2. **描边折线** —— 圆角拐角 + 圆端帽
3. **径向渐变圆** —— 渐变句柄，且**绘制后立刻销毁**
4. **8×8 棋盘位图** —— 位图句柄，同样**绘制后立刻销毁**
5. **离屏层 + 高斯模糊** —— `begin_layer` / `end_layer` 配对
6. **文本** —— 字体加载 + 度量 + shaping（找不到字体就跳过）
7. **错误路径** —— 越界枚举必须被拒绝，且不崩
8. **写出 BMP** —— 不依赖任何图像库，便于肉眼与程序双重确认

---

## 关于 Node 绑定的选型（为什么是 koffi）

题目本来可以是 **napi-rs** 或 **WASM**，我选了第三条路。理由：

| 方案 | 为什么不选 |
|---|---|
| **napi-rs** | 需要在**构建期下载 Node 头文件**。本项目此前上过一次当 —— `static-dxc` 的构建期下载直接挂住，最后只能放弃。不想再引入同类风险 |
| **WASM** | **到不了 GPU 后端**。引擎的主力路径是 wgpu，`wasm32` 上拿不到 —— 用 WASM 演示等于只演示了一半能力 |
| **koffi** ✓ | 纯 FFI，`npm install` 即用，**不需要改 Rust 侧一行代码** |

而第三个示例的**目的**是证明"这个 ABI 能被非 C / 非 .NET 的运行时调进来" ——
koffi 完整地证明了这一点。**这一版要的是能跑、能验证，而不是绑定层选型的最优解。**

将来若要做正式的分发绑定，`napi-rs` 仍是更合适的选择（原生 addon、无额外运行时依赖）；
那时这份 koffi 示例可以直接当作 ABI 是否可用的快速探针。

---

## 手工构建（不用脚本）

```bash
# 1. 先构建原生库（三个示例共用）
cargo build -p amiki-capi --release

# 2. C
cd crates/amiki-capi/examples/c
cl /utf-8 /W4 /I..\..\include minimal.c /Fe:minimal.exe ^
   /link /LIBPATH:..\..\..\..\target\release amiki2d.dll.lib
minimal.exe

# 3. C#
cd ../csharp
dotnet run -c Release -p:AmikiNativeDir=<repo>/target/release

# 4. Node
cd ../node
npm install
node minimal.js
```

**MSVC 用户注意**：`amiki.h` 是 UTF-8 无 BOM，在非 UTF-8 代码页的系统上
需要 `/utf-8`，否则会报 `warning C4819`（开了 `/WX` 就变成错误）。

---

## 构建产物（已 gitignore）

| 文件 | 说明 |
|---|---|
| `minimal.exe` / `*.obj` / `amiki2d.dll` | C 示例的产物 |
| `csharp/bin/`、`csharp/obj/` | .NET 产物 |
| `node/node_modules/` | npm 依赖 |
| `*_demo.bmp` / `*_demo.png` | 渲染输出 |

**`amiki2d.dll` 会被脚本复制到示例目录** —— 运行期需要它在 exe 旁边（或 PATH 上）。
