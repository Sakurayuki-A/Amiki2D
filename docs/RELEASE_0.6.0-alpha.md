# Amiki2D 0.6.0-alpha 发布快照

**版本**：`0.6.0-alpha`（`workspace.package.version`，工作区 8 个 crate 统一继承）
**主题**：**mipmap**（关闭 B4）+ **WASM 三浏览器字节验证**
**性质**：**快照**，不是判据清单。0.3.0 之后不再逐版列"发布判据 6 条"，
本文件记录这一版**交付了什么、拿什么证据证明、还留着什么问题** ——
细节与推导在 [`CHANGELOG.md`](../CHANGELOG.md) 的 `[0.6.0-alpha]` 条目。

---

## 一、交付物

| 项 | 内容 |
|---|---|
| **mipmap（B4 关闭）** | 大幅缩小的位图不再走样：2×2 盒式滤波在**线性光 + 预乘 alpha** 空间生成 mip 链，采样按缩小倍数取层（三线性） |
| **一致性设计** | 链**只在 CPU 生成**、LOD **只在 CPU 计算**（打包进放置缓冲，WGSL 只读不算）、GPU **只上传** —— 用"单一实现"换跨端一致 |
| **平铺接缝 bug（潜伏）** | 双线性路径一直存在：GPU 钳边、CPU 回绕 → 新增 **Repeat 地址模式采样器**（binding 17） |
| **`mip_bias` 增量路径 bug（本版引入、本版修掉）** | 增量路径没把 `mip_bias` 用在主画布上（只有层内容用了）→ 修法见 CHANGELOG |
| **WASM 浏览器验证页** | `crates/amiki-capi/examples/wasm/browser_test.html`：页内写死期望哈希，Chrome / Edge / Firefox（均无头）三方比对 |
| **API** | Rust：`Image::from_rgba8_with_mipmap`、`generate_mip_chain`、`compute_lod`、`generate_padded_mip_levels`、`mip_level_count`；C ABI：`amiki_image_set_mipmap_mode`、`amiki_canvas_set_mip_bias`（符号数 **72 → 74**） |

**兼容性硬约束（逐条兑现）**：`Image::from_rgba8` 行为不变；1:1 与放大的绘制
带链/不带链**逐字节一致**；`AMIKI_MIPMAP_DISABLED` 回到 0.5.0 行为；
AMK1 编码流不变（mip 链是运行时派生物，场景文件仍只存原图）。

---

## 二、验证证据（都可复现）

| 判据 | 证据 |
|---|---|
| **双端逐像素** | `gpu_image_mipmap_matches_cpu`：缩小 / 缩小+旋转 / 缩小+平铺，最大通道差 **1** |
| **摩尔纹消除（量化判定，不靠目视）** | 8px 棋盘缩到奈奎斯特之后（LOD ≈ 4.5）：带链整幅均匀灰 **188**（线性平均的正确值），无链是 0/255 椒盐噪声 |
| **1:1 与放大不变** | `one_to_one_render_is_byte_identical_with_without_mip` 断言逐字节相等 |
| **纯 CPU 构建** | `--no-default-features` 下 **151 测试 / 0 失败**；四个示例按 `required-features = ["gpu"]` 自动跳过 |
| **WASM ↔ 原生 CPU** | `examples/wasm/build.cmd`：Node 运行 + 原生 CPU 参照**字节完全相同**，SHA-256 `d565cc36…` |
| **三个浏览器** | 同一场景在 Chrome / Edge / Firefox 渲染并哈希，均与原生 CPU 输出**逐字节一致**（同上哈希） |
| **测试总数** | `cargo test --workspace`：**221 passed / 0 failed** |
| **静态检查** | `cargo clippy --workspace --all-targets` 与 `cargo doc --workspace --no-deps` **零警告** |
| **C ABI 符号** | header 声明 **74** / Rust 导出 **74**，无缺失（`verify.cmd` 的符号项） |

> **口径注记**：`d565cc36…` 这个哈希在 0.6.1 的工作里被**独立复算过一次**
> （Node 演示 + 原生 CPU 对比），与浏览器页里写死的期望值一致 —— 也就是说
> "跨架构字节一致"这条不是只跑过一次的历史结论。

---

## 三、性能与口径

### mip 链生成（判据：生成 ≤ 2× 原图上传）

| 口径 | 实测 |
|---|---|
| 链生成（1024²，多次取最小） | **2.4–3.2 ms** |
| 原图上传（`write_texture` 主机侧） | **1.1–1.6 ms**（本身在波动） |
| 比值 | **~1.6–2.9×** |

**结论：边际达成**（多次运行 3 过 2 未过）。分母自身在波动，判据处在门槛上 ——
**照实记为边际**，不写成"通过"。测量示例：`cargo run --release -p amiki --example mipmap_perf`。

### 其它两条

| 判据 | 结果 |
|---|---|
| 采样回退 ≤ 5% | **−1.9%**（fine pass GPU 口径）✅ |
| 内存 = 原图 1/3 | **33.3%** ✅（4096 KB → 1365 KB） |

### 链生成从 56 ms 优化到 ~2.5 ms（22×）

四步：sRGB 编解码**查表**（解码 256 项、编码 16384 项量化边界对齐）；
**配对求和** `(a+b)+(c+d)`；**行级并行**（原生目标）；**不透明快路径**。
每一步都要求"字节与慢路径逐位一致"。详见 CHANGELOG。

---

## 四、已知问题（**本快照时点**的状态）

| 编号 | 状态 |
|---|---|
| **B4** 位图缩小无多级过滤 | ✅ **本版关闭** |
| **A1** 跨厂商硬件未验证 | 仍为**已知未验证项**（同一块 RTX 3050；不承诺跨厂商像素一致） |
| **A2** WASM 的 WebGPU 构建未验证 | 仍为已知限制。本版**查明**了真正的阻塞点不是体积（实测 2.5 MB）而是**同步/异步 ABI**：`amiki_context_create` 同步等待异步的 WebGPU 初始化 |
| **B2** 字体格式有限 | 不支持 `.ttc` / `.woff` / `.woff2` |
| **无字体回退** | 一段文本只能用一种字体：CJK 需调用方显式用 `NotoSansSC-Regular.ttf`（随仓库分发）；只加载 Inter 时中文画成方框。**这是"声称能力"上的缺口，下一步补** |
| **B5** even-odd 填充未实现 | 只有 nonzero |
| **§8.5 一次不可复现的测试失败** | 本快照时点仍记为"无法复现"，**未定位** |

> 本快照写于 0.6.1 的开发周期内：**§8.5 那条后来复现并定位**（根因不是测试跑法，
> 而是并发创建 GPU 设备会崩进程 —— 属运行时约束），将在下一版的变更记录与已知问题
> 登记表里写清。

---

## 五、明确不做（本版时点）

双向文本、Knuth-Plass 全局最优断行、宽色域、各向异性过滤、GPU 生成 mip 链、
纹理压缩、动画系统。理由见 [`RELEASE_PLAN.md`](RELEASE_PLAN.md) 第六节。

---

## 六、复现命令

```bash
cargo test  --workspace                                # 221 测试
cargo clippy --workspace --all-targets                 # 零警告
cargo doc   --workspace --no-deps                      # 零警告
cargo run --release -p amiki --example mipmap_perf     # 三条判据的数字
cargo run --release -p amiki --example mipmap_demo     # out/mipmap_demo.png
crates/amiki-capi/examples/wasm/build.cmd              # wasm ↔ 原生 CPU 字节对比
```

浏览器验证：起一个静态服务器指向 `examples/wasm/`，打开 `browser_test.html`
（页面自校验 SHA-256）；或先跑 `make_standalone.py` 生成单文件版本。

---

## 七、这一版留下的、值得记住的两件事

1. **"只在 CPU 做"是一致性的手段，不是保守**：mip 链、LOD、sRGB 查表全部
   只在宿主侧算一次，着色器只读 —— 于是 `log2` 的 1 ULP 漂移、两套滤波实现
   的漂移这两个经典坑**从结构上不存在**。
2. **对拍会连坐出旧 bug**：平铺接缝那条是 0.3.0 就存在、被 mipmap 对拍首次
   照出来的。**新的对拍要把旧的边界一起拉进来**才划算。
