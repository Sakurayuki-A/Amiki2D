# AMK1 —— Amiki2D 的紧凑二进制场景格式

**用途**：场景的单次紧凑上传 / 磁盘缓存 / 网络传输 / 跨语言绑定。
实现在 `crates/amiki-scene/src/lib.rs`（`encode_scene` / `decode_scene`）。

## 布局（小端）

```text
"AMK1" | version u32 | width u32 | height u32 | n_clips u32 | n_draws u32
每 clip：transform 6×f32 | path
每 draw：transform 6×f32 | paint | blend/flags u32 | n_refs u32 | refs | path
path = count u32 | count × (tag u8 + 坐标 f32)
tag：0 MoveTo, 1 LineTo, 2 QuadTo, 3 CubicTo, 4 Close
```

## 版本

| 版本 | 状态 | 说明 |
|---|---|---|
| `1` | **只解不编** | v2 之前的格式 |
| `2` | **当前** | 与 v1 **布局完全相同**，唯一差别是下面那个 bit 12 的定义 |

`STREAM_VERSION = 2`。**编码器统一写 v2**（没有单独的 v1 编码器，
避免"两份编码器漂移"）。解码器同时接受 v1 与 v2；**未知版本（≥ 3）显式报错**，
不静默按某个版本解读。

## flags bit 12：`fill_rule`

每个 draw 的那个 u32 原本只是 **blend id**（只占 bit 0..3），
v2 把它升级为**标志字**：

```text
bit 0..3  = blend id（0 = Normal … 15 = Luminosity）
bit 12    = fill_rule
              0 = NonZero（默认）
              1 = EvenOdd
其余位    保留（写出时为 0）
```

- **v1 场景此 bit 保留**，解码恒为 `NonZero`（即使字节里该位是 1 也被忽略 ——
  这正是"保留位"的语义）。
- 写入时：`NonZero` 写 0，`EvenOdd` 写 1。
- **布局零变更**：没有加字段、没有扩宽任何字段；因此新旧解码器读同一份字节
  时，只有"对 bit 12 的解释"不同。

阅读代码时对应的常量：`STREAM_VERSION` / `STREAM_VERSION_V1` / `FILL_RULE_BIT`
（都在 `amiki-scene/src/lib.rs` 的 AMK1 段）。

## 明确不支持（显式报错，不静默降级）

| 情形 | 行为 |
|---|---|
| 内嵌位图（`Paint::Image`） | 解码时**显式报错**（编码流不携带位图数据），见 B3 |
| magic 不是 `AMK1` | 显式报错 |
| 版本 ≥ 3 | 显式报错 |
| 未知 blend id / paint 类型 / tag | 显式报错 |

## 测试

- `stream_round_trip`：v2 往返（含裁剪、渐变、混合模式）
- `stream_v2_round_trip_keeps_fill_rule`：4 条 draw（含 2 条 EvenOdd）顺序与值不变，
  且 blend 不被标志位污染
- `stream_v1_bytes_decode_as_nonzero`：同一份字节按 v1 解 ⇒ `NonZero`
- `stream_rejects_unknown_version` / `stream_rejects_garbage`：显式拒绝
