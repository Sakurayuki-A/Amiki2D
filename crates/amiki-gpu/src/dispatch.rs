//! 派发维度与脏 tile 表（含 0.10.9 的 4KB 行主序修正）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。


/// 校验脏区与画布 tile 尺寸一致。
///
/// `Damage` 用的是它自己的画布 tile 数，而派发范围按当前场景的 tile 数钳制。
/// 若两者不一致（例如调用方复用了为不同尺寸画布建的 `Damage`），
/// "全脏"会被误判成"只脏左上角"，从而**静默留下陈旧像素**。
/// 这类错误必须炸出来，不能靠钳制掩盖。
pub(crate) fn check_damage_bounds(damage: Option<&amiki_scene::Damage>, tiles_x: u32, tiles_y: u32) {
    if let Some(d) = damage {
        assert_eq!(
            (d.total_tiles()),
            tiles_x.max(1) * tiles_y.max(1),
            "damage 的 tile 数与当前画布不一致（damage 可能是为别的画布尺寸创建的）—— 否则「全脏」会被误当作局部脏区，导致陈旧像素残留"
        );
    }
}

/// fine 阶段的**脏 tile 索引表**（1D 派发用）：每项打包成 `tx | ty << 16`。
///
/// - `damage` 为 `None` 或全脏 ⇒ 整幅画布的 tile（row-major）
/// - 否则 ⇒ 脏矩形覆盖的 tile（row-major；矩形是 `Damage` 目前唯一的表达）
///
/// # 为什么把 (tx, ty) 放进缓冲，而不是让内核从 `global_invocation_id` 推
///
/// wgpu 的 `dispatch_workgroups` **没有起点偏移**。早先的实现把脏区换算成
/// `(x_start, x_count, y_start, y_count)`，却只用了后两个 —— 于是脏区不在
/// 左上角时，被重算的是 `[0,x_count) × [0,y_count)`，**真正的脏 tile 一个都没
/// 重算**，画面静默保留上一帧像素（`window_demo` 的滚动就会踩到）。
///
/// 查表之后，派发只与"要重算的 tile 数"有关，与画布尺寸、与脏区起点都无关；
/// 未来要支持非矩形脏区（多个小矩形）也只需要往这张表里多塞几项。
pub(crate) fn dirty_tile_list(
    damage: Option<&amiki_scene::Damage>,
    tiles_x: u32,
    tiles_y: u32,
) -> Vec<u32> {
    let (tx0, ty0, tx1, ty1) = match damage.and_then(|d| d.tiles()) {
        Some(r) => r,
        None => (0, 0, tiles_x, tiles_y),
    };
    let tx0 = tx0.min(tiles_x);
    let tx1 = tx1.min(tiles_x).max(tx0);
    let ty0 = ty0.min(tiles_y);
    let ty1 = ty1.min(tiles_y).max(ty0);
    debug_assert!(tx1 <= 0xFFFF && ty1 <= 0xFFFF, "tile 坐标超出一个 u16 的打包范围");
    let mut out = Vec::with_capacity(((tx1 - tx0) as usize) * ((ty1 - ty0) as usize));
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            out.push(tx | (ty << 16));
        }
    }
    out
}

/// fine 的 1D 派发维度：`(x_workgroups, z_workgroups)`，`y` 恒为 1。
///
/// 每个 workgroup = 一个 tile（`workgroup_size(16,16)`）。单维 workgroup 上限
/// 是 WebGPU 规范保证的 **65535**，所以用 z 维分批 —— 这样任意画布尺寸
/// （含 8K：480×270 = 129 600 tile）都在一张表里放得下。
pub(crate) fn fine_dispatch_dims(n_tiles: u32) -> (u32, u32) {
    /// WebGPU 规范保证的 `maxComputeWorkgroupsPerDimension` 下限。
    const MAX_PER_DIM: u32 = 65535;
    // `clamp` 而非 `min().max()`：后者在 clippy 眼里是手写的 clamp。
    // 下限 1 是因为 workgroup 数不能为 0（`n_tiles == 0` 时调用方本来就跳过派发）。
    let x = n_tiles.clamp(1, MAX_PER_DIM);
    let z = n_tiles.div_ceil(MAX_PER_DIM).max(1);
    (x, z)
}

/// 稀疏路径单维派发（64 线程/工作组）的维度：`(x_workgroups, y_workgroups)`。
///
/// `n_elements == 0` 返回 `(0, 0)`（调用方跳过派发，`dispatch_workgroups(0, ..)` 非法）。
///
/// # 与内核 `lin()` 的契约（**改一处必须改另一处**）
///
/// ```text
/// 行距 = 1024 个工作组 × 64 线程 = 65536 个元素
/// x    = min(⌈n/64⌉, 1024)      y = ⌈⌈n/64⌉ / 1024⌉
/// ```
///
/// 内核的 `lin(gid) = gid.x + gid.y * 65536u` 必须与这里的 `1024` 一致 ——
/// 两侧步长不一致就是这个式子最贵的错法：旧写法按 **65535 个工作组**推进、
/// 内核按 **65535 个元素**解码（差 64 倍），于是行间重叠 95%（`atomicAdd`
/// 的内核重复计数）、尾部元素**永不执行** ⇒ 4K 画布最下面两行瓦片整片空白。
/// 事故全记录见 `CHANGELOG.md` 的 0.10.9 ③。
/// 一致性由 `dispatch_dims_1d_covers_every_element_exactly_once`（纯算术，
/// 不需要 GPU）钉住：任何元素都必须**恰好被处理一次**。
pub(crate) fn dispatch_dims_1d(n_elements: u32) -> (u32, u32) {
    /// 与所有稀疏内核的 `@workgroup_size(64)` 一致。
    const WG: u32 = 64;
    /// 一行的工作组数（= 内核 `lin()` 的行距 65536 个元素）。
    const ROW_WG: u32 = 1024;
    if n_elements == 0 {
        return (0, 0);
    }
    let wg = n_elements.div_ceil(WG);
    let x = wg.min(ROW_WG);
    let y = wg.div_ceil(ROW_WG);
    // 元素上限 = `65535 行 × 65536 元素 ≈ 4.29e9`，而 `n_elements` 是 u32
    // （最大 4.29e9）：只有理论极值会超出 y 维上限，那时 wgpu 报派发维度非法
    // （**响亮地失败**，不是静默少算）。
    debug_assert!(y <= 65_535, "1D 派发的行数超出 WebGPU 的 y 维上限");
    (x, y)
}

