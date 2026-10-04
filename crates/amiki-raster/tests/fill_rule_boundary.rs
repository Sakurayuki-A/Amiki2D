//! **段与 tile / 像素的归属约定**（0.8.0 阶段 1 的 spike 产物）。
//!
//! 结论：现有分箱用的是一套**自洽**的约定，**不要改**：
//!
//! | 量 | 现有实现 | 为什么不能按"直觉"改 |
//! |---|---|---|
//! | 段归属 tile（行表） | `floor(x_min/tf) ..= floor(x_max/tf)`（**偏包含**） | 它保证"正好结束在 tile 左边界"的段被放进该 tile 的行表 |
//! | carry 前缀排除 | 严格 `x_max < T_l` | 与上面互补：那个边界段只经行表计一次，不再进 carry |
//! | 像素覆盖 | `xc >= xl && xc < xr`（由 `seg_area` 的 `clamp` 隐式给出） | 半像素列 + `clamp` 已是左闭右开语义 |
//!
//! 换句话说：**行表偏包含 + carry 严格排除**是一对互补约定，两者一起才让
//! `carry[row] + Σ_{本 tile 段}` 恒等于"全部段的贡献"。
//! 若按"统一起见"把前缀排除改成 `x_max <= T_l`，那个边界段就会被
//! **行表与 carry 各计一次** ⇒ 逐像素偏差 `s·len`，破坏既有的逐字节一致。
//!
//! 这段没有做成生产代码里的函数：转换器（`even_odd.rs`）在**整条段**上工作，
//! 不涉及 tile 划分；而分箱里的这两行是既有的、被 86 个测试钉住的实现。
//! 所以这里把约定**写成可执行的断言**，防止将来"顺手统一"改坏它。

/// 段是否应进入 tile `[t_l, t_r)` 的行表（现有实现：`floor` 两端都算）。
fn belongs_to_tile(x_min: f32, x_max: f32, t_l: f32, t_r: f32, tf: f32) -> bool {
    let lo = (x_min / tf).floor() * tf;
    let hi = (x_max / tf).floor() * tf;
    lo <= hi && hi >= t_l && lo <= t_r - tf + 1e-6
}

/// carry 前缀是否排除该段（现有实现：严格小于）。
fn excluded_by_prefix(x_max: f32, t_l: f32) -> bool {
    x_max < t_l
}

#[test]
fn boundary_convention() {
    let tf = 8.0f32; // tile 宽
    let t_l = 8.0f32;
    let t_r = 16.0f32;

    // ① 段正好在 tile 左边界结束 → **仍进行表**（偏包含），且**不进 carry**
    assert!(belongs_to_tile(0.0, t_l, t_l, t_r, tf), "边界段应进行表");
    assert!(!excluded_by_prefix(t_l, t_l), "边界段不该被前缀排除（严格小于）");
    // 两者互补：行表计一次，carry 不计 ⇒ 总数正确
    // （若把前缀排除改成 <=，就会行表 + carry 各计一次）

    // ② 段完全在 tile 左边（x_max < t_l）→ 不进行表、进前缀
    assert!(!belongs_to_tile(0.0, t_l - 0.001, t_l, t_r, tf));
    assert!(excluded_by_prefix(t_l - 0.001, t_l));

    // ③ 段越过左边界 → 进行表
    assert!(belongs_to_tile(0.0, t_l + 0.001, t_l, t_r, tf));

    // ④ 段正好在 tile 右边界开始 → 进的是**右边那个** tile 的行表
    assert!(belongs_to_tile(t_r, t_r + 8.0, t_r, t_r + tf, tf));
    assert!(!belongs_to_tile(t_r, t_r + 8.0, t_l - tf, t_l, tf));

    // ⑤ 段跨整个 tile → 进行表
    assert!(belongs_to_tile(0.0, 32.0, t_l, t_r, tf));

    // ⑥ 退化段（x_min == x_max == t_l）→ 仍然进行表（它贡献恒为 −s·len），
    //    且不进前缀 —— 与 ① 同一条规则
    assert!(belongs_to_tile(t_l, t_l, t_l, t_r, tf));
    assert!(!excluded_by_prefix(t_l, t_l));
}
