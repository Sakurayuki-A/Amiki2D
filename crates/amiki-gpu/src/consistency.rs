//! 稀疏路径的**自洽检查**（M3.5）。
//!
//! 从 `lib.rs` 拆出：这一簇只依赖 `GpuScene` / `SparseStats` / `PipelineDump`
//! 这三个**数据结构**，与渲染实现本身无关 —— 放在一起只会让"实现"与
//! "验证实现的手段"混在同一个文件里。

use crate::{err, GpuScene, PipelineDump, RenderError};

fn need<'a, T>(v: &'a Option<T>, field: &str) -> Result<&'a T, RenderError> {
    v.as_ref().ok_or_else(|| {
        RenderError(format!(
            "稀疏自洽检查缺少转储字段 {field}，请以 Some(PipelineDump::default()) 调用 \
             run_pipeline_sparse（需要 strip_meta / strip_cell_off / strip_bucket_off / cell_meta \
             / cell_aux / cell_csr_off / cell_segs / sparse_buckets / sparse_stats 全部字段）"
        ))
    })
}

fn need_len<T>(name: &str, v: &[T], want: usize) -> Result<(), RenderError> {
    if v.len() < want {
        return err(format!("转储字段 {name} 长度不足：{} < {want}", v.len()));
    }
    Ok(())
}

/// M3.5：**稀疏路径自洽检查**——在无稠密参照的情况下独立验证稀疏分箱结构。
///
/// # 为什么需要它
///
/// 稀疏（M3b）是默认路径，但它此前的正确性完全依赖"稀疏 vs 稠密逐字节对拍"
/// （`sparse_matches_dense`）。而稠密网格显存 ∝ `tiles × draws`，在 720p 约 280 draw、
/// 1080p 约 31 draw 就撞 `max_storage_buffer_binding_size`（默认 128 MiB）——
/// **真实规模下稠密根本跑不起来，于是稀疏在大场景没有裁判**。本函数填补这个空洞：
/// 只读稀疏自身的缓冲与原始几何，逐级断言各级结构互相吻合。
///
/// # 检查项（全部由原始几何独立重算，不依赖任何稠密路径产物）
///
/// 1. **strip 区间布局**：`strip_meta[si = tr*n_draws + d]` 的 `(lo, hi, y0, y1)` 在界内，
///    且 `lo == tiles_x` 哨兵 ⟺ 区间为空（与 `strips.wgsl` 判空语义一致）；
/// 2. **逐 tile 桶表一致性**：对每个 tile，`tile_count` 必须等于"区间覆盖该列的 strip 数"，
///    `tile_list[tile_start .. +count]` 必须与 `tr_list` 顺序完全相同，且 `d` 严格递增
///    （画家序），每条的 `w == hi − lo + 1`——这是"fine 不丢 draw"的前提；
/// 3. **单元 CSR 分区**：`cell_csr_off` 必须是 `[0, n_items]` 上的单调整数分区
///    （每 cell 的 `base + count == 下一 cell 的 base`）；
/// 4. **carry 桶前缀**：`cell_meta` 的 carry 位模式必须逐位等于"含当前桶的定点前缀取负"
///    （`sparse_emit` 的桶布局 + `sparse_carry` 的累加语义重算）——细粒度绕数正确性的核心；
/// 5. **段端点与成员计数**：`cell_meta` 的段数量必须等于 `cell_aux` 的成员计数，
///    且总成员数等于 `cell_segs` 的条目数。
///
/// 与 `check_strips`（M3a，依赖稠密转储）互补：本函数**不需要稠密路径可运行**。
pub fn check_sparse_consistency(
    gs: &GpuScene,
    d: &PipelineDump,
) -> Result<(), RenderError> {
    let tiles_x = gs.tiles_x;
    let tiles_y = gs.tiles_y;
    let n_draws = gs.colors.len() as u32;
    let n_strips = (tiles_y as u64) * (n_draws as u64);
    let n_tiles = (tiles_x as u64) * (tiles_y as u64);

    let meta = need(&d.strip_meta, "strip_meta")?;
    let sc_off = need(&d.strip_cell_off, "strip_cell_off")?;
    let sb_off = need(&d.strip_bucket_off, "strip_bucket_off")?;
    let cell_meta = need(&d.cell_meta, "cell_meta")?;
    let cell_aux = need(&d.cell_aux, "cell_aux")?;
    let csr = need(&d.cell_csr_off, "cell_csr_off")?;
    let buckets = need(&d.sparse_buckets, "sparse_buckets")?;
    let stats = need(&d.sparse_stats, "sparse_stats")?;
    let tile_count = need(&d.tile_count, "tile_count")?;
    let tile_start = need(&d.tile_start, "tile_start")?;
    let tile_list = need(&d.tile_list, "tile_list")?;
    let tr_offsets = need(&d.tr_offsets, "tr_offsets")?;
    let tr_list = need(&d.tr_list, "tr_list")?;

    let (n_cells, n_buckets, n_tile_entries) = (stats[0], stats[1], stats[2]);
    let n_items = stats[3];

    // 空场景：无内容覆盖画布，结构应为全空
    if n_cells == 0 {
        if n_buckets != 0 || n_tile_entries != 0 || n_items != 0 {
            return err(format!(
                "空场景统计不自洽：cells=0 但 buckets={n_buckets} \
                 tile_entries={n_tile_entries} items={n_items}"
            ));
        }
        return Ok(());
    }

    // ---- 长度前置检查：避免后续索引越界 panic ----
    need_len("strip_meta", meta, n_strips as usize * 4)?;
    need_len("strip_cell_off", sc_off, n_strips as usize)?;
    need_len("strip_bucket_off", sb_off, n_strips as usize)?;
    need_len("cell_meta", cell_meta, n_cells as usize)?;
    need_len("cell_aux", cell_aux, n_cells as usize * 2)?;
    need_len("cell_csr_off", csr, n_cells as usize + 1)?;
    need_len("sparse_buckets", buckets, n_buckets as usize)?;
    need_len("tile_count", tile_count, n_tiles as usize)?;
    need_len("tile_start", tile_start, n_tiles as usize)?;
    need_len("tile_list", tile_list, n_tile_entries as usize)?;
    need_len("tr_offsets", tr_offsets, tiles_y as usize)?;
    need_len("tr_list", tr_list, n_strips as usize)?;

    // ================= 1 + 2：strip 布局与逐 tile 桶表 =================
    let mut expected_entries: u64 = 0;
    for tr in 0..tiles_y {
        let off = tr_offsets[tr as usize];
        let (list_start, list_cnt) = (off[0] as usize, off[1] as usize);
        for col in 0..tiles_x {
            let mut want: Vec<(u32, u32, u32)> = Vec::new(); // (d, lo, hi)
            for k in 0..list_cnt {
                let si = tr_list[list_start + k];
                if si as u64 >= n_strips {
                    return err(format!("tr_list[{list_start}+{k}] = {si} 越出 strip 数 {n_strips}"));
                }
                let m = &meta[si as usize * 4..si as usize * 4 + 4];
                let (lo, hi, y0, y1) = (m[0], m[1], m[2], m[3]);
                if lo > tiles_x || (lo < tiles_x && (hi >= tiles_x || hi < lo)) {
                    return err(format!(
                        "strip {si}（tr={tr}）区间非法：lo={lo} hi={hi} tiles_x={tiles_x}"
                    ));
                }
                if lo == tiles_x {
                    return err(format!(
                        "strip {si}（tr={tr}）判空哨兵 lo==tiles_x 却出现在 tr_list（幻影 strip）"
                    ));
                }
                if y0 > y1 || y1 >= 16 {
                    return err(format!("strip {si} 局部行区间非法：y0={y0} y1={y1}（应 ⊆ [0,16)）"));
                }
                let d = (si as u64 % n_draws as u64) as u32;
                if lo <= col && col <= hi {
                    want.push((d, lo, hi));
                }
            }
            let tile = (tr as u64 * tiles_x as u64 + col as u64) as usize;
            let got_cnt = tile_count[tile] as usize;
            if got_cnt != want.len() {
                return err(format!(
                    "tile(tr={tr},col={col}) 桶表条目数不一致：GPU tile_count={got_cnt}，\
                     由 strip 区间重算={}（fine 会漏 draw 或多算 draw）",
                    want.len()
                ));
            }
            let start = tile_start[tile] as usize;
            if start + got_cnt > n_tile_entries as usize {
                return err(format!(
                    "tile(tr={tr},col={col}) 桶表越界：start={start} count={got_cnt} \
                     总条目={n_tile_entries}"
                ));
            }
            let mut prev_d: Option<u32> = None;
            for k in 0..got_cnt {
                let e = tile_list[start + k];
                let (w, d_got) = (e[1], e[2]);
                let (want_d, lo, hi) = want[k];
                if d_got != want_d {
                    return err(format!(
                        "tile(tr={tr},col={col}) 第 {k} 条 draw 序不符：GPU d={d_got}，期望 d={want_d}\
                         （画家序错乱会导致混合顺序错误）"
                    ));
                }
                if w != hi - lo + 1 {
                    return err(format!(
                        "tile(tr={tr},col={col}) draw={d_got} 的 w={w} 与区间宽 {} 不符",
                        hi - lo + 1
                    ));
                }
                if let Some(p) = prev_d {
                    if d_got <= p {
                        return err(format!(
                            "tile(tr={tr},col={col}) 桶表未按 d 严格递增：{p} → {d_got}"
                        ));
                    }
                }
                prev_d = Some(d_got);
            }
            expected_entries += got_cnt as u64;
        }
    }
    if expected_entries != n_tile_entries as u64 {
        return err(format!(
            "桶表条目总数不一致：统计 {n_tile_entries}，逐 tile 重算 {expected_entries}"
        ));
    }

    // ================= 3：单元 CSR 分区 =================
    if csr[0] != 0 {
        return err(format!("cell_csr_off[0] 应为 0，实际 {}", csr[0]));
    }
    let mut cell_refs: Vec<(u32, u32, u32, u32)> = Vec::new(); // (cell, base, count, carry_bits)
    for strip in 0..n_strips {
        let si = strip as usize;
        let m = &meta[si * 4..si * 4 + 4];
        let (lo, hi, y0, y1) = (m[0], m[1], m[2], m[3]);
        if lo == tiles_x {
            continue; // 空 strip
        }
        let w = hi - lo + 1;
        for row in y0..=y1 {
            for col in lo..=hi {
                let cell = sc_off[si] + (row - y0) * w + (col - lo);
                if cell as u64 >= n_cells as u64 {
                    return err(format!(
                        "单元索引越界：strip {si}(lo={lo},hi={hi},y0={y0},y1={y1}) \
                         row={row} col={col} → cell={cell} ≥ n_cells={n_cells}"
                    ));
                }
                let cm = cell_meta[cell as usize];
                let (base, count) = (cm[1], cm[2]);
                if base + count > n_items {
                    return err(format!(
                        "单元 {cell} 段范围越界：base={base} count={count} > n_items={n_items}"
                    ));
                }
                if csr[cell as usize + 1] != base + count {
                    return err(format!(
                        "单元 CSR 非连续分区：cell={cell} 的 base+count={} 与 \
                         cell_csr_off[{}]={} 不符",
                        base + count,
                        cell + 1,
                        csr[cell as usize + 1]
                    ));
                }
                let cnt_aux = cell_aux[cell as usize];
                if count != cnt_aux {
                    return err(format!(
                        "单元 {cell} 段数量与 emit 成员计数不符：cell_meta={count} cell_aux={cnt_aux}"
                    ));
                }
                cell_refs.push((cell, base, count, cm[0]));
            }
        }
    }
    if csr[n_cells as usize] != n_items {
        return err(format!(
            "cell_csr_off 末项应为 n_items={n_items}，实际 {}",
            csr[n_cells as usize]
        ));
    }
    cell_refs.sort_unstable_by_key(|c| c.0);

    // ================= 4 + 5：carry 桶前缀与段列表 =================
    let mut members: u64 = 0;
    let mut cursor: u32 = 0; // 段槽位必须按 cell 索引顺序填满
    for &(cell, base, count, carry_bits) in &cell_refs {
        members += count as u64;
        if base != cursor {
            return err(format!(
                "段槽位未按 cell 索引顺序连续：cell={cell} base={base}，期望 {cursor}\
                 （scatter 游标语义错乱）"
            ));
        }
        cursor = base + count;

        // 由 cell 反推所属 strip（strip_cell_off 单调递增，二分定位）
        let si = match sc_off.binary_search(&cell) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let m = &meta[si * 4..si * 4 + 4];
        let (lo, hi, y0, _y1) = (m[0], m[1], m[2], m[3]);
        let w = hi - lo + 1;
        let rel = cell - sc_off[si];
        let row = y0 + rel / w;
        let col = lo + rel % w;

        // carry = −(x_max 严格落在该 cell 左界之前的段的定点前缀)
        // 桶布局（sparse_emit）：strip_bucket_off[si] 即 strip 的桶基址，
        // 局部位移 = (row − y0) * (w + 1) + (bucket − lo)，其中 bucket = ⌊x_max/16⌋ + 1。
        // 故 bucket_k 对应 tile 列 lo + k：桶 lo 自身装"右界 ≤ tile lo 左界"的段，
        // 而 cell(col) 的左界正是 tile col 的左界 ⇒ 前缀需含桶 col 但**排除桶 col+1**。
        // 这点极易写错（含当前桶会多算一项，直接把覆盖率整体偏移）。
        let bi = (sb_off[si] as u64) + ((row - y0) as u64) * (w as u64 + 1);
        let mut running: i32 = 0;
        for k in 0..=(col as u64).saturating_sub(lo as u64) {
            running = running.wrapping_add(buckets[(bi + k) as usize]);
        }
        let want_carry = -(running as f32) / 65536.0;
        let got_carry = f32::from_bits(carry_bits);
        if want_carry.to_bits() != got_carry.to_bits() {
            return err(format!(
                "单元 {cell} 的 carry 与桶前缀不符：cell_meta={got_carry}（位模式 {carry_bits:#x}），\
                 由 buckets 前缀重算={want_carry}（strip={si} row={row} col={col} \
                 前缀桶数={}）—— 该 cell 的绕数会整体偏移，导致覆盖率错误",
                (col as u64).saturating_sub(lo as u64) + 1
            ));
        }
    }
    if members != n_items as u64 {
        return err(format!("成员总数不一致：统计 n_items={n_items}，由 cell_meta 重算 {members}"));
    }
    if cursor != n_items {
        return err(format!(
            "段槽位总数 {cursor} 与 n_items={n_items} 不符（cell_segs 存在空洞或重叠）"
        ));
    }
    // 桶的定点量化误差说明：|carry| ≤ 16 故 |前缀| ≤ 16·65536，远小于 f32 精确整数范围，
    // 因此上式用 f32 重算不会引入舍入误差（逐位比较是严格的）
    Ok(())
}

