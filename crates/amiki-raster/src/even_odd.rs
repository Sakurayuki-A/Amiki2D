//! **even-odd 填充规则 → 等价 NonZero 几何**（扫描线跨度分解）。
//!
//! # 为什么不是"改累加符号"
//!
//! 引擎的覆盖率来自 `Σ seg_area` 取 `min(|Σ|, 1)`，这个量在几何上等于
//! **该像素的绕数加权面积** `∬ W dA`。even-odd 要的是**奇偶区域**的面积 ——
//! 它是绕数的**非线性**函数（`W mod 2`），所以无论怎么改每个段的符号，
//! 都不可能从 `∬ W dA` 得到它。
//!
//! 这条在 `tests/fill_rule_spike.rs` 里用独立超采样裁判量过：
//! 对普通方块取"Σ 的奇偶性"，边缘像素偏差达 **0.7207**（≈184/255）；
//! 同向双环达 **0.8477**（≈216/255）—— 解析抗锯齿会整个丢掉。
//!
//! # 做法：把 even-odd 变成一组**互不重叠**的梯形
//!
//! 按 y 扫描：把平面切成若干"带"（band），在每个带内把与带相交的段按 x 排序、
//! **两两配对**（第 1↔2、3↔4…），每一对就是一个"内部跨度"，输出成一个梯形。
//! 这些梯形**互不重叠**，于是 NonZero 的填充结果恰好等于 even-odd 的填充结果 ——
//! 覆盖率数学（`seg_area` / `min(|Σ|,1)`）**一个字都不用改**。
//!
//! ## 为什么每个梯形只输出两条边
//!
//! 梯形 = 左边界、右边界、上下两条水平边。水平段的 `seg_area` 恒为 0
//! （`dy ≈ 0` 直接返回），所以水平边可以**直接丢掉**。而剩下的一对竖直边
//! （左边向上、右边向下）贡献之和是：
//!
//! ```text
//! ∫(clamp(x_right) − clamp(x_left)) dy
//! ```
//!
//! 正是该梯形在像素列内的**精确裁剪覆盖面积**：`−s·len` 那些常数项自动相消。
//! 这带来两个好处：
//!
//! 1. 无需补偿前缀（不像 NonZero 那样依赖 `carry` 抵消 `−s·len`）；
//! 2. 多个互不重叠的梯形之和 ≤ 1，因此**不会触发 `|W| ≥ 2` 的饱和**。
//!
//! # 精确度
//!
//! 带的边界取**所有顶点 y**；若带内有两条段**交换了 x 次序**（即它们在该带内相交），
//! 就把带在交点处切开再处理 —— 因此配对用的是"该带内实际次序"，不是近似。
//! 交点在带内但**极靠近端点**（< [`EPS_Y`]）时不再切分，误差有界且远小于 1 像素。
//!
//! # 代价（写在明处）
//!
//! 这是**每次绘制都要跑**的 CPU 侧预处理（只对 even-odd 的 draw 生效）：
//! 输出段数随带数与跨度数增长，通常远大于原路径的段数。
//! 因此 even-odd 的**分箱成本高于 nonzero**；想省就要在调用方缓存结果。
//! （后续可加"按路径哈希缓存"，见 `docs/KNOWN_ISSUES.md` 的技术债。）

/// y 方向的最小带高：比它还薄的带不再切分（避免病态递归）。
const EPS_Y: f32 = 1e-4;
/// 一次转换最多输出多少条边（防御病态输入；超了就截断并告警一次）。
const MAX_OUT_SEGS: usize = 400_000;
/// 带的分裂次数上限（同上）。
const MAX_BANDS: usize = 200_000;

/// 一条竖直边：`[x0, y0, x1, y1]`，`y0 < y1` 表示向下（`s = +1`）。
#[inline]
fn edge(x_top: f32, y_top: f32, x_bot: f32, y_bot: f32, downward: bool) -> [f32; 4] {
    if downward {
        [x_top, y_top, x_bot, y_bot]
    } else {
        [x_bot, y_bot, x_top, y_top]
    }
}

/// 把 even-odd 的线段表转换成**等价 NonZero 几何**（一组竖边）。
///
/// 输入是设备空间的线段表（`[x0, y0, x1, y1]`，自动闭合过的子路径），
/// 与 [`crate::flatten_draw`] 的输出同构；水平段会被忽略（对覆盖率无贡献）。
///
/// 返回的线段表交给**原有**的 NonZero 分箱/着色路径即可，二者结果一致。
pub fn even_odd_spans(segs: &[[f32; 4]]) -> Vec<[f32; 4]> {
    // 1) 归一化成 y 递增、丢掉水平段与退化段
    let mut ys: Vec<f32> = Vec::with_capacity(segs.len() * 2);
    let mut norm: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(segs.len()); // (x_top, y_top, x_bot, y_bot)
    for s in segs {
        let (mut xa, mut ya, mut xb, mut yb) = (s[0], s[1], s[2], s[3]);
        if ya > yb {
            std::mem::swap(&mut xa, &mut xb);
            std::mem::swap(&mut ya, &mut yb);
        }
        if yb - ya <= EPS_Y {
            continue; // 水平段 / 退化段：`seg_area` 恒为 0
        }
        norm.push((xa, ya, xb, yb));
        ys.push(ya);
        ys.push(yb);
    }
    if norm.len() < 2 {
        return Vec::new();
    }

    // 2) 顶点 y 去重 → 初始带
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.dedup();
    let mut bands: Vec<(f32, f32)> = Vec::with_capacity(ys.len());
    for w in ys.windows(2) {
        if w[1] - w[0] > EPS_Y {
            bands.push((w[0], w[1]));
        }
    }

    let mut out: Vec<[f32; 4]> = Vec::new();
    let mut warned_odd = false;
    let mut splits = 0usize;

    // 3) 逐带处理；遇到带内交点就把带切开，把子带推回队列
    let mut i = 0usize;
    while i < bands.len() {
        let (y0, y1) = bands[i];
        i += 1;
        if out.len() >= MAX_OUT_SEGS || splits > MAX_BANDS {
            eprintln!("amiki-raster：even-odd 跨度分解达到上限，已截断（路径过于复杂？）");
            break;
        }

        // 本带的活跃段（`ya < y1 && yb > y0`，与 `seg_area` 的 dy 判定一致）
        let mut xs: Vec<(f32, f32)> = Vec::new(); // (x@y0, x@y1)
        for &(xa, ya, xb, yb) in &norm {
            if ya < y1 && yb > y0 {
                let k = (xb - xa) / (yb - ya);
                xs.push((xa + k * (y0 - ya), xa + k * (y1 - ya)));
            }
        }
        if xs.len() < 2 {
            continue;
        }

        // 带内是否有点对交换了次序？收集交点 y 并切分
        let mut cuts: Vec<f32> = Vec::new();
        for a in 0..xs.len() {
            for b in (a + 1)..xs.len() {
                let d0 = xs[a].0 - xs[b].0;
                let d1 = xs[a].1 - xs[b].1;
                if d0 == 0.0 || d1 == 0.0 || (d0 > 0.0) == (d1 > 0.0) {
                    continue; // 该带内次序未变
                }
                let t = d0 / (d0 - d1); // 交点参数（0..1）
                let yc = y0 + t * (y1 - y0);
                if yc > y0 + EPS_Y && yc < y1 - EPS_Y {
                    cuts.push(yc);
                }
            }
        }
        if !cuts.is_empty() {
            cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            cuts.dedup();
            // 先把带在**所有**交点处切成连续子带，再逐个处理。
            // （初版这里写错了：对每个交点都推 (y0,yc) 与 (yc,y1)，于是子带相互重叠、
            //   梯形被重复累加 —— 五角星用例把它抓了出来：偏差 0.63。）
            cuts.push(y1);
            let mut lo = y0;
            for yc in &cuts {
                if *yc - lo > EPS_Y {
                    bands.push((lo, *yc));
                }
                lo = *yc;
            }
            splits += 1;
            continue;
        }

        // 无交换 ⇒ 按带中点的 x 排序，两两配对
        let xm = |p: &(f32, f32)| 0.5 * (p.0 + p.1);
        xs.sort_by(|a, b| xm(a).partial_cmp(&xm(b)).unwrap());
        if !xs.len().is_multiple_of(2) && !warned_odd {
            eprintln!(
                "amiki-raster：even-odd 在某条带里遇到奇数个交点（数值退化），已丢掉最后一个"
            );
            warned_odd = true;
        }
        let mut k = 0usize;
        while k + 1 < xs.len() {
            let (lt, lb) = xs[k]; // 左边界：x@y0, x@y1
            let (rt, rb) = xs[k + 1]; // 右边界
            // 左边界向上、右边界向下 ⇒ 贡献为 +∫(clamp(x_r) − clamp(x_l))dy
            out.push(edge(lt, y0, lb, y1, false));
            out.push(edge(rt, y0, rb, y1, true));
            k += 2;
        }
    }

    out
}
