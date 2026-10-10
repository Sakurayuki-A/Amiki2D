//! WGSL 着色器清单（`include_str!` 的落点，路径相对本文件）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。


pub(crate) const FINE_MATH_WGSL: &str = include_str!("fine_math.wgsl");
pub(crate) const FINE_WGSL: &str = include_str!("fine.wgsl");
pub(crate) const SPARSE_FINE_WGSL: &str = include_str!("sparse_fine.wgsl");

pub(crate) const SCAN_WGSL: &str = include_str!("scan.wgsl");
pub(crate) const FLATTEN_WGSL: &str = include_str!("flatten.wgsl");
pub(crate) const BINS_WGSL: &str = include_str!("bins.wgsl");
pub(crate) const STRIPS_WGSL: &str = include_str!("strips.wgsl");
pub(crate) const SPARSE_MEASURE_WGSL: &str = include_str!("sparse_measure.wgsl");
pub(crate) const SPARSE_TILES_WGSL: &str = include_str!("sparse_tiles.wgsl");
pub(crate) const SPARSE_EMIT_WGSL: &str = include_str!("sparse_emit.wgsl");
pub(crate) const SPARSE_SCATTER_WGSL: &str = include_str!("sparse_scatter.wgsl");
pub(crate) const SPARSE_CARRY_WGSL: &str = include_str!("sparse_carry.wgsl");

