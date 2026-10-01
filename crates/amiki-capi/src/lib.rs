//! # amiki-capi —— Amiki2D 的 C ABI
//!
//! **契约是手写的 [`include/amiki.h`](https://github.com/) ，本 crate 去适配它。**
//! 之所以不由 `cbindgen` 之类的工具从 Rust 类型生成：那样 header 会随 Rust
//! 类型变动而漂移，绑定作者拿不到稳定界面。Skia / Blend2D / wgpu-native
//! 都是手写 header。
//!
//! ## 三条硬保证
//!
//! 1. **绝不 panic 跨 FFI 边界** —— 每个导出函数都在 `catch_unwind` 之内。
//!    跨 FFI 的 panic 是未定义行为，不能让 C 调用方承担。
//! 2. **panic 之后对象中毒** —— 出事的对象后续调用一律返回
//!    `AMIKI_ERROR_PANIC`，而不是继续用一个状态不明的对象。
//!    理由见 `objects` 模块文档。
//! 3. **失败时不写出参** —— `amiki.h` 里对每个 out 参数都做了这个承诺。
//!
//! ## 为什么本 crate 允许 `unsafe`
//!
//! 仓库其余 crate 全都 `#![forbid(unsafe_code)]`。C ABI 的边界**必然**
//! 涉及裸指针解引用，所以这里是唯一的例外 —— 也因此把 `unsafe` 集中在
//! 少数几个辅助函数里（`api::ffi` 宏、`image_arc`、`paint_from_raw` 等），
//! 并逐处写明 SAFETY 依据。
//!
//! ## 模块
//!
//! - [`types`] —— `amiki.h` 里所有 `#[repr(C)]` 类型的镜像 + **布局断言**
//! - [`objects`] —— 句柄背后的真实对象 + 中毒机制
//! - [`api`] —— panic 守卫、版本、纯值构造
//! - [`api_obj`] —— 上下文 / 画布 / 路径 / 资源的导出函数

// C ABI 边界必然有 unsafe；把"unsafe 函数体内也要显式 unsafe 块"打开，
// 保证每一处解引用都写着 SAFETY 依据。
#![deny(unsafe_op_in_unsafe_fn)]

// 导出函数接收裸指针并解引用，但**不**标成 `unsafe fn`。
//
// 理由是 FFI 边界的定义：C 里没有 `unsafe` 概念，标记对 C 调用方毫无意义，
// 反而会让 Rust 侧的测试多一层无谓包装。指针有效性的保证写在
// `include/amiki.h` 的所有权一节里 —— 那是**契约**，也正是本 crate 的全部意义。
//
// 指针安全由三件事兜住：每个入口都判空、每个对象都跟踪中毒、
// 每次调用都在 `catch_unwind` 之内。
#![allow(clippy::not_unsafe_ptr_arg_deref)]

// 链接器在生成导入库时会往 stdout 写一句"正在创建库 …"，
// cargo 把它当成 warning（`linker_messages`）报出来。那是**提示信息**，
// 不是问题；但留着会让"零警告"这条门禁失效，所以显式允许。
#![allow(linker_messages)]

pub mod api;
pub mod api_obj;
pub mod objects;
pub mod types;

#[cfg(test)]
mod tests;
