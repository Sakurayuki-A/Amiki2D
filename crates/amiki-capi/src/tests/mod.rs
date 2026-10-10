//! ABI 层自测。
//!
//! 这些测试**从 C 侧的角度**调用导出函数（裸指针、out 参数、错误码），
//! 因此它们验证的是 `include/amiki.h` 承诺的那些行为：
//!
//! - 版本号两处一致
//! - 空指针 / 越界枚举 / 非法数值**都被拒绝**，且不崩溃
//! - **失败时不写出参**（header 的明确承诺）
//! - 中毒机制真的会挡住后续调用
//! - 位图 / 渐变可以**在绘制后立刻销毁**（header 所有权第 4 条）
//! - 端到端的 CPU 渲染真的画出东西
//!
//! 这些测试不替代 C / C# / Node 三个示例 —— 那三个验证的是"别的语言真能调进来"，
//! 而这里验证的是"契约本身成立"。

use crate::api::*;
use crate::api_obj::*;
use crate::objects::*;
use crate::types::*;


use helpers::*;

mod helpers;
mod docs_sync;
mod abi;
mod null_and_out;
mod gradient;
mod canvas_layer;
mod canvas;
mod path;
mod poison;
mod backend;
mod tone_map;
mod mipmap;
mod fill_rule;
mod text_cache;
// 编解码三个入口只在 `codec` feature 下存在。这条门控是 CI 加"特性矩阵"时补的：
// 在此之前 `--no-default-features` 只构建过 lib（能过），**测试目标其实一直编不过**。
#[cfg(feature = "codec")]
mod codec;
mod state_stack;
mod font;
mod async_render;
