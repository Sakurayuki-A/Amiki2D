//! **P4 回归**（嵌入式集成反馈）：后端初始化失败时**绝不能**报 `PANIC`。
//!
//! # 背景
//!
//! `AMIKI_BACKENDS=gl` 且进程无窗口时，wgpu 25 在**创建 compute pipeline**
//! 那一步内部 panic（实测 `wgpu-25.0.2/src/backend/wgpu_core.rs:1410`）。
//! 该 panic 一路逃到 C ABI 的兜底捕获，于是调用方看到的是
//! `AMIKI_ERROR_PANIC`（"引擎有 bug，请上报"），而 `amiki.h` 的契约是
//! "请求的后端不可用 ⇒ `AMIKI_ERROR_BACKEND_UNAVAILABLE`"。
//!
//! 对嵌入式宿主来说这不是小事：它会把"这台机器跑不了这个后端"误判成
//! "引擎有 bug"，从而在错误的方向上排查。
//!
//! # 这个测试钉什么
//!
//! 只有一条**契约**：结果是
//! - `AMIKI_OK`（这台机器/这个进程真的能起 GL），或
//! - `AMIKI_ERROR_BACKEND_UNAVAILABLE` / `AMIKI_ERROR_NOT_SUPPORTED`（起不来，但**说清了是后端问题**），
//!
//! **绝不能是 `AMIKI_ERROR_PANIC`**。
//!
//! # 为什么用独立文件
//!
//! `AMIKI_BACKENDS` 是**进程级**环境变量，而 cargo 让每个集成测试文件跑在
//! 独立进程里 —— 放在 `src/tests.rs` 会污染同进程的其它测试（那里有一条
//! 用 `AMIKI_BACKEND_AUTO` 的用例）。

use amiki2d::api_obj::{amiki_context_create, amiki_context_destroy};
use amiki2d::types::{
    AMIKI_BACKEND_GPU, AMIKI_ERROR_BACKEND_UNAVAILABLE, AMIKI_ERROR_NOT_SUPPORTED,
    AMIKI_ERROR_PANIC, AMIKI_OK,
};

#[test]
fn gl_backend_init_never_reports_panic() {
    std::env::set_var("AMIKI_BACKENDS", "gl");
    let mut ctx: *mut amiki2d::objects::AmikiContext = std::ptr::null_mut();
    let r = amiki_context_create(AMIKI_BACKEND_GPU, &mut ctx);
    eprintln!("AMIKI_BACKENDS=gl → amiki_context_create 返回 {r}");

    if r == AMIKI_OK {
        assert!(!ctx.is_null(), "返回 OK 就必须给出有效 context");
        amiki_context_destroy(ctx);
    } else {
        assert_ne!(
            r, AMIKI_ERROR_PANIC,
            "后端初始化失败泄漏成了 PANIC —— 调用方会误判为引擎 bug"
        );
        assert!(
            r == AMIKI_ERROR_BACKEND_UNAVAILABLE || r == AMIKI_ERROR_NOT_SUPPORTED,
            "失败时必须是 BACKEND_UNAVAILABLE({AMIKI_ERROR_BACKEND_UNAVAILABLE}) 或 \
             NOT_SUPPORTED({AMIKI_ERROR_NOT_SUPPORTED})，实际 {r}"
        );
    }
    std::env::remove_var("AMIKI_BACKENDS");
}
