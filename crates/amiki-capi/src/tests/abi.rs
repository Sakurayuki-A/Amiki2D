//! 契约与版本：版本串、错误码、header 镜像。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

#[test]
fn version_string_is_taken_from_cargo_toml() {
    // SAFETY: `amiki_version_string` 返回静态 C 字符串
    let s = unsafe { std::ffi::CStr::from_ptr(amiki_version_string()) };
    // 直接与包版本比 —— 这条断言**不可能因为"忘了改某处"而漂移**，
    // 因为实现就是编译期从 Cargo.toml 取的。
    assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
}

#[test]
fn numeric_version_matches_package_version() {
    // 数值版本没法在 const 上下文里从字符串解析，所以只能各写一份 ——
    // 这条测试就是保证两份同步的东西。
    let pkg = env!("CARGO_PKG_VERSION");
    let core = pkg.split('-').next().unwrap(); // 去掉 -alpha 之类
    let parts: Vec<u32> = core.split('.').map(|x| x.parse().expect("版本段不是数字")).collect();
    assert_eq!(parts.len(), 3, "包版本应为 major.minor.patch：{pkg}");

    let v = amiki_version();
    assert_eq!(v >> 16, parts[0], "major 与 Cargo.toml 不一致（包版本 {pkg}）");
    assert_eq!((v >> 8) & 0xFF, parts[1], "minor 与 Cargo.toml 不一致（包版本 {pkg}）");
    assert_eq!(v & 0xFF, parts[2], "patch 与 Cargo.toml 不一致（包版本 {pkg}）");
}

#[test]
fn result_string_covers_every_value_and_rejects_out_of_range() {
    for r in [
        AMIKI_OK,
        AMIKI_ERROR_INVALID_ARGUMENT,
        AMIKI_ERROR_OUT_OF_MEMORY,
        AMIKI_ERROR_BACKEND_UNAVAILABLE,
        AMIKI_ERROR_NOT_SUPPORTED,
        AMIKI_ERROR_PANIC,
        AMIKI_ERROR_UNKNOWN,
        AMIKI_ERROR_NOT_FOUND,
        AMIKI_ERROR_STACK_OVERFLOW,
        AMIKI_ERROR_STACK_UNDERFLOW,
        AMIKI_ERROR_BUSY,
    ] {
        // SAFETY: 返回静态 C 字符串
        let s = unsafe { std::ffi::CStr::from_ptr(amiki_result_string(r)) };
        assert!(!s.to_bytes().is_empty(), "错误码 {r} 没有说明文字");
        // `"unknown"` 只留给**越界的错误码**（那是调用方自己的 bug）；
        // 每个已定义的错误码都要有自己的说明。
        assert_ne!(s.to_str().unwrap(), "unknown", "错误码 {r} 落到了 unknown");
    }
    // 越界值不能崩，且明确返回 "unknown"
    // SAFETY: 同上
    let s = unsafe { std::ffi::CStr::from_ptr(amiki_result_string(9999)) };
    assert_eq!(s.to_str().unwrap(), "unknown");
}

/// **错误码取值必须在 header 与 Rust 两侧一致**。
///
/// 这是个"镜像"：`AMIKI_ERROR_*` 在 `include/amiki.h`（C 侧的契约）与
/// `src/types.rs`（实现）各写一份。任何一侧改了数字，调用方就会按**错的含义**
/// 处理错误 —— 与 0.9.0 抓到的 `clip_depth` / `fill_rule` 镜像漂移是同一类问题。
///
/// 只扫 `amiki_result` 枚举体那一段（范围要窄，不能让相邻段落命中）。
#[test]
fn error_codes_match_the_header() {
    let header = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/include/amiki.h"))
        .expect("读 amiki.h");
    let start = header.find("typedef enum amiki_result {").expect("header 里有 amiki_result");
    let end = header[start..].find("} amiki_result;").expect("枚举体有结束") + start;
    let body = &header[start..end];

    for (name, value) in [
        ("AMIKI_OK", AMIKI_OK),
        ("AMIKI_ERROR_INVALID_ARGUMENT", AMIKI_ERROR_INVALID_ARGUMENT),
        ("AMIKI_ERROR_OUT_OF_MEMORY", AMIKI_ERROR_OUT_OF_MEMORY),
        ("AMIKI_ERROR_BACKEND_UNAVAILABLE", AMIKI_ERROR_BACKEND_UNAVAILABLE),
        ("AMIKI_ERROR_NOT_SUPPORTED", AMIKI_ERROR_NOT_SUPPORTED),
        ("AMIKI_ERROR_PANIC", AMIKI_ERROR_PANIC),
        ("AMIKI_ERROR_UNKNOWN", AMIKI_ERROR_UNKNOWN),
        ("AMIKI_ERROR_NOT_FOUND", AMIKI_ERROR_NOT_FOUND),
        ("AMIKI_ERROR_STACK_OVERFLOW", AMIKI_ERROR_STACK_OVERFLOW),
        ("AMIKI_ERROR_STACK_UNDERFLOW", AMIKI_ERROR_STACK_UNDERFLOW),
        ("AMIKI_ERROR_BUSY", AMIKI_ERROR_BUSY),
    ] {
        let needle = format!("{name} = {value}");
        let at = body
            .find(&needle)
            .unwrap_or_else(|| panic!("header 的 amiki_result 里没有 `{needle}`"))
            + needle.len();
        // 防"值前几位相同"造成的假通过（例如 header 写 100，needle 是 10）
        assert_ne!(
            body[at..].chars().next(),
            Some('0'),
            "header 里 {name} 的取值看起来不是 {value}"
        );
    }
}

