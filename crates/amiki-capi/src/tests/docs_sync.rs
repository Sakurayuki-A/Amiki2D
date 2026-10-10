//! 文档 ↔ 契约 ↔ 代码的**计数与版本**对账。
//!
//! # 为什么需要这个模块
//!
//! 本项目反复出现过同一类漂移：改了代码，忘了改文档里的数字。已发生过的例子 ——
//! `docs/OVERVIEW.md` 的"验证状态"表里长期写着 `345 passed`（实际 360）、
//! 以及"符号数 80/80 核对"（实际 90/90）；`include/amiki.h` 的版本宏与
//! `Cargo.toml` 也曾不一致（0.10.9 那次：header 是 9、`api.rs` 还写着 8）。
//!
//! 漂移的根因不是粗心，而是**同一个事实被放在多处**。这里的做法是承认无法消灭
//! 副本，但可以让副本**不一致时立刻失败**：
//!
//! - **能钉死的**：ABI 符号集合（header 声明 ↔ `#[no_mangle]` 导出 ↔ 文档里的数字）
//!   与版本宏（header ↔ `Cargo.toml`）—— 这些变化只发生在"有意改契约"时，
//!   那一刻本来就必须同时改文档，失败信息会指到哪里不一致。
//! - **钉不住的**：测试条数、wasm 模块体积这类"跑一次才知道"的数字。
//!   它们不在这里断言（否则加一条测试就要改一次文档），而是标注"实测于某版本"，
//!   以 CI 输出 / `verify.cmd` 为准。
//!
//! 本模块只读文件、不调外部命令，因此与 `verify.cmd` 相比能在任何时候（包括
//! 无 MSVC、无 GPU 的机器上）跑 —— CI 的 `test` job 就靠它先把这两条挡住。

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// 一个测试里要用到的仓库路径（相对 crate 根）。
fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn read(rel: &str) -> String {
    let p = repo_path(rel);
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("读不到 {}：{e}", p.display()))
}

/// 递归收集 `dir` 下所有 `.rs` 文件。
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("读不到目录 {}：{e}", dir.display())) {
        let path = entry.expect("read_dir 项").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
}

/// `include/amiki.h` 里 `AMIKI_API ... amiki_xxx(` 声明出来的符号名。
///
/// 注意一行里可能出现**多个** `amiki_` 标记：返回类型 `amiki_result`、调用约定宏
/// `AMIKI_CALL`、最后才是函数名 —— 所以要对每个出现位置试一遍，
/// 取"后面紧跟 `(`"的那个（这正是 `verify.ps1` 那条正则做的事）。
fn symbols_declared_in_the_header(header: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in header.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("AMIKI_API") else {
            continue;
        };
        let mut from = 0;
        while let Some(rel) = rest[from..].find("amiki_") {
            let tail = &rest[from + rel..];
            let name: String = tail
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if tail[name.len()..].trim_start().starts_with('(') {
                out.insert(name);
                break;
            }
            from += rel + "amiki_".len();
        }
    }
    out
}

/// 本 crate 的源码里真正带 `#[no_mangle]` 的 `pub extern "C" fn amiki_xxx`。
///
/// 顺手也验证了"每条导出上面都有 `#[no_mangle]`"—— 漏写 `#[no_mangle]` 会让
/// 函数悄悄变成 Rust 内部符号，`verify.cmd` 只能在链接后才发现。
fn symbols_exported_by_this_crate() -> BTreeSet<String> {
    let mut files = Vec::new();
    rust_files(&repo_path("src"), &mut files);
    let mut out = BTreeSet::new();
    for file in files {
        // 测试模块自己会调用导出函数，但不定义它们
        if file.components().any(|c| c.as_os_str() == "tests") {
            continue;
        }
        let text = fs::read_to_string(&file).expect("读源码");
        let mut prev_nonempty = String::new();
        for line in text.lines() {
            let t = line.trim();
            if let Some(pos) = t.find("pub extern \"C\" fn amiki_") {
                let tail = &t[pos + "pub extern \"C\" fn ".len()..];
                let name: String = tail
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                assert_eq!(
                    prev_nonempty, "#[no_mangle]",
                    "{} 里的 `{name}` 上面不是 #[no_mangle]（而是 `{prev_nonempty}`）—— \
                     漏写它函数就不会被导出",
                    file.display()
                );
                out.insert(name);
            }
            if !t.is_empty() {
                prev_nonempty = t.to_string();
            }
        }
    }
    out
}

/// 从一行里取"某个标记之前的那个整数"。
fn number_before(line: &str, marker: &str) -> Option<u32> {
    let at = line.find(marker)?;
    let head = &line[..at];
    let digits: String = head
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == ' ')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.trim().parse().ok()
}

/// 在 `text` 里找同时含有 `needles` 的那一行，再取 `marker` 之前的数字。
fn documented_number(text: &str, needles: &[&str], marker: &str, where_: &str) -> u32 {
    for line in text.lines() {
        if needles.iter().all(|n| line.contains(n)) {
            if let Some(n) = number_before(line, marker) {
                return n;
            }
        }
    }
    panic!("在 {where_} 里找不到含 {needles:?} 且 `{marker}` 前带数字的行 —— 措辞改了就要同步本测试的解析");
}

/// 从一行里取"某个标记之后的第一个整数"。
fn number_after(line: &str, marker: &str) -> Option<u32> {
    let rest = &line[line.find(marker)? + marker.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// 在 `text` 里找同时含有 `needles` 的那一行，再取 `marker` 之后的数字。
fn documented_number_after(text: &str, needles: &[&str], marker: &str, where_: &str) -> u32 {
    for line in text.lines() {
        if needles.iter().all(|n| line.contains(n)) {
            if let Some(n) = number_after(line, marker) {
                return n;
            }
        }
    }
    panic!("在 {where_} 里找不到含 {needles:?} 且 `{marker}` 后带数字的行 —— 措辞改了就要同步本测试的解析");
}

#[test]
fn symbol_count_matches_the_header_the_code_and_the_docs() {
    let declared = symbols_declared_in_the_header(&read("include/amiki.h"));
    let exported = symbols_exported_by_this_crate();
    assert!(!declared.is_empty(), "从 amiki.h 里解析出 0 条声明 —— 是解析错了，不是 ABI 空了");

    let missing: Vec<_> = declared.difference(&exported).collect();
    let extra: Vec<_> = exported.difference(&declared).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "header 与实现不一致：header 有而实现没有 {missing:?}；实现有而 header 没声明 {extra:?}"
    );

    let overview = read("../../docs/OVERVIEW.md");
    let capi_readme = read("README.md");
    let in_overview = documented_number(
        &overview,
        &["个符号", "verify.cmd"],
        "个符号",
        "docs/OVERVIEW.md 的 C ABI 行",
    );
    let in_readme = documented_number_after(
        &capi_readme,
        &["header 声明"],
        "header 声明",
        "crates/amiki-capi/README.md 的 verify 说明",
    );

    let n = declared.len() as u32;
    assert_eq!(
        (in_overview, in_readme, n),
        (n, n, n),
        "符号数三处不一致：OVERVIEW 写 {in_overview}、capi/README 写 {in_readme}、\
         header 声明 {n}、实现导出 {} —— 改契约时四处要一起改",
        exported.len()
    );

    // 根 README（仓库主页）也是符号数的一个副本 —— 它最显眼，也最容易漂：
    // 重写主页时实测发现它还写着 0.3.0 时代的 "72 个导出符号"。
    let root = read("../../README.md");
    let in_root = documented_number(&root, &["个符号", "C ABI"], "个符号", "根 README 的 C ABI 一句");
    assert_eq!(
        in_root, n,
        "根 README 写 {in_root} 个符号，而 header 声明 {n} 个 —— 主页的这句话也要跟着改"
    );

    // "符号数 N/M" 这种写法只在**描述 verify.cmd 本次输出**时出现（历史数字会写
    // "80 个导出符号（0.4.0）"这类带版本的措辞），所以它必须永远是当前值。
    for (file, text) in [("docs/OVERVIEW.md", &overview), ("crates/amiki-capi/README.md", &capi_readme)] {
        for line in text.lines() {
            let Some(at) = line.find("符号数") else { continue };
            let rest = &line[at + "符号数".len()..];
            let digits: String = rest
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            let Some(slash) = rest.find('/') else { continue };
            let after: String = rest[slash + 1..]
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if digits.is_empty() || after.is_empty() {
                continue;
            }
            assert_eq!(
                (digits.as_str(), after.as_str()),
                (n.to_string().as_str(), n.to_string().as_str()),
                "{file} 里写着「符号数 {digits}/{after}」—— 这是 verify.cmd 的本次输出，\
                 必须是 {n}/{n}；写历史数字请改成「N 个导出符号（哪个版本）」的说法"
            );
        }
    }
}

#[test]
fn license_declaration_matches_the_license_file() {
    // 补这一条的原因：仓库里曾出现 "LICENSE 文本是 MIT，而 Cargo.toml 声明
    // MIT OR Apache-2.0" 的不一致。两者不是等价的 —— 双许可要求**两份完整文本**
    // 加标准声明，只有一份 MIT 文本时声明就必须是 MIT。改一处忘另一处就会红。
    let cargo = read("../../Cargo.toml");
    let declared = cargo
        .lines()
        .find_map(|l| l.trim().strip_prefix("license = "))
        .map(|v| v.trim().trim_matches('"').to_string())
        .expect("Cargo.toml 的 [workspace.package] 里应有 license = \"...\"");
    let first = read("../../LICENSE")
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    let spdx = if first.starts_with("MIT") {
        "MIT"
    } else if first.starts_with("Apache") {
        "Apache-2.0"
    } else {
        ""
    };
    assert!(
        !spdx.is_empty(),
        "LICENSE 首行是 `{first}`：本测试只认识 MIT / Apache 两种 —— \
         换成别的许可证时请同步这个映射"
    );
    assert_eq!(
        declared, spdx,
        "Cargo.toml 声明 `{declared}`，而 LICENSE 文件是 `{spdx}`：两者必须一致"
    );
}

#[test]
fn header_version_macros_match_cargo_toml() {
    let header = read("include/amiki.h");
    let mut got = [None, None, None];
    for line in header.lines() {
        let t = line.trim();
        for (i, name) in ["AMIKI_VERSION_MAJOR", "AMIKI_VERSION_MINOR", "AMIKI_VERSION_PATCH"]
            .iter()
            .enumerate()
        {
            if let Some(rest) = t.strip_prefix(&format!("#define {name}")) {
                got[i] = rest.trim().parse::<u32>().ok();
            }
        }
    }
    let pkg = env!("CARGO_PKG_VERSION");
    let parts: Vec<u32> = pkg
        .split(['.', '-'])
        .take(3)
        .map(|x| x.parse().expect("Cargo.toml 版本必须形如 x.y.z"))
        .collect();
    assert_eq!(
        got,
        [Some(parts[0]), Some(parts[1]), Some(parts[2])],
        "amiki.h 的版本宏 {got:?} 与 Cargo.toml 的 {pkg} 不一致 —— \
         这是 ABI 的一部分，调用方会用宏做编译期判断"
    );
}
