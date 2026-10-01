#!/usr/bin/env bash
# ============================================================================
# Amiki2D 跨厂商验证 —— 一条命令跑完并产出可直接贴进 issue 的结果
# ============================================================================
#
# ## 这个脚本是给谁的
#
# 作者只有一块 NVIDIA RTX 3050，所以**跨厂商验证是已知限制而不是待办**
# （见 docs/KNOWN_ISSUES.md 的 A1）。但这件事在**技术上是可验证的**，
# 只是需要硬件。
#
# 所以它是**钩子**，不是任务：
#   · 你有 AMD / Intel / Apple Silicon / 移动 GPU → 跑它，把输出贴进 issue
#   · 没人跑 → 也没有任何损失
#
# ## 它做什么
#
#   1. 记录环境（OS、GPU、驱动）—— 没有这些，结果无法归因
#   2. 逐个后端跑 amiki-gpu 的全部测试
#   3. 额外跑一次全工作区测试（默认后端）
#   4. 打印一张可以直接复制的表
#
# ## 用法
#
#   bash tools/verify_gpu.sh              # 全部
#   bash tools/verify_gpu.sh --quick      # 跳过全工作区那一遍（快很多）
#
# ## 退出码
#
#   0 = 全部通过
#   1 = 有失败（**这一般是引擎的问题，不是你的问题** —— 请把输出贴进 issue）
#
# ============================================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数：$arg（用 --help 看用法）" >&2; exit 2 ;;
    esac
done

hr() { printf '%s\n' "----------------------------------------------------------------------"; }

# ============================================================================
# 1. 环境
# ============================================================================

OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
OS_ARCH="$(uname -m 2>/dev/null || echo unknown)"
OS_RELEASE="$(uname -r 2>/dev/null || echo unknown)"

gpu_info() {
    # 尽力而为：每个平台能拿到多少算多少。**拿不到不是错误** ——
    # 结果表里写 unknown 也比不跑强。
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
        return
    fi
    case "$OS_NAME" in
        Linux)
            if command -v lspci >/dev/null 2>&1; then
                lspci 2>/dev/null | grep -Ei 'vga|3d|display' | sed 's/^[0-9a-f:.]* //'
            fi
            ;;
        Darwin)
            system_profiler SPDisplaysDataType 2>/dev/null \
                | grep -E 'Chipset Model|Metal' | sed 's/^ *//'
            ;;
        MINGW*|MSYS*|CYGWIN*)
            # Git Bash / MSYS：用 wmic（已弃用但仍在）或 PowerShell 兜底
            if command -v powershell >/dev/null 2>&1; then
                powershell -NoProfile -Command \
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name" \
                    2>/dev/null
            fi
            ;;
    esac
}

GPU_TEXT="$(gpu_info | sed '/^$/d' | head -4)"
[ -z "$GPU_TEXT" ] && GPU_TEXT="unknown（脚本没能识别；请手工补上）"

RUSTC_VER="$(rustc --version 2>/dev/null || echo 'rustc 不可用')"
CARGO_VER="$(cargo --version 2>/dev/null || echo 'cargo 不可用')"

echo "======================================================================"
echo " Amiki2D 跨厂商验证"
echo "======================================================================"
echo " 操作系统   : $OS_NAME $OS_RELEASE ($OS_ARCH)"
echo " GPU        :"
echo "$GPU_TEXT" | sed 's/^/              /'
echo " 工具链     : $RUSTC_VER"
echo "              $CARGO_VER"
echo " 仓库       : $REPO_ROOT"
echo " 提交       : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo " 版本       : $((grep -m1 '^version' Cargo.toml 2>/dev/null || echo '?') | tr -d ' ')"
hr

# ============================================================================
# 2. 编译（失败就没必要往下跑）
# ============================================================================

echo "[构建] cargo build --workspace --release"
if ! cargo build --workspace --release 2>&1 | tail -3; then
    echo
    echo "构建失败 —— 这本身就是一个 bug，请把上面的输出贴进 issue。" >&2
    exit 1
fi
hr

# ============================================================================
# 3. 逐后端跑测试
# ============================================================================

# macOS 上没有 AMIKI_BACKENDS 可用的值（wgpu 走 Metal），所以只跑默认一遍。
# 这不是遗漏 —— 引擎的 `AMIKI_BACKENDS` 只认 vulkan / dx12 / gl / all。
BACKENDS=""
case "$OS_NAME" in
    Darwin) BACKENDS="(default)" ;;
    *)      BACKENDS="vulkan dx12 gl (default)" ;;
esac

# 结果收集：每行是 "标签|通过|失败组|说明"
RESULTS=""
OVERALL_FAIL=0

run_one() {
    local label="$1" env_backend="$2" pkg="$3" extra="$4"
    local log
    log="$(mktemp)"

    if [ "$env_backend" = "(default)" ]; then
        # shellcheck disable=SC2086
        cargo test $pkg --release $extra >"$log" 2>&1
    else
        # shellcheck disable=SC2086
        AMIKI_BACKENDS="$env_backend" cargo test $pkg --release $extra >"$log" 2>&1
    fi
    local rc=$?

    local passed failed note
    # 用 awk 而不是 `paste | bc`：bc 在 Git Bash / 精简容器里**不一定存在**，
    # 而它不存在时上面的写法会静默退化成 0 —— 那会reporting "通过 0" 却其实跑了
    # 一堆测试。这一类"静默错误读数"正是本项目最不能接受的，所以换成一定有
    # awk 的写法。（第一次跑就是踩了这个：四个后端全报"通过 0"。）
    # 注意 "25 passed;" 里的分号：直接比较字段会**永远不匹配**，然后静默报 0。
    # 先把 `;` `,` 换成空格再拆字段。（第一版就是这么错的。）
    passed="$(awk '
        /test result:/ {
            line = $0
            gsub(/[;,]/, " ", line)
            n = split(line, a, " ")
            for (i = 1; i <= n; i++) {
                if (a[i] == "passed") { total += a[i - 1] }
            }
        }
        END { printf "%d", total + 0 }
    ' "$log")"
    [ -z "$passed" ] && passed=0
    # 顺带记下"跑了几个测试二进制" —— 若为 0，说明 cargo 根本没跑测试
    # （例如 -p 写错），那和"全通过"是两件事。
    local ran note_ran
    ran="$(grep -cE 'test result:' "$log" || true)"
    if [ "$ran" -eq 0 ]; then
        passed=0
        note_ran="**没有测试被执行** —— cargo 可能没找到目标"
    fi
    failed="$(grep -cE 'test result: FAILED' "$log" || true)"

    note="${note_ran:-}"
    if [ -n "${note_ran:-}" ]; then OVERALL_FAIL=1; fi
    if [ "$rc" -ne 0 ]; then
        note="见下方日志"
        OVERALL_FAIL=1
    fi
    # 已知例外：GL 不支持位图，那条测试会**按预期**报错
    if [ "$env_backend" = "gl" ] && grep -q 'GL 后端无法正确读取位图' "$log"; then
        note="位图按预期显式拒绝"
    fi

    RESULTS="${RESULTS}${label}|${passed}|${failed}|${note}
"

    printf '  %-22s 通过 %-4s 失败组 %-3s %s\n' "$label" "$passed" "$failed" "$note"

    # 失败时把关键行打出来 —— 但要过滤掉已经预期的那条
    if [ "$rc" -ne 0 ]; then
        echo "    ---- 失败摘要 ----"
        grep -E 'panicked|FAILED|error\[|^error|Validation Error|X3511|C1115' "$log" \
            | grep -v 'GL 后端无法正确读取位图' \
            | head -12 | sed 's/^/    /'
        echo "    （完整日志：$log）"
    else
        rm -f "$log"
    fi
}

echo "[逐后端] amiki-gpu 的全部测试"
for b in $BACKENDS; do
    if [ "$b" = "(default)" ]; then
        run_one "默认后端" "(default)" "-p amiki-gpu" ""
    else
        run_one "$b" "$b" "-p amiki-gpu" ""
    fi
done
hr

# ============================================================================
# 4. 全工作区（可选）
# ============================================================================

if [ "$QUICK" -eq 0 ]; then
    echo "[全工作区] 默认后端"
    run_one "全工作区" "(default)" "--workspace" ""
    hr
fi

# ============================================================================
# 5. 结果表（可直接贴进 issue）
# ============================================================================

echo
echo "======================================================================"
echo " 结果 —— 请把下面整块复制进 issue"
echo "======================================================================"
echo
echo "| 项 | 值 |"
echo "|---|---|"
echo "| 操作系统 | $OS_NAME $OS_RELEASE ($OS_ARCH) |"
echo "| GPU | $(echo "$GPU_TEXT" | paste -sd'; ' -) |"
echo "| 工具链 | $RUSTC_VER |"
echo "| 提交 | $(git rev-parse --short HEAD 2>/dev/null || echo unknown) |"
echo
echo "| 后端 | 通过 | 失败组 | 说明 |"
echo "|---|---|---|---|"
printf '%s' "$RESULTS" | while IFS='|' read -r label passed failed note; do
    [ -z "$label" ] && continue
    echo "| $label | $passed | $failed | $note |"
done
echo
echo "clippy："
if cargo clippy --workspace --all-targets --message-format=short 2>&1 | grep -qE 'warning:|error'; then
    echo "  有警告/错误（请一并贴出）"
else
    echo "  零警告"
fi
echo
hr

if [ "$OVERALL_FAIL" -eq 0 ]; then
    echo "全部通过。"
    echo
    echo "请把上面那张表贴进 issue（标题建议："
    echo "  \"Cross-vendor validation: <你的 GPU> on <OS>\"）。"
    echo "即使全部通过也**很有价值** —— 它把 A1 从「未验证」变成「已验证」。"
    exit 0
else
    echo "有失败。**这一般是引擎的问题，不是你的问题。**" >&2
    echo "请把上面的失败摘要与结果表一起贴进 issue。" >&2
    exit 1
fi
