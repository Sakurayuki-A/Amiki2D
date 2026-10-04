# -*- coding: utf-8 -*-
"""生成独立版浏览器验证页 browser_test_standalone.html。

把 wasm 模块与**两个字体**以 base64 内嵌进页面，双击（file://）即可运行，
不依赖本地 HTTP 服务器 —— 浏览器禁止 file:// 页面用 fetch 读文件，
所以 HTTP 版（browser_test.html）在 file:// 下必然 "Failed to fetch"。

0.7.0 起字体有两个（Inter 管拉丁、Noto Sans SC 管汉字）：**两个都必须内嵌**，
否则独立版画不出同样的画面，哈希必然对不上。

用法：python make_standalone.py   （在 examples/wasm 目录下运行）
wasm 或字体更新后需要重新生成。生成物不入库（.gitignore）。
"""
import base64
import io
import pathlib
import sys

here = pathlib.Path(__file__).parent
root = (here / ".." / ".." / ".." / "..").resolve()
wasm = (here / "amiki2d.wasm").read_bytes()
font = (root / "assets" / "fonts" / "Inter-Regular.ttf").read_bytes()
font_cjk = (root / "assets" / "fonts" / "NotoSansSC-Regular.ttf").read_bytes()
src = (here / "browser_test.html").read_text(encoding="utf-8")

# 1. wasm 的 fetch → 内嵌 base64
old = "const bytes = new Uint8Array(await (await fetch('amiki2d.wasm')).arrayBuffer());"
new = "const bytes = B64_U8(WASM_B64);"
assert src.count(old) == 1, "wasm fetch 站点没找到"
src = src.replace(old, new)

# 2. 注入数据与解码函数（放在垫片脚本之后）。
#    字体那一段**不需要**改写：页面自己就是"内嵌优先，否则 fetch"（见 browser_test.html），
#    这样 HTTP 版与独立版共用同一段字体加载代码，不会两边漂移。
inject = """
<script>
// 内嵌数据（由 make_standalone.py 生成）
const WASM_B64 = "%s";
const FONT_B64 = "%s";
const FONT_B64_CJK = "%s";
function B64_U8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
}
</script>
<script src="amiki.js">""" % (
    base64.b64encode(wasm).decode(),
    base64.b64encode(font).decode(),
    base64.b64encode(font_cjk).decode(),
)
old = '<script src="amiki.js">'
assert src.count(old) == 1, "amiki.js 标签没找到"
src = src.replace(old, inject)

out = here / "browser_test_standalone.html"
io.open(out, "w", encoding="utf-8", newline="").write(src)
print(
    "生成 %s（%.0f KB，含 wasm %.0f KB + 字体 %.0f KB + 中文字体 %.0f KB）"
    % (
        out.name,
        out.stat().st_size / 1024,
        len(wasm) / 1024,
        len(font) / 1024,
        len(font_cjk) / 1024,
    )
)
