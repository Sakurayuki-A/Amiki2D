# -*- coding: utf-8 -*-
"""生成独立版浏览器验证页 browser_test_standalone.html。

把 wasm 模块与 Inter 字体以 base64 内嵌进页面，双击（file://）即可运行，
不依赖本地 HTTP 服务器 —— 浏览器禁止 file:// 页面用 fetch 读文件，
所以 HTTP 版（browser_test.html）在 file:// 下必然 "Failed to fetch"。

用法：python make_standalone.py   （在 examples/wasm 目录下运行）
wasm 或字体更新后需要重新生成。生成物不入库（.gitignore）。
"""
import base64
import io
import pathlib
import re
import sys

here = pathlib.Path(__file__).parent
wasm = (here / "amiki2d.wasm").read_bytes()
font = (here / ".." / ".." / ".." / ".." / "assets" / "fonts" / "Inter-Regular.ttf").resolve().read_bytes()
src = (here / "browser_test.html").read_text(encoding="utf-8")

# 1. wasm 的 fetch → 内嵌 base64
old = "const bytes = new Uint8Array(await (await fetch('amiki2d.wasm')).arrayBuffer());"
new = "const bytes = B64_U8(WASM_B64);"
assert src.count(old) == 1, "wasm fetch 站点没找到"
src = src.replace(old, new)

# 2. 字体的 fetch → 内嵌 base64
old = """        // 字体走 HTTP 取（与 Node 的相对路径同一份文件）
        const resp = await fetch('../../../../assets/fonts/Inter-Regular.ttf');
        if (resp.ok) {
            const fontBytes = new Uint8Array(await resp.arrayBuffer());"""
new = """        // 字体内嵌（与 Node 读的是同一份文件）
        if (FONT_B64.length > 0) {
            const fontBytes = B64_U8(FONT_B64);"""
assert src.count(old) == 1, "字体 fetch 站点没找到"
src = src.replace(old, new)

# 3. 注入数据与解码函数（放在垫片脚本之后）
inject = """
<script>
// 内嵌数据（由 make_standalone.py 生成）
const WASM_B64 = "%s";
const FONT_B64 = "%s";
function B64_U8(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
}
</script>
<script src="amiki.js">""" % (base64.b64encode(wasm).decode(), base64.b64encode(font).decode())
old = '<script src="amiki.js">'
assert src.count(old) == 1, "amiki.js 标签没找到"
src = src.replace(old, inject)

out = here / "browser_test_standalone.html"
io.open(out, "w", encoding="utf-8", newline="").write(src)
print(f"生成 {out.name}（{out.stat().st_size/1024:.0f} KB，含 wasm {len(wasm)/1024:.0f} KB + 字体 {len(font)/1024:.0f} KB）")
