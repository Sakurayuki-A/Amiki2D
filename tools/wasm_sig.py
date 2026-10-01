"""解析 wasm 模块的函数签名，看清结构体按值传递被降级成了什么。

WebAssembly 没有"结构体按值"这种概念，所以 C ABI 里的
`AmikiPaint f(AmikiColor)` 会被编译器降级 —— 通常是：

    大结构体返回 → sret（调用方传一个隐藏的返回空间指针）
    大结构体参数 → 按引用传（内部改成指针）

这个脚本把真实的签名打出来，加载器按它写，不靠猜。
"""
import struct
import sys


def leb(data, i):
    """无符号 LEB128。"""
    r = 0
    shift = 0
    while True:
        b = data[i]
        i += 1
        r |= (b & 0x7F) << shift
        if not (b & 0x80):
            return r, i
        shift += 7


VAL = {0x7F: "i32", 0x7E: "i64", 0x7D: "f32", 0x7C: "f64",
       0x7B: "v128", 0x70: "funcref", 0x6F: "externref"}


def parse(path):
    data = open(path, "rb").read()
    assert data[:4] == b"\x00asm", "不是 wasm 模块"
    i = 8

    types = []          # [(params, results)]
    func_types = []     # 每个函数的 type index
    exports = []        # (name, kind, index)

    while i < len(data):
        sid = data[i]
        i += 1
        size, i = leb(data, i)
        end = i + size
        body = data[i:end]

        if sid == 1:  # type
            j = 0
            n, j = leb(body, j)
            for _ in range(n):
                assert body[j] == 0x60, "只支持 functype"
                j += 1
                np_, j = leb(body, j)
                ps = []
                for _ in range(np_):
                    ps.append(VAL.get(body[j], hex(body[j])))
                    j += 1
                nr, j = leb(body, j)
                rs = []
                for _ in range(nr):
                    rs.append(VAL.get(body[j], hex(body[j])))
                    j += 1
                types.append((ps, rs))

        elif sid == 3:  # function
            j = 0
            n, j = leb(body, j)
            for _ in range(n):
                t, j = leb(body, j)
                func_types.append(t)

        elif sid == 7:  # export
            j = 0
            n, j = leb(body, j)
            for _ in range(n):
                ln, j = leb(body, j)
                nm = body[j:j + ln].decode("utf-8", "replace")
                j += ln
                kind = body[j]
                j += 1
                idx, j = leb(body, j)
                exports.append((nm, kind, idx))

        i = end

    return types, func_types, exports


def main():
    path = sys.argv[1]
    types, func_types, exports = parse(path)
    print(f"模块: {path}")
    print(f"函数类型 {len(types)} 种；定义函数 {len(func_types)} 个；导出 {len(exports)} 个\n")

    # 只看 amiki_* 且是函数(0)的
    want = sys.argv[2:] or None
    print(f"{'导出名':<36} {'参数':<44} 返回")
    print("-" * 100)
    for nm, kind, idx in sorted(exports):
        if kind != 0:
            continue
        if not nm.startswith("amiki_"):
            continue
        if want and not any(w in nm for w in want):
            continue
        # 导出索引 = 定义函数索引（没有 import 函数时；这个模块确实没有）
        if idx >= len(func_types):
            print(f"{nm:<36} (idx {idx} 超出定义函数范围)")
            continue
        ps, rs = types[func_types[idx]]
        print(f"{nm:<36} {', '.join(ps) or '(无)':<44} {', '.join(rs) or '(无)'}")


if __name__ == "__main__":
    main()
