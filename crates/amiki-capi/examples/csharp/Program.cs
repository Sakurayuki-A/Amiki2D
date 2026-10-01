// ============================================================================
// Amiki2D C ABI —— C# 示例
// ============================================================================
//
// 它做与 C 版示例相同的事，但写法是 .NET 风格的：
//   · 用 `using` 表达式接管所有权（header 的 `_destroy` 规则）
//   · 失败抛异常而不是查返回码
//   · 写出一张 BMP 供肉眼确认
//
// 运行：build.cmd
// ============================================================================

using System;
using System.IO;
using Amiki;
// `Path` 与 `System.IO.Path` 撞名 —— 用别名把引擎的路径类型区分出来
using AmikiPath = Amiki.Path;

const uint W = 640, H = 400;

Console.WriteLine($"Amiki2D {AmikiHandle.VersionString}（运行期编码 0x{AmikiHandle.Version:X6}）");

// 所有权：context 与 canvas 都用 using 包住 —— header 要求这样释放。
using var ctx = Context.Create(Backend.Auto);
Console.WriteLine($"后端：{ctx.ActualBackend}（{ctx.AdapterName}）");

using var cv = ctx.CreateCanvas(W, H);

// ---- 背景 ----
cv.FillRect(0, 0, W, H, Paint.Solid(Color.Rgb8(24, 26, 32)));

// ---- 描边折线（圆角 + 圆端帽）----
using (var poly = AmikiPath.Create())
{
    poly.MoveTo(60, 300).LineTo(160, 220).LineTo(260, 320).LineTo(360, 200);
    var style = StrokeStyle.New(8.0f, LineCap.Round, LineJoin.Round);
    cv.StrokePath(poly, style, Paint.Solid(Color.Rgb8(90, 200, 250)));
}

// ---- 径向渐变圆 ----
using (var grad = Gradient.Radial(470, 150, 110,
           new[]
           {
               new GradientStop(0.0f, Color.Rgb8(255, 214, 102)),
               new GradientStop(0.5f, Color.Rgb8(255, 122, 89)),
               new GradientStop(1.0f, Color.Rgb8(160, 80, 200)),
           }))
{
    cv.FillCircle(470, 150, 100, Paint.FromGradient(grad));
}
// 注意：`grad` 在 using 块结束时销毁 —— 这是**允许**的，
// 因为引擎内部对渐变持计数引用（header 所有权第 4 条）。

// ---- 位图：8×8 棋盘 ----
{
    var checker = new byte[8 * 8 * 4];
    for (var y = 0; y < 8; y++)
    for (var x = 0; x < 8; x++)
    {
        var on = ((x ^ y) & 1) != 0;
        var o = (y * 8 + x) * 4;
        checker[o + 0] = checker[o + 1] = checker[o + 2] = (byte)(on ? 255 : 40);
        checker[o + 3] = 255;
    }
    using var img = Image.FromRgba8(checker, 8, 8);
    Console.WriteLine($"位图：{img.Size.W}×{img.Size.H}");
    cv.DrawImage(img, 40, 40, 200, 160);
    // `img` 在 using 块结束时销毁，但它在这次绘制里已经被引擎引用 —— 安全。
}

// ---- 离屏层 + 模糊 ----
cv.BeginLayer(new LayerBounds(300, 240, 300, 140), Filter.Blur(6.0f), 1.0f, BlendMode.Normal);
cv.FillRect(330, 280, 430, 340, Paint.Solid(Color.Rgb8(120, 255, 170)));
cv.FillCircle(480, 310, 40, Paint.Solid(Color.Rgb8(255, 210, 90)));
cv.EndLayer();

// ---- 文本（字体找不到就跳过，示例不假设机器上有哪个字体）----
{
    var candidates = new[]
    {
        "../../../../assets/fonts/Inter-Regular.ttf",
        "../../assets/fonts/Inter-Regular.ttf",
        "assets/fonts/Inter-Regular.ttf",
        "../assets/fonts/Inter-Regular.ttf",
    };
    Font? font = null;
    foreach (var path in candidates)
    {
        try
        {
            font = Font.FromFile(path);
            Console.WriteLine($"字体：{path}");
            break;
        }
        catch (AmikiException)
        {
            // 这个路径没有字体，试下一个
        }
    }

    if (font is not null)
    {
        using (font)
        {
            var (ascent, descent, gap) = font.Metrics;
            Console.WriteLine($"字体度量（每 em）：ascent {ascent:F3} / descent {descent:F3} / line_gap {gap:F3}");
            cv.DrawText(font, "Amiki2D C ABI", 40, 120, 42, Paint.Solid(Color.Rgb8(240, 244, 255)));
        }
    }
    else
    {
        Console.WriteLine("字体：未找到，跳过文本");
    }
}

// ---- 渲染 ----
var pixels = cv.RenderRgba8();
WriteBmp("amiki_csharp_demo.bmp", pixels, W, H);
Console.WriteLine($"已写出 amiki_csharp_demo.bmp（{W}×{H}，{pixels.Length} 字节）");

// ---- 错误路径：托管层把返回码变成了异常 ----
try
{
    cv.SetBlend((BlendMode)9999);
    Console.WriteLine("越界枚举竟然被接受了 —— 这是 bug");
    return 1;
}
catch (AmikiException e)
{
    Console.WriteLine($"越界枚举被正确拒绝：{e.Code} / {AmikiException.ResultString(e.Code)}");
}

Console.WriteLine("完成");
return 0;

// ---------------------------------------------------------------------------
// 写 24 位 BMP。不引入任何图像库 —— 这个示例要能单独编译运行。
// BMP 行序自下而上、每行 4 字节对齐；引擎给的是自上而下的非预乘 RGBA。
// ---------------------------------------------------------------------------
static void WriteBmp(string path, byte[] rgba, uint w, uint h)
{
    var rowRaw = (int)(w * 3);
    var pad = (4 - (rowRaw % 4)) % 4;
    var row = rowRaw + pad;
    var dataSize = row * (int)h;
    var fileSize = 54 + dataSize;

    using var fs = File.Create(path);
    using var bw = new BinaryWriter(fs);

    bw.Write((byte)'B');
    bw.Write((byte)'M');
    bw.Write(fileSize);
    bw.Write(0);            // 保留
    bw.Write(54);           // 像素数据偏移
    bw.Write(40);           // DIB 头大小
    bw.Write((int)w);
    bw.Write((int)h);
    bw.Write((short)1);     // planes
    bw.Write((short)24);    // bpp
    bw.Write(0);            // 压缩
    bw.Write(dataSize);
    bw.Write(2835);         // 水平分辨率
    bw.Write(2835);
    bw.Write(0);
    bw.Write(0);

    var line = new byte[row];
    for (var y = (int)h - 1; y >= 0; y--)
    {
        Array.Clear(line);
        for (var x = 0; x < w; x++)
        {
            var o = (y * (int)w + x) * 4;
            line[x * 3 + 0] = rgba[o + 2]; // B
            line[x * 3 + 1] = rgba[o + 1]; // G
            line[x * 3 + 2] = rgba[o + 0]; // R
        }
        bw.Write(line);
    }
}
