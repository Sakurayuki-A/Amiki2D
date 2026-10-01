// ============================================================================
// Amiki2D C ABI 的 C# 绑定（手写 P/Invoke）
// ============================================================================
//
// 这份文件是**从 include/amiki.h 手工映射**过来的，不是生成的。
// 它演示了绑定作者需要处理的四件事：
//
//   1. 结构体布局：`[StructLayout(LayoutKind.Sequential)]` 对应 header 的
//      `typedef struct`。**字段顺序必须与 header 完全一致** —— 顺序错了
//      不会编译报错，只会静默读到错位的值。
//
//   2. 调用约定：Rust 的 `extern "C"` 在 Windows x64 上就是 cdecl 的等价物，
//      `CallingConvention.Cdecl` 是显式写法（也是 32 位下的正确选择）。
//
//   3. 字符串：header 明确要求 UTF-8，所以是本机 UTF-8 封送
//      （`StringMarshalling.Utf8`），**不是** .NET 默认的 UTF-16。
//      用错的后果是字体路径变成乱码、加载失败。
//
//   4. 所有权：`_destroy` 必须被调用。这里用 `IDisposable` 包装，
//      让 `using` 表达式接管 —— header 的所有权规则在 C# 里落到这一点上。
//
// 版本：Amiki2D 0.4.0
// ============================================================================

using System;
using System.Runtime.InteropServices;

namespace Amiki;

// ============================================================================
// 枚举（取值与 header 逐一对应）
// ============================================================================

public enum Result
{
    Ok = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    BackendUnavailable = 3,
    NotSupported = 4,
    /// <summary>引擎内部 panic 被捕获 —— 这是引擎的 bug，请上报。</summary>
    Panic = 5,
    Unknown = 6,
}

public enum BlendMode
{
    Normal = 0, Multiply = 1, Screen = 2, Overlay = 3,
    Darken = 4, Lighten = 5, ColorDodge = 6, ColorBurn = 7,
    HardLight = 8, SoftLight = 9, Difference = 10, Exclusion = 11,
    Hue = 12, Saturation = 13, Color = 14, Luminosity = 15,
}

public enum LineCap { Butt = 0, Round = 1, Square = 2 }

public enum LineJoin { Bevel = 0, Miter = 1, Round = 2 }

public enum FilterKind { None = 0, Blur = 1, DropShadow = 2 }

public enum PaintKind { Solid = 0, Linear = 1, Radial = 2, Sweep = 3 }

public enum Backend { Auto = 0, Gpu = 1, Cpu = 2 }

// ============================================================================
// 结构体（字段顺序与 header **逐字段**一致）
// ============================================================================

[StructLayout(LayoutKind.Sequential)]
public struct Vec2
{
    public float X, Y;
    public Vec2(float x, float y) { X = x; Y = y; }
}

[StructLayout(LayoutKind.Sequential)]
public struct Affine
{
    public float A, B, C, D, E, F;
    public static Affine Identity() => Native.amiki_affine_identity();
    public static Affine Translate(float dx, float dy) => Native.amiki_affine_translate(dx, dy);
    public static Affine Rotate(float radians) => Native.amiki_affine_rotate(radians);
    public static Affine Scale(float s) => Native.amiki_affine_scale(s);
}

[StructLayout(LayoutKind.Sequential)]
public struct Color
{
    public float R, G, B, A;
    public static Color Rgba(float r, float g, float b, float a) => Native.amiki_color_rgba(r, g, b, a);
    public static Color Rgb8(byte r, byte g, byte b) => Native.amiki_color_rgb8(r, g, b);
    public static Color Rgba8(byte r, byte g, byte b, byte a) => Native.amiki_color_rgba8(r, g, b, a);
}

[StructLayout(LayoutKind.Sequential)]
public struct LayerBounds
{
    public int X, Y;
    public uint Width, Height;
    public LayerBounds(int x, int y, uint w, uint h) { X = x; Y = y; Width = w; Height = h; }
}

[StructLayout(LayoutKind.Sequential)]
public struct GradientStop
{
    public float T;
    public Color Color;
    public GradientStop(float t, Color c) { T = t; Color = c; }
}

[StructLayout(LayoutKind.Sequential)]
public struct Placement
{
    public Vec2 Origin, UAxis, VAxis;
    public int Repeat;
    public static Placement Rect(float x0, float y0, float x1, float y1) => new()
    {
        Origin = new Vec2(x0, y0),
        UAxis = new Vec2(x1 - x0, 0),
        VAxis = new Vec2(0, y1 - y0),
        Repeat = 0,
    };
}

[StructLayout(LayoutKind.Sequential)]
public struct StrokeStyle
{
    public float Width;
    public LineCap Cap;
    public LineJoin Join;
    public float MiterLimit;
    public static StrokeStyle Default() => Native.amiki_stroke_style_default();
    public static StrokeStyle New(float width, LineCap cap, LineJoin join) =>
        Native.amiki_stroke_style_new(width, cap, join);
}

[StructLayout(LayoutKind.Sequential)]
public struct Filter
{
    public FilterKind Kind;
    public float Sigma;
    public float Dx, Dy;
    public Color Color;
    public static Filter None() => Native.amiki_filter_none();
    public static Filter Blur(float sigma) => Native.amiki_filter_blur(sigma);
    public static Filter DropShadow(float dx, float dy, float sigma, Color c) =>
        Native.amiki_filter_drop_shadow(dx, dy, sigma, c);
}

[StructLayout(LayoutKind.Sequential)]
public struct Paint
{
    public PaintKind Kind;
    public Color Color;
    public IntPtr Gradient;
    public static Paint Solid(Color c) => Native.amiki_paint_solid(c);
    public static Paint FromGradient(Gradient g) => Native.amiki_paint_gradient(g.DangerousGetHandle());
}

// ============================================================================
// P/Invoke 声明
// ============================================================================

internal static partial class Native
{
    private const string Lib = "amiki2d";

    // ---- 版本 ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint amiki_version();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr amiki_version_string();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr amiki_result_string(Result r);

    // ---- 值构造 ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Affine amiki_affine_identity();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Affine amiki_affine_translate(float dx, float dy);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Affine amiki_affine_rotate(float radians);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Affine amiki_affine_scale(float s);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Color amiki_color_rgba(float r, float g, float b, float a);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Color amiki_color_rgb8(byte r, byte g, byte b);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Color amiki_color_rgba8(byte r, byte g, byte b, byte a);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern StrokeStyle amiki_stroke_style_default();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern StrokeStyle amiki_stroke_style_new(float width, LineCap cap, LineJoin join);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Filter amiki_filter_none();
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Filter amiki_filter_blur(float sigma);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Filter amiki_filter_drop_shadow(float dx, float dy, float sigma, Color c);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Paint amiki_paint_solid(Color c);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Paint amiki_paint_gradient(IntPtr gradient);

    // ---- context ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_context_create(Backend backend, out IntPtr ctx);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void amiki_context_destroy(IntPtr ctx);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Backend amiki_context_backend(IntPtr ctx);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr amiki_context_adapter_name(IntPtr ctx);

    // ---- canvas ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_create(IntPtr ctx, uint w, uint h, out IntPtr cv);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void amiki_canvas_destroy(IntPtr cv);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_clear(IntPtr cv);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_size(IntPtr cv, out uint w, out uint h);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_set_transform(IntPtr cv, Affine t);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_reset_transform(IntPtr cv);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_translate(IntPtr cv, float dx, float dy);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_rotate(IntPtr cv, float radians);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_scale(IntPtr cv, float s);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_set_blend(IntPtr cv, BlendMode blend);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_push_clip(IntPtr cv, IntPtr path);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_pop_clip(IntPtr cv);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_fill_path(IntPtr cv, IntPtr path, Paint paint);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_stroke_path(IntPtr cv, IntPtr path, StrokeStyle style, Paint paint);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_draw_rect(IntPtr cv, float x0, float y0, float x1, float y1, Paint paint);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_draw_circle(IntPtr cv, float cx, float cy, float r, Paint paint);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_draw_pentagram(IntPtr cv, float cx, float cy, float r, Paint paint);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_draw_image(IntPtr cv, IntPtr img, float x0, float y0, float x1, float y1);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_draw_image_placed(IntPtr cv, IntPtr img, Placement p);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_fill_path_image(IntPtr cv, IntPtr path, IntPtr img, Placement p);

    // UTF-8 显式封送：header 要求 UTF-8，不是 .NET 默认的 UTF-16。
    // 用 `MarshalAs(UnmanagedType.LPUTF8Str)` 而不是 `StringMarshalling.Utf8` ——
    // 后者在 .NET 10 的 `DllImportAttribute` 上不可用，前者各版本都有且语义相同。
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_draw_text(IntPtr cv, IntPtr font,
                                                         [MarshalAs(UnmanagedType.LPUTF8Str)] string utf8,
                                                         UIntPtr utf8Len,
                                                         float x, float y, float size, Paint paint);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_begin_layer(IntPtr cv, LayerBounds b, Filter f, float opacity, BlendMode blend);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_end_layer(IntPtr cv);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_render_rgba8(IntPtr cv, byte[] outPixels, UIntPtr outLen);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_canvas_set_flatten_tolerance(IntPtr cv, float eps);

    // ---- path ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_create(out IntPtr p);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void amiki_path_destroy(IntPtr p);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_clear(IntPtr p);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_move_to(IntPtr p, float x, float y);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_line_to(IntPtr p, float x, float y);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_quad_to(IntPtr p, float cx, float cy, float x, float y);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_cubic_to(IntPtr p, float c1x, float c1y, float c2x, float c2y, float x, float y);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_close(IntPtr p);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_rect(IntPtr p, float x0, float y0, float x1, float y1);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_path_circle(IntPtr p, float cx, float cy, float r);

    // ---- image ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_image_create_rgba8(byte[] pixels, uint w, uint h, UIntPtr stride, out IntPtr img);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void amiki_image_destroy(IntPtr img);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_image_size(IntPtr img, out uint w, out uint h);

    // ---- font ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_font_create_from_file(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path, out IntPtr font);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void amiki_font_destroy(IntPtr font);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_font_metrics(IntPtr font, out float ascent, out float descent, out float lineGap);

    // ---- gradient ----
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_gradient_create_linear(float x0, float y0, float x1, float y1,
                                                               GradientStop[] stops, uint count, out IntPtr g);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_gradient_create_radial(float cx, float cy, float radius,
                                                               GradientStop[] stops, uint count, out IntPtr g);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern Result amiki_gradient_create_sweep(float cx, float cy, float start,
                                                              GradientStop[] stops, uint count, out IntPtr g);
    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void amiki_gradient_destroy(IntPtr g);
}

// ============================================================================
// 托管包装：让所有权规则落到 IDisposable 上
// ============================================================================

/// <summary>把 <see cref="Result"/> 变成异常 —— C 里靠返回码，C# 里靠异常更自然。</summary>
public sealed class AmikiException : Exception
{
    public Result Code { get; }

    public AmikiException(Result code, string call)
        : base($"{call} 失败：{ResultString(code)}（{code}）")
    {
        Code = code;
    }

    /// <summary>把错误码转成人类可读的说明（静态字符串，来自引擎）。</summary>
    public static string ResultString(Result r)
    {
        var p = Native.amiki_result_string(r);
        return p == IntPtr.Zero ? "(null)" : Marshal.PtrToStringUTF8(p) ?? "(null)";
    }

    internal static void Check(Result r, string call)
    {
        if (r != Result.Ok) throw new AmikiException(r, call);
    }
}

/// <summary>所有句柄的公共骨架：持有指针 + <c>Dispose</c> 时调对应的 <c>_destroy</c>。</summary>
public abstract class AmikiHandle : IDisposable
{
    internal IntPtr Handle;
    private bool _disposed;

    internal AmikiHandle(IntPtr handle) { Handle = handle; }

    public static uint Version => Native.amiki_version();

    public static string VersionString =>
        Marshal.PtrToStringUTF8(Native.amiki_version_string()) ?? "?";

    /// <summary>引擎在渲染中用到的原生长度（只用于校验指针未失效）。</summary>
    internal IntPtr DangerousGetHandle()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return Handle;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Destroy();
        Handle = IntPtr.Zero;
        GC.SuppressFinalize(this);
    }

    /// <summary>派生类实现：调用对应的 <c>amiki_*_destroy</c>。</summary>
    protected abstract void Destroy();

    ~AmikiHandle() => Dispose();
}

public sealed class Context : AmikiHandle
{
    private Context(IntPtr h) : base(h) { }

    public static Context Create(Backend backend = Backend.Auto)
    {
        var r = Native.amiki_context_create(backend, out var h);
        // 注意：失败时 h 不被写入（header 承诺），所以这里不能依赖它。
        if (r != Result.Ok) throw new AmikiException(r, "amiki_context_create");
        return new Context(h);
    }

    /// <summary>实际生效的后端（<see cref="Backend.Auto"/> 下可用来查看最终选了哪个）。</summary>
    public Backend ActualBackend => Native.amiki_context_backend(DangerousGetHandle());

    public string AdapterName
    {
        get
        {
            var p = Native.amiki_context_adapter_name(DangerousGetHandle());
            return p == IntPtr.Zero ? "(null)" : Marshal.PtrToStringUTF8(p) ?? "(null)";
        }
    }

    public Canvas CreateCanvas(uint width, uint height)
    {
        var r = Native.amiki_canvas_create(DangerousGetHandle(), width, height, out var h);
        AmikiException.Check(r, "amiki_canvas_create");
        return new Canvas(h);
    }

    protected override void Destroy() => Native.amiki_context_destroy(Handle);
}

public sealed class Canvas : AmikiHandle
{
    internal Canvas(IntPtr h) : base(h) { }

    public (uint W, uint H) Size
    {
        get
        {
            AmikiException.Check(Native.amiki_canvas_size(DangerousGetHandle(), out var w, out var h),
                                 "amiki_canvas_size");
            return (w, h);
        }
    }

    public void Clear() => AmikiException.Check(Native.amiki_canvas_clear(DangerousGetHandle()), "amiki_canvas_clear");

    public void SetTransform(Affine t) =>
        AmikiException.Check(Native.amiki_canvas_set_transform(DangerousGetHandle(), t), "amiki_canvas_set_transform");
    public void ResetTransform() =>
        AmikiException.Check(Native.amiki_canvas_reset_transform(DangerousGetHandle()), "amiki_canvas_reset_transform");
    public void Translate(float dx, float dy) =>
        AmikiException.Check(Native.amiki_canvas_translate(DangerousGetHandle(), dx, dy), "amiki_canvas_translate");
    public void Rotate(float radians) =>
        AmikiException.Check(Native.amiki_canvas_rotate(DangerousGetHandle(), radians), "amiki_canvas_rotate");
    public void Scale(float s) =>
        AmikiException.Check(Native.amiki_canvas_scale(DangerousGetHandle(), s), "amiki_canvas_scale");
    public void SetBlend(BlendMode b) =>
        AmikiException.Check(Native.amiki_canvas_set_blend(DangerousGetHandle(), b), "amiki_canvas_set_blend");

    public void PushClip(Path p) =>
        AmikiException.Check(Native.amiki_canvas_push_clip(DangerousGetHandle(), p.DangerousGetHandle()),
                             "amiki_canvas_push_clip");
    public void PopClip() => AmikiException.Check(Native.amiki_canvas_pop_clip(DangerousGetHandle()), "amiki_canvas_pop_clip");

    public void FillPath(Path p, Paint paint) =>
        AmikiException.Check(Native.amiki_canvas_fill_path(DangerousGetHandle(), p.DangerousGetHandle(), paint),
                             "amiki_canvas_fill_path");

    public void StrokePath(Path p, StrokeStyle style, Paint paint) =>
        AmikiException.Check(Native.amiki_canvas_stroke_path(DangerousGetHandle(), p.DangerousGetHandle(), style, paint),
                             "amiki_canvas_stroke_path");

    public void FillRect(float x0, float y0, float x1, float y1, Paint paint) =>
        AmikiException.Check(Native.amiki_canvas_draw_rect(DangerousGetHandle(), x0, y0, x1, y1, paint),
                             "amiki_canvas_draw_rect");

    public void FillCircle(float cx, float cy, float r, Paint paint) =>
        AmikiException.Check(Native.amiki_canvas_draw_circle(DangerousGetHandle(), cx, cy, r, paint),
                             "amiki_canvas_draw_circle");

    public void DrawImage(Image img, float x0, float y0, float x1, float y1) =>
        AmikiException.Check(Native.amiki_canvas_draw_image(DangerousGetHandle(), img.DangerousGetHandle(), x0, y0, x1, y1),
                             "amiki_canvas_draw_image");

    /// <summary>画一段 UTF-8 文本。<paramref name="utf8Len"/> 为 0 表示按 NUL 求长。</summary>
    public void DrawText(Font font, string text, float x, float y, float size, Paint paint)
    {
        AmikiException.Check(
            Native.amiki_canvas_draw_text(DangerousGetHandle(), font.DangerousGetHandle(), text,
                                          UIntPtr.Zero, x, y, size, paint),
            "amiki_canvas_draw_text");
    }

    public void BeginLayer(LayerBounds b, Filter f, float opacity, BlendMode blend) =>
        AmikiException.Check(Native.amiki_canvas_begin_layer(DangerousGetHandle(), b, f, opacity, blend),
                             "amiki_canvas_begin_layer");
    public void EndLayer() =>
        AmikiException.Check(Native.amiki_canvas_end_layer(DangerousGetHandle()), "amiki_canvas_end_layer");

    /// <summary>渲染为 RGBA8（非预乘、行序从上到下）。返回新数组。</summary>
    public byte[] RenderRgba8()
    {
        var (w, h) = Size;
        var buf = new byte[(long)w * h * 4];
        AmikiException.Check(
            Native.amiki_canvas_render_rgba8(DangerousGetHandle(), buf, (UIntPtr)buf.Length),
            "amiki_canvas_render_rgba8");
        return buf;
    }

    protected override void Destroy() => Native.amiki_canvas_destroy(Handle);
}

public sealed class Path : AmikiHandle
{
    private Path(IntPtr h) : base(h) { }

    public static Path Create()
    {
        AmikiException.Check(Native.amiki_path_create(out var h), "amiki_path_create");
        return new Path(h);
    }

    public Path MoveTo(float x, float y) { Check(Native.amiki_path_move_to(DangerousGetHandle(), x, y)); return this; }
    public Path LineTo(float x, float y) { Check(Native.amiki_path_line_to(DangerousGetHandle(), x, y)); return this; }
    public Path QuadTo(float cx, float cy, float x, float y) { Check(Native.amiki_path_quad_to(DangerousGetHandle(), cx, cy, x, y)); return this; }
    public Path CubicTo(float c1x, float c1y, float c2x, float c2y, float x, float y)
    {
        Check(Native.amiki_path_cubic_to(DangerousGetHandle(), c1x, c1y, c2x, c2y, x, y)); return this;
    }
    public Path Close() { Check(Native.amiki_path_close(DangerousGetHandle())); return this; }
    public Path Rect(float x0, float y0, float x1, float y1) { Check(Native.amiki_path_rect(DangerousGetHandle(), x0, y0, x1, y1)); return this; }
    public Path Circle(float cx, float cy, float r) { Check(Native.amiki_path_circle(DangerousGetHandle(), cx, cy, r)); return this; }

    private static void Check(Result r) => AmikiException.Check(r, "amiki_path_*");

    protected override void Destroy() => Native.amiki_path_destroy(Handle);
}

public sealed class Image : AmikiHandle
{
    private Image(IntPtr h) : base(h) { }

    /// <summary>从 RGBA8 像素创建。像素**被拷贝**，返回后托管数组即可释放。</summary>
    public static Image FromRgba8(byte[] pixels, uint width, uint height, uint stride = 0)
    {
        AmikiException.Check(
            Native.amiki_image_create_rgba8(pixels, width, height, (UIntPtr)stride, out var h),
            "amiki_image_create_rgba8");
        return new Image(h);
    }

    public (uint W, uint H) Size
    {
        get
        {
            AmikiException.Check(Native.amiki_image_size(DangerousGetHandle(), out var w, out var h), "amiki_image_size");
            return (w, h);
        }
    }

    protected override void Destroy() => Native.amiki_image_destroy(Handle);
}

public sealed class Font : AmikiHandle
{
    private Font(IntPtr h) : base(h) { }

    public static Font FromFile(string path)
    {
        AmikiException.Check(Native.amiki_font_create_from_file(path, out var h), "amiki_font_create_from_file");
        return new Font(h);
    }

    /// <summary>度量（每 em 的相对值，乘以字号即得像素）。</summary>
    public (float Ascent, float Descent, float LineGap) Metrics
    {
        get
        {
            AmikiException.Check(Native.amiki_font_metrics(DangerousGetHandle(), out var a, out var d, out var g),
                                 "amiki_font_metrics");
            return (a, d, g);
        }
    }

    protected override void Destroy() => Native.amiki_font_destroy(Handle);
}

public sealed class Gradient : AmikiHandle
{
    private Gradient(IntPtr h) : base(h) { }

    public static Gradient Linear(float x0, float y0, float x1, float y1, GradientStop[] stops)
    {
        AmikiException.Check(
            Native.amiki_gradient_create_linear(x0, y0, x1, y1, stops, (uint)stops.Length, out var h),
            "amiki_gradient_create_linear");
        return new Gradient(h);
    }

    public static Gradient Radial(float cx, float cy, float radius, GradientStop[] stops)
    {
        AmikiException.Check(
            Native.amiki_gradient_create_radial(cx, cy, radius, stops, (uint)stops.Length, out var h),
            "amiki_gradient_create_radial");
        return new Gradient(h);
    }

    protected override void Destroy() => Native.amiki_gradient_destroy(Handle);
}
