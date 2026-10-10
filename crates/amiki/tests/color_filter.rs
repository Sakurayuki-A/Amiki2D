//! 颜色矩阵滤镜（0.10.5）：**逐像素**判据。
//!
//! 判据原则：给**互不相同的输入**、断言**已知结果**。
//! 这里用纯红/纯绿/纯蓝三个矩形 —— 去色后的灰度就是 Rec.709 权重 × 255，
//! 是能算出来的确定值，不需要猜。

use amiki::Canvas;
use amiki_scene::{Color, ColorMatrix, Gradient, GradientStop, Image, Paint, PathBuilder};
use amiki_math::Vec2;

fn rect(c: &mut Canvas, x0: f32, y0: f32, x1: f32, y1: f32, paint: Paint) {
    let mut pb = PathBuilder::new();
    pb.rect(x0, y0, x1, y1);
    c.fill_path(&pb.build(), paint);
}

fn at(img: &(u32, Vec<u8>), x: u32, y: u32) -> [u8; 3] {
    let i = ((y * img.0 + x) * 4) as usize;
    [img.1[i], img.1[i + 1], img.1[i + 2]]
}

fn render(c: &Canvas, w: u32) -> (u32, Vec<u8>) {
    (w, amiki::render_cpu(c, 0.05))
}

/// 不设滤镜 ⇒ 与"从未调用过 set_color_filter"逐字节相同（零破坏承诺）。
#[test]
fn no_filter_is_byte_identical() {
    let build = |with_none: bool| {
        let mut c = Canvas::new(64, 64);
        if with_none {
            c.set_color_filter(None);
        }
        rect(&mut c, 4.0, 4.0, 30.0, 30.0, Paint::Solid(Color::rgb8(200, 60, 30)));
        rect(&mut c, 32.0, 4.0, 60.0, 30.0, Paint::Solid(Color::rgb8(30, 200, 60)));
        amiki::render_cpu(&c, 0.05)
    };
    assert_eq!(build(false), build(true), "set_color_filter(None) 必须是无操作");
    eprintln!("无滤镜：两次渲染逐字节相同");
}

/// 去色：纯红/绿/蓝三个矩形 → Rec.709 灰度（可算的确定值）。
#[test]
// Channel indexing reads clearer here than iterator adapters; same kind of local,
// commented allow as elsewhere (see docs/KNOWN_ISSUES.md D3).
#[allow(clippy::needless_range_loop)]
fn greyscale_applies_to_solids() {
    let mut c = Canvas::new(96, 32);
    c.set_color_filter(Some(ColorMatrix::greyscale()));
    rect(&mut c, 0.0, 0.0, 32.0, 32.0, Paint::Solid(Color::rgb8(255, 0, 0)));
    rect(&mut c, 32.0, 0.0, 64.0, 32.0, Paint::Solid(Color::rgb8(0, 255, 0)));
    rect(&mut c, 64.0, 0.0, 96.0, 32.0, Paint::Solid(Color::rgb8(0, 0, 255)));
    let img = render(&c, 96);
    let expect: [i32; 3] = [
        (0.2126f32 * 255.0).round() as i32,
        (0.7152f32 * 255.0).round() as i32,
        (0.0722f32 * 255.0).round() as i32,
    ];
    for (i, e) in expect.iter().enumerate() {
        let p = at(&img, (i as u32) * 32 + 16, 16);
        for ch in 0..3 {
            assert!(
                (p[ch] as i32 - e).abs() <= 2,
                "第 {i} 块通道 {ch}: 实测 {} 期望 {e}",
                p[ch]
            );
        }
    }
    eprintln!("去色纯色块：{}/{}/{}（期望 Rec.709 × 255）", expect[0], expect[1], expect[2]);
}

/// 渐变**每个色标**都要被变换：红→绿去色后是"暗灰→亮灰"。
#[test]
fn greyscale_applies_to_gradient_stops() {
    let g = Gradient::linear(
        Vec2::new(0.0, 0.0),
        Vec2::new(64.0, 0.0),
        vec![
            GradientStop::new(0.0, Color::rgb8(255, 0, 0)),
            GradientStop::new(1.0, Color::rgb8(0, 255, 0)),
        ],
    )
    .expect("两个色标");
    let mut c = Canvas::new(64, 16);
    c.set_color_filter(Some(ColorMatrix::greyscale()));
    rect(&mut c, 0.0, 0.0, 64.0, 16.0, Paint::Gradient(std::sync::Arc::new(g)));
    let img = render(&c, 64);
    let left = at(&img, 1, 8);
    let right = at(&img, 62, 8);
    assert!(left[0] == left[1] && left[1] == left[2], "左端应为灰，实测 {left:?}");
    assert!(right[0] == right[1] && right[1] == right[2], "右端应为灰，实测 {right:?}");
    assert!(
        (right[0] as i32 - left[0] as i32) > 80,
        "右端应明显更亮（红→绿去色后亮度差），实测 {left:?} / {right:?}"
    );
    eprintln!("渐变去色：左 {:?} → 右 {:?}（都是灰、且亮度递增）", left, right);
}

/// 图像逐像素变换，且**源图不被修改**（no aliasing）。
#[test]
fn image_pixels_filtered_source_untouched() {
    // 2×2：纯红 / 纯绿 / 纯蓝 / 白
    let mut data = Vec::new();
    for c in [[255u8, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]] {
        data.extend_from_slice(&c);
    }
    let src = Image::from_rgba8(2, 2, data).expect("2x2");
    let before = src.data.clone();

    let mut c = Canvas::new(64, 64);
    c.set_color_filter(Some(ColorMatrix::greyscale()));
    c.image(std::sync::Arc::new(src.clone()), 0.0, 0.0, 64.0, 64.0);
    let img = render(&c, 64);
    // 与**块顺序无关**的判据：四块都必须变成灰，且灰度集合 = 四个预期值。
    // （"第 i 块 = 第 i 个灰度"会隐含假设行序/翻转方向 —— 那是没被测过的约定。
    //   这里只断言"变了灰 + 集合正确"，把顺序留给实现。）
    let mut got: Vec<i32> = Vec::new();
    for i in 0..4u32 {
        let (x, y) = ((i % 2) * 32 + 16, (i / 2) * 32 + 16);
        let p = at(&img, x, y);
        assert!(p[0] == p[1] && p[1] == p[2], "块 {i} 应为灰，实测 {p:?}");
        got.push(p[0] as i32);
    }
    got.sort_unstable();
    let mut want = vec![18i32, 54, 182, 255];
    want.sort_unstable();
    for (g, w) in got.iter().zip(want.iter()) {
        // 容差 4：2 源像素铺到 64 设备像素 ⇒ 一个 texel 32px，**任何整数像素中心
        // 都落在 texel 中心外半像素** ⇒ 双线性采样必然掺入约 1.6% 邻 texel
        // （实测 [22,55,183,255] vs 解析值 [18,54,182,255]）。这是采样几何，不是
        // 滤镜误差 —— 纯色块那条判据（无采样）才是 ±2。
        assert!((g - w).abs() <= 4, "灰度值不符：实测 {got:?} 期望 {want:?}");
    }
    assert_eq!(src.data, before, "源图必须原样不变");
    eprintln!("图像去色：四块灰度集合 {got:?}（期望 {want:?}；源图未被修改）");
}

/// 两个不同矩阵给出不同结果；saturate(0) 与 greyscale 等价（交叉验证）。
#[test]
fn saturate_zero_matches_greyscale_on_scene() {
    let build = |m: ColorMatrix| {
        let mut c = Canvas::new(48, 48);
        c.set_color_filter(Some(m));
        rect(&mut c, 0.0, 0.0, 48.0, 48.0, Paint::Solid(Color::rgb8(220, 40, 90)));
        amiki::render_cpu(&c, 0.05)
    };
    assert_eq!(
        build(ColorMatrix::saturate(0.0)),
        build(ColorMatrix::greyscale()),
        "saturate(0) 应与 greyscale 渲染结果一致"
    );
    assert_ne!(
        build(ColorMatrix::IDENTITY),
        build(ColorMatrix::greyscale()),
        "不同矩阵必须给出不同结果"
    );
    eprintln!("saturate(0) 与 greyscale 渲染一致；与恒等不同");
}
