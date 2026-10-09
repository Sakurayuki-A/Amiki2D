//! GPU 路径固定开销探针（**只测量**，不做判断）。
//!
//! 要分开的两种解释：
//!   (a) **每次调用**都付固定代价（重建渲染器 / 设备 / 缓冲 / 管线）
//!       ⇒ 每次调用都慢、且与尺寸内容无关
//!   (b) 只有**首次**付冷启动代价（设备与管线初始化），之后应显著更快
//!       ⇒ 第 1 次慢、后续快
//!
//! 用法：`cargo test -p amiki --test gpu_fixed_cost --release -- --nocapture`

use amiki::Canvas;
use amiki_scene::{Color, Paint, PathBuilder};

fn scene(w: u32, h: u32, n: u32) -> Canvas {
    let mut c = Canvas::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(250, 250, 252)));
    for i in 0..n {
        let x = (i % 20) as f32 * (w as f32 / 20.0);
        let y = (i / 20) as f32 * (h as f32 / 20.0);
        let mut pb = PathBuilder::new();
        pb.rect(x, y, x + w as f32 / 40.0, y + h as f32 / 40.0);
        c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(30, 90, 180)));
    }
    c
}

fn one(c: &Canvas) -> f64 {
    let t = std::time::Instant::now();
    let r = amiki::render_to_rgba(c, 0.05);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(r.is_ok(), "GPU 渲染失败：{:?}", r.err());
    ms
}

#[test]
fn probe_gpu_per_call_vs_cold_start() {
    // 很小的场景：如果固定开销来自"每次重建"，这里也会是大数字
    let small = scene(64, 64, 2);
    let mut t = Vec::new();
    for _ in 0..6 {
        t.push(one(&small));
    }
    println!("small 64x64/2rects: first={:.1} rest={:?} (ms)", t[0],
        t[1..].iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>());

    // 中等场景：同尺寸重复，看是否稳定
    let mid = scene(640, 360, 40);
    let mut t2 = Vec::new();
    for _ in 0..4 {
        t2.push(one(&mid));
    }
    println!("mid 640x360/40rects: first={:.1} rest={:?} (ms)", t2[0],
        t2[1..].iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>());

    // CPU 对照（同场景同次数）
    let t3 = std::time::Instant::now();
    for _ in 0..6 {
        let _ = amiki::render_cpu(&small, 0.05);
    }
    println!("cpu 64x64/2rects x6 total={:.2} ms", t3.elapsed().as_secs_f64() * 1000.0);

    println!("read: if 'rest' is as large as 'first' => per-call fixed cost (a); \
              if 'rest' is small => cold start only (b)");
}
