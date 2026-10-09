//! CPU vs GPU（**含回读**）交叉点测量 —— 只测量，不做判断（交接文档第 7/11 条）。
//!
//! 为什么先测这个：P3（异步渲染）是五项目标里**唯一要动 GPU 后端**的，而它的
//! 必要性取决于一个我们从未测过的数 —— **GPU 路径（渲染 + staging 拷贝 + map 回读）
//! 到底在什么规模下才比 CPU 快**。没有这个数就做 P3，等于赌。
//!
//! 用法（输出为 ASCII，避免控制台编码干扰）：
//!   cargo test -p amiki --test backend_crossover --release -- --nocapture
//!
//! 判据：**没有断言**。耗时数据在这里只被打印；严格判据交给独立基准。

use amiki::Canvas;
use amiki_math::Vec2;
use amiki_scene::{Color, Paint, PathBuilder};

const SIZES: [(u32, u32, &str); 4] = [
    (320, 180, "320x180"),
    (1280, 720, "1280x720"),
    (1920, 1080, "1920x1080"),
    (3840, 2160, "3840x2160"),
];

const CONTENTS: [(u32, &str); 3] = [(20, "20rects"), (200, "200rects"), (2000, "2000rects")];

fn scene(w: u32, h: u32, n: u32) -> Canvas {
    let mut c = Canvas::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(250, 250, 252)));
    // 伪随机但确定的网格布局（每帧同一张图，公平比较）
    let side = (n as f32).sqrt().ceil() as u32;
    for i in 0..n {
        let col = i % side;
        let row = i / side;
        let cw = w as f32 / side as f32;
        let ch = h as f32 / side as f32;
        let x0 = col as f32 * cw + cw * 0.15;
        let y0 = row as f32 * ch + ch * 0.15;
        let mut pb = PathBuilder::new();
        pb.rect(x0, y0, x0 + cw * 0.7, y0 + ch * 0.7);
        // 每块颜色不同：颜色滤镜/填充都不会被优化掉
        let t = i as f32 / n.max(1) as f32;
        c.fill_path(
            &pb.build(),
            Paint::Solid(Color {
                r: t,
                g: 1.0 - t,
                b: (t * 3.0) % 1.0,
                a: 1.0,
            }),
        );
    }
    c
}

fn best_of<T>(n: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..n {
        let t = std::time::Instant::now();
        let v = f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best {
            best = ms;
        }
        out = Some(v);
    }
    (out.expect("至少跑一次"), best)
}

#[test]
// 默认忽略：这里要渲染到 4K（约 11 秒），不该进日常套件。
// 需要时：cargo test -p amiki --test backend_crossover --release -- --ignored --nocapture
#[ignore]
fn measure_cpu_vs_gpu_crossover() {
    println!("scene,cpu_ms,gpu_ms,gpu_over_cpu,bytes_equal");
    for (w, h, sname) in SIZES {
        for (n, cname) in CONTENTS {
            let c = scene(w, h, n);

            let (cpu_px, cpu_ms) = best_of(3, || amiki::render_cpu(&c, 0.05));
            let (gpu_res, gpu_ms) = best_of(3, || amiki::render_to_rgba(&c, 0.05));

            let (gpu_ms, equal) = match gpu_res {
                Ok(gpu_px) => {
                    // 两侧一致性顺带量一下（既有承诺是 ≤2/255）
                    let maxdiff = if gpu_px.len() == cpu_px.len() {
                        cpu_px
                            .iter()
                            .zip(gpu_px.iter())
                            .map(|(a, b)| (*a as i32 - *b as i32).abs())
                            .max()
                            .unwrap_or(0)
                    } else {
                        -1
                    };
                    let equal = if maxdiff == 0 {
                        "exact"
                    } else if maxdiff > 0 && maxdiff <= 2 {
                        "le2"
                    } else if maxdiff < 0 {
                        "len_diff"
                    } else {
                        "gt2"
                    };
                    (gpu_ms, equal)
                }
                Err(_) => (f64::NAN, "no_gpu"),
            };

            println!(
                "{}/{} ({:.1} Mpx),{:.2},{:.2},{:.2},{}",
                sname,
                cname,
                (w as f64 * h as f64) / 1e6,
                cpu_ms,
                gpu_ms,
                gpu_ms / cpu_ms,
                equal
            );
        }
    }
    println!("note: cpu = amiki::render_cpu, gpu = amiki::render_to_rgba (render + readback), best of 3");
}
