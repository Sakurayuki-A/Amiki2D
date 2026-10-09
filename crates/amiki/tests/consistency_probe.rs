//! 定位 CPU↔GPU 一致性偏差（**只测量**）：4K / 2000 矩形是已知的 >2/255 格。
//!
//! 输出三类信息，用来**区分**原因：
//!   1. 差值直方图 —— 是"极少数像素越界"还是"整体系统性偏移"
//!   2. 最差像素的坐标与两侧取值 —— 落在哪
//!   3. 按 256×256 分块的越界计数 —— 是**聚集在矩形边缘**（覆盖率/AA 差异）
//!      还是**散在特定条带**（分箱/平铺边界）还是**均匀分布**（精度）
//!
//! 用法：`cargo test -p amiki --test consistency_probe --release -- --ignored --nocapture`

use amiki::Canvas;
use amiki_scene::{Color, Paint, PathBuilder};

fn scene(w: u32, h: u32, n: u32) -> Canvas {
    let mut c = Canvas::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(250, 250, 252)));
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
        let t = i as f32 / n.max(1) as f32;
        c.fill_path(
            &pb.build(),
            Paint::Solid(Color { r: t, g: 1.0 - t, b: (t * 3.0) % 1.0, a: 1.0 }),
        );
    }
    c
}

#[test]
#[ignore]
fn locate_consistency_deviation() {
    let (w, h, n) = (3840u32, 2160u32, 2000u32);
    let c = scene(w, h, n);
    let cpu = amiki::render_cpu(&c, 0.05);
    let gpu = amiki::render_to_rgba(&c, 0.05).expect("gpu");

    let mut hist = [0usize; 8]; // 0 / 1 / 2 / 3 / 4 / 5-8 / 9-16 / >16
    let bucket = |d: i32| -> usize {
        match d {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5..=8 => 5,
            9..=16 => 6,
            _ => 7,
        }
    };
    let mut worst = (0i32, 0u32, 0u32, 0u8, 0u8, 0u8); // diff,x,y,cpu,gpu,ch
    let mut gt2 = 0usize;
    let tw = (w as usize).div_ceil(256);
    let mut tiles = vec![0u32; tw * (h as usize).div_ceil(256)];
    let mut on_edge_ish = 0usize; // 越界像素里，其左侧或上侧邻居差值为 0 的（近似"边缘"）
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            for ch in 0..3 {
                let d = (cpu[i + ch] as i32 - gpu[i + ch] as i32).abs();
                hist[bucket(d)] += 1;
                if d > worst.0 {
                    worst = (d, x, y, cpu[i + ch], gpu[i + ch], ch as u8);
                }
                if d > 2 {
                    gt2 += 1;
                    tiles[(y as usize / 256) * tw + (x as usize / 256)] += 1;
                    let left = if x > 0 {
                        let j = ((y * w + x - 1) * 4) as usize;
                        (cpu[j + ch] as i32 - gpu[j + ch] as i32).abs()
                    } else {
                        0
                    };
                    let up = if y > 0 {
                        let j = (((y - 1) * w + x) * 4) as usize;
                        (cpu[j + ch] as i32 - gpu[j + ch] as i32).abs()
                    } else {
                        0
                    };
                    if left == 0 || up == 0 {
                        on_edge_ish += 1;
                    }
                }
            }
        }
    }
    let total = (w as usize) * (h as usize) * 3;
    println!("config {w}x{h}/{n}rects, channels={total}");
    println!(
        "hist diff 0/1/2/3/4/5-8/9-16/>16 = {}/{}/{}/{}/{}/{}/{}/{}",
        hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]
    );
    println!(
        "gt2 = {gt2} ({:.6}% of channels), of which adjacent-to-equal (edge-ish) = {on_edge_ish}",
        100.0 * gt2 as f64 / total as f64
    );
    println!(
        "worst diff={} at ({},{}) ch={} cpu={} gpu={}",
        worst.0, worst.1, worst.2, worst.5, worst.3, worst.4
    );
    let mut top: Vec<(usize, u32)> = tiles.iter().cloned().enumerate().collect();
    top.sort_by_key(|t| std::cmp::Reverse(t.1));
    println!("hottest 256x256 tiles (tile_x,tile_y,count):");
    for (idx, cnt) in top.iter().take(6) {
        if *cnt == 0 {
            break;
        }
        println!("  ({},{}) -> {}", *idx % tw, *idx / tw, cnt);
    }
    println!("read: edge-ish share high => coverage/AA difference; uniform+tiny => precision; \
              few tiles => binning boundary");
}
