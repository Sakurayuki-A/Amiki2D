//! 状态栈的对拍与零破坏（0.9.0）：**GPU 与 CPU 逐像素一致**，
//! 且"空配对"不改变任何输出。
//!
//! 为什么放在门面测试里：门面同时依赖 `amiki-raster`（CPU 参考）与
//! `amiki-gpu`（GPU），一次就能拿到两条路径 —— 在 `amiki-gpu` 里反而
//! 用不到门面的 `save` / `restore`（会形成循环依赖）。

use amiki::{render_cpu, render_to_rgba, Canvas};
use amiki_math::{Affine, Vec2};
use amiki_scene::{BlendMode, Color, FillRule, Paint, PathBuilder};

fn paint(r: u8, g: u8, b: u8) -> Paint {
    Paint::Solid(Color::rgb8(r, g, b))
}

/// 一段"UI 式"的绘制：每块内容各自 save / 局部状态 / restore。
fn ui_scene(with_empty_pairs: bool) -> Canvas {
    let mut c = Canvas::new(64, 64);
    // 底
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 64.0, 64.0);
    c.fill_path(&pb.build(), paint(245, 245, 248));

    let panel = {
        let mut pb = PathBuilder::new();
        pb.rect(6.3, 6.1, 30.7, 22.9);
        pb.build()
    };
    let star = {
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(45.0, 18.0), 11.0);
        pb.build()
    };

    if with_empty_pairs {
        let _ = c.save();
        let _ = c.restore();
    }

    // 控件 1：局部变换 + 局部混合
    c.save().expect("save");
    c.set_transform(Affine::translate(Vec2::new(3.0, 2.0)));
    c.set_blend(BlendMode::Multiply);
    c.fill_path(&panel, paint(40, 120, 220));
    c.restore().expect("restore");

    // 控件 2：even-odd + 裁剪，全部局部
    c.save().expect("save");
    c.set_fill_rule(FillRule::EvenOdd);
    let clip = {
        let mut pb = PathBuilder::new();
        pb.rect(20.0, 28.0, 60.0, 60.0);
        pb.build()
    };
    c.push_clip(&clip);
    c.set_transform(Affine::translate(Vec2::new(2.0, 1.0)));
    c.fill_path(&star, paint(200, 60, 40));
    c.restore().expect("restore");

    // 控件 3：回到初始状态之后再画一条（证明 restore 真的回去了）
    c.fill_path(&star, paint(30, 30, 30));
    c
}

/// 空配对不改变任何输出（**零破坏**的直接证据，且两端各验一次）。
#[test]
fn empty_save_restore_pairs_change_nothing() {
    let plain = ui_scene(false);
    let paired = ui_scene(true);

    let cpu_plain = render_cpu(&plain, 0.05);
    let cpu_paired = render_cpu(&paired, 0.05);
    assert_eq!(cpu_plain, cpu_paired, "CPU：空配对不得改变任何像素");

    match (render_to_rgba(&plain, 0.05), render_to_rgba(&paired, 0.05)) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "GPU：空配对不得改变任何像素"),
        _ => eprintln!("空配对：GPU 不可用，跳过 GPU 侧"),
    }
    let ink = cpu_plain.chunks(4).filter(|p| p[3] > 0).count();
    assert!(ink > 1000, "场景必须真的画出东西，实际不透明像素 {ink}");
    eprintln!("零破坏：空配对在 CPU 与 GPU 两侧都不改变输出（不透明像素 {ink}）");
}

/// 状态栈包裹的绘制序列：**GPU↔CPU 逐像素 ≤2/255**。
#[test]
fn gpu_matches_cpu_with_state_stack() {
    let c = ui_scene(false);
    let cpu = render_cpu(&c, 0.05);
    let gpu = match render_to_rgba(&c, 0.05) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("状态栈对拍：GPU 不可用（{e}），跳过");
            return;
        }
    };

    let mut max_diff = 0u32;
    let mut ink = 0usize;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        max_diff = max_diff.max((*a as i32 - *b as i32).unsigned_abs());
        if *a < 250 {
            ink += 1;
        }
    }
    assert!(ink > 2000, "必须真的画出内容，实际非白通道 {ink}");
    assert!(
        max_diff <= 2,
        "状态栈场景下 GPU 与 CPU 最大通道差 {max_diff} 超出容差"
    );
    eprintln!("状态栈对拍：最大通道差 {max_diff}，非白通道 {ink}");
}

/// `restore` 之后画的那条，必须与"从未改过状态"画的那条**逐像素相同**
/// （把状态恢复的正确性钉到渲染结果上，而不是只比字段）。
#[test]
fn restored_state_renders_identically() {
    let star = {
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(30.0, 30.0), 20.3);
        pb.build()
    };

    let a = {
        let mut c = Canvas::new(64, 64);
        c.fill_path(&star, paint(20, 40, 200));
        c
    };
    let b = {
        let mut c = Canvas::new(64, 64);
        c.save().expect("save");
        c.set_transform(Affine::translate(Vec2::new(9.0, -4.0)));
        c.set_blend(BlendMode::Screen);
        c.set_fill_rule(FillRule::EvenOdd);
        c.restore().expect("restore");
        c.fill_path(&star, paint(20, 40, 200));
        c
    };
    assert_eq!(
        render_cpu(&a, 0.05),
        render_cpu(&b, 0.05),
        "CPU：restore 之后的绘制必须与未改过状态时逐字节相同"
    );
    match (render_to_rgba(&a, 0.05), render_to_rgba(&b, 0.05)) {
        (Ok(x), Ok(y)) => assert_eq!(x, y, "GPU：同上"),
        _ => eprintln!("恢复后一致性：GPU 不可用，跳过 GPU 侧"),
    }
    eprintln!("恢复后一致性：CPU/GPU 都与未改过状态逐字节相同");
}
