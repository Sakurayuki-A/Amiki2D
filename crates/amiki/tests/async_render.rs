//! P3 第 ③ 步：**异步入口与同步入口必须逐字节相同**（判据 1、2）。
//!
//! 判据（先写死再实现）：
//! 1. `begin_render_to_rgba` + `poll` + `try_read` 的结果 == `render_to_rgba` ✓✓
//! 2. CPU 分支（超范围 / 无 GPU）**第一次 poll 即 Ready** ✓（确定性，不做计时断言 ✗）

use amiki::Canvas;
use amiki_scene::{Color, Paint, PathBuilder};

fn scene() -> Canvas {
    let mut c = Canvas::new(200, 120);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, 200.0, 120.0);
    c.fill_path(&pb.build(), Paint::Solid(Color::rgb8(250, 250, 252)));
    for i in 0..20u32 {
        let x = (i % 5) as f32 * 38.0 + 6.0;
        let y = (i / 5) as f32 * 28.0 + 4.0;
        let mut pb = PathBuilder::new();
        pb.rect(x, y, x + 28.0, y + 18.0);
        c.fill_path(
            &pb.build(),
            Paint::Solid(Color::rgb8(30 + i as u8 * 7, 90, 180 - i as u8 * 5)),
        );
    }
    c
}

/// 判据 1 + 2 一起测：异步结果与同步逐字节相同，且轮询能走到 Ready。
#[test]
fn async_matches_sync_and_reaches_ready() {
    let c = scene();
    let sync = amiki::render_to_rgba(&c, 0.05).expect("同步渲染（GPU 或 CPU 回落）");

    let mut pending = amiki::begin_render_to_rgba(&c, 0.05).expect("begin");
    let mut polls = 0u32;
    while pending.poll() == amiki::gpu::PollStatus::Pending && polls < 200_000 {
        polls += 1;
    }
    let got = pending.try_read().expect("到达 Ready 后应能取值").expect("无错误");

    assert_eq!(got.len(), sync.len(), "长度必须一致");
    assert_eq!(got, sync, "异步与同步必须逐字节相同");
    eprintln!("async == sync（{} 字节），轮询 {polls} 次", got.len());
}

/// 判据 2：**超出 GPU 已验证范围**时走 CPU 分支，第一次 poll 就必须是 Ready。
#[test]
fn cpu_branch_is_ready_on_first_poll() {
    // 2049 > GPU_MAX_SUPPORTED_SIDE ⇒ 明确回落 CPU（与同步路径同一判断）
    let c = Canvas::new(8, 2049);
    let mut pending = amiki::begin_render_to_rgba(&c, 0.05).expect("begin");
    assert_eq!(
        pending.poll(),
        amiki::gpu::PollStatus::Ready,
        "CPU 分支应首帧即就绪"
    );
    let got = pending.try_read().expect("应能立即取值").expect("无错误");
    assert_eq!(got.len(), 8 * 2049 * 4);
    eprintln!("CPU 分支：首帧即 Ready，{} 字节", got.len());
}

/// 判据 3（0.10.9）：`take_ready` **取不到不算数** —— 未就绪时返回 `None`
/// 且那一帧留在原地，轮询到就绪后仍能取到同一张图；取走之后再取得到 `None`
/// （**不能是一张零字节的"合法空图"**）。
///
/// C ABI 的三入口就建立在这条语义上（见 `amiki_canvas_try_read_render_rgba8`）。
#[test]
fn take_ready_does_not_consume_a_frame_that_is_not_ready() {
    let c = scene();
    let sync = amiki::render_to_rgba(&c, 0.05).expect("同步渲染");
    let mut pending = amiki::begin_render_to_rgba(&c, 0.05).expect("begin");

    let mut misses = 0u32;
    let got = loop {
        if let Some(r) = pending.take_ready() {
            break r.expect("无错误");
        }
        misses += 1;
        assert!(misses < 200_000, "map 回调一直没触发");
        let _ = pending.poll();
    };
    assert_eq!(got, sync, "取不到不算数的路径也必须给出同一张图");
    assert!(pending.take_ready().is_none(), "取走之后应当得到 None");
    eprintln!("take_ready：未就绪 {misses} 次后取到，与同步逐字节相同");
}
