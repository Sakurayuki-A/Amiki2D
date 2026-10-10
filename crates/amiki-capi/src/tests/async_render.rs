//! 异步渲染三入口（0.10.9）。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

/// 判据 1（CPU 后端）：三入口能走通，且与**同步入口逐字节相同**。
#[test]
fn async_trio_matches_sync_on_the_cpu_backend() {
    let (ctx, _) = cpu_context();
    let (w, h) = (64u32, 48u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_sample(cv, w as f32, h as f32);
    let sync = render(cv, w, h);
    assert!(sync.iter().any(|&b| b != 0), "有牙的前提：画面不是空的");

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut ready = 0i32;
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
    assert_eq!(ready, 1, "CPU 分支应首帧即就绪（与门面的异步入口同一语义）");

    let mut out = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert_eq!(out, sync, "异步与同步必须逐字节相同");

    // 取走之后就没有在飞的帧了：三条入口按"没 begin"处理
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    // 重新 begin 是允许的（旧帧已被取走，这里验证"可以反复提交"）
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut again = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, again.as_mut_ptr(), again.len()), AMIKI_OK);
    assert_eq!(again, sync, "第二次提交的产物必须与第一次相同");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 没有 `begin` 就 `poll` / `try_read`：**显式报错**（与 `restore` 多于 `save` 同类），
/// 且失败时不写出参、不写一个字节。
#[test]
fn poll_and_try_read_without_begin_are_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let mut ready = 0x1234_5678i32;
    assert_eq!(
        amiki_canvas_poll_render_rgba8(cv, &mut ready),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "没提交过就没有'算完没有'可问"
    );
    assert_eq!(ready, 0x1234_5678, "失败时不写出参");

    let mut out = vec![0xAAu8; 16 * 16 * 4];
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(out.iter().all(|&b| b == 0xAA), "失败时不应写入任何字节");

    // 空指针 / 空出参
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, ptr::null_mut(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(amiki_canvas_begin_render_rgba8(ptr::null_mut()), AMIKI_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        amiki_canvas_poll_render_rgba8(ptr::null_mut(), &mut ready),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(ptr::null_mut(), out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// `out_len` 不足 ⇒ 报错、不写字节、**且不消费在飞的那一帧**
/// （长度不够是调用方的 bug，不该顺手毁掉它），补一个够长的缓冲仍能取到同一张图。
#[test]
fn try_read_rejects_short_buffer_and_keeps_the_frame() {
    let (ctx, _) = cpu_context();
    let (w, h) = (32u32, 24u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_sample(cv, w as f32, h as f32);
    let sync = render(cv, w, h);

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut short = vec![0xAAu8; sync.len()];
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, short.as_mut_ptr(), short.len() - 1),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(short.iter().all(|&b| b == 0xAA), "失败时不应写入任何字节");

    // 帧还在：补一个够长的缓冲就能取到
    let mut out = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert_eq!(out, sync, "短缓冲失败后这一帧不应被毁掉");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 打开中的层没配平就 `begin`：与同步入口同一条理由 —— 直接渲染会静默丢掉那些绘制。
#[test]
fn begin_with_an_open_layer_is_rejected() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let b = AmikiLayerBounds { x: 0, y: 0, width: 8, height: 8 };
    assert_eq!(
        amiki_canvas_begin_layer(cv, b, amiki_filter_blur(2.0), 1.0, AMIKI_BLEND_NORMAL),
        AMIKI_OK
    );
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_ERROR_INVALID_ARGUMENT);
    // 被拒绝 ⇒ 没有在飞的帧，poll 也应当报"没提交过"
    let mut ready = 0i32;
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_ERROR_INVALID_ARGUMENT);

    // 配平之后就能提交了
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// `amiki_canvas_clear` 必须把在飞的那一帧一起丢掉：
/// 它算的是**清空之前**的内容，留着只会让调用方取到过期像素。
#[test]
fn clear_drops_the_in_flight_frame() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);
    draw_sample(cv, 16.0, 16.0);

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_clear(cv), AMIKI_OK);

    let mut ready = 0i32;
    assert_eq!(
        amiki_canvas_poll_render_rgba8(cv, &mut ready),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "clear 之后不该还有帧可取（那会是清空前的过期像素）"
    );
    let mut out = vec![0xAAu8; 16 * 16 * 4];
    assert_eq!(
        amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()),
        AMIKI_ERROR_INVALID_ARGUMENT
    );
    assert!(out.iter().all(|&b| b == 0xAA));

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 判据 2（GPU 后端）：异步与同步**逐字节相同**；
/// 并且"还没算完"时 `try_read` 必须报 `AMIKI_ERROR_BUSY`、不写一个字节、
/// **也不消费那一帧**（后面轮询到就绪仍能取到）。
#[test]
fn async_matches_sync_on_the_gpu_backend() {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    assert_eq!(amiki_context_create(AMIKI_BACKEND_AUTO, &mut ctx), AMIKI_OK);
    if amiki_context_backend(ctx) != AMIKI_BACKEND_GPU {
        eprintln!("无可用 GPU，跳过（这条只在 GPU 后端上有意义）");
        amiki_context_destroy(ctx);
        return;
    }

    let (w, h) = (96u32, 64u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);
    draw_sample(cv, w as f32, h as f32);
    let sync = render(cv, w, h);

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut out = vec![0xAAu8; sync.len()];
    let first = amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len());
    if first == AMIKI_OK {
        // 回调已经跑完了（合法，只是少见）
        assert_eq!(out, sync, "首帧即就绪时也要逐字节相同");
    } else {
        assert_eq!(first, AMIKI_ERROR_BUSY, "未就绪只能是 BUSY（不是别的错误）");
        assert!(out.iter().all(|&b| b == 0xAA), "BUSY 时不应写入任何字节");

        // 轮询到就绪；**刚刚 BUSY 过的那一帧还在**
        let mut ready = 0i32;
        let mut polls = 0u32;
        loop {
            assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
            if ready != 0 {
                break;
            }
            polls += 1;
            assert!(polls < 1_000_000, "轮询次数异常：map 回调一直没触发");
        }
        let mut out2 = vec![0u8; sync.len()];
        assert_eq!(
            amiki_canvas_try_read_render_rgba8(cv, out2.as_mut_ptr(), out2.len()),
            AMIKI_OK
        );
        assert_eq!(out2, sync, "GPU 上异步与同步必须逐字节相同（轮询 {polls} 次）");
    }

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// **0.10.9 的回归钉子（C ABI 侧）**：`draws × tile 行` 够大时，稀疏路径的 1D 派发
/// 会拆成二维 —— 宿主与内核的行距不一致会让**尾部 strip 永远不被处理**，
/// 于是画面最下面若干行整片透明（`0.10.8` 记为"GPU 在 `y ≥ 2048` 会算错"，
/// 当时的处置是"超范围回落 CPU"；真因查明后那条限制已删除）。
///
/// 判据：GPU 上下文与纯 CPU 上下文渲染**同一场景**，最大通道差 ≤ 2，
/// 且**最下面一行不是透明的**（旧 bug 时那里什么都没有）。
#[test]
fn tall_canvas_with_many_draws_is_correct_on_the_gpu_backend() {
    let mut ctx: *mut AmikiContext = ptr::null_mut();
    assert_eq!(amiki_context_create(AMIKI_BACKEND_AUTO, &mut ctx), AMIKI_OK);
    if amiki_context_backend(ctx) != AMIKI_BACKEND_GPU {
        eprintln!("无可用 GPU，跳过");
        amiki_context_destroy(ctx);
        return;
    }

    // 256×2160（135 个 tile 行）× 2000 个矩形 ⇒ draws × tile 行 = 270 135 > 262 140
    // ⇒ carry 需要 4 322 160 个元素（> 65535 个工作组）⇒ 必然触发二维分批。
    let (w, h, n) = (256u32, 2160u32, 2000u32);
    let side = (n as f32).sqrt().ceil() as u32;

    let mut cv_gpu: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv_gpu), AMIKI_OK);
    let (cctx, _) = cpu_context();
    let mut cv_cpu: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(cctx, w, h, &mut cv_cpu), AMIKI_OK);

    // 同一个场景画到两块画布上（背景 + 45×45 的彩色矩形网格）
    for cv in [cv_gpu, cv_cpu] {
        assert_eq!(
            amiki_canvas_draw_rect(
                cv,
                0.0,
                0.0,
                w as f32,
                h as f32,
                amiki_paint_solid(amiki_color_rgb8(250, 250, 252))
            ),
            AMIKI_OK
        );
        for i in 0..n {
            let col = i % side;
            let row = i / side;
            let cw = w as f32 / side as f32;
            let ch = h as f32 / side as f32;
            let (x0, y0) = (col as f32 * cw + cw * 0.15, row as f32 * ch + ch * 0.15);
            let t = i as f32 / n as f32;
            assert_eq!(
                amiki_canvas_draw_rect(
                    cv,
                    x0,
                    y0,
                    x0 + cw * 0.7,
                    y0 + ch * 0.7,
                    amiki_paint_solid(amiki_color_rgba(t, 1.0 - t, (t * 3.0) % 1.0, 1.0))
                ),
                AMIKI_OK
            );
        }
    }

    let gpu_px = render(cv_gpu, w, h);
    let cpu_px = render(cv_cpu, w, h);
    let max_diff = gpu_px
        .iter()
        .zip(cpu_px.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 2,
        "GPU 与 CPU 最大通道差 {max_diff} 超出容差 —— 尾部 tile 可能又被丢掉了"
    );

    // 最下面一行必须真的画了东西（旧 bug 时这里是全透明）
    let bottom = (((h - 1) * w + w / 2) as usize) * 4;
    assert!(gpu_px[bottom + 3] > 0, "最下面一行是透明的：尾部 strip 又被丢掉了");
    eprintln!("C ABI 4K 尾部回归：{w}x{h}/{n} 矩形，最大通道差 {max_diff}");

    amiki_canvas_destroy(cv_gpu);
    amiki_canvas_destroy(cv_cpu);
    amiki_context_destroy(ctx);
    amiki_context_destroy(cctx);
}

/// 同一张 HDR 画布的异步读法（`render_hot_square` 的异步版）。
fn hot_square_canvas(tone: i32) -> (*mut AmikiContext, *mut AmikiCanvas) {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 64, 64, &mut cv), AMIKI_OK);
    assert_eq!(amiki_canvas_set_tone_map(cv, tone), AMIKI_OK);
    let hot = amiki_color_rgba(1.8, 1.8, 1.8, 1.0);
    assert_eq!(
        amiki_canvas_draw_rect(cv, 8.0, 8.0, 56.0, 56.0, amiki_paint_solid(hot)),
        AMIKI_OK
    );
    (ctx, cv)
}

/// 走完异步三入口（轮询到就绪再取），返回像素。
fn read_async(cv: *mut AmikiCanvas, w: u32, h: u32) -> Vec<u8> {
    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut ready = 0i32;
    let mut polls = 0u32;
    loop {
        assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
        if ready != 0 {
            break;
        }
        polls += 1;
        assert!(polls < 1_000_000, "map 回调一直没触发");
    }
    let mut out = vec![0u8; (w * h * 4) as usize];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    out
}

/// **色调映射、mip bias 必须一起接进异步入口**：否则两条入口在 HDR 画布上就会不同，
/// 而"逐字节相同"是这条 API 唯一的卖点。
#[test]
fn async_trio_honours_the_tone_map() {
    let sync_raw = render_hot_square(AMIKI_TONEMAP_RAW);
    let sync_aces = render_hot_square(AMIKI_TONEMAP_ACES);

    let (ctx, cv) = hot_square_canvas(AMIKI_TONEMAP_ACES);
    let async_aces = read_async(cv, 64, 64);
    assert_eq!(async_aces, sync_aces, "异步入口必须与同步入口的 ACES 输出逐字节相同");
    assert_ne!(async_aces, sync_raw, "有牙的前提：ACES 与 RAW 本来就该不同");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}

/// 层场景在 `begin` 里就同步算完了（层合成需要根图像，异步没有收益）⇒
/// **第一次 `poll` 即就绪**，且产物与同步入口逐字节相同。
#[test]
fn async_layer_scene_is_ready_on_the_first_poll() {
    let (ctx, _) = cpu_context();
    let (w, h) = (64u32, 48u32);
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut cv), AMIKI_OK);

    let b = AmikiLayerBounds { x: 0, y: 0, width: w, height: h };
    assert_eq!(
        amiki_canvas_begin_layer(cv, b, amiki_filter_blur(3.0), 0.8, AMIKI_BLEND_NORMAL),
        AMIKI_OK
    );
    draw_sample(cv, w as f32, h as f32);
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    let sync = render(cv, w, h);
    assert!(sync.iter().any(|&b| b != 0), "有牙的前提：画面不是空的");

    assert_eq!(amiki_canvas_begin_render_rgba8(cv), AMIKI_OK);
    let mut ready = 0i32;
    assert_eq!(amiki_canvas_poll_render_rgba8(cv, &mut ready), AMIKI_OK);
    assert_eq!(ready, 1, "层场景没有在飞可言，第一次 poll 就必须就绪");

    let mut out = vec![0u8; sync.len()];
    assert_eq!(amiki_canvas_try_read_render_rgba8(cv, out.as_mut_ptr(), out.len()), AMIKI_OK);
    assert_eq!(out, sync, "层场景的异步产物也要与同步逐字节相同");

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
}
