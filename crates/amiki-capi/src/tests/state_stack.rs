//! 状态栈 save / restore。
//!
//! 从 `tests.rs` 分出（只搬代码，未改逻辑）。

use std::ptr;
use super::*;

/// 往返：save → 改状态 → restore ⇒ 画出来的东西与"从未改过"完全相同。
///
/// 用**渲染结果**比对，而不是只看 getter：这直接证明"绘制时用的确实是那套状态"。
#[test]
fn canvas_save_restore_roundtrip() {
    let (ctx, _) = cpu_context();
    let (w, h) = (32u32, 32u32);
    let ink = amiki_paint_solid(amiki_color_rgb8(0, 0, 0));

    let mut p: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut p), AMIKI_OK);
    assert_eq!(amiki_path_rect(p, 4.3, 4.1, 28.7, 28.9), AMIKI_OK);

    // A：只在初始状态下画
    let mut a: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut a), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(a, p, ink), AMIKI_OK);
    let pa = render(a, w, h);

    // B：save → 改变换/混合/填充规则 → 画一条（应不同）→ restore → 再画
    let mut b: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut b), AMIKI_OK);
    assert_eq!(amiki_canvas_save(b), AMIKI_OK);
    assert_eq!(amiki_canvas_translate(b, 7.0, 5.0), AMIKI_OK);
    assert_eq!(amiki_canvas_set_blend(b, 1), AMIKI_OK); // Multiply
    assert_eq!(amiki_canvas_set_fill_rule(b, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(b, p, ink), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(b), AMIKI_OK);
    assert_eq!(amiki_canvas_get_fill_rule(b), AMIKI_FILL_NONZERO, "恢复后应回到 NonZero");
    assert_eq!(amiki_canvas_fill_path(b, p, ink), AMIKI_OK);
    let pb_img = render(b, w, h);
    assert_ne!(pa, pb_img, "改过状态的那条必须真的画得不一样（否则测试没牙）");

    // 只做 save/改状态/restore，然后画 —— 应与 A 逐字节一致
    let mut c: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, w, h, &mut c), AMIKI_OK);
    assert_eq!(amiki_canvas_save(c), AMIKI_OK);
    assert_eq!(amiki_canvas_translate(c, 7.0, 5.0), AMIKI_OK);
    assert_eq!(amiki_canvas_set_blend(c, 1), AMIKI_OK);
    assert_eq!(amiki_canvas_set_fill_rule(c, AMIKI_FILL_EVENODD), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(c), AMIKI_OK);
    assert_eq!(amiki_canvas_fill_path(c, p, ink), AMIKI_OK);
    let pc = render(c, w, h);
    assert_eq!(pa, pc, "restore 之后的状态必须与从未改过时逐字节一致");

    amiki_path_destroy(p);
    amiki_canvas_destroy(a);
    amiki_canvas_destroy(b);
    amiki_canvas_destroy(c);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 往返：恢复后与初始状态逐字节一致");
}

/// 溢出与下溢：32 层成功、第 33 层 `STACK_OVERFLOW`；清空后多 restore `STACK_UNDERFLOW`。
#[test]
fn canvas_save_overflows_and_restore_underflows() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    for i in 0..32 {
        assert_eq!(amiki_canvas_save(cv), AMIKI_OK, "第 {} 次 save 应成功", i + 1);
    }
    assert_eq!(
        amiki_canvas_save(cv),
        AMIKI_ERROR_STACK_OVERFLOW,
        "第 33 次必须显式溢出"
    );
    for i in 0..32 {
        assert_eq!(amiki_canvas_restore(cv), AMIKI_OK, "第 {} 次 restore 应成功", i + 1);
    }
    assert_eq!(
        amiki_canvas_restore(cv),
        AMIKI_ERROR_STACK_UNDERFLOW,
        "空栈 restore 必须显式下溢"
    );
    assert_eq!(
        unsafe { std::ffi::CStr::from_ptr(amiki_result_string(AMIKI_ERROR_STACK_OVERFLOW)) }
            .to_str()
            .unwrap(),
        "state stack overflow (missing restore)"
    );

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 溢出/下溢：32 层上限与两个错误码都对");
}

/// `restore` 会回退裁剪深度 ⇒ C 侧那本账必须同步，否则 `pop_clip` 会误判。
#[test]
fn canvas_restore_syncs_clip_depth() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    let mut p: *mut AmikiPath = ptr::null_mut();
    assert_eq!(amiki_path_create(&mut p), AMIKI_OK);
    assert_eq!(amiki_path_rect(p, 0.0, 0.0, 10.0, 10.0), AMIKI_OK);

    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_push_clip(cv, p), AMIKI_OK);
    assert_eq!(amiki_canvas_push_clip(cv, p), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK);
    assert_eq!(
        amiki_canvas_pop_clip(cv),
        AMIKI_ERROR_INVALID_ARGUMENT,
        "restore 已把裁剪深度退回 0，此时 pop 必须报错（说明 C 侧的账同步了）"
    );

    assert_eq!(amiki_canvas_push_clip(cv, p), AMIKI_OK);
    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK);
    assert_eq!(
        amiki_canvas_pop_clip(cv),
        AMIKI_OK,
        "这一层是 save 之前压的，restore 不该把它算掉"
    );

    amiki_path_destroy(p);
    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 裁剪同步：restore 回退深度后 pop 的判定正确");
}

/// 层内的栈独立（C ABI 侧）。
#[test]
fn canvas_layer_has_independent_stack() {
    let (ctx, _) = cpu_context();
    let mut cv: *mut AmikiCanvas = ptr::null_mut();
    assert_eq!(amiki_canvas_create(ctx, 16, 16, &mut cv), AMIKI_OK);

    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(
        amiki_canvas_begin_layer(
            cv,
            AmikiLayerBounds { x: 0, y: 0, width: 16, height: 16 },
            AmikiFilter::default(),
            1.0,
            0,
        ),
        AMIKI_OK
    );
    assert_eq!(
        amiki_canvas_restore(cv),
        AMIKI_ERROR_STACK_UNDERFLOW,
        "层内栈不继承外层栈"
    );
    assert_eq!(amiki_canvas_save(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_end_layer(cv), AMIKI_OK);
    assert_eq!(amiki_canvas_restore(cv), AMIKI_OK, "外层那层还在");
    assert_eq!(amiki_canvas_restore(cv), AMIKI_ERROR_STACK_UNDERFLOW);

    amiki_canvas_destroy(cv);
    amiki_context_destroy(ctx);
    eprintln!("C ABI 层交互：层内栈从空开始，外层计数不受影响");
}

// ============================================================================
// 字体栈 / 字体回退（0.7.0）
// ============================================================================

