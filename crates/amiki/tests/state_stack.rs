//! 状态栈（0.9.0）：`save` / `restore` 保存与恢复四项状态。
//!
//! # 验证方式：比 `Draw`，而不是比 getter
//!
//! 每次绘制都会把**当时的完整状态**烙进 `Draw`（变换 / 混合模式 / 裁剪引用 /
//! 填充规则）。所以"状态恢复得对不对"最直接的判据是：
//! **`save`/`restore` 包裹后画出的 Draw，与不做任何状态操作直接画出的 Draw
//! 逐字段相等**。这比读 getter 强 —— getter 只能证明字段被写回，
//! 而 Draw 相等证明的是"绘制时用的确实是那套状态"。

use amiki::Canvas;
use amiki_math::{Affine, Vec2};
use amiki_scene::{BlendMode, Color, Draw, FillRule, Paint, Path, PathBuilder};

fn paint() -> Paint {
    Paint::Solid(Color::rgb8(10, 20, 30))
}

fn rect_path(x: f32, y: f32, w: f32, h: f32) -> Path {
    let mut pb = PathBuilder::new();
    pb.rect(x, y, x + w, y + h);
    pb.build()
}

/// 跑一段绘制脚本，取出它记录的 Draw 列表。
fn draws(build: impl FnOnce(&mut Canvas)) -> Vec<Draw> {
    let mut c = Canvas::new(32, 32);
    build(&mut c);
    c.scene.draws.clone()
}

// ============================================================ 四项状态各自往返

/// **变换**：save → 改变换 → restore ⇒ 之后的 Draw 变换回到保存点。
#[test]
fn restore_brings_back_transform() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let plain = draws(|c| {
        c.fill_path(&p, paint());
    });
    let wrapped = draws(|c| {
        c.save().expect("save");
        c.set_transform(Affine::translate(Vec2::new(11.0, 7.0)));
        c.fill_path(&p, paint());
        c.restore().expect("restore");
        c.fill_path(&p, paint());
    });

    assert_eq!(wrapped.len(), 2);
    assert_ne!(
        wrapped[0].transform, plain[0].transform,
        "改了变换的那条必须真的不同（否则测试没牙）"
    );
    assert_eq!(
        wrapped[1].transform, plain[0].transform,
        "restore 后变换必须回到保存点"
    );
    eprintln!(
        "变换往返：改后 tx={:.1} 恢复后 tx={:.1}",
        wrapped[0].transform.e, wrapped[1].transform.e
    );
}

/// **混合模式**：save → set_blend(Multiply) → restore ⇒ 回到 Normal。
#[test]
fn restore_brings_back_blend_mode() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let plain = draws(|c| {
        c.fill_path(&p, paint());
    });
    let wrapped = draws(|c| {
        c.save().expect("save");
        c.set_blend(BlendMode::Multiply);
        c.fill_path(&p, paint());
        c.restore().expect("restore");
        c.fill_path(&p, paint());
    });

    assert_eq!(wrapped[0].blend, BlendMode::Multiply);
    assert_eq!(wrapped[1].blend, plain[0].blend, "restore 后混合模式必须回来");
    assert_eq!(wrapped[1].blend, BlendMode::Normal);
}

/// **裁剪栈**：save → push_clip ×2 → restore ⇒ 深度回到保存点，
/// 且之后画出的 Draw **不带**那两层裁剪。
#[test]
fn restore_brings_back_clip_stack_depth() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let clip = rect_path(0.0, 0.0, 20.0, 20.0);

    let mut c = Canvas::new(32, 32);
    assert_eq!(c.clip_depth(), 0);
    c.save().expect("save");
    c.push_clip(&clip).push_clip(&clip);
    assert_eq!(c.clip_depth(), 2);
    c.fill_path(&p, paint());
    c.restore().expect("restore");

    assert_eq!(c.clip_depth(), 0, "restore 后裁剪深度必须回到保存点");
    c.fill_path(&p, paint());

    assert_eq!(c.scene.draws.len(), 2);
    assert_eq!(c.scene.draws[0].clips.len(), 2, "被包裹的那条应带两层裁剪");
    assert!(
        c.scene.draws[1].clips.is_empty(),
        "restore 之后画出的不应再带裁剪"
    );
    eprintln!(
        "裁剪往返：被包裹 clips={} / 恢复后 clips={}",
        c.scene.draws[0].clips.len(),
        c.scene.draws[1].clips.len()
    );
}

/// **填充规则**（0.8.0 验收 6 的原文）：save → EvenOdd → restore ⇒ 回到 NonZero。
#[test]
fn restore_brings_back_fill_rule() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let plain = draws(|c| {
        c.fill_path(&p, paint());
    });
    let wrapped = draws(|c| {
        c.save().expect("save");
        c.set_fill_rule(FillRule::EvenOdd);
        c.fill_path(&p, paint());
        c.restore().expect("restore");
        c.fill_path(&p, paint());
    });

    assert_eq!(wrapped[0].fill_rule, FillRule::EvenOdd);
    assert_eq!(
        wrapped[1].fill_rule,
        FillRule::NonZero,
        "restore 后填充规则必须回到 NonZero"
    );
    assert_eq!(wrapped[1], plain[0], "恢复后的整条 Draw 应与不曾改过完全相同");
}

/// **整体覆盖而非叠加**：两层 save 各自改不同字段，逐层 restore 时
/// 状态应当**逐层退回**，而不是把改动累积。
#[test]
fn restore_overwrites_instead_of_merging() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let s0 = draws(|c| {
        c.fill_path(&p, paint());
    });

    let mut c = Canvas::new(32, 32);
    c.fill_path(&p, paint()); // 0：初始
    c.save().expect("save#1");
    c.set_blend(BlendMode::Multiply);
    c.fill_path(&p, paint()); // 1：blend=Multiply
    c.save().expect("save#2");
    c.set_fill_rule(FillRule::EvenOdd);
    c.set_transform(Affine::translate(Vec2::new(3.0, 4.0)));
    c.fill_path(&p, paint()); // 2：blend=Multiply + EvenOdd + 平移
    c.restore().expect("restore#2");
    c.fill_path(&p, paint()); // 3：应退回"blend=Multiply"，EvenOdd 与平移都要消失
    c.restore().expect("restore#1");
    c.fill_path(&p, paint()); // 4：应完全回到初始

    let d = &c.scene.draws;
    assert_eq!(d.len(), 5);
    assert_eq!(d[0], s0[0], "0 应为初始");
    assert_eq!(d[1].blend, BlendMode::Multiply);
    assert_eq!(d[1].fill_rule, FillRule::NonZero);
    assert_eq!(d[1].transform, Affine::IDENTITY);
    assert_eq!(d[2].blend, BlendMode::Multiply);
    assert_eq!(d[2].fill_rule, FillRule::EvenOdd);
    assert_ne!(d[2].transform, Affine::IDENTITY);
    assert_eq!(d[3].blend, BlendMode::Multiply, "restore 后应保留更早那层设的 blend");
    assert_eq!(d[3].fill_rule, FillRule::NonZero, "EvenOdd 应被回退");
    assert_eq!(d[3].transform, Affine::IDENTITY, "平移应被回退");
    assert_eq!(d[4], s0[0], "全部 restore 后应完全回到初始状态");
    eprintln!("整体覆盖：三处状态逐层退回正确（未叠加）");
}

// ============================================================ 嵌套 / 深度

/// 三层嵌套：每层改一项，回退顺序与设置顺序相反且各自独立。
#[test]
fn three_level_nesting_is_independent() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let mut c = Canvas::new(32, 32);

    c.save().expect("L1"); // 初始
    c.set_blend(BlendMode::Screen);
    c.save().expect("L2");
    c.set_fill_rule(FillRule::EvenOdd);
    c.save().expect("L3");
    c.set_transform(Affine::translate(Vec2::new(5.0, 6.0)));
    assert_eq!(c.save_depth(), 3);

    c.fill_path(&p, paint()); // 最深：三项都生效
    c.restore().expect("L3");
    c.fill_path(&p, paint()); // 退回变换
    c.restore().expect("L2");
    c.fill_path(&p, paint()); // 退回填充规则
    c.restore().expect("L1");
    c.fill_path(&p, paint()); // 全部退回

    let d = &c.scene.draws;
    assert_eq!(d[0].transform.e, 5.0);
    assert_eq!(d[0].fill_rule, FillRule::EvenOdd);
    assert_eq!(d[0].blend, BlendMode::Screen);
    assert_eq!(d[1].transform, Affine::IDENTITY);
    assert_eq!(d[1].fill_rule, FillRule::EvenOdd, "第二层设的 EvenOdd 此时还在");
    assert_eq!(d[2].fill_rule, FillRule::NonZero, "退回第二层后 EvenOdd 应消失");
    assert_eq!(d[2].blend, BlendMode::Screen, "第一层设的 Screen 此时还在");
    assert_eq!(d[3], draws(|x| {
        x.fill_path(&p, paint());
    })[0]);
    assert_eq!(c.save_depth(), 0);
    eprintln!("三层嵌套：逐层独立、回退顺序正确");
}

/// 深度上限 32：第 32 次成功，第 33 次返回 `StackOverflow` 且**不改变深度**。
#[test]
fn stack_overflows_on_33rd_save() {
    let mut c = Canvas::new(16, 16);
    for i in 0..amiki::MAX_STACK_DEPTH {
        c.save().unwrap_or_else(|e| panic!("第 {} 次 save 不该失败：{e}", i + 1));
    }
    assert_eq!(c.save_depth(), amiki::MAX_STACK_DEPTH);
    assert_eq!(amiki::MAX_STACK_DEPTH, 32, "上限是 32（契约）");

    let err = c.save().expect_err("第 33 次必须失败");
    assert_eq!(err, amiki::CanvasError::StackOverflow);
    assert_eq!(c.save_depth(), 32, "失败的 save 不得改变深度");

    // 全部 restore 之后又能继续 save
    for _ in 0..32 {
        c.restore().expect("restore");
    }
    assert!(c.save().is_ok(), "清空后又应能 save");
    eprintln!("溢出：32 成功 / 第 33 次显式 StackOverflow");
}

/// 空栈 restore 返回 `StackUnderflow`，**不 panic、不静默 no-op**。
#[test]
fn empty_stack_underflows() {
    let mut c = Canvas::new(16, 16);
    assert_eq!(
        c.restore().expect_err("空栈必须失败"),
        amiki::CanvasError::StackUnderflow
    );
    c.save().expect("save");
    c.restore().expect("restore");
    assert_eq!(
        c.restore().expect_err("多 restore 一次必须失败"),
        amiki::CanvasError::StackUnderflow
    );
    assert_eq!(c.save_depth(), 0);
    eprintln!("下溢：空栈与多 restore 都显式报错");
}

// ============================================================ 与层 / 裁剪的交互

/// **层内栈独立**：层内 save/restore 不影响外层，深度也不共享。
#[test]
fn layer_has_an_independent_stack() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let mut c = Canvas::new(32, 32);
    c.save().expect("外层 save#1");
    c.set_blend(BlendMode::Multiply);
    c.save().expect("外层 save#2");

    let mut inner_depth_seen = Vec::new();
    let mut inner_err_seen = None;
    let mut inner_fill_rule = None;
    c.push_layer_filtered(
        amiki_scene::LayerBounds::new(0, 0, 32, 32),
        amiki_scene::Filter::None,
        1.0,
        BlendMode::Normal,
        |lc| {
            assert_eq!(lc.save_depth(), 0, "层内栈必须是空的（不继承外层栈）");
            inner_depth_seen.push(lc.save_depth());
            // 层内自己的 save / restore
            lc.save().expect("层内 save");
            inner_depth_seen.push(lc.save_depth());
            lc.restore().expect("层内 restore");
            // 层内独立下溢
            inner_err_seen = Some(lc.restore().expect_err("层内空栈必须失败"));
            lc.fill_path(&p, paint());
            inner_fill_rule = Some(lc.fill_rule());
        },
    );

    assert_eq!(inner_depth_seen, vec![0, 1]);
    assert_eq!(inner_err_seen, Some(amiki::CanvasError::StackUnderflow));
    assert_eq!(
        c.save_depth(),
        2,
        "外层栈深度不应被层内的 save/restore 影响"
    );
    assert_eq!(c.fill_rule(), FillRule::NonZero, "外层状态也不受影响");
    assert_eq!(inner_fill_rule, Some(FillRule::NonZero));

    // 外层继续 restore 仍然正确
    c.restore().expect("外层 restore#2");
    assert_eq!(c.save_depth(), 1);
    c.restore().expect("外层 restore#1");
    assert_eq!(c.save_depth(), 0);
    eprintln!("层交互：层内栈从 0 起算，独立计数、独立报错，外层不受影响");
}

/// 层的**初始状态**继承父画布的填充规则（0.8.0 门面漏掉的那一条）。
#[test]
fn layer_inherits_current_fill_rule() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let mut c = Canvas::new(32, 32);
    c.set_fill_rule(FillRule::EvenOdd);
    c.push_layer_filtered(
        amiki_scene::LayerBounds::new(0, 0, 32, 32),
        amiki_scene::Filter::None,
        1.0,
        BlendMode::Normal,
        |lc| {
            assert_eq!(lc.fill_rule(), FillRule::EvenOdd, "层应继承当前填充规则");
            lc.fill_path(&p, paint());
        },
    );
    let layer_draw = &c.scene.layers[0].scene.draws[0];
    assert_eq!(
        layer_draw.fill_rule,
        FillRule::EvenOdd,
        "层内画出的 Draw 必须带 EvenOdd"
    );
    eprintln!("层继承：初始状态继承填充规则，栈仍是空的");
}

/// 裁剪可以**独立**使用（save/restore 不是裁剪的唯一控制手段）。
#[test]
fn clip_can_still_be_used_independently() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let clip = rect_path(0.0, 0.0, 20.0, 20.0);

    let mut c = Canvas::new(32, 32);
    c.push_clip(&clip); // 不经 save 直接压
    c.fill_path(&p, paint());
    c.pop_clip();
    c.fill_path(&p, paint());

    assert_eq!(c.clip_depth(), 0);
    assert_eq!(c.scene.draws[0].clips.len(), 1);
    assert!(c.scene.draws[1].clips.is_empty());

    // 与 save/restore 混用：save 记深度 → push → restore 回到该深度 → 再 pop
    c.save().expect("save");
    c.push_clip(&clip);
    c.restore().expect("restore");
    assert_eq!(c.clip_depth(), 0, "restore 应把深度退回保存点");
    c.pop_clip(); // 仍可独立使用：此时深度为 0，pop 是 no-op（不 panic）
    assert_eq!(c.clip_depth(), 0);
    eprintln!("裁剪独立使用：push/pop 与 save/restore 正交");
}

/// **零破坏**的旁证：不调 save/restore 时，一串绘制与加了
/// "save+restore 空配对"的版本逐字段相同。
#[test]
fn no_save_no_restore_is_byte_identical() {
    let p = rect_path(1.0, 2.0, 8.0, 6.0);
    let c2 = rect_path(3.0, 3.0, 10.0, 10.0);

    let plain = draws(|c| {
        c.set_blend(BlendMode::Overlay).set_fill_rule(FillRule::EvenOdd);
        c.push_clip(&c2);
        c.fill_path(&p, paint());
        c.pop_clip();
        c.translate(Vec2::new(2.0, 3.0));
        c.fill_path(&p, paint());
    });
    let with_empty_pairs = draws(|c| {
        c.save().expect("save");
        c.restore().expect("restore");
        c.set_blend(BlendMode::Overlay).set_fill_rule(FillRule::EvenOdd);
        c.save().expect("save");
        c.restore().expect("restore");
        c.push_clip(&c2);
        c.fill_path(&p, paint());
        c.pop_clip();
        c.translate(Vec2::new(2.0, 3.0));
        c.save().expect("save");
        c.restore().expect("restore");
        c.fill_path(&p, paint());
    });

    assert_eq!(plain, with_empty_pairs, "空配对不得改变任何 Draw");
    assert_eq!(plain.len(), 2);
    eprintln!("零破坏：save+restore 空配对不改变绘制结果");
}
