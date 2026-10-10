//! 画布状态：变换 / 混合 / 裁剪 / 状态栈 / 填充规则 / 配色与 mip 偏置。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::ffi;
use crate::objects::*;
use crate::types::*;


#[no_mangle]
pub extern "C" fn amiki_canvas_set_transform(canvas: *mut AmikiCanvas, t: AmikiAffine) -> i32 {
    ffi!(canvas, |c| {
        if !t.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().set_transform(t.to_affine());
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_reset_transform(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        c.target().reset();
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_translate(canvas: *mut AmikiCanvas, dx: f32, dy: f32) -> i32 {
    ffi!(canvas, |c| {
        if !dx.is_finite() || !dy.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().translate(amiki_math::Vec2::new(dx, dy));
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_rotate(canvas: *mut AmikiCanvas, radians: f32) -> i32 {
    ffi!(canvas, |c| {
        if !radians.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().rotate(radians);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_scale(canvas: *mut AmikiCanvas, s: f32) -> i32 {
    ffi!(canvas, |c| {
        if !s.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().scale(s);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_set_blend(canvas: *mut AmikiCanvas, blend: i32) -> i32 {
    ffi!(canvas, |c| {
        let Some(b) = blend_from_i32(blend) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        c.target().set_blend(b);
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_push_clip(
    canvas: *mut AmikiCanvas,
    path: *const AmikiPath,
) -> i32 {
    if path.is_null() {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    }
    ffi!(canvas, |c| {
        // SAFETY: 已判空；契约要求指针有效
        let p = unsafe { &*path };
        if let Err(r) = p.poison.check() {
            return r;
        }
        c.target().push_clip(&p.path);
        c.clip_depth += 1;
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_pop_clip(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        // `Canvas::pop_clip` 在栈空时是静默空操作 —— 那会让"压了没弹/弹多了"
        // 这类错误无声通过。C ABI 这一层显式拒绝。
        if c.clip_depth == 0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.target().pop_clip();
        c.clip_depth -= 1;
        AMIKI_OK
    })
}

// ---- 绘制 ----


#[no_mangle]
pub extern "C" fn amiki_canvas_set_flatten_tolerance(
    canvas: *mut AmikiCanvas,
    eps: f32,
) -> i32 {
    ffi!(canvas, |c| {
        if !eps.is_finite() || eps <= 0.0 {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.eps = eps;
        AMIKI_OK
    })
}

// ============================================================================
// 色调映射
// ============================================================================

/// 设置输出量化时用的色调映射。默认 `AMIKI_TONEMAP_RAW`（只裁剪）。
#[no_mangle]
pub extern "C" fn amiki_canvas_set_tone_map(cv: *mut AmikiCanvas, tone: i32) -> i32 {
    ffi!(cv, |c| {
        // 越界值**显式失败**，不静默退回默认 ——
        // 静默退回会让"设错了"看起来像"设上了"
        let Some(t) = crate::types::tone_map_from_raw(tone) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        c.tone_map = t;
        AMIKI_OK
    })
}

/// 读回当前的色调映射设置。空画布返回 `AMIKI_TONEMAP_RAW`。
#[no_mangle]
pub extern "C" fn amiki_canvas_tone_map(cv: *const AmikiCanvas) -> i32 {
    if cv.is_null() {
        return crate::types::AMIKI_TONEMAP_RAW;
    }
    crate::types::tone_map_to_raw(unsafe { &*cv }.tone_map)
}

// ============================================================================
// 路径
// ============================================================================

/// 设置画布的 mip 采样偏移（B4）。默认 0。
///
/// 正值取更糊的层（适合先放大后缩小的场合），负值取更锐的层；
/// 采样层级会被钳到 mip 链范围内，过大/过小的值只是失效、不会出错。
/// 非有限值（NaN / inf）被拒绝。只影响**缩小**的位图采样，不影响绘制本身。
#[no_mangle]
pub extern "C" fn amiki_canvas_set_mip_bias(cv: *mut AmikiCanvas, bias: f32) -> i32 {
    ffi!(cv, |c| {
        if !bias.is_finite() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        c.mip_bias = bias;
        AMIKI_OK
    })
}

// ---- 填充规则（even-odd，0.8.0）----

/// 设置**后续填充**的填充规则。状态式、粘性；默认 `AMIKI_FILL_NONZERO`。
///
/// 越界枚举值被拒绝（返回 `AMIKI_ERROR_INVALID_ARGUMENT`），状态不变。
/// 只作用于填充，不影响描边（描边是轮廓化，与填充规则无关）。
#[no_mangle]
pub extern "C" fn amiki_canvas_set_fill_rule(cv: *mut AmikiCanvas, rule: i32) -> i32 {
    let Some(rule) = fill_rule_from_i32(rule) else {
        return AMIKI_ERROR_INVALID_ARGUMENT;
    };
    ffi!(cv, |c| {
        c.fill_rule = rule;
        // 立刻写进当前绘制目标（根画布或最内层打开中的层）。
        // 层是**之后**才打开的，所以在 `begin_layer` 里也要把规则传下去。
        c.target().set_fill_rule(rule);
        AMIKI_OK
    })
}

/// 当前填充规则。传 NULL 返回 `AMIKI_FILL_NONZERO`（与 `amiki_context_backend`
/// 对 NULL 的处理一致：不崩、给默认值）。
#[no_mangle]
pub extern "C" fn amiki_canvas_get_fill_rule(cv: *const AmikiCanvas) -> i32 {
    if cv.is_null() {
        return AMIKI_FILL_NONZERO;
    }
    // SAFETY: 已判空
    let c = unsafe { &*cv };
    fill_rule_to_i32(c.fill_rule)
}

// ---- 状态栈（0.9.0）----

/// 保存当前状态（变换 / 混合模式 / 裁剪栈 / 填充规则）到状态栈。
///
/// 已存 32 层时返回 `AMIKI_ERROR_STACK_OVERFLOW`，栈不变。
#[no_mangle]
pub extern "C" fn amiki_canvas_save(cv: *mut AmikiCanvas) -> i32 {
    ffi!(cv, |c| {
        match c.target().save() {
            Ok(()) => AMIKI_OK,
            Err(e) => canvas_error_to_i32(e),
        }
    })
}

/// 从状态栈恢复：四项状态**整体覆盖**为最近一次 save 时的值。
///
/// 栈空返回 `AMIKI_ERROR_STACK_UNDERFLOW`（状态不变）。
///
/// ⚠️ **两侧镜像都要重新同步**。C 侧独立维护着两份门面状态的镜像：
/// `clip_depth`（`push_clip` 加、`pop_clip` 减）与 `fill_rule`
/// （`amiki_canvas_set_fill_rule` 写）。恢复会把门面的这两项一起回退，
/// 但**不会**动 C 侧的镜像 —— 不同步的话：
/// - `clip_depth` 停在旧值 ⇒ 下一次 `amiki_canvas_pop_clip` 误判；
/// - `fill_rule` 停在旧值 ⇒ `amiki_canvas_get_fill_rule` 报出一个
///   与实际绘制不一致的规则（本测试抓到的就是这个：恢复后仍报 EvenOdd）。
#[no_mangle]
pub extern "C" fn amiki_canvas_restore(cv: *mut AmikiCanvas) -> i32 {
    ffi!(cv, |c| {
        match c.target().restore() {
            Ok(()) => {
                c.clip_depth = c.target().clip_depth();
                c.fill_rule = c.target().fill_rule();
                AMIKI_OK
            }
            Err(e) => canvas_error_to_i32(e),
        }
    })
}

