//! 离屏层：`begin_layer` / `end_layer`。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::ffi;
use crate::objects::*;
use crate::types::*;

use amiki_scene::Layer;

#[no_mangle]
pub extern "C" fn amiki_canvas_begin_layer(
    canvas: *mut AmikiCanvas,
    bounds: AmikiLayerBounds,
    filter: AmikiFilter,
    opacity: f32,
    blend: i32,
) -> i32 {
    ffi!(canvas, |c| {
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let Some(blend) = blend_from_i32(blend) else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let Some(filter) = filter.to_filter() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 层里再开层会让"谁在谁里面"变得难以推理，而 C 侧没有作用域语法兜底。
        // 这里放开嵌套（引擎支持），但深度设上限以免调用方写错时无限增长。
        const MAX_LAYER_DEPTH: usize = 32;
        if c.layers.len() >= MAX_LAYER_DEPTH {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let (w, h) = (c.width, c.height);
        let mut sub = amiki::Canvas::new(w, h);
        sub.set_blend(blend);
        // 填充规则是画布级状态：新开的层要继承当前规则，
        // 否则"设了 even-odd 再开层"会在这层里静默退回 NonZero。
        sub.set_fill_rule(c.fill_rule);
        c.layers.push(PendingLayer {
            canvas: sub,
            bounds: layer_bounds_from_raw(bounds),
            filter,
            opacity,
            blend,
        });
        AMIKI_OK
    })
}

#[no_mangle]
pub extern "C" fn amiki_canvas_end_layer(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        let Some(l) = c.layers.pop() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 把子场景作为一个 layer 挂到**当前目标**（可能是更外层）上。
        // 这里用 `Scene::push_layer` 而不是门面的 `push_layer_filtered`：
        // 后者接收闭包，而 C 侧没有闭包 —— 绘制是在 begin/end 之间发生的。
        let target = c.target();
        target.scene.push_layer(
            Layer {
                bounds: l.bounds,
                scene: l.canvas.scene,
                filter: l.filter,
                opacity: l.opacity,
                blend: l.blend,
            },
            |_| {},
        );
        AMIKI_OK
    })
}

// ---- 输出 ----

