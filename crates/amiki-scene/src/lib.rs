//! amiki-scene —— 路径、绘制与场景模型。
//!
//! Scene 是引擎的中间表示：上层 API 产出 Scene，后端（GPU / CPU）消费 Scene。
//! M1 采用直接结构化表示；M2 演进为 Vello 式 tag+data 紧凑编码流（接口保持稳定）。

#![forbid(unsafe_code)]

use amiki_math::{Affine, CubicBez, QuadBez, Vec2};

mod damage;
pub use damage::{DAMAGE_TILE, Damage, DamageRect, changed_bounds, draw_device_bounds, edit_damage};
mod mipmap;
pub use mipmap::{
    compute_lod, generate_mip_chain, generate_padded_mip_levels, mip_level_count, MipChain,
    MipLevel,
};
mod stroke;
pub use stroke::{
    dash_polyline, flatten_path, stroke_path, stroke_path_dashed, stroke_polyline, LineCap,
    LineJoin, StrokeStyle,
};
mod tone;
pub use tone::{linear_to_srgb, rgba8_from_f32_with, srgb_to_linear, ToneMap};


// ---- 数据模型与编码：0.10.9 起按类型簇分文件（原 lib.rs 2493 行 → 见各模块）----
mod amk1;
pub use amk1::{FILL_RULE_BIT, STREAM_MAGIC, STREAM_VERSION, STREAM_VERSION_V1, decode_scene, encode_scene};
mod color;
pub use color::{Color, ColorMatrix};
mod gradient;
pub use gradient::{Gradient, GradientKind, GradientStop};
mod image;
pub use image::{Image, ImagePlacement, NinePatchSlice, nine_patch, sub_image};
mod paint;
pub use paint::{BlendMode, Clip, FillRule, Insets, MAX_DRAW_CLIPS, Paint};
mod path;
pub use path::{Path, PathBuilder, PathEl};
mod scene;
pub use scene::{Draw, Filter, Layer, LayerBounds, Scene, cubic_of, curve_points, quad_of};
mod text;
pub use text::{FontId, GlyphKey, GlyphOutline, PositionedGlyph, TextCache, TextRun, expand_text_run, expand_text_run_with};

#[cfg(test)]
mod tests;
