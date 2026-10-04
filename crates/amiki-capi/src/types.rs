//! `include/amiki.h` 里所有 `#[repr(C)]` 类型的 Rust 镜像，以及枚举的双向映射。
//!
//! # 这一层必须与 header **逐字段对应**
//!
//! header 是契约，这里是实现。任何字段顺序 / 类型 / 对齐的偏差都会让绑定
//! 静默踩内存 —— 所以本文件末尾有**布局断言**（`size_of` 与 `offset_of`），
//! 它们不是形式主义：C 侧的 `sizeof` 必须与这里一致。
//!
//! 枚举取值也**逐一对齐**（`BlendMode` 的 id、`LineCap`/`LineJoin` 的顺序）。
//! 那些 id 直接进 WGSL，改一个数字就会同时影响渲染结果。

use amiki_scene::{
    BlendMode, Color, Filter, Gradient, GradientKind, GradientStop, Image, ImagePlacement, LayerBounds,
    LineCap, LineJoin, Paint, StrokeStyle,
};
use std::sync::Arc;

// ============================================================================
// amiki_result
// ============================================================================

pub const AMIKI_OK: i32 = 0;
pub const AMIKI_ERROR_INVALID_ARGUMENT: i32 = 1;
pub const AMIKI_ERROR_OUT_OF_MEMORY: i32 = 2;
pub const AMIKI_ERROR_BACKEND_UNAVAILABLE: i32 = 3;
pub const AMIKI_ERROR_NOT_SUPPORTED: i32 = 4;
pub const AMIKI_ERROR_PANIC: i32 = 5;
pub const AMIKI_ERROR_UNKNOWN: i32 = 6;
/// 0.7.0 新增：**找得到对象类型、但里面没有要找的东西** ——
/// 目前只用于 `amiki_font_stack_resolve`（字体栈里没有一个字体含该码点）。
pub const AMIKI_ERROR_NOT_FOUND: i32 = 7;

/// 人类可读的说明。与 header 里 `amiki_result_string` 的返回值一致。
///
/// 全部是 **C 字符串字面量**（`c"..."`）—— 它们本身就是 `&'static CStr`，
/// 不需要分配，也就不会在 drop 时释放（header 承诺"不可释放、永远有效"）。
pub(crate) fn result_message(r: i32) -> &'static std::ffi::CStr {
    match r {
        AMIKI_OK => c"ok",
        AMIKI_ERROR_INVALID_ARGUMENT => c"invalid argument",
        AMIKI_ERROR_OUT_OF_MEMORY => c"out of memory",
        AMIKI_ERROR_BACKEND_UNAVAILABLE => c"backend unavailable",
        AMIKI_ERROR_NOT_SUPPORTED => c"not supported by this backend",
        AMIKI_ERROR_PANIC => c"engine panicked (this is a bug in Amiki2D)",
        AMIKI_ERROR_NOT_FOUND => c"not found in this font stack",
        // 给出比 "unknown" 更有用的信息：调用方看到它就知道"引擎报了个未分类错误"，
        // 而 "unknown" 只留给**越界的错误码**（那是调用方自己的 bug）。
        AMIKI_ERROR_UNKNOWN => c"unspecified engine error",
        _ => c"unknown",
    }
}

// ============================================================================
// 文本 shaping 选项（字体回退，0.7.0）
// ============================================================================

/// 缺字策略：与 header 的 `amiki_missing_glyph_policy` 一一对应。
pub const AMIKI_MISSING_GLYPH_ERROR: i32 = 0;
pub const AMIKI_MISSING_GLYPH_NOTDEF: i32 = 1;
pub const AMIKI_MISSING_GLYPH_SKIP: i32 = 2;

/// shaping 选项的 C 镜像（header 的 `amiki_text_options`）。
///
/// `bool` 在 C 里是 1 字节、Rust 里也是 1 字节，`_pad` 显式补齐到 4 字节对齐 ——
/// 布局由 `tests::text_options_layout_matches_the_contract` 钉住。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmikiTextOptions {
    pub missing_glyph: i32,
    pub kerning: bool,
    pub ligatures: bool,
    pub _pad: u32,
}

impl AmikiTextOptions {
    /// header 承诺的默认值（缺字报错、字距与连字都开）。
    pub(crate) fn default_options() -> AmikiTextOptions {
        AmikiTextOptions {
            missing_glyph: AMIKI_MISSING_GLYPH_ERROR,
            kerning: true,
            ligatures: true,
            _pad: 0,
        }
    }

    /// 转成引擎的选项。越界的 `missing_glyph` ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`
    /// （与其它枚举一样：越界值被拒绝，不猜）。
    pub(crate) fn to_options(self) -> Result<amiki_text::TextOptions, i32> {
        let missing_glyph = match self.missing_glyph {
            AMIKI_MISSING_GLYPH_ERROR => amiki_text::MissingGlyphPolicy::Error,
            AMIKI_MISSING_GLYPH_NOTDEF => amiki_text::MissingGlyphPolicy::NotDef,
            AMIKI_MISSING_GLYPH_SKIP => amiki_text::MissingGlyphPolicy::Skip,
            _ => return Err(AMIKI_ERROR_INVALID_ARGUMENT),
        };
        Ok(amiki_text::TextOptions {
            missing_glyph,
            kerning: self.kerning,
            ligatures: self.ligatures,
        })
    }
}

// ============================================================================
// POD 值类型
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmikiVec2 {
    pub x: f32,
    pub y: f32,
}

/// 仿射变换：`x' = a*x + c*y + e`、`y' = b*x + d*y + f`。
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmikiAffine {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmikiColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AmikiLayerBounds {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmikiGradientStop {
    pub t: f32,
    pub color: AmikiColor,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmikiPlacement {
    pub origin: AmikiVec2,
    pub u_axis: AmikiVec2,
    pub v_axis: AmikiVec2,
    /// 非 0 = 平铺
    pub repeat: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmikiStrokeStyle {
    pub width: f32,
    pub cap: i32,
    pub join: i32,
    pub miter_limit: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmikiFilter {
    pub kind: i32,
    pub sigma: f32,
    pub dx: f32,
    pub dy: f32,
    pub color: AmikiColor,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmikiPaint {
    pub kind: i32,
    pub color: AmikiColor,
    /// `SOLID` 时为 null；否则指向 `AmikiGradient`（**借用**，见 header 所有权第 3 条）。
    pub gradient: *const std::ffi::c_void,
}

// ---- 枚举常量（与 header 逐一对齐）----

// 16 种混合模式的常量。取值必须与 `include/amiki.h` 和引擎内部 id **三方一致** ——
// 这些数字直接进 WGSL，改一个就是渲染错误。
pub const AMIKI_BLEND_NORMAL: i32 = 0;
pub const AMIKI_BLEND_MULTIPLY: i32 = 1;
pub const AMIKI_BLEND_SCREEN: i32 = 2;
pub const AMIKI_BLEND_OVERLAY: i32 = 3;
pub const AMIKI_BLEND_DARKEN: i32 = 4;
pub const AMIKI_BLEND_LIGHTEN: i32 = 5;
pub const AMIKI_BLEND_COLOR_DODGE: i32 = 6;
pub const AMIKI_BLEND_COLOR_BURN: i32 = 7;
pub const AMIKI_BLEND_HARD_LIGHT: i32 = 8;
pub const AMIKI_BLEND_SOFT_LIGHT: i32 = 9;
pub const AMIKI_BLEND_DIFFERENCE: i32 = 10;
pub const AMIKI_BLEND_EXCLUSION: i32 = 11;
pub const AMIKI_BLEND_HUE: i32 = 12;
pub const AMIKI_BLEND_SATURATION: i32 = 13;
pub const AMIKI_BLEND_COLOR: i32 = 14;
pub const AMIKI_BLEND_LUMINOSITY: i32 = 15;

pub const AMIKI_CAP_BUTT: i32 = 0;
pub const AMIKI_CAP_ROUND: i32 = 1;
pub const AMIKI_CAP_SQUARE: i32 = 2;

pub const AMIKI_JOIN_BEVEL: i32 = 0;
pub const AMIKI_JOIN_MITER: i32 = 1;
pub const AMIKI_JOIN_ROUND: i32 = 2;

pub const AMIKI_FILTER_NONE: i32 = 0;
pub const AMIKI_FILTER_BLUR: i32 = 1;
pub const AMIKI_FILTER_DROP_SHADOW: i32 = 2;

pub const AMIKI_PAINT_SOLID: i32 = 0;
pub const AMIKI_PAINT_LINEAR: i32 = 1;
pub const AMIKI_PAINT_RADIAL: i32 = 2;
pub const AMIKI_PAINT_SWEEP: i32 = 3;

// ---- 色调映射（header 的 `amiki_tone_map`）----
pub const AMIKI_TONEMAP_RAW: i32 = 0;
pub const AMIKI_TONEMAP_REINHARD: i32 = 1;
pub const AMIKI_TONEMAP_ACES: i32 = 2;

/// C 侧的 `amiki_tone_map` → 引擎的 [`amiki_scene::ToneMap`]。
///
/// 越界值返回 `None` —— 调用方据此报 `AMIKI_ERROR_INVALID_ARGUMENT`，
/// **不静默退回默认值**（静默退回会让"设错了"看起来像"设上了"）。
pub(crate) fn tone_map_from_raw(v: i32) -> Option<amiki_scene::ToneMap> {
    match v {
        AMIKI_TONEMAP_RAW => Some(amiki_scene::ToneMap::Raw),
        AMIKI_TONEMAP_REINHARD => Some(amiki_scene::ToneMap::Reinhard),
        AMIKI_TONEMAP_ACES => Some(amiki_scene::ToneMap::Aces),
        _ => None,
    }
}

/// 引擎的 [`amiki_scene::ToneMap`] → C 侧取值。
pub(crate) fn tone_map_to_raw(t: amiki_scene::ToneMap) -> i32 {
    match t {
        amiki_scene::ToneMap::Raw => AMIKI_TONEMAP_RAW,
        amiki_scene::ToneMap::Reinhard => AMIKI_TONEMAP_REINHARD,
        amiki_scene::ToneMap::Aces => AMIKI_TONEMAP_ACES,
    }
}

/// C 侧的 `amiki_mipmap_mode`（B4）。
pub const AMIKI_MIPMAP_DISABLED: i32 = 0;
pub const AMIKI_MIPMAP_BOX: i32 = 1;

/// C 侧的 `amiki_mipmap_mode` → 是否生成 mip 链。
///
/// 与 tone map 同一约定：越界值返回 `None`，调用方显式报错且**不改动
/// 已有设置** —— 静默退回默认会让"设错了"看起来像"设上了"。
pub(crate) fn mipmap_mode_from_raw(v: i32) -> Option<bool> {
    match v {
        AMIKI_MIPMAP_DISABLED => Some(false),
        AMIKI_MIPMAP_BOX => Some(true),
        _ => None,
    }
}

pub const AMIKI_BACKEND_AUTO: i32 = 0;
pub const AMIKI_BACKEND_GPU: i32 = 1;
pub const AMIKI_BACKEND_CPU: i32 = 2;

// ============================================================================
// 到引擎类型的转换
// ============================================================================

impl AmikiColor {
    pub(crate) fn to_color(self) -> Color {
        Color { r: self.r, g: self.g, b: self.b, a: self.a }
    }
    pub(crate) fn from_color(c: Color) -> Self {
        AmikiColor { r: c.r, g: c.g, b: c.b, a: c.a }
    }
    pub(crate) fn is_finite(self) -> bool {
        self.r.is_finite() && self.g.is_finite() && self.b.is_finite() && self.a.is_finite()
    }
}

impl AmikiAffine {
    pub(crate) fn to_affine(self) -> amiki_math::Affine {
        amiki_math::Affine { a: self.a, b: self.b, c: self.c, d: self.d, e: self.e, f: self.f }
    }
    pub(crate) fn from_affine(t: amiki_math::Affine) -> Self {
        AmikiAffine { a: t.a, b: t.b, c: t.c, d: t.d, e: t.e, f: t.f }
    }
    pub(crate) fn is_finite(self) -> bool {
        self.a.is_finite()
            && self.b.is_finite()
            && self.c.is_finite()
            && self.d.is_finite()
            && self.e.is_finite()
            && self.f.is_finite()
    }
}

impl AmikiPlacement {
    pub(crate) fn to_placement(self) -> ImagePlacement {
        ImagePlacement {
            origin: amiki_math::Vec2::new(self.origin.x, self.origin.y),
            u_axis: amiki_math::Vec2::new(self.u_axis.x, self.u_axis.y),
            v_axis: amiki_math::Vec2::new(self.v_axis.x, self.v_axis.y),
            repeat: self.repeat != 0,
        }
    }
}

/// `amiki_blend_mode` → [`BlendMode`]。越界返回 `None`。
pub(crate) fn blend_from_i32(v: i32) -> Option<BlendMode> {
    Some(match v {
        0 => BlendMode::Normal,
        1 => BlendMode::Multiply,
        2 => BlendMode::Screen,
        3 => BlendMode::Overlay,
        4 => BlendMode::Darken,
        5 => BlendMode::Lighten,
        6 => BlendMode::ColorDodge,
        7 => BlendMode::ColorBurn,
        8 => BlendMode::HardLight,
        9 => BlendMode::SoftLight,
        10 => BlendMode::Difference,
        11 => BlendMode::Exclusion,
        12 => BlendMode::Hue,
        13 => BlendMode::Saturation,
        14 => BlendMode::Color,
        15 => BlendMode::Luminosity,
        _ => return None,
    })
}

pub(crate) fn cap_from_i32(v: i32) -> Option<LineCap> {
    Some(match v {
        AMIKI_CAP_BUTT => LineCap::Butt,
        AMIKI_CAP_ROUND => LineCap::Round,
        AMIKI_CAP_SQUARE => LineCap::Square,
        _ => return None,
    })
}

pub(crate) fn join_from_i32(v: i32, miter_limit: f32) -> Option<LineJoin> {
    Some(match v {
        AMIKI_JOIN_BEVEL => LineJoin::Bevel,
        // `<= 0` 视为 Canvas/SVG 默认值 10（header 承诺的语义）
        AMIKI_JOIN_MITER => LineJoin::Miter {
            limit: if miter_limit > 0.0 { miter_limit } else { 10.0 },
        },
        AMIKI_JOIN_ROUND => LineJoin::Round,
        _ => return None,
    })
}

impl AmikiStrokeStyle {
    pub(crate) fn to_style(self) -> Option<StrokeStyle> {
        if !self.width.is_finite() || self.width < 0.0 {
            return None;
        }
        Some(StrokeStyle {
            width: self.width,
            cap: cap_from_i32(self.cap)?,
            join: join_from_i32(self.join, self.miter_limit)?,
        })
    }
}

impl AmikiFilter {
    pub(crate) fn to_filter(self) -> Option<Filter> {
        Some(match self.kind {
            AMIKI_FILTER_NONE => Filter::None,
            AMIKI_FILTER_BLUR => {
                if !self.sigma.is_finite() || self.sigma < 0.0 {
                    return None;
                }
                Filter::Blur { sigma: self.sigma }
            }
            AMIKI_FILTER_DROP_SHADOW => {
                if !self.sigma.is_finite() || self.sigma < 0.0 {
                    return None;
                }
                if !self.dx.is_finite() || !self.dy.is_finite() || !self.color.is_finite() {
                    return None;
                }
                Filter::DropShadow {
                    dx: self.dx,
                    dy: self.dy,
                    sigma: self.sigma,
                    color: self.color.to_color(),
                }
            }
            _ => return None,
        })
    }
}

// ============================================================================
// 渐变 / 色标
// ============================================================================

/// 把 C 侧的色标数组转成引擎类型。少于 2 个返回 `None`。
///
/// # Safety
/// `stops` 必须指向至少 `count` 个有效元素（或 `count == 0`）。
pub(crate) unsafe fn stops_from_raw(
    stops: *const AmikiGradientStop,
    count: u32,
) -> Option<Vec<GradientStop>> {
    if count < 2 {
        return None;
    }
    if stops.is_null() {
        return None;
    }
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count as usize {
        // SAFETY: 调用方保证 [0, count) 都有效（见本函数的契约）
        let s = unsafe { *stops.add(i) };
        if !s.t.is_finite() || !s.color.is_finite() {
            return None;
        }
        out.push(GradientStop { t: s.t, color: s.color.to_color() });
    }
    Some(out)
}

pub(crate) fn make_gradient(kind: GradientKind, mut stops: Vec<GradientStop>) -> Arc<Gradient> {
    stops.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    Arc::new(Gradient { kind, stops, repeat: false })
}

/// 由 C 侧 paint 构造引擎 paint。句柄为空 / 已中毒 / 类型越界时返回 `None`。
///
/// # Safety
/// `p.gradient` 若非空，必须指向一个有效的 `AmikiGradient`（契约见 header）。
pub(crate) unsafe fn paint_from_raw(p: AmikiPaint) -> Option<Paint> {
    match p.kind {
        AMIKI_PAINT_SOLID => {
            if !p.color.is_finite() {
                return None;
            }
            Some(Paint::Solid(p.color.to_color()))
        }
        AMIKI_PAINT_LINEAR | AMIKI_PAINT_RADIAL | AMIKI_PAINT_SWEEP => {
            if p.gradient.is_null() {
                return None;
            }
            // SAFETY: 见函数契约；指针来自 `amiki_gradient_create_*`
            let g = unsafe { &*(p.gradient as *const crate::objects::AmikiGradient) };
            if g.poison.is_poisoned() {
                return None;
            }
            // `Arc::clone` 而不是借用：场景要独立持有一份，
            // 于是 C 侧句柄可以在绘制后立刻销毁（header 所有权第 4 条）。
            Some(Paint::Gradient(Arc::clone(&g.gradient)))
        }
        _ => None,
    }
}

/// 位图 paint：把 `Arc<Image>` 与放置组合成 `Paint::Image`。
pub(crate) fn image_paint(image: Arc<Image>, placement: ImagePlacement) -> Paint {
    Paint::Image { image, placement }
}

pub(crate) fn layer_bounds_from_raw(b: AmikiLayerBounds) -> LayerBounds {
    LayerBounds { x: b.x, y: b.y, width: b.width, height: b.height }
}

impl Default for AmikiStrokeStyle {
    fn default() -> Self {
        AmikiStrokeStyle { width: 1.0, cap: AMIKI_CAP_BUTT, join: AMIKI_JOIN_BEVEL, miter_limit: 10.0 }
    }
}

impl Default for AmikiFilter {
    fn default() -> Self {
        AmikiFilter {
            kind: AMIKI_FILTER_NONE,
            sigma: 0.0,
            dx: 0.0,
            dy: 0.0,
            color: AmikiColor::default(),
        }
    }
}

impl Default for AmikiPaint {
    fn default() -> Self {
        AmikiPaint {
            kind: AMIKI_PAINT_SOLID,
            color: AmikiColor::default(),
            gradient: std::ptr::null(),
        }
    }
}

// ============================================================================
// 布局断言：**C 侧的 sizeof/offsetof 必须与这里一致**
// ============================================================================

#[cfg(test)]
mod layout_tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    /// 这些数字必须与 `include/amiki.h` 里 C 的 `sizeof` 一致。
    /// 改动任何一个 `#[repr(C)]` 结构都必须同步改这里 **和** header 里的注释。
    #[test]
    fn repr_c_layouts_match_the_contract() {
        assert_eq!(size_of::<AmikiVec2>(), 8);
        assert_eq!(size_of::<AmikiAffine>(), 24);
        assert_eq!(size_of::<AmikiColor>(), 16);
        assert_eq!(size_of::<AmikiLayerBounds>(), 16);
        assert_eq!(size_of::<AmikiGradientStop>(), 20);
        assert_eq!(size_of::<AmikiPlacement>(), 28);
        assert_eq!(size_of::<AmikiStrokeStyle>(), 16);
        assert_eq!(size_of::<AmikiFilter>(), 32);
        // `AmikiPaint` 里有指针：`kind`(4) + `color`(16) 到 20，指针要对齐到 8
        // 于是补 4 字节 padding，总长 = 24 + 指针宽度。
        // 这个 padding 是 C 与 Rust **共同遵守**的对齐规则，不是巧合。
        assert_eq!(
            size_of::<AmikiPaint>(),
            24 + size_of::<*const std::ffi::c_void>()
        );
    }

    #[test]
    fn float_only_types_are_4_byte_aligned() {
        assert_eq!(align_of::<AmikiVec2>(), 4);
        assert_eq!(align_of::<AmikiAffine>(), 4);
        assert_eq!(align_of::<AmikiColor>(), 4);
        assert_eq!(align_of::<AmikiLayerBounds>(), 4);
        assert_eq!(align_of::<AmikiGradientStop>(), 4);
        assert_eq!(align_of::<AmikiPlacement>(), 4);
        assert_eq!(align_of::<AmikiStrokeStyle>(), 4);
        assert_eq!(align_of::<AmikiFilter>(), 4);
    }

    #[test]
    fn field_offsets_match_the_contract() {
        assert_eq!(offset_of!(AmikiAffine, a), 0);
        assert_eq!(offset_of!(AmikiAffine, c), 8);
        assert_eq!(offset_of!(AmikiAffine, e), 16);
        assert_eq!(offset_of!(AmikiStrokeStyle, cap), 4);
        assert_eq!(offset_of!(AmikiStrokeStyle, join), 8);
        assert_eq!(offset_of!(AmikiStrokeStyle, miter_limit), 12);
        assert_eq!(offset_of!(AmikiFilter, sigma), 4);
        assert_eq!(offset_of!(AmikiFilter, dx), 8);
        assert_eq!(offset_of!(AmikiFilter, color), 16);
        assert_eq!(offset_of!(AmikiPlacement, repeat), 24);
        assert_eq!(offset_of!(AmikiGradientStop, color), 4);
    }

    /// 枚举取值必须与 header 一致 —— 它们直接进 WGSL，改一个数字就是渲染错误。
    #[test]
    fn blend_ids_match_the_engine() {
        for (v, expect) in [
            (0, BlendMode::Normal),
            (1, BlendMode::Multiply),
            (15, BlendMode::Luminosity),
        ] {
            assert_eq!(blend_from_i32(v), Some(expect));
        }
        assert_eq!(blend_from_i32(16), None);
        assert_eq!(blend_from_i32(-1), None);
    }

    #[test]
    fn out_of_range_enums_are_rejected() {
        assert_eq!(cap_from_i32(3), None);
        assert_eq!(join_from_i32(3, 0.0), None);
        assert_eq!(AmikiFilter { kind: 99, ..Default::default() }.to_filter(), None);
    }

    /// 缺字策略的**布局与取值**都要与 header 一致（C 侧按 `sizeof` 传值）。
    #[test]
    fn text_options_layout_matches_the_contract() {
        assert_eq!(size_of::<AmikiTextOptions>(), 12);
        assert_eq!(offset_of!(AmikiTextOptions, missing_glyph), 0);
        assert_eq!(offset_of!(AmikiTextOptions, kerning), 4);
        assert_eq!(offset_of!(AmikiTextOptions, ligatures), 5);
        assert_eq!(offset_of!(AmikiTextOptions, _pad), 8);
    }

    /// 三个策略值必须映射到引擎的三种行为；越界值必须被拒绝。
    #[test]
    fn missing_glyph_policy_is_mapped_one_to_one() {
        use amiki_text::MissingGlyphPolicy;
        for (v, expect) in [
            (AMIKI_MISSING_GLYPH_ERROR, MissingGlyphPolicy::Error),
            (AMIKI_MISSING_GLYPH_NOTDEF, MissingGlyphPolicy::NotDef),
            (AMIKI_MISSING_GLYPH_SKIP, MissingGlyphPolicy::Skip),
        ] {
            let o = AmikiTextOptions { missing_glyph: v, ..AmikiTextOptions::default_options() };
            assert_eq!(o.to_options().expect("合法取值").missing_glyph, expect);
        }
        let bad = AmikiTextOptions { missing_glyph: 3, ..AmikiTextOptions::default_options() };
        assert_eq!(bad.to_options().unwrap_err(), AMIKI_ERROR_INVALID_ARGUMENT);
        // 默认值 = header 写的那三个
        let d = AmikiTextOptions::default_options();
        assert_eq!(d.missing_glyph, AMIKI_MISSING_GLYPH_ERROR);
        assert!(d.kerning && d.ligatures);
    }

    /// 新的错误码必须有可读说明（header 承诺 `amiki_result_string` 覆盖全部取值）。
    #[test]
    fn not_found_has_a_message() {
        assert_eq!(result_message(AMIKI_ERROR_NOT_FOUND).to_str().unwrap(), "not found in this font stack");
        assert_ne!(result_message(AMIKI_ERROR_NOT_FOUND), c"unknown");
    }

    /// `miter_limit <= 0` 要变成 Canvas/SVG 的默认值 10（header 承诺的语义）。
    #[test]
    fn non_positive_miter_limit_becomes_default_ten() {
        assert_eq!(join_from_i32(AMIKI_JOIN_MITER, 0.0), Some(LineJoin::Miter { limit: 10.0 }));
        assert_eq!(join_from_i32(AMIKI_JOIN_MITER, -1.0), Some(LineJoin::Miter { limit: 10.0 }));
        assert_eq!(join_from_i32(AMIKI_JOIN_MITER, 4.0), Some(LineJoin::Miter { limit: 4.0 }));
    }
}
