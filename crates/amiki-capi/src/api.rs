//! ABI 层的通用部件：**panic 守卫**、版本查询、错误串、以及纯值构造。
//!
//! # `ffi!` 宏是全部导出函数的统一外壳
//!
//! 它做三件事，顺序不能变：
//! 1. 空指针检查 → `AMIKI_ERROR_INVALID_ARGUMENT`（**不 deref 空指针**）
//! 2. 中毒检查 → `AMIKI_ERROR_PANIC`
//! 3. `catch_unwind` 包住函数体；真的 panic 了就**标记中毒**并返回
//!    `AMIKI_ERROR_PANIC`
//!
//! 第 3 条是本 crate 存在的核心理由之一：**跨 FFI 边界的 panic 是未定义行为**，
//! 绝不能让 C 调用方承担。`amiki.h` 把这条写成了承诺。
//!
//! 用 `AssertUnwindSafe` 是**有意的**：对象在 panic 后可能自相矛盾，
//! 而这正是我们要标记中毒的原因 —— 标记一个 `bool` 本身是安全的。

use crate::objects::*;
use crate::types::*;

/// 导出函数的统一外壳。`$body` 是一个接收 `&mut T` 的闭包式表达式。
macro_rules! ffi {
    ($ptr:expr, |$o:ident| $body:block) => {{
        let p = $ptr;
        if p.is_null() {
            return $crate::types::AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let r = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
            // SAFETY: 上面已判空；调用方保证指针来自对应的 `_create` 且未被销毁
            // （这是 C ABI 的固有约定，写在 `amiki.h` 的所有权一节里）。
            let $o: &mut _ = unsafe { &mut *p };
            if $o.poison.is_poisoned() {
                return $crate::types::AMIKI_ERROR_PANIC;
            }
            $body
        }));
        match r {
            ::std::result::Result::Ok(v) => v,
            ::std::result::Result::Err(_) => {
                // 标记中毒：对象可能已被 panic 打断在修改中途，继续用它有静默出错的风险。
                // 标记一个 bool 不涉及对象其余状态，是安全的。
                unsafe { (*p).poison.mark() };
                $crate::types::AMIKI_ERROR_PANIC
            }
        }
    }};
}
pub(crate) use ffi;

/// 同 [`ffi`]，但用于**不接收对象**的导出函数（例如各 `_create`）。
macro_rules! ffi_free {
    ($body:block) => {{
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| $body)) {
            ::std::result::Result::Ok(v) => v,
            ::std::result::Result::Err(_) => $crate::types::AMIKI_ERROR_PANIC,
        }
    }};
}
pub(crate) use ffi_free;

/// 写入出参指针。出参为空是**允许**的（header 里多处写明"可为 NULL"）。
///
/// # Safety
/// `out` 若非空，必须指向一个有效的可写位置。
#[inline]
pub(crate) unsafe fn write_out<T>(out: *mut T, v: T) {
    if !out.is_null() {
        // SAFETY: 见函数契约
        unsafe { *out = v };
    }
}

// ============================================================================
// 堆分配（WASM 必需；原生下也有用）
// ============================================================================

/// 分配对齐的缓冲，供调用方写入数据后再传给其它函数。
///
/// # 为什么 C ABI 需要这个
///
/// 在 **WASM** 上，调用方（JS）无法自己分配引擎线性内存里的空间 ——
/// 它只能通过导出的分配器。没有这个函数，就无法把像素、字体字节、
/// 渐变色标这类数组传进来。
///
/// 原生场景下它也有用：调用方可以用**同一个分配器**分配与释放，
/// 避免跨 CRT / 跨 `malloc` 实现释放造成的未定义行为。
///
/// # 契约
///
/// - 返回的指针对齐到 **16 字节**（够放下任何标量类型与 SIMD 对齐需求）
/// - `size == 0` 返回空指针
/// - 必须用 [`amiki_free`] 释放，**且传入与分配时相同的 `size`** ——
///   释放时需要它来重建 layout
/// - 分配失败返回空指针（WASM 上通常意味着内存耗尽）
#[no_mangle]
pub extern "C" fn amiki_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return std::ptr::null_mut();
    }
    // `from_size_align` 会检查对齐是不是 2 的幂、以及 size 会不会溢出 isize。
    // 对齐是常量 16，所以只会因 size 过大而失败 —— 那时返回 NULL 是正确行为。
    let Ok(layout) = std::alloc::Layout::from_size_align(size, ALLOC_ALIGN) else {
        return std::ptr::null_mut();
    };
    // SAFETY: layout 非零大小（size > 0 已判）
    unsafe { std::alloc::alloc(layout) }
}

/// 释放 [`amiki_alloc`] 返回的缓冲。传空指针是空操作。
///
/// `size` 必须是**分配时的同一个值** —— 重建 layout 需要它。
#[no_mangle]
pub extern "C" fn amiki_free(ptr: *mut u8, size: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    // 用**安全检查版**重建 layout：契约要求 size 与分配时一致，所以这里不会
    // 失败；万一调用方传错，我们也只是提前返回而不是进入未定义行为。
    let Ok(layout) = std::alloc::Layout::from_size_align(size, ALLOC_ALIGN) else {
        return;
    };
    // SAFETY: 契约要求 `ptr` 来自 `amiki_alloc`，且 layout 与分配时相同
    unsafe { std::alloc::dealloc(ptr, layout) };
}

/// [`amiki_alloc`] 使用的对齐。header 里写明了这个值。
const ALLOC_ALIGN: usize = 16;

// ============================================================================
// 版本
// ============================================================================

/// 运行期库版本，编码为 `(major << 16) | (minor << 8) | patch`。
///
/// 与 header 的 `AMIKI_VERSION` 对照；不一致说明头文件与二进制不是同一版。
#[no_mangle]
pub extern "C" fn amiki_version() -> u32 {
    // 数值版本必须与 `Cargo.toml` 一致，但 const 上下文里没法解析字符串，
    // 所以只能各写一份。**一致性由测试保证**：
    // `tests::numeric_version_matches_package_version` 会解析
    // `CARGO_PKG_VERSION` 逐段比对 —— 不同步就红。
    const MAJOR: u32 = 0;
    const MINOR: u32 = 9;
    const PATCH: u32 = 0;
    (MAJOR << 16) | (MINOR << 8) | PATCH
}

/// 完整版本串，形如 `"0.4.0-alpha"`。**不可释放**，永远有效。
///
/// 与 [`amiki_version`] 的分工：数值只含 major/minor/patch（便于比较），
/// 本字符串额外带预发布标签（便于展示与上报）。
#[no_mangle]
pub extern "C" fn amiki_version_string() -> *const std::ffi::c_char {
    // **编译期**从 `Cargo.toml` 取，`concat!` 补上结尾的 NUL ——
    // 于是它不可能与包版本漂移。
    //
    // 这一条是踩出来的：我先前把版本串**硬编码**成 "0.4.0-alpha"，
    // 后来把 workspace 版本升到 0.5.0-alpha 时就漏改了，于是 Node 示例
    // 打印出旧版本。两处各写一份必然漂移，干脆只留一处。
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// 人类可读的错误说明（静态字符串，**不可释放**）。
#[no_mangle]
pub extern "C" fn amiki_result_string(r: i32) -> *const std::ffi::c_char {
    result_message(r).as_ptr()
}

// ============================================================================
// 纯值构造（不分配、不会失败，所以直接返回值而不是 result）
// ============================================================================

#[no_mangle]
pub extern "C" fn amiki_affine_identity() -> AmikiAffine {
    AmikiAffine::from_affine(amiki_math::Affine::IDENTITY)
}

#[no_mangle]
pub extern "C" fn amiki_affine_translate(dx: f32, dy: f32) -> AmikiAffine {
    AmikiAffine::from_affine(amiki_math::Affine::translate(amiki_math::Vec2::new(dx, dy)))
}

#[no_mangle]
pub extern "C" fn amiki_affine_rotate(radians: f32) -> AmikiAffine {
    AmikiAffine::from_affine(amiki_math::Affine::rotate(radians))
}

#[no_mangle]
pub extern "C" fn amiki_affine_scale(s: f32) -> AmikiAffine {
    AmikiAffine::from_affine(amiki_math::Affine::scale(s))
}

/// `result = a ∘ b`（先应用 `b`，再应用 `a`）。
#[no_mangle]
pub extern "C" fn amiki_affine_concat(a: AmikiAffine, b: AmikiAffine) -> AmikiAffine {
    AmikiAffine::from_affine(a.to_affine().then(b.to_affine()))
}

#[no_mangle]
pub extern "C" fn amiki_color_rgba(r: f32, g: f32, b: f32, a: f32) -> AmikiColor {
    AmikiColor { r, g, b, a }
}

#[no_mangle]
pub extern "C" fn amiki_color_rgb8(r: u8, g: u8, b: u8) -> AmikiColor {
    AmikiColor::from_color(amiki_scene::Color::rgb8(r, g, b))
}

#[no_mangle]
pub extern "C" fn amiki_color_rgba8(r: u8, g: u8, b: u8, a: u8) -> AmikiColor {
    AmikiColor::from_color(amiki_scene::Color::rgba8(r, g, b, a))
}

/// 默认描边样式：宽 1.0、butt 端帽、bevel 拐角、miter limit 10。
#[no_mangle]
pub extern "C" fn amiki_stroke_style_default() -> AmikiStrokeStyle {
    AmikiStrokeStyle::default()
}

/// 构造描边样式。`miter_limit` 取 Canvas/SVG 默认值 10。
#[no_mangle]
pub extern "C" fn amiki_stroke_style_new(
    width: f32,
    cap: i32,
    join: i32,
) -> AmikiStrokeStyle {
    AmikiStrokeStyle { width, cap, join, miter_limit: 10.0 }
}

#[no_mangle]
pub extern "C" fn amiki_filter_none() -> AmikiFilter {
    AmikiFilter { kind: AMIKI_FILTER_NONE, ..Default::default() }
}

#[no_mangle]
pub extern "C" fn amiki_filter_blur(sigma: f32) -> AmikiFilter {
    AmikiFilter { kind: AMIKI_FILTER_BLUR, sigma, ..Default::default() }
}

#[no_mangle]
pub extern "C" fn amiki_filter_drop_shadow(
    dx: f32,
    dy: f32,
    sigma: f32,
    color: AmikiColor,
) -> AmikiFilter {
    AmikiFilter { kind: AMIKI_FILTER_DROP_SHADOW, sigma, dx, dy, color }
}

#[no_mangle]
pub extern "C" fn amiki_paint_solid(color: AmikiColor) -> AmikiPaint {
    AmikiPaint { kind: AMIKI_PAINT_SOLID, color, gradient: std::ptr::null() }
}

/// 渐变 paint。**借用** `gradient`；引擎在绘制时会 `Arc::clone` 一份，
/// 所以绘制后你就可以销毁自己的渐变句柄（见 header 所有权第 4 条）。
///
/// `kind` **由渐变自身的几何决定**（线性 / 径向 / 扫描），不是调用方指定的。
/// 传空指针不会崩：返回一个"用到时必然报 `AMIKI_ERROR_INVALID_ARGUMENT`"的 paint。
#[no_mangle]
pub extern "C" fn amiki_paint_gradient(gradient: *const AmikiGradient) -> AmikiPaint {
    let kind = ffi_free!({
        if gradient.is_null() {
            AMIKI_PAINT_LINEAR
        } else {
            // SAFETY: 非空指针按契约来自 `amiki_gradient_create_*`，仍然有效。
            let g = unsafe { &*gradient };
            match g.gradient.kind {
                amiki_scene::GradientKind::Linear { .. } => AMIKI_PAINT_LINEAR,
                amiki_scene::GradientKind::Radial { .. } => AMIKI_PAINT_RADIAL,
                amiki_scene::GradientKind::Sweep { .. } => AMIKI_PAINT_SWEEP,
            }
        }
    });
    AmikiPaint { kind, color: AmikiColor::default(), gradient: gradient.cast() }
}
