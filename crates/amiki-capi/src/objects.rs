//! C 侧不透明句柄背后的真实对象，以及**中毒**机制。
//!
//! # 为什么要有「中毒」
//!
//! panic 可能发生在一次修改的中途，对象内部状态未必自洽。继续用它
//! 有可能**静默画出错误结果** —— 而"不静默出错"是本项目最硬的一条取舍。
//! 所以捕获 panic 之后把对象标记为中毒，后续调用一律立刻返回
//! `AMIKI_ERROR_PANIC`，调用方只能 `_destroy` 后重建。
//!
//! 这条语义写在 `include/amiki.h` 的 `AMIKI_ERROR_PANIC` 一节里，是契约的一部分。

use amiki_scene::{BlendMode, Filter, Gradient, Image, LayerBounds, Path};
use std::ffi::CString;
use std::sync::Arc;

use crate::types::AmikiLayerBounds;

/// 所有句柄共用的中毒标志。
///
/// `true` 之后这个对象的每个入口都直接返回 [`crate::types::AMIKI_ERROR_PANIC`]。
#[derive(Default)]
pub(crate) struct Poison {
    poisoned: bool,
}

impl Poison {
    /// 已中毒则返回 `Err(PANIC)`，调用方应立刻返回它。
    #[inline]
    pub(crate) fn check(&self) -> Result<(), i32> {
        if self.poisoned {
            Err(crate::types::AMIKI_ERROR_PANIC)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub(crate) fn mark(&mut self) {
        self.poisoned = true;
    }

    #[inline]
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

// ============================================================================
// 上下文
// ============================================================================

/// 渲染后端。**GPU 渲染器在这里只创建一次**，跨帧复用。
///
/// 这一点是 C ABI 相对 `amiki::render_to_rgba` 的关键改进：后者每次都
/// `GpuRenderer::new()`（实测冷启动约 61 ms），逐帧调用会让它成为主要开销。
///
/// 0.10.9 起用 `Arc` 而不是 `Box`：异步入口要调
/// `GpuRenderer::begin_with(self: &Arc<Self>, ...)`（在飞帧自己持一份渲染器引用，
/// 于是它没有生命周期、可以放进画布状态里）。
pub(crate) enum Backend {
    /// wgpu 后端。`gpu` feature 关掉时**这个变体不存在** ——
    /// 于是"误用 GPU"会变成**编译错误**而不是运行时错误。
    #[cfg(feature = "gpu")]
    Gpu(Arc<amiki_gpu::GpuRenderer>),
    /// CPU 引擎（`amiki-raster`）。**零第三方依赖**，因此 wasm32 上也能跑。
    Cpu,
}

pub struct AmikiContext {
    pub(crate) backend: Backend,
    /// 实际适配器名。**静态字符串的持有者** —— 它必须活到 context 销毁，
    /// 因为 `amiki_context_adapter_name` 返回的是它的裸指针。
    pub(crate) adapter_name: CString,
    pub(crate) poison: Poison,
}

// ============================================================================
// 画布
// ============================================================================

/// 一个**打开中的**离屏层。
///
/// C 侧没有闭包，所以 `push_layer_filtered(|c| ...)` 被拆成
/// `begin_layer` / `end_layer`：`begin` 时压入这里的子画布，
/// 之后的绘制都路由到它，`end` 时把整个子场景作为一个 layer 挂到父场景上。
pub(crate) struct PendingLayer {
    pub(crate) canvas: amiki::Canvas,
    pub(crate) bounds: LayerBounds,
    pub(crate) filter: Filter,
    pub(crate) opacity: f32,
    pub(crate) blend: BlendMode,
}

pub struct AmikiCanvas {
    /// **借用**，不拥有：`include/amiki.h` 要求 context 比 canvas 活得久。
    pub(crate) ctx: *mut AmikiContext,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) eps: f32,
    pub(crate) root: amiki::Canvas,
    pub(crate) layers: Vec<PendingLayer>,
    /// 裁剪栈深度。**由本层自己跟踪**：引擎的 `Canvas` 没暴露 `clip_stack`，
    /// 而"弹多了"必须显式报错（引擎那边是静默空操作）。
    pub(crate) clip_depth: usize,
    /// 输出量化时用的色调映射（`amiki_canvas_set_tone_map`）。
    /// 只影响 `render_rgba8` 的**出口**，不影响绘制本身。
    pub(crate) tone_map: amiki_scene::ToneMap,
    /// mip 采样偏移（`amiki_canvas_set_mip_bias`，B4）。
    /// 影响缩小位图的采样层级；0 = 不偏移。
    pub(crate) mip_bias: f32,
    /// 填充规则（`amiki_canvas_set_fill_rule`，0.8.0）。状态式、粘性：
    /// 影响**后续**填充，直到再次设置。默认 `NonZero`。
    pub(crate) fill_rule: amiki_scene::FillRule,
    /// **在飞的那一帧**（异步入口，P3 第 ③b 步）。`None` = 没有提交过 / 已经取走。
    ///
    /// 挂在画布上而不是做成新的不透明句柄：C 侧没有所有权语义，多一个句柄就要多一个
    /// `_destroy`（符号 +1，且忘了销毁就是泄漏）。取完自动清空，**没有泄漏路径**。
    pub(crate) pending: Option<PendingCanvasRender>,
    pub(crate) poison: Poison,
}

/// 异步渲染提交后**在飞的那一帧**（P3 第 ③b 步，C ABI 侧的状态）。
///
/// 与 `amiki::PendingRender` 的区别：这里的 CPU 分支带上画布自己的**色调映射**
/// 与 **mip bias**，与同步入口 `amiki_canvas_render_rgba8` 用的是同一份实现 ——
/// "两条入口逐字节相同"这条承诺的根据就在这里。
pub(crate) enum PendingCanvasRender {
    /// **已经算好**（CPU 后端；或 GPU 超出已验证范围而回落 CPU）⇒ 首帧即就绪。
    Ready(Vec<u8>),
    /// GPU 上真在飞。`PendingFrame` 自己持 `Arc<GpuRenderer>`，所以无生命周期参数。
    #[cfg(feature = "gpu")]
    InFlight(amiki_gpu::PendingFrame),
}

impl AmikiCanvas {
    /// 当前绘制目标：最内层打开中的层，或根画布。
    pub(crate) fn target(&mut self) -> &mut amiki::Canvas {
        match self.layers.last_mut() {
            Some(l) => &mut l.canvas,
            None => &mut self.root,
        }
    }

    /// 当前层的范围（用于 `begin_layer` 时把子画布建得和父画布一样大 ——
    /// 层的 `bounds` 只决定**合成范围**，不决定子场景尺寸）。
    pub(crate) fn reset_state(&mut self) {
        self.root = amiki::Canvas::new(self.width, self.height);
        self.layers.clear();
        self.clip_depth = 0;
        self.tone_map = amiki_scene::ToneMap::Raw;
        self.mip_bias = 0.0;
        self.fill_rule = amiki_scene::FillRule::NonZero;
        self.root.set_fill_rule(amiki_scene::FillRule::NonZero);
        // 清空画布 = 连在飞的那一帧一起丢（它算的是旧内容，留着只会让调用方取到过期像素）
        self.pending = None;
        self.poison = Poison::default();
    }
}

// ============================================================================
// 资源句柄
// ============================================================================

pub struct AmikiPath {
    pub(crate) path: Path,
    pub(crate) poison: Poison,
}

pub struct AmikiImage {
    /// 引用计数：绘制时 `Arc::clone` 给场景一份，于是 C 侧句柄
    /// **可以在绘制后立刻销毁**（契约见 header 所有权第 4 条）。
    pub(crate) image: Arc<Image>,
    pub(crate) poison: Poison,
}

pub struct AmikiGradient {
    pub(crate) gradient: Arc<Gradient>,
    pub(crate) poison: Poison,
}

pub struct AmikiFont {
    /// 字体字节在 `Font` 内部持有（`from_bytes` 会拿走 `Vec<u8>`），
    /// 所以 C 侧的输入缓冲用完即可释放。
    pub(crate) font: amiki_text::Font,
    pub(crate) poison: Poison,

    /// **每个字体句柄自己的字形轮廓缓存**（0.10.6，解决 `KNOWN_ISSUES` 的 D8）。
    ///
    /// 为什么不新增"缓存句柄"：C 侧调用方给的就是 `const amiki_font*` —— 一个
    /// **稳定的字体身份**。把缓存挂在字体上，键天然是 `(字体, 字形 id, 字号)`
    /// ⇒ **构造上不可能把 A 字体的字形当成 B 字体的**（D8 担心的静默错误），
    /// 而且**不需要任何新 API**：`amiki_canvas_draw_text` 自动受益。
    ///
    /// 用 `Mutex` 是因为绘制入口收的是 `const amiki_font*`；锁只在一次绘制期间
    /// 持有（无竞争）。锁中毒时取回内部值继续，**不 panic**。
    pub cache: std::sync::Mutex<amiki_scene::TextCache>,
}

/// **有序字体栈**（字体回退，0.7.0）。
///
/// 与 [`AmikiFont`] 的关系：`amiki_font_stack_push` 把字体**克隆进栈**，
/// 而 `amiki_text::Font` 内部用 `Arc` 共享字节 —— 所以压栈不复制那几 MB 数据，
/// 且**句柄仍归调用方**（`amiki_font_destroy` 该调还得调）。
pub struct AmikiFontStack {
    pub(crate) stack: amiki_text::FontStack,
    pub(crate) poison: Poison,
}

impl AmikiFontStack {
    pub(crate) fn new() -> AmikiFontStack {
        AmikiFontStack { stack: amiki_text::FontStack::new(), poison: Poison::default() }
    }
}

// ============================================================================
// 从 `AmikiLayerBounds`（C 值类型）到内部用的转换
// ============================================================================

impl From<AmikiLayerBounds> for LayerBounds {
    fn from(b: AmikiLayerBounds) -> Self {
        LayerBounds { x: b.x, y: b.y, width: b.width, height: b.height }
    }
}
