//! 字体栈（**字体回退**）与 shaping 选项。
//!
//! # 为什么需要它
//!
//! 一段文本里的字符不总在同一个字体里：拉丁字母在 Inter 里，汉字在 Noto Sans SC 里。
//! 此前 [`crate::shape`] 只吃**一个**字体，于是"支持 CJK"实际上要求调用方自己为
//! 每段文本挑对字体 —— 挑错了（比如只加载 Inter）就整段画成方框。
//!
//! 本模块把"按字符挑字体"交给引擎：调用方给出一个**有序**的字体列表，
//! 引擎按序解析、逐段 shaping。
//!
//! # 设计取舍
//!
//! - **顺序显式**：`resolve` 返回**第一个**含该字形的字体下标，不做任何
//!   "按脚本优先级猜"的隐式规则 —— 谁先谁赢，可预测、可测试。
//! - **缺字不静默**：默认 [`MissingGlyphPolicy::Error`]。方框在很多界面里
//!   "看起来像是对的"，所以这个决定必须由调用方做（见 [`TextOptions`]）。
//! - **零破坏**：`shape` / `layout` / `expand_text_run` 等老入口全部保留、
//!   行为不变；新能力只走 `*_with_stack` 入口。

use crate::Font;

/// 缺字（字体栈里**没有任何**字体含该字符）的处理策略。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MissingGlyphPolicy {
    /// 默认：返回 [`crate::TextError::MissingGlyph`]（payload 是**码点**）。
    ///
    /// 符合本项目"不支持就显式报错，不静默画错"的一贯取舍。
    #[default]
    Error,
    /// 用 `.notdef`（字形 0）替代：画出一个方框，位置与前进量按该字体算。
    ///
    /// 未解析的字符会被归到**栈里第一个字体**（顺序即优先级）。
    NotDef,
    /// 跳过该字符：不产生字形，前进量为 0，后续字形紧接着排。
    Skip,
}

/// shaping 选项。
///
/// 默认值与 0.6.1 的单字体路径**一致**（字距与连字都开）；
/// 唯一的差别是缺字策略：栈路径默认 `Error`，而老单字体路径保持
/// "交给 shaping 映射成 `.notdef`"（见 [`TextOptions::LEGACY`]）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextOptions {
    pub missing_glyph: MissingGlyphPolicy,
    /// 启用字距调整（`kern`）。默认开。
    pub kerning: bool,
    /// 启用连字（`liga` / `clig`）。默认开。
    pub ligatures: bool,
}

impl Default for TextOptions {
    fn default() -> Self {
        TextOptions {
            missing_glyph: MissingGlyphPolicy::Error,
            kerning: true,
            ligatures: true,
        }
    }
}

impl TextOptions {
    /// 与 0.6.1 单字体路径等价的选项：缺字交给 shaping 映射成 `.notdef`。
    ///
    /// 单字体栈 + 这套选项 ⇒ [`crate::shape_with_stack`] 直接走老的
    /// [`crate::shape`]，因此结果**逐字节一致**（这也是"零破坏"的可执行证据）。
    /// 从老 API 迁过来的调用方可以直接用它。
    pub const LEGACY: TextOptions = TextOptions {
        missing_glyph: MissingGlyphPolicy::NotDef,
        kerning: true,
        ligatures: true,
    };

    pub fn with_missing_glyph(mut self, policy: MissingGlyphPolicy) -> Self {
        self.missing_glyph = policy;
        self
    }

    pub fn without_kerning(mut self) -> Self {
        self.kerning = false;
        self
    }

    pub fn without_ligatures(mut self) -> Self {
        self.ligatures = false;
        self
    }
}

/// **有序**字体栈：按顺序解析，第一个含该字形的字体胜出。
///
/// 不持有 `&Font` 而是持有 `Font` 本身：字体数据（`Vec<u8>`）在 [`Font`] 里，
/// 借用会让生命周期爬到调用方的每个 API 上。若确实只想借用，请调用方自己
/// 用 `Vec<Font>` + 索引管理。
pub struct FontStack {
    fonts: Vec<Font>,
}

impl FontStack {
    pub fn new() -> FontStack {
        FontStack { fonts: Vec::new() }
    }

    /// 单字体栈：给"只有一个字体"的调用方用（行为等价于老 API）。
    pub fn single(font: Font) -> FontStack {
        FontStack { fonts: vec![font] }
    }

    /// 追加一个字体，返回它在栈中的下标（**顺序即优先级**）。
    pub fn push(&mut self, font: Font) -> usize {
        self.fonts.push(font);
        self.fonts.len() - 1
    }

    pub fn len(&self) -> usize {
        self.fonts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fonts.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&Font> {
        self.fonts.get(idx)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Font> {
        self.fonts.iter()
    }

    /// 栈中每个字体的内部 ID（顺序与下标一致）。
    pub fn ids(&self) -> Vec<amiki_scene::FontId> {
        self.fonts.iter().map(|f| f.id()).collect()
    }

    /// 返回**第一个**含该字形的字体下标；都不含则 `None`。
    pub fn resolve(&self, codepoint: char) -> Option<usize> {
        self.fonts.iter().position(|f| f.glyph_index(codepoint).is_some())
    }

    /// 同 [`FontStack::resolve`]，但用**已经解析好的 face** 做查询。
    ///
    /// `faces` 必须与栈内字体**一一对应**（`faces[i]` 属于 `self.fonts[i]`）。
    /// shaping 内部走这条路：一个字符一次 cmap 查询，若每次都重新解析字体表，
    /// 长文本会白花掉几十次解析。
    pub(crate) fn resolve_in(
        &self,
        faces: &[ttf_parser::Face<'_>],
        codepoint: char,
    ) -> Option<usize> {
        debug_assert_eq!(faces.len(), self.fonts.len(), "faces 与字体栈必须一一对应");
        faces.iter().position(|f| f.glyph_index(codepoint).is_some())
    }

    /// 栈内字体的切片（`crate` 内部用：shaping 需要访问字体数据）。
    pub(crate) fn fonts(&self) -> &[Font] {
        &self.fonts
    }
}

impl Default for FontStack {
    fn default() -> Self {
        FontStack::new()
    }
}

impl std::fmt::Debug for FontStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Font` 不实现 Debug（它只是一袋字节 + 几个数），这里给身份信息就够定位问题。
        f.debug_struct("FontStack")
            .field("len", &self.fonts.len())
            .field("ids", &self.ids())
            .finish()
    }
}
