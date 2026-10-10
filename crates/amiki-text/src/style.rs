//! 字体**样式属性**与"按样式取最近字体"的取舍（0.10.7）。
//!
//! 单独一个模块、并用**扩展 impl** 给 `Font` 加方法：既有 `Font` 的 impl 很完整，
//! 往中间插代码风险高（本项目为此栽过两次）。Rust 允许同 crate 内任意模块写
//! `impl Font`，于是这里可以**零侵入**地加能力。
//!
//! 分工要说清（两者不是一回事）：
//! - **字体回退**（`FontStack::resolve_in`）问的是"这个**字形**谁有"；
//! - **样式匹配**（本模块）问的是"我要的**字重/斜体**谁最接近"。

use crate::Font;

/// 字体的**样式属性**：字重（100..900，400 = Regular、700 = Bold）与是否斜体。
///
/// 取值约定同 OpenType（读 `OS/2` 表的 `usWeightClass` 与 `fsSelection` 斜体位）；
/// 表缺失时兜底为常规值 —— 那是"字体没声明样式"，不是猜测。
///
/// **不做合成加粗/合成倾斜**：那要改字形轮廓，是另一件事，且容易与"任意缩放都不糊"
/// 的既有性质冲突。字体文件里没有的样式，将来是"按距离选最近的"或"显式报错"，
/// 而不是**伪造**。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct FontStyle {
    /// 字重（100..900）
    pub weight: u16,
    /// 是否斜体（含 Oblique）
    pub italic: bool,
}

impl FontStyle {
    pub const REGULAR: FontStyle = FontStyle { weight: 400, italic: false };
    pub const BOLD: FontStyle = FontStyle { weight: 700, italic: false };
    pub const ITALIC: FontStyle = FontStyle { weight: 400, italic: true };
    pub const BOLD_ITALIC: FontStyle = FontStyle { weight: 700, italic: true };

    /// 与"想要的样式"的**距离**（越小越接近）；`0` = 完全命中。
    ///
    /// # 取舍（写在明处，且有测试固定）
    ///
    /// `distance = |字重差| + （候选或目标任一方为斜体而另一方不是 ? 1000 : 0）`
    ///
    /// **斜体不匹配比字重差更严重**：用正体冒充斜体在排版上更刺眼，而请求 Regular
    /// 却拿到 SemiBold 只是略重。字重差最大 800（100..900），所以"斜体命中但字重差最大"
    /// 依然**优先于**"字重完全一致但斜体不符"。
    pub fn distance(&self, want: FontStyle) -> u32 {
        let dw = (self.weight as i32 - want.weight as i32).unsigned_abs();
        let di = if self.italic == want.italic { 0 } else { 1000 };
        dw + di
    }

    /// 是否比 `other` 更接近 `want`（相等时返回 `false`，便于稳定地保留先出现的那个）。
    pub fn is_closer_than(&self, other: FontStyle, want: FontStyle) -> bool {
        self.distance(want) < other.distance(want)
    }
}

impl Font {
    /// 从字体文件读出的样式（`OS/2` 表；缺失时兜底为 `REGULAR`）。
    pub fn style(&self) -> FontStyle {
        let Ok(face) = self.face() else {
            return FontStyle::REGULAR;
        };
        FontStyle {
            weight: face.weight().to_number(),
            italic: face.is_italic(),
        }
    }

    /// 字重（100..900）。
    pub fn weight(&self) -> u16 {
        self.style().weight
    }

    /// 是否斜体。
    pub fn is_italic(&self) -> bool {
        self.style().italic
    }
}
