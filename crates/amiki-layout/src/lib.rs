//! amiki-layout —— Amiki2D 的排版层：断行、行高、对齐、段落。
//!
//! # 定位（这一层为什么独立）
//!
//! ```text
//! 字符 → [amiki-text] shaping → 字形 + 位置（单行）
//!                    ↓
//!         [amiki-layout] 断行 / 行高 / 对齐 / 段落
//!                    ↓
//!         → Vec<TextRun>（每行一段）→ [引擎] 渲染
//! ```
//!
//! 渲染器只接收**已定位的字形**（`TextRun`），因此它不需要、也不该有排版概念。
//! 业界共识亦然：Skia 的 `SkTextBlob` 只承载预先定位好的字形，HarfBuzz 明确
//! 声明不做断行。本层**独立且可选** —— 只想画一行字的调用方不必引入它。
//!
//! # 输出仍然只是 `TextRun`
//!
//! 排版的结果是"每行一个 `TextRun` + 该行的基线位置"。于是排版层与渲染层之间
//! 的接口就是引擎已有的那个，没有新增耦合。
//!
//! # 断行策略
//!
//! 采用**贪心 + 可断点分类**：
//! - 空白（U+0020 等）之后可断；
//! - 连字符之后可断；
//! - **CJK 逐字可断**（汉字之间没有空格，必须允许字间断行）；
//! - 超长不可断片段（如长 URL）**溢出**而非硬切 —— 硬切会破坏内容语义，
//!   而溢出是可被调用方检测并处理的（见 `LayoutResult::overflow`）。
//!
//! 贪心不是最优（不做 Knuth-Plass 全局最优断行），但对绝大多数文本与 UI
//! 场景足够，且行为可预测。全局最优点是后续增强，不影响本层的接口。

#![forbid(unsafe_code)]

use amiki_math::Vec2;
use amiki_scene::TextRun;
use amiki_text::{shape, Font, FontMetrics};

/// 排版错误。
#[derive(Debug)]
pub enum LayoutError {
    /// 字体层错误（解析/shaping）
    Text(String),
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::Text(e) => write!(f, "文本层错误：{e}"),
        }
    }
}

impl std::error::Error for LayoutError {}

impl From<amiki_text::TextError> for LayoutError {
    fn from(e: amiki_text::TextError) -> Self {
        LayoutError::Text(format!("{e}"))
    }
}

/// 水平对齐方式。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Align {
    #[default]
    Left,
    Center,
    Right,
}

/// 一个可断点的种类（决定断行后是否"吃掉"该字符）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BreakKind {
    /// 不可断
    No,
    /// 空白后可断，且**断行时吞掉**该空白（不让它出现在行尾或行首）
    Space,
    /// 字符后可断，且断行时**保留**该字符（如连字符）
    Keep,
    /// CJK 等：字与字之间可断，不吞任何字符
    Anywhere,
}

/// 该字符之后是否可断，以及断点类型。
fn break_kind_after(c: char, prev_is_cjk: bool) -> BreakKind {
    if c == ' ' || c == '\t' || c == '\u{3000}' {
        return BreakKind::Space;
    }
    if c == '-' || c == '\u{2010}' {
        return BreakKind::Keep;
    }
    if is_cjk(c) || prev_is_cjk {
        // CJK 之间（含 CJK 与拉丁交界）逐字可断
        return BreakKind::Anywhere;
    }
    BreakKind::No
}

/// 是否为需要逐字断行的表意文字（CJK 及常见全角标点范围）。
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF      // 谚文字母
        | 0x2E80..=0x2EFF    // 部首补充
        | 0x3000..=0x303F    // CJK 标点
        | 0x3040..=0x30FF    // 平假名/片假名
        | 0x3130..=0x318F    // 谚文兼容
        | 0x3400..=0x4DBF    // 扩展 A
        | 0x4E00..=0x9FFF    // 基本汉字
        | 0xAC00..=0xD7AF    // 谚文音节
        | 0xF900..=0xFAFF    // 兼容汉字
        | 0xFF00..=0xFF60    // 全角形式
        | 0x20000..=0x2FA1F  // 扩展 B 及以后
    )
}

/// 排版参数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayoutOptions {
    /// 字号（设备像素）
    pub size: f32,
    /// 最大行宽（设备像素）；`f32::INFINITY` 表示不折行
    pub max_width: f32,
    /// 行距倍数（× 字体行高）。1.0 = 紧贴字体自身的行高
    pub line_spacing: f32,
    /// 段间距（设备像素），仅用于一次排版多段文本时
    pub paragraph_spacing: f32,
    pub align: Align,
}

impl LayoutOptions {
    pub fn new(size: f32, max_width: f32) -> LayoutOptions {
        LayoutOptions {
            size,
            max_width,
            line_spacing: 1.0,
            paragraph_spacing: 0.0,
            align: Align::Left,
        }
    }
}

impl Default for LayoutOptions {
    fn default() -> Self {
        LayoutOptions::new(16.0, f32::INFINITY)
    }
}

/// 排版结果中的一行。
#[derive(Clone, Debug)]
pub struct Line {
    /// 该行的字形（位置为**行内** em 单位，相对该行基线原点）
    pub run: TextRun,
    /// 相对排版原点（左上角）的**基线**偏移（设备像素）
    pub baseline: Vec2,
    /// 行宽（设备像素）
    pub width: f32,
    /// 该行在原文中的字符范围（用于调试与高亮）
    pub text_range: std::ops::Range<usize>,
}

/// 排版结果。
#[derive(Clone, Debug)]
pub struct LayoutResult {
    pub lines: Vec<Line>,
    /// 整个排版块的尺寸（设备像素）
    pub size: Vec2,
    /// 是否存在**无法放入 max_width 的片段**（如超长单词）。
    ///
    /// 排版不会硬切这类片段（硬切破坏内容语义），而是让它溢出。
    /// 调用方可用此标志决定换行策略或提示用户。
    pub overflow: bool,
}

impl LayoutResult {
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// 对一段文本做排版。
///
/// `text` 中的 `\n` 强制换行；`max_width` 为 `INFINITY` 时不自动折行。
pub fn layout(
    font: &Font,
    text: &str,
    opts: &LayoutOptions,
) -> Result<LayoutResult, LayoutError> {
    if text.is_empty() {
        return Ok(LayoutResult { lines: Vec::new(), size: Vec2::ZERO, overflow: false });
    }
    let metrics: FontMetrics = font.metrics()?;
    let line_h = metrics.line_height() * opts.line_spacing * opts.size;
    let baseline_in_line = metrics.ascent * opts.size;

    let mut lines: Vec<Line> = Vec::new();
    let mut overflow = false;

    // 逐段（以 \n 分段）；段内贪心断行
    let mut para_start = 0usize;
    let mut cursor_y = 0.0f32;
    for para in text.split('\n') {
        let para_len = para.len();
        let breaks = greedy_break(font, para, opts, &mut overflow);

        for (line_text, range) in breaks {
            let run = shape(font, line_text, opts.size)?;
            // 行宽 = 末字形位置 + 其前进量（更准确；shaping 已含 kerning）
            let width = measure_run(font, &run, opts.size);
            let x = match opts.align {
                Align::Left => 0.0,
                Align::Center => (opts.max_width - width) / 2.0,
                Align::Right => opts.max_width - width,
            };
            lines.push(Line {
                run,
                baseline: Vec2::new(x, cursor_y + baseline_in_line),
                width,
                text_range: (para_start + range.start)..(para_start + range.end),
            });
            cursor_y += line_h;
        }
        para_start += para_len + 1; // +1 吃掉 '\n'
        if opts.paragraph_spacing > 0.0 && para_start <= text.len() {
            cursor_y += opts.paragraph_spacing;
        }
    }

    let max_line_w = lines.iter().map(|l| l.width).fold(0.0f32, f32::max);
    let total_h = if lines.is_empty() { 0.0 } else { cursor_y };
    Ok(LayoutResult {
        lines,
        size: Vec2::new(max_line_w, total_h),
        overflow,
    })
}

/// 用 shaping 结果测行宽：末字形位置 + 其前进量。
///
/// 不用逐字符累加 advance —— 那样会丢掉 kerning 与连字带来的宽度变化。
fn measure_run(font: &Font, run: &TextRun, size: f32) -> f32 {
    let Some(last) = run.glyphs.last() else { return 0.0 };
    let adv = font.advance(last.id).unwrap_or(0.0);
    (last.x + adv) * size
}

/// 预处理：每个字符的 `(字节偏移, 字节长度, 其后断点类型)`，
/// 以及前缀宽度数组 `prefix[k]` = 前 k 个字符的总宽（设备像素）。
///
/// 用**前缀宽度**而不是"边累加边回退"：后者要靠减法还原断点处的宽度，
/// 容易在空白吞并、制表符等情况下算错，且很难验证。
/// 前缀数组让"任意切片宽度 = prefix[b] - prefix[a]"恒等成立。
struct CharMetrics {
    /// (字节偏移, 字节长度, 其后断点类型)
    chars: Vec<(usize, usize, BreakKind)>,
    /// prefix[k] = 前 k 个字符的累计宽度
    prefix: Vec<f32>,
}

fn measure_chars(font: &Font, para: &str, size: f32) -> CharMetrics {
    let mut chars = Vec::new();
    let mut prefix = vec![0.0f32];
    let mut prev_cjk = false;
    let mut acc = 0.0f32;
    for (i, c) in para.char_indices() {
        let w = if c == '\t' {
            font.advance_width("    ") * size
        } else {
            let mut buf = [0u8; 4];
            font.advance_width(c.encode_utf8(&mut buf)) * size
        };
        let kind = break_kind_after(c, prev_cjk);
        prev_cjk = is_cjk(c);
        chars.push((i, c.len_utf8(), kind));
        acc += w;
        prefix.push(acc);
    }
    CharMetrics { chars, prefix }
}

impl CharMetrics {
    fn n(&self) -> usize {
        self.chars.len()
    }
    /// 第 k 个字符的字节偏移（k == n 时返回段末）
    fn byte_at(&self, k: usize) -> usize {
        if k >= self.chars.len() {
            self.chars.last().map(|(b, l, _)| b + l).unwrap_or(0)
        } else {
            self.chars[k].0
        }
    }
    /// 字符区间 `[a, b)` 的宽度
    fn width(&self, a: usize, b: usize) -> f32 {
        let hi = b.min(self.prefix.len() - 1);
        let lo = a.min(hi);
        self.prefix[hi] - self.prefix[lo]
    }
    fn kind(&self, k: usize) -> BreakKind {
        self.chars.get(k).map(|(_, _, t)| *t).unwrap_or(BreakKind::No)
    }
}

/// 贪心断行：返回 `(该行文本, 该行在段内的字节范围)`。
///
/// 贪心地"尽可能多放"：从左往右吃字符，直到再加一个就超宽；此时回退到
/// **本行内最后一个可断点**处断行。若本行内根本没有可断点（超长单词），
/// 则让它**溢出**而不是硬切 —— 硬切破坏内容语义，溢出是可被调用方检测的。
///
/// # 为什么是贪心
///
/// Knuth-Plass 全局最优断行能给出更均匀的行宽，但需要为每个可断点定义惩罚值
/// 并做动态规划。对 UI 与大多数正文场景，贪心已足够，且**行为可预测**
/// （调试排版问题时这一点很重要）。全局最优是后续增强，不改变本层接口。
fn greedy_break<'a>(
    font: &Font,
    para: &'a str,
    opts: &LayoutOptions,
    overflow: &mut bool,
) -> Vec<(&'a str, std::ops::Range<usize>)> {
    let mut out = Vec::new();
    if para.is_empty() {
        // 空段仍占一行（保持 \n\n 的空行语义）
        out.push(("", 0..0));
        return out;
    }
    if opts.max_width.is_infinite() {
        out.push((para, 0..para.len()));
        return out;
    }

    let m = measure_chars(font, para, opts.size);
    let n = m.n();
    let mut start = 0usize; // 当前行的起始字符下标

    while start < n {
        // 本行最多能容纳到哪个字符下标（不含）
        let mut end = start;
        while end < n && m.width(start, end + 1) <= opts.max_width {
            end += 1;
        }

        if end == n {
            let (a, b) = (m.byte_at(start), m.byte_at(n));
            out.push((&para[a..b], a..b));
            break;
        }

        // 在本行 [start, end) 内找最后一个可断点
        let mut brk: Option<usize> = None;
        for k in (start + 1)..end {
            if m.kind(k - 1) != BreakKind::No {
                brk = Some(k);
            }
        }

        let Some(cut) = brk else {
            // 本行内**根本没有可断点**（超长不可断片段，如长 URL）。
            //
            // 这里绝不能退化成 `cut = end` —— 那正是**硬切**：会把 URL 截成两段，
            // 破坏内容语义（截断的 URL 就失效了）。初版正是这么写的，
            // 被"内容不得丢失 + 必须置 overflow"的测试抓出。
            // 正确做法：整段留到下一个可断点（或段尾）作为一行**溢出**。
            *overflow = true;
            let mut k = end;
            while k < n && m.kind(k - 1) == BreakKind::No {
                k += 1;
            }
            let (a, b) = (m.byte_at(start), m.byte_at(k));
            out.push((&para[a..b], a..b));
            start = k;
            continue;
        };

        // 空白处断行：**吞掉**空白（不留在行尾，也不放到行首）
        let mut trim = cut;
        while trim > start && m.kind(trim - 1) == BreakKind::Space {
            trim -= 1;
        }
        let (a, b) = (m.byte_at(start), m.byte_at(trim));
        // 空行保护：若 trim 退回到 start（整行都是空白），则至少吃到 cut
        if trim == start {
            let b2 = m.byte_at(cut);
            out.push((&para[a..b2], a..b2));
        } else {
            out.push((&para[a..b], a..b));
        }
        start = cut;
    }
    out
}
