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

//! API 稳定性：稳定面为 `layout` / `LayoutOptions` / `LayoutResult` / `Line` /
//! `Align` / `is_no_line_start` / `is_no_line_end`。
//! 完整划分见仓库 `docs/API.md`。

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

/// **行首禁则**：这些字符不得出现在行首（收尾类标点与右括号）。
///
/// 中日文排版的"禁则处理"（kinsoku shori）。不做的话，`「你好。」` 断行后
/// 可能变成下一行以 `」` 开头 —— 这是明显不合格的排版，但**只有中文内容
/// 才会暴露**，英文测试永远看不到。
pub fn is_no_line_start(c: char) -> bool {
    matches!(
        c,
        '。' | '、' | '，' | '．' | '！' | '？' | '：' | '；'
            | '）' | '〕' | '］' | '｝' | '〉' | '》' | '」' | '』' | '】' | '〗' | '〙' | '〛'
            | '｀' | '´' | '％' | '‰' | '°' | '℃' | '々' | 'ー' | '～' | '·'
    )
}

/// **行尾禁则**：这些字符不得出现在行尾（起始类括号与开引号）。
pub fn is_no_line_end(c: char) -> bool {
    matches!(
        c,
        '（' | '〔' | '［' | '｛' | '〈' | '《' | '「' | '『' | '【' | '〖' | '〘' | '〚'
            | '“' | '‘' | '＄' | '￥' | '＃'
    )
}

/// 在 `before` 与 `after` 之间断行是否**被禁则禁止**。
///
/// 禁则是**排版偏好而非硬约束**：当整行都找不到合规断点时，调用方会退回
/// "最后一个可断点"（宁可破坏禁则，也不要溢出成超长行）。
fn kinsoku_forbids_break(before: char, after: char) -> bool {
    is_no_line_start(after) || is_no_line_end(before)
}

/// 在一行的 `[start, end)` 内挑选断点。
///
/// 返回 `(选中的断点, 本行是否存在任何可断点)`：
/// - 优先选**满足禁则**的最后一个断点；
/// - 若可断点全被禁则否决，退回**最后一个可断点**（禁则不是硬约束）；
/// - `false` 表示本行根本没有可断点（调用方走"不可断片段"的溢出路径）。
///
/// 抽成**纯函数**是为了可测：禁则只与字符有关、与字体无关，
/// 而这个仓库里没有 CJK 字体，靠端到端排版是验证不了它的。
fn pick_break(kinds: &[(BreakKind, char)], start: usize, end: usize) -> (Option<usize>, bool) {
    let mut any: Option<usize> = None;
    let mut kinsoku_ok: Option<usize> = None;
    for k in (start + 1)..end {
        let Some(&(kind, _)) = kinds.get(k - 1) else { break };
        if kind == BreakKind::No {
            continue;
        }
        any = Some(k);
        let before = kinds.get(k - 1).map(|x| x.1).unwrap_or(' ');
        let after = kinds.get(k).map(|x| x.1).unwrap_or(' ');
        if !kinsoku_forbids_break(before, after) {
            kinsoku_ok = Some(k);
        }
    }
    (kinsoku_ok.or(any), any.is_some())
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
    /// (字节偏移, 字节长度, 其后断点类型, 字符本身)
    ///
    /// 带上字符本身是因为**禁则要判断断点两侧是什么字**（如 `」` 不得在行首）。
    chars: Vec<(usize, usize, BreakKind, char)>,
    /// prefix[k] = 前 k 个字符的累计宽度
    prefix: Vec<f32>,
}

fn measure_chars(font: &Font, para: &str, size: f32) -> CharMetrics {
    // ① 字符表与断点类型（与宽度无关，先算）
    let mut chars = Vec::new();
    let mut prev_cjk = false;
    for (i, c) in para.char_indices() {
        let kind = break_kind_after(c, prev_cjk);
        prev_cjk = is_cjk(c);
        chars.push((i, c.len_utf8(), kind, c));
    }

    // ② 前缀宽度**必须来自 shaping 的结果**，而不是逐字符累加 `advance_width`。
    //
    // 逐字符累加会丢掉**字距与连字**：`AV` 渲染时 V 会向左收（Inter 实测
    // 0.0684 em），而逐字符累加把它算成 0.69 em。后果是**排版宽度与渲染宽度
    // 不一致** —— 断行偏保守、`LayoutResult::size` 与画面也对不上。
    //
    // 做法：整段 shaping 一次，用字形自带的 `cluster`（字节偏移）把推进量映射回
    // 字符边界。连字会让多个字符落在同一 cluster，于是它们的前缀宽度相同 ——
    // 这正是断行需要的语义（连字不可拆）。
    let mut prefix = vec![0.0f32; chars.len() + 1];
    match amiki_text::shape_clustered(font, para) {
        Ok(glyphs) if !glyphs.is_empty() => {
            // 段尾宽度 = 末字形起点 + 其推进量（推进量要查字体，TextRun 不带）
            let last = glyphs[glyphs.len() - 1];
            let end_pen = (last.x + font.advance(last.id).unwrap_or(0.0)) * size;
            // 每个字形的**起点**推进量即它 cluster 处的边界宽度
            let mut gi = 0usize;
            for (k, &(off, _, _, _)) in chars.iter().enumerate() {
                let off = off as u32;
                // 沿字形前进，直到找到一个 cluster ≥ 当前字符偏移的字形
                while gi < glyphs.len() && glyphs[gi].cluster < off {
                    gi += 1;
                }
                prefix[k] = if gi < glyphs.len() { glyphs[gi].x * size } else { end_pen };
            }
            // 段尾宽度：末字形起点 + 其推进量
            prefix[chars.len()] = end_pen;
        }
        // shaping 失败：**回退到逐字符累加**并明确记录（不静默给出不同结果）
        _ => {
            eprintln!("amiki-layout：shaping 失败，回退到逐字符 advance（字距与连字不参与排版）");
            let mut acc = 0.0f32;
            for (k, &(off, len, _, _)) in chars.iter().enumerate() {
                let c = para[off..off + len].chars().next().unwrap_or(' ');
                let w = if c == '\t' {
                    font.advance_width("    ") * size
                } else {
                    let mut buf = [0u8; 4];
                    font.advance_width(c.encode_utf8(&mut buf)) * size
                };
                acc += w;
                prefix[k + 1] = acc;
            }
        }
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
            self.chars.last().map(|(b, l, _, _)| b + l).unwrap_or(0)
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
        self.chars.get(k).map(|(_, _, t, _)| *t).unwrap_or(BreakKind::No)
    }

    /// `(断点类型, 字符)` 视图 —— `pick_break` 只需要这两样。
    fn chars_kinds(&self) -> Vec<(BreakKind, char)> {
        self.chars.iter().map(|(_, _, k, c)| (*k, *c)).collect()
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

        // 在本行 [start, end) 内挑断点（含禁则处理）
        let (brk, has_any) = pick_break(&m.chars_kinds(), start, end);

        debug_assert_eq!(has_any, brk.is_some());
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

#[cfg(test)]
mod kinsoku_tests {
    use super::*;

    /// 把字符串变成 `pick_break` 需要的 `(断点类型, 字符)` 数组。
    ///
    /// 这里**不涉及字体** —— 禁则只与字符有关，所以能在没有 CJK 字体的
    /// 仓库里被测到。端到端的中文排版验证仍需要一份 CJK 字体，
    /// 那是 `assets/fonts/` 里目前缺的东西。
    fn kinds(text: &str) -> Vec<(BreakKind, char)> {
        let mut out = Vec::new();
        let mut prev_cjk = false;
        for c in text.chars() {
            out.push((break_kind_after(c, prev_cjk), c));
            prev_cjk = is_cjk(c);
        }
        out
    }

    /// 在 `text` 的 `[0, end)` 范围内挑断点，返回选中的位置。
    ///
    /// 断点 `k` 表示"在第 k 个字符**之前**断行"。
    fn chosen(text: &str, end: usize) -> Option<usize> {
        let ks = kinds(text);
        pick_break(&ks, 0, end).0
    }

    /// 行首禁则：**被禁的断点恰好是"最后一个候选"**，因此开/关禁则会给出不同结果。
    ///
    /// `你好。」世` 在 `end = 4`（即只排到 `」`）时有三个候选：
    /// `k=1`（好之后，合规）、`k=2`（。之后 → 下一行以 `。` 开头，禁）、
    /// `k=3`（」之后 → 下一行以 `」` 开头，禁）。
    /// 无禁则时会取最后一个（3）；有禁则时退到最后一个**合规**的（1）。
    #[test]
    fn closing_punctuation_never_starts_a_line() {
        assert_eq!(
            chosen("你好。」世", 4),
            Some(1),
            "`。`/`」` 不得出现在行首：应退到 `好` 之后，而不是 `」` 之后"
        );
    }

    /// 行尾禁则：同样让被禁断点成为最后一个候选。
    ///
    /// `你好「世` 在 `end = 4` 时候选为 `k=1`、`k=2`、`k=3`；
    /// `k=3` 是「`「` 之后」→ 本行以 `「` 结尾，禁。有禁则时应取 `k=2`。
    #[test]
    fn opening_bracket_never_ends_a_line() {
        assert_eq!(
            chosen("你好「世", 4),
            Some(2),
            "`「` 不得出现在行尾：应退到 `好` 之后，而不是 `「` 之后"
        );
    }

    #[test]
    fn kinsoku_is_a_preference_not_a_hard_rule() {
        // 整行都是禁则字符时，不能因为禁则就**拒绝断行** ——
        // 那会让行溢出画面。此时退回最后一个可断点。
        let ks = kinds("。。。");
        let (chosen, any) = pick_break(&ks, 0, ks.len());
        assert!(any, "应有可断点");
        assert!(chosen.is_some(), "禁则不是硬约束：找不到合规断点时应退回可断点");
    }

    #[test]
    fn no_breakable_position_reports_none() {
        // 纯拉丁、无空格与连字符：没有任何可断点，调用方应走溢出路径
        let ks = kinds("abcdef");
        let (chosen, any) = pick_break(&ks, 0, ks.len());
        assert!(!any, "不应存在可断点");
        assert_eq!(chosen, None);
    }

    #[test]
    fn kinsoku_predicates_cover_the_common_sets() {
        for c in ['。', '、', '，', '！', '？', '」', '』', '）', '】', '》'] {
            assert!(is_no_line_start(c), "`{c}` 应在行首禁则集合里");
        }
        for c in ['「', '『', '（', '【', '《', '“'] {
            assert!(is_no_line_end(c), "`{c}` 应在行尾禁则集合里");
        }
        // 普通汉字不受禁则约束
        assert!(!is_no_line_start('好') && !is_no_line_end('好'));
        // 两类集合不应相交（同一个字不能既禁行首又禁行尾）
        for c in ['。', '、', '」', '「', '（'] {
            assert!(
                !(is_no_line_start(c) && is_no_line_end(c)),
                "`{c}` 不应同时属于两个禁则集合"
            );
        }
    }

    #[test]
    fn mixed_cjk_latin_breaks_between_cjk_chars() {
        // 「中文abc中文」：CJK 之间逐字可断（这是 Anywhere 的语义）
        let ks = kinds("中文");
        let (chosen, any) = pick_break(&ks, 0, ks.len());
        assert!(any, "CJK 之间应可断");
        assert_eq!(chosen, Some(1), "应在两字之间断");
    }
}
