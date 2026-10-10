//! amiki-text —— Amiki2D 的文本层：字体解析与 shaping。
//!
//! # 定位
//!
//! 本 crate 是项目中**唯一**引入第三方依赖的地方（`ttf-parser` + `rustybuzz`）。
//! 它把文本转成引擎原生认识的 [`amiki_scene::GlyphOutline`]（即 `PathEl` 序列），
//! 于是**排版与渲染彻底解耦**：
//!
//! ```text
//! 文本 + 字体 → [本 crate] → em 单位轮廓（PathEl）→ 既有管线（压平/分箱/解析覆盖率）
//! ```
//!
//! 引擎侧因此不需要任何文本专用代码，字体栈也可替换。
//!
//! # 为什么字形是轮廓而不是位图
//!
//! 传统方案预渲染字形位图（glyph atlas），一放大就糊。本引擎把轮廓直接喂给
//! 解析式覆盖率光栅器，**任意缩放下边缘都是精确面积积分** —— 这是架构带来的
//! 质量优势，不是额外优化。

//! API 稳定性：稳定面为 `Font` / `shape` / `shape_clustered` / `ShapedGlyph` /
//! `TextRun` / `TextError`；0.7.0 起新增 `FontStack` / `FontId` /
//! `MissingGlyphPolicy` / `TextOptions` / `shape_with_stack`（**新入口**，
//! 老入口签名与行为都不变）。完整划分见仓库 `docs/API.md`。

#![forbid(unsafe_code)]

mod stack;

pub use amiki_scene::FontId;
pub use stack::{FontStack, MissingGlyphPolicy, TextOptions};

use amiki_math::Vec2;
use amiki_scene::{GlyphOutline, PathEl, TextRun};

/// 文本层错误。
#[derive(Debug)]
pub enum TextError {
    /// 字体数据无法解析
    FontParse(String),
    /// 请求的字形或字符不存在。
    ///
    /// payload 的含义随调用路径而定：[`Font::outline`] / [`Font::advance`] 里是
    /// **字形 id**；[`shape_with_stack`] 里是**码点**（该字符在字体栈里找不到）。
    MissingGlyph(u32),
}

impl std::fmt::Display for TextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextError::FontParse(e) => write!(f, "字体解析失败：{e}"),
            TextError::MissingGlyph(g) => write!(f, "字形缺失：{g}"),
        }
    }
}

impl std::error::Error for TextError {}

/// 一个已加载的字体面。
///
/// 持有字体数据的**所有权**——因为 `ttf_parser::Face` 借用底层数据，
/// 自引用结构在安全 Rust 里无法直接表达，故这里按需创建 `Face`。
///
/// 字节用 `Arc` 共享，因此 [`Font`] 是**廉价可克隆**的：把一个已加载的字体
/// 同时放进字体栈和单字体路径不会复制那几 MB 数据（C ABI 的
/// `amiki_font_stack_push` 正依赖这一点）。
#[derive(Clone)]
pub struct Font {
    data: std::sync::Arc<Vec<u8>>,
    /// 每 em 的字体单位数（用于把字体单位归一化为 em）
    units_per_em: f32,
    glyph_count: u32,
    /// **内部稳定标识**（字体自己声明的名字，见 [`Font::id`]）
    id: FontId,
    /// `id` 是从哪儿来的（诊断用：`PostScript 名` 等）
    id_name: String,
}

impl std::fmt::Debug for Font {
    /// 手写：默认实现会把几 MB 字节全打出来。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Font")
            .field("id", &self.id)
            .field("id_name", &self.id_name)
            .field("units_per_em", &self.units_per_em)
            .field("glyph_count", &self.glyph_count)
            .field("bytes", &self.data.len())
            .finish()
    }
}

impl Font {
    /// 从字体文件字节加载。
    ///
    /// 支持 TrueType / OpenType。**不支持** `.ttc` 集合与 `.woff/.woff2`
    /// （前者需按索引取面，后者是压缩网页格式）——按需再扩展。
    pub fn from_bytes(data: Vec<u8>) -> Result<Font, TextError> {
        let face = ttf_parser::Face::parse(&data, 0)
            .map_err(|e| TextError::FontParse(format!("{e:?}")))?;
        let units_per_em = face.units_per_em() as f32;
        let glyph_count = face.number_of_glyphs() as u32;
        let (id, id_name) = font_identity(&face, &data);
        Ok(Font {
            data: std::sync::Arc::new(data),
            units_per_em: units_per_em.max(1.0),
            glyph_count,
            id,
            id_name,
        })
    }

    /// 从文件加载。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Font, TextError> {
        let data = std::fs::read(path)
            .map_err(|e| TextError::FontParse(format!("读取字体文件失败：{e}")))?;
        Font::from_bytes(data)
    }

    /// 字体的**内部稳定标识** —— 与路径、加载顺序都无关。
    ///
    /// 取字体自己声明的名字，优先级：PostScript 名（`name` 表 ID 6）→
    /// Unique ID（ID 3）→ Full name（ID 4）→ Family + Subfamily（ID 1+2）→
    /// 字体字节哈希（都没有时）。字形缓存用它分键，所以"同一份字体"必须
    /// 稳定得到同一个 ID：换目录、换文件名、重排字体栈都不影响。
    pub fn id(&self) -> FontId {
        self.id
    }

    /// [`Font::id`] 的来源描述（如 `PostScript 名 "Inter-Regular"`）——诊断用。
    pub fn id_name(&self) -> &str {
        &self.id_name
    }

    /// 每 em 的字体单位数。
    pub fn units_per_em(&self) -> f32 {
        self.units_per_em
    }

    /// 字形总数。
    pub fn glyph_count(&self) -> u32 {
        self.glyph_count
    }

    /// 字符 → 字形 id（未经 shaping 的简单映射）。
    ///
    /// **每次调用都会解析一遍字体表**。要按字符批量查询（字体回退的解析阶段），
    /// 请用 `FontStack::resolve_in`，它复用一次解析的结果。
    pub fn glyph_index(&self, c: char) -> Option<u32> {
        let face = ttf_parser::Face::parse(&self.data, 0).ok()?;
        face.glyph_index(c).map(|g| g.0 as u32)
    }

    pub(crate) fn face(&self) -> Result<ttf_parser::Face<'_>, TextError> {
        ttf_parser::Face::parse(&self.data, 0)
            .map_err(|e| TextError::FontParse(format!("{e:?}")))
    }

    /// 取字形轮廓，坐标归一化为 **em 单位**（1.0 = 字号），y 轴向下。
    ///
    /// 字体坐标系 y 向上，而引擎与屏幕坐标一致（y 向下），故此处取负翻转 ——
    /// 转换只做一次，之后所有下游代码都只面对一种朝向。
    pub fn outline(&self, glyph: u32) -> Result<GlyphOutline, TextError> {
        let face = self.face()?;
        let mut b = OutlineToPathEl::new(self.units_per_em);
        let gid = ttf_parser::GlyphId(glyph as u16);
        face.outline_glyph(gid, &mut b)
            .ok_or(TextError::MissingGlyph(glyph))?;
        Ok(GlyphOutline::new(b.els))
    }
}

/// 把 `ttf_parser::OutlineBuilder` 的回调攒成 `PathEl` 序列。
///
/// 这个 trait 的五个方法与 [`PathEl`] 的五个变体**一一对应**，
/// 因此这里的实现是纯粹的搬运 + 单位换算。
struct OutlineToPathEl {
    els: Vec<PathEl>,
    /// 字体单位 → em 的缩放系数
    scale: f32,
}

impl OutlineToPathEl {
    fn new(units_per_em: f32) -> Self {
        OutlineToPathEl { els: Vec::new(), scale: 1.0 / units_per_em.max(1.0) }
    }

    /// 字体坐标 → em 坐标（y 取负：字体 y 向上，屏幕 y 向下）
    fn p(&self, x: f32, y: f32) -> Vec2 {
        Vec2::new(x * self.scale, -y * self.scale)
    }
}

impl ttf_parser::OutlineBuilder for OutlineToPathEl {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.p(x, y);
        self.els.push(PathEl::MoveTo(p));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.p(x, y);
        self.els.push(PathEl::LineTo(p));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let c = self.p(x1, y1);
        let e = self.p(x, y);
        self.els.push(PathEl::QuadTo(c, e));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let c1 = self.p(x1, y1);
        let c2 = self.p(x2, y2);
        let e = self.p(x, y);
        self.els.push(PathEl::CubicTo(c1, c2, e));
    }
    fn close(&mut self) {
        self.els.push(PathEl::Close);
    }
}

/// 字体度量：排版（断行、行高、基线）所需的一切。
///
/// 全部归一化为 **em 单位**（1.0 = 字号），因此与字号无关，可直接乘以字号使用。
/// 这正是"为什么布局层不需要新依赖"的原因：`ttf-parser` 的 `Face` 已提供全部数据。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FontMetrics {
    /// 基线以上高度（正值）
    pub ascent: f32,
    /// 基线以下深度（值为正，表示"向下多少"）
    pub descent: f32,
    /// 行间额外留白
    pub line_gap: f32,
    /// 小写 x 高度（用于光学对齐/竖排居中）
    pub x_height: f32,
    /// 大写高度（cap height，用于光学对齐）
    pub cap_height: f32,
    /// 建议行高 = ascent + descent + line_gap
    pub line_height: f32,
}

impl FontMetrics {
    /// 排版用行高。若字体给了 `use_typographic_metrics` 之外的度量，
    /// 这里取 hhea 的 ascent/descent/line_gap —— 这是绝大多数排版系统
    /// （含浏览器默认行为）采用的一组。
    pub fn line_height(&self) -> f32 {
        self.line_height
    }
}

impl Font {
    /// 字体度量（em 单位）。
    ///
    /// 坐标系约定与本 crate 一致：**y 向下**，因此 `descent` 以正值表示
    /// "基线向下的深度"；排版时"下一行基线 = 当前基线 + line_height"。
    pub fn metrics(&self) -> Result<FontMetrics, TextError> {
        let face = self.face()?;
        let upem = self.units_per_em.max(1.0);
        let ascent = face.ascender() as f32 / upem;
        let descent = -(face.descender() as f32) / upem; // ttf 的 descender 通常为负
        let line_gap = face.line_gap() as f32 / upem;
        let x_height = face.x_height().unwrap_or(0) as f32 / upem;
        let cap_height = face.capital_height().unwrap_or(0) as f32 / upem;
        Ok(FontMetrics {
            ascent,
            descent,
            line_gap,
            x_height,
            cap_height,
            line_height: ascent + descent + line_gap,
        })
    }

    /// 某字形的水平前进量（em 单位）。
    ///
    /// 供布局在**不做 shaping** 时估算宽度（例如快速测量），
    /// 或对已 shaping 的结果做逐字形校验。
    pub fn advance(&self, glyph: u32) -> Result<f32, TextError> {
        let face = self.face()?;
        let gid = ttf_parser::GlyphId(glyph as u16);
        match face.glyph_hor_advance(gid) {
            Some(a) => Ok(a as f32 / self.units_per_em.max(1.0)),
            None => Err(TextError::MissingGlyph(glyph)),
        }
    }

    /// 一串文本的**排版宽度**（em 单位），不做 shaping。
    ///
    /// 只累加 simple 字形映射的前进量，因此**不含 kerning 与连字**，
    /// 对宽度是近似（通常略偏宽）。用于快速测量/断行预估；
    /// 需要精确宽度时请用 [`shape`] 后取末字形位置。
    /// 字体中没有对应字形的字符按 `.notdef` 处理（跳过，不计宽）。
    pub fn advance_width(&self, text: &str) -> f32 {
        let mut w = 0.0;
        for c in text.chars() {
            if let Some(g) = self.glyph_index(c) {
                if let Ok(a) = self.advance(g) {
                    w += a;
                }
            }
        }
        w
    }
}

/// 一段文本的 shaping 结果：字形序列（位置为 em 单位，相对基线原点）。
///
/// 直接可转成 [`TextRun`] 交给引擎。
///
/// # 位置是怎么来的（容易搞错）
///
/// shaping 给每个字形的是 **advance（前进量）** 而非绝对位置。位置必须由
/// **游标累加**得到：
///
/// ```text
/// pen = 起点 0
/// 对每个字形：位置 = pen + offset；然后 pen += advance
/// ```
///
/// 忽略 advance 会让所有字形叠在同一处；忽略 offset 会丢掉 kerning 与 mark 定位。
/// 两者都由 shaping 给出，都必须用上。
/// shaping 产出的单个字形：位置（em 单位）**加上它对应的字符字节偏移**。
///
/// `cluster` 是 rustybuzz 给出的**字节偏移**（`text` 中的位置）。排版需要它才能
/// 把字形的推进量映射回字符边界 —— 没有它就只能在"逐字符累加 advance"
/// （丢掉字距与连字）和"对每个前缀各 shaping 一次"（O(n²)）之间二选一。
///
/// 连字会把多个字符映射到**同一个** cluster，这正是断行需要的语义：连字不可拆。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedGlyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub cluster: u32,
}

/// shaping（含 cluster）。[`shape`] 是它的便捷包装。
pub fn shape_clustered(font: &Font, text: &str) -> Result<Vec<ShapedGlyph>, TextError> {
    let face = ttf_parser::Face::parse(&font.data, 0)
        .map_err(|e| TextError::FontParse(format!("{e:?}")))?;
    let upem = face.units_per_em() as f32;
    let rb_face = rustybuzz::Face::from_face(face);

    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(text);
    let out = rustybuzz::shape(&rb_face, &[], buf);

    let infos = out.glyph_infos();
    let positions = out.glyph_positions();
    // 排版单位为字体单位 → 归一化成 em
    let k = 1.0 / upem.max(1.0);

    let mut glyphs = Vec::with_capacity(positions.len());
    let mut pen_x = 0.0f32;
    let mut pen_y = 0.0f32;
    for (i, p) in positions.iter().enumerate() {
        let x = pen_x + p.x_offset as f32 * k;
        let y = pen_y - p.y_offset as f32 * k; // y 取负：与轮廓一致地翻转到屏幕朝向
        glyphs.push(ShapedGlyph { id: infos[i].glyph_id, x, y, cluster: infos[i].cluster });
        pen_x += p.x_advance as f32 * k;
        pen_y -= p.y_advance as f32 * k;
    }
    Ok(glyphs)
}

pub fn shape(font: &Font, text: &str, size: f32) -> Result<TextRun, TextError> {
    let face = ttf_parser::Face::parse(&font.data, 0)
        .map_err(|e| TextError::FontParse(format!("{e:?}")))?;
    let upem = face.units_per_em() as f32;
    let rb_face = rustybuzz::Face::from_face(face);

    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(text);
    let out = rustybuzz::shape(&rb_face, &[], buf);

    let infos = out.glyph_infos();
    let positions = out.glyph_positions();
    // 排版单位为字体单位 → 归一化成 em
    let k = 1.0 / upem.max(1.0);

    let mut run = TextRun::new(size);
    let mut pen_x = 0.0f32;
    let mut pen_y = 0.0f32;
    for (i, p) in positions.iter().enumerate() {
        // 字形 id 取自 info：shaping 可能做了替换（连字/变体），不能用原始字符映射
        let id = infos[i].glyph_id;
        let x = pen_x + p.x_offset as f32 * k;
        let y = pen_y - p.y_offset as f32 * k; // y 取负：与轮廓一致地翻转到屏幕朝向
        run.push_glyph(id, x, y);
        pen_x += p.x_advance as f32 * k;
        pen_y -= p.y_advance as f32 * k;
    }
    Ok(run)
}

// ============================================================================
// 字体身份（字形缓存的键要用它）
// ============================================================================

/// FNV-1a 64 的偏移基数。
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// 带来源标签的哈希 → [`FontId`]。标签让不同来源（名字 / 字节）不会互相碰撞。
fn hash_id(tag: &str, s: &str) -> FontId {
    let h = fnv1a(s.as_bytes(), fnv1a(tag.as_bytes(), FNV_OFFSET));
    FontId::from_bits(if h == 0 { 1 } else { h })
}

/// 取某个 name ID 的**第一条 Unicode 名字**（同一 ID 会有多语言、多平台记录）。
fn name_by_id(face: &ttf_parser::Face<'_>, want: u16) -> Option<String> {
    face.names()
        .into_iter()
        .filter(|n| n.name_id == want && n.is_unicode())
        .find_map(|n| n.to_string())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// 字体的内部身份：优先用它**自己声明的名字**，都没有才退到字节哈希。
///
/// 用名字而不是路径/字节做默认身份，是因为"同一份字体"经常以不同文件名、
/// 不同目录出现（打包、子集化前的原件、系统字体），而那些都应当命中同一份缓存。
fn font_identity(face: &ttf_parser::Face<'_>, data: &[u8]) -> (FontId, String) {
    use ttf_parser::name_id;
    // ① 单值型：PostScript 名（标准唯一名）/ Unique ID / Full name
    for (want, tag, label) in [
        (name_id::POST_SCRIPT_NAME, "ps", "PostScript 名"),
        (name_id::UNIQUE_ID, "uid", "Unique ID"),
        (name_id::FULL_NAME, "full", "Full name"),
    ] {
        if let Some(s) = name_by_id(face, want) {
            return (hash_id(tag, &s), format!("{label} \"{s}\""));
        }
    }
    // ② 组合型：Family + Subfamily
    if let Some(family) = name_by_id(face, name_id::FAMILY) {
        let s = match name_by_id(face, name_id::SUBFAMILY) {
            Some(sub) => format!("{family} {sub}"),
            None => family,
        };
        return (hash_id("fam", &s), format!("Family/Subfamily \"{s}\""));
    }
    // ③ 兜底：整个字体字节的哈希（同一份字节总得到同一个 ID）
    let h = fnv1a(data, fnv1a(b"bytes", FNV_OFFSET));
    (
        FontId::from_bits(if h == 0 { 1 } else { h }),
        format!("字节哈希（{} 字节）", data.len()),
    )
}

// ============================================================================
// 字体回退 shaping
// ============================================================================

/// **字体回退** shaping：一段文本里不同字符可以由不同字体渲染。
///
/// # 行为
///
/// 1. 逐字符按 [`FontStack::resolve`] 解析出字体下标（顺序即优先级）；
/// 2. 相邻的**同字体且文本连续**的字符合并成一个 run，每个 run 单独 shaping
///    —— 各自保留字距与连字；
/// 3. 前进量（pen）**跨 run 连续累加**，因此字形位置是整段的绝对位置。
///
/// 代价写在明处：run 边界处**不做字距调整**（`AV` 跨两个字体时不会收窄），
/// 因为那是字体内部的信息。同字体的整段文本不受影响。
///
/// # 缺字（栈里没有任何字体含该字符）
///
/// 由 [`MissingGlyphPolicy`] 决定：`Error` 返回 [`TextError::MissingGlyph`]
/// （payload 是**码点**）、`NotDef` 归到栈里第一个字体并渲染 `.notdef`、
/// `Skip` 不产生字形且前进量为 0。
///
/// # 单字体栈
///
/// `stack` 只有一个字体**且**选项等价于 [`TextOptions::LEGACY`] 时，直接走老的
/// [`shape`]，结果**逐字节一致** —— 这条快速路径就是"零破坏"的保证。
pub fn shape_with_stack(
    stack: &FontStack,
    text: &str,
    size: f32,
    options: &TextOptions,
) -> Result<TextRun, TextError> {
    if text.is_empty() {
        return Ok(TextRun::new(size));
    }
    if stack.is_empty() {
        return Err(TextError::FontParse("字体栈为空".into()));
    }
    // 快速路径：单字体 + 缺字交给 shaping（= 0.6.1 行为）
    if stack.len() == 1 && *options == TextOptions::LEGACY {
        return shape(&stack.fonts()[0], text, size);
    }

    // 每个字体解析一次，供逐字符 cmap 查询复用（shaping 时按 run 再解析）。
    let faces: Vec<ttf_parser::Face<'_>> =
        stack.fonts().iter().map(|f| f.face()).collect::<Result<_, _>>()?;

    // ① 逐字符解析 → run
    let mut runs: Vec<Run> = Vec::new();
    for (byte_off, c) in text.char_indices() {
        let font_index = match stack.resolve_in(&faces, c) {
            Some(i) => i,
            None => match options.missing_glyph {
                MissingGlyphPolicy::Error => return Err(TextError::MissingGlyph(c as u32)),
                MissingGlyphPolicy::Skip => continue,
                // 归到第一个字体：把它交给 shaping，自然得到 `.notdef`
                MissingGlyphPolicy::NotDef => 0,
            },
        };
        let mut buf = [0u8; 4];
        let s = c.encode_utf8(&mut buf);
        let append = match runs.last() {
            Some(r) => r.font_index == font_index && r.next_byte == byte_off,
            None => false,
        };
        if append {
            let r = runs.last_mut().expect("刚刚判定过非空");
            r.orig.push((r.text.len() as u32, byte_off as u32));
            r.text.push_str(s);
            r.next_byte = byte_off + s.len();
        } else {
            runs.push(Run {
                font_index,
                text: s.to_string(),
                orig: vec![(0, byte_off as u32)],
                next_byte: byte_off + s.len(),
            });
        }
    }
    if runs.is_empty() {
        return Ok(TextRun::new(size)); // 整段都被 Skip 掉了
    }

    // ② 逐 run shaping，pen 跨 run 累加
    let mut out = TextRun::new(size);
    let mut pen = (0.0f32, 0.0f32);
    for r in &runs {
        let font = &stack.fonts()[r.font_index];
        let glyphs = shape_run_into(font, &r.text, options, &mut pen)?;
        for (id, x, y, cluster) in glyphs {
            out.push_glyph_in(r.font_index, id, x, y, translate_cluster(&r.orig, cluster));
        }
    }
    Ok(out)
}

/// 一个 run：同一字体、**文本连续**的片段。
struct Run {
    font_index: usize,
    /// 交给 shaping 的文本（`Skip` 会挖掉缺字，故不一定等于原文切片）
    text: String,
    /// `(本 run 内的字节偏移, 原文中的字节偏移)`，按前者升序 —— 用来把 shaping
    /// 给出的 cluster 翻回**原文**偏移（`Skip` 挖洞后二者不再相同）。
    orig: Vec<(u32, u32)>,
    /// 原文中紧跟本 run 的下一个字节偏移（判断"文本连续"用）
    next_byte: usize,
}

/// 把 run 内的 cluster 翻回原文偏移。
fn translate_cluster(orig: &[(u32, u32)], cluster: u32) -> u32 {
    // 线性扫描足够：一个 run 的字符数有限，且 cluster 单调不减
    let mut last = orig.first().map(|(_, o)| *o).unwrap_or(0);
    for (run_off, orig_off) in orig {
        if *run_off > cluster {
            break;
        }
        last = *orig_off;
    }
    last
}

/// 用**一个**字体 shaping 一段文本；字形位置按当前 `pen` 累加（返回整段绝对位置）。
fn shape_run_into(
    font: &Font,
    text: &str,
    options: &TextOptions,
    pen: &mut (f32, f32),
) -> Result<Vec<(u32, f32, f32, u32)>, TextError> {
    let face = font.face()?;
    let upem = face.units_per_em() as f32;
    let rb_face = rustybuzz::Face::from_face(face);

    let mut features = Vec::new();
    if !options.kerning {
        features.push(rustybuzz::Feature::new(
            ttf_parser::Tag::from_bytes(b"kern"),
            0,
            ..,
        ));
    }
    if !options.ligatures {
        for tag in [b"liga", b"clig"] {
            features.push(rustybuzz::Feature::new(
                ttf_parser::Tag::from_bytes(tag),
                0,
                ..,
            ));
        }
    }

    let mut buf = rustybuzz::UnicodeBuffer::new();
    buf.push_str(text);
    let out = rustybuzz::shape(&rb_face, &features, buf);

    let infos = out.glyph_infos();
    let positions = out.glyph_positions();
    // 排版单位为字体单位 → 归一化成 em
    let k = 1.0 / upem.max(1.0);

    let mut glyphs = Vec::with_capacity(positions.len());
    for (i, p) in positions.iter().enumerate() {
        let x = pen.0 + p.x_offset as f32 * k;
        let y = pen.1 - p.y_offset as f32 * k; // y 取负：与轮廓一致地翻转到屏幕朝向
        glyphs.push((infos[i].glyph_id, x, y, infos[i].cluster));
        pen.0 += p.x_advance as f32 * k;
        pen.1 -= p.y_advance as f32 * k;
    }
    Ok(glyphs)
}

mod style;
pub use style::FontStyle;
