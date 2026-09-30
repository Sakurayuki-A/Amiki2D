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

#![forbid(unsafe_code)]

use amiki_math::Vec2;
use amiki_scene::{GlyphOutline, PathEl, TextRun};

/// 文本层错误。
#[derive(Debug)]
pub enum TextError {
    /// 字体数据无法解析
    FontParse(String),
    /// 请求的字形或字符不存在
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
/// 持有字体数据的**所有权**（`Vec<u8>`）——因为 `ttf_parser::Face` 借用底层数据，
/// 自引用结构在安全 Rust 里无法直接表达，故这里按需创建 `Face`。
pub struct Font {
    data: Vec<u8>,
    /// 每 em 的字体单位数（用于把字体单位归一化为 em）
    units_per_em: f32,
    glyph_count: u32,
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
        Ok(Font { data, units_per_em: units_per_em.max(1.0), glyph_count })
    }

    /// 从文件加载。
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Font, TextError> {
        let data = std::fs::read(path)
            .map_err(|e| TextError::FontParse(format!("读取字体文件失败：{e}")))?;
        Font::from_bytes(data)
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
    pub fn glyph_index(&self, c: char) -> Option<u32> {
        let face = ttf_parser::Face::parse(&self.data, 0).ok()?;
        face.glyph_index(c).map(|g| g.0 as u32)
    }

    fn face(&self) -> Result<ttf_parser::Face<'_>, TextError> {
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
