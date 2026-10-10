//! 文本模型：字形轮廓、字体身份、`TextRun` / `GlyphKey` / `TextCache` 与展开。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

/// 一个字形的轮廓（**em 单位**：1.0 = 字号，原点在字形基线左端）。
///
/// 用曲线元素序列表达，与 [`Path`] 同构——因此可零成本喂给既有管线。
/// 引擎不关心它来自 TrueType/OpenType 的哪种表，只要求坐标是 em 单位的
/// 轮廓坐标（后续按 `font_size` 与变换缩放）。
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphOutline {
    /// 轮廓元素；坐标单位为 em（相对字号归一化）
    pub els: Vec<PathEl>,
}

impl GlyphOutline {
    pub fn new(els: Vec<PathEl>) -> GlyphOutline {
        GlyphOutline { els }
    }
}

/// 字体的**内部稳定标识** —— 不是路径，也不是"栈里的下标"。
///
/// # 为什么必须是内部 ID
///
/// 它有两个用途：字形缓存的键（见 [`GlyphKey`]），以及"是不是同一个字体"的判定。
/// 路径会变（换机器、换部署目录）、下标会变（栈顺序调整），拿它们做键会让缓存
/// **悄悄命中错误的字形**（画出来是另一个字体的字）。所以这里用字体**自己声明的**
/// 标识，按优先级取第一个可用的：
///
/// 1. PostScript 名（`name` 表 ID 6）—— 字体厂商给的标准唯一名
/// 2. Unique ID（ID 3）
/// 3. Full name（ID 4）
/// 4. Family + Subfamily（ID 1 + 2）
/// 5. 都没有时退到**字体字节的哈希**（同一份字节仍稳定，但两份"内容相同的不同文件"会撞成一个）
///
/// 计算是 FNV-1a 64，并带一个**来源标签前缀**（`ps:` / `uid:` / …），
/// 因此不同来源不会互相碰撞。
///
/// 定义在 `amiki-scene` 而不是 `amiki-text`：`GlyphKey` 在这里，
/// 而依赖方向是 `text → scene`（不能反过来）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FontId(u64);

impl FontId {
    /// 未知/未指定字体：**旧单字体路径**的缓存键用它（那时调用方只给字形 id，
    /// 拿不到字体身份）。真实字体 ID 永远不是 0。
    pub const UNKNOWN: FontId = FontId(0);

    pub const fn from_bits(bits: u64) -> FontId {
        FontId(bits)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for FontId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == 0 {
            write!(f, "FontId(unknown)")
        } else {
            write!(f, "FontId({:016x})", self.0)
        }
    }
}

/// 一个已就位的字形：id 用于缓存，位置以**em 单位**给出（基线原点向左为 0）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionedGlyph {
    /// 字形标识（调用方定义：字体内 glyph id 即可），供缓存分键
    pub id: u32,
    /// 该字形来自**字体栈里的第几个字体**。
    ///
    /// 单字体路径恒为 0（老代码仍用 [`TextRun::push_glyph`]，行为不变）。
    /// 字体回退场景下，`expand` 必须靠它找到正确的字体取轮廓 ——
    /// 同一个 `id` 在不同字体里是完全不同的字形。
    pub font_index: usize,
    /// 该字形在原文中的**字符字节偏移**（rustybuzz 的 cluster）。
    ///
    /// 连字会把多个字符映射到同一个 cluster —— 这正是断行需要的语义（连字不可拆）。
    /// 旧单字体路径不填（0）。
    pub cluster: u32,
    /// 基线原点（em 单位）
    pub x: f32,
    pub y: f32,
}

/// 一段文本：字号 + 已 shaping 的字形序列。
///
/// `size` 是**字号**（em 到设备空间的缩放基准）。若调用方已把字号折进变换，
/// 传 `1.0` 即可——两条路等价，引擎不做假设。
#[derive(Clone, Debug, PartialEq)]
pub struct TextRun {
    pub size: f32,
    pub glyphs: Vec<PositionedGlyph>,
}

impl TextRun {
    pub fn new(size: f32) -> TextRun {
        TextRun { size, glyphs: Vec::new() }
    }

    /// 追加一个字形（**旧入口**：字体下标 0、不带 cluster，行为与 0.6.1 一致）。
    pub fn push_glyph(&mut self, id: u32, x: f32, y: f32) -> &mut Self {
        self.push_glyph_in(0, id, x, y, 0)
    }

    /// 追加一个字形，并指明它来自哪个字体、对应原文的哪个字节偏移。
    pub fn push_glyph_in(
        &mut self,
        font_index: usize,
        id: u32,
        x: f32,
        y: f32,
        cluster: u32,
    ) -> &mut Self {
        self.glyphs.push(PositionedGlyph { id, font_index, cluster, x, y });
        self
    }
}

/// 字形轮廓缓存键：`(字体, glyph id, 量化后的字号)`。
///
/// 为什么必须缓存：一个字形会被成百上千次复用（每个字符位置一次），
/// 每次都重新构造 `Path` 会让 CPU 侧成为瓶颈。
/// 为什么按**量化**字号分键：同一段文本常以若干离散字号出现，
/// 量化后命中率远高于精确浮点相等，且误差只是一次查表偏移。
/// 为什么带**字体 ID**：同一个 glyph id 在不同字体里是不同字形，
/// 不带字体维度就会跨字体误命中（字体回退场景下必然发生）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GlyphKey {
    /// 字体内部 ID（旧路径为 [`FontId::UNKNOWN`]）
    pub font_id: FontId,
    pub id: u32,
    /// 字号 × 16 后取整（1/16 px 粒度）
    pub size_q: u32,
}

impl GlyphKey {
    pub const SIZE_QUANT: f32 = 16.0;

    /// 旧入口：不带字体身份（等价于 0.6.1 的键）。
    pub fn new(id: u32, size: f32) -> GlyphKey {
        GlyphKey::with_font(FontId::UNKNOWN, id, size)
    }

    /// 带字体身份的键（字体回退路径用它）。
    pub fn with_font(font_id: FontId, id: u32, size: f32) -> GlyphKey {
        GlyphKey { font_id, id, size_q: (size.max(0.0) * Self::SIZE_QUANT).round() as u32 }
    }
}

/// 把 [`TextRun`] 展开为若干 `(Path, 变换)`——每个字形一个。
///
/// `lookup` 提供字形轮廓。返回的变换把 em 单位轮廓映射到设备空间：
/// 缩放 `size`、平移字形位置。调用方再乘上文本自身的外部变换。
///
/// 展开成**多个独立 path** 而不是合并成一条大路径，有两个理由：
/// 1. 每个字形各自成 draw，可被分箱与 damage region 独立处理（改一个字只脏一小块）；
/// 2. 避免合并后产生大量子路径带来的绕数判定开销。
pub fn expand_text_run(
    run: &TextRun,
    mut lookup: impl FnMut(u32) -> Option<GlyphOutline>,
) -> Vec<(Path, Affine)> {
    expand_text_run_with(run, |_font_index, id| lookup(id))
}

/// 字体回退版 [`expand_text_run`]：`lookup` 同时收到**字体下标**与字形 id。
///
/// 多字体的一段文本里，同一个 `glyph id` 在不同字体中是**不同字形**，
/// 因此取轮廓必须带上 [`PositionedGlyph::font_index`]；`lookup` 通常实现为
/// `|fi, id| stack.get(fi)?.outline(id).ok()`。
pub fn expand_text_run_with(
    run: &TextRun,
    mut lookup: impl FnMut(usize, u32) -> Option<GlyphOutline>,
) -> Vec<(Path, Affine)> {
    let mut out = Vec::with_capacity(run.glyphs.len());
    for g in &run.glyphs {
        let Some(o) = lookup(g.font_index, g.id) else { continue };
        if o.els.is_empty() {
            continue;
        }
        // em → 设备：先按字号缩放，再平移到字形基线原点
        let scale = Affine::scale(run.size);
        let shift = Affine::translate(Vec2::new(g.x, g.y));
        out.push((Path::from_els(o.els.clone()), scale.then(shift)));
    }
    out
}

/// 带缓存的文本展开器：由调用方持有，跨帧复用。
///
/// 缓存的是**已按字号缩放的路径**（`Affine::scale(size)` 折入坐标），
/// 因此同一 `(id, size)` 第二次出现时不再重复缩放整条轮廓。
#[derive(Default)]
pub struct TextCache {
    /// (键 → 已缩放轮廓)。用 Vec 而非 HashMap：字形数量通常很小（几十到几百），
    /// 线性查找比哈希更快，也避免引入哈希依赖。
    entries: Vec<(GlyphKey, Path)>,
    hits: u64,
    misses: u64,
}

impl TextCache {
    pub fn new() -> TextCache {
        TextCache::default()
    }

    /// 命中率（诊断用）。
    pub fn hit_rate(&self) -> f64 {
        let n = self.hits + self.misses;
        if n == 0 {
            0.0
        } else {
            self.hits as f64 / n as f64
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 展开一段文本，命中缓存则复用已缩放轮廓。
    ///
    /// `outline` 只在未命中时调用（避免无谓的轮廓提取开销）。
    pub fn expand<F>(&mut self, run: &TextRun, mut outline: F) -> Vec<(Path, Affine)>
    where
        F: FnMut(u32) -> Option<GlyphOutline>,
    {
        self.expand_with(run, |_| FontId::UNKNOWN, |_fi, id| outline(id))
    }

    /// 字体回退版 [`TextCache::expand`]：缓存键带**字体内部 ID**。
    ///
    /// 两个闭包分开是有意的：`font_of` 很便宜（`stack.get(i).map(Font::id)`），
    /// 而键必须在查缓存**之前**算出来；`outline` 才是贵的那个，只在未命中时调用。
    /// 若只给一个闭包，就得为了拿字体 ID 而先把轮廓提出来 —— 缓存也就白做了。
    pub fn expand_with<FF, FO>(
        &mut self,
        run: &TextRun,
        mut font_of: FF,
        mut outline: FO,
    ) -> Vec<(Path, Affine)>
    where
        FF: FnMut(usize) -> FontId,
        FO: FnMut(usize, u32) -> Option<GlyphOutline>,
    {
        let mut out = Vec::with_capacity(run.glyphs.len());
        for g in &run.glyphs {
            let key = GlyphKey::with_font(font_of(g.font_index), g.id, run.size);
            // 命中：线性查已缩放轮廓
            let idx = self.entries.iter().position(|(k, _)| *k == key);
            let path = match idx {
                Some(i) => {
                    self.hits += 1;
                    self.entries[i].1.clone()
                }
                None => {
                    self.misses += 1;
                    let Some(o) = outline(g.font_index, g.id) else { continue };
                    if o.els.is_empty() {
                        continue;
                    }
                    // 把字号折进坐标，之后只需平移 → 同一字号的复用不再缩放
                    let s = run.size;
                    let scaled: Vec<PathEl> = o
                        .els
                        .iter()
                        .map(|el| match *el {
                            PathEl::MoveTo(p) => PathEl::MoveTo(p * s),
                            PathEl::LineTo(p) => PathEl::LineTo(p * s),
                            PathEl::QuadTo(c, p) => PathEl::QuadTo(c * s, p * s),
                            PathEl::CubicTo(a, b, p) => PathEl::CubicTo(a * s, b * s, p * s),
                            PathEl::Close => PathEl::Close,
                        })
                        .collect();
                    let p = Path::from_els(scaled);
                    self.entries.push((key, p.clone()));
                    p
                }
            };
            // 位置也要乘以字号：`expand_text_run` 用 `scale.then(translate)`，
            // 平移量同样被缩放。缓存把字号折进了**坐标**，因此这里必须显式
            // 缩放位置，否则 size≠1 时字形会被摆到错误位置（曾用测试抓出此 bug）。
            out.push((
                path,
                Affine::translate(Vec2::new(g.x * run.size, g.y * run.size)),
            ));
        }
        out
    }
}

