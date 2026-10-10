//! AMK1 编码流：`encode_scene` / `decode_scene`（格式见 `docs/AMK1.md`）。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

pub const STREAM_MAGIC: &[u8; 4] = b"AMK1";
/// 当前编码版本。
///
/// # v1 → v2（0.8.0）
///
/// v2 与 v1 的**布局完全相同**，唯一差别是每个 draw 的 blend 字里
/// **bit 12 被定义为 `fill_rule`**（v1 该位恒为 0 ⇒ 解码为 `NonZero`）。
///
/// - **只解不编**：编码器统一写 v2（v1 编码器已不存在，避免死代码漂移）；
///   解码器同时接受 v1 与 v2。
/// - 未知版本（≥3）**显式报错**，不静默错解。
pub const STREAM_VERSION: u32 = 2;
/// v1 的版本号（只解不编）。
pub const STREAM_VERSION_V1: u32 = 1;
/// blend 字里 `fill_rule` 占的位（与计划的 bit 12 一致）。
pub const FILL_RULE_BIT: u32 = 12;

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

/// 写入一个颜色（4×f32：r, g, b, a）。
fn put_color(buf: &mut Vec<u8>, c: &Color) {
    put_f32(buf, c.r);
    put_f32(buf, c.g);
    put_f32(buf, c.b);
    put_f32(buf, c.a);
}

fn put_path(buf: &mut Vec<u8>, path: &Path) {
    put_u32(buf, path.els.len() as u32);
    for el in &path.els {
        match *el {
            PathEl::MoveTo(p) => {
                buf.push(0);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::LineTo(p) => {
                buf.push(1);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::QuadTo(c, p) => {
                buf.push(2);
                put_f32(buf, c.x);
                put_f32(buf, c.y);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::CubicTo(c1, c2, p) => {
                buf.push(3);
                put_f32(buf, c1.x);
                put_f32(buf, c1.y);
                put_f32(buf, c2.x);
                put_f32(buf, c2.y);
                put_f32(buf, p.x);
                put_f32(buf, p.y);
            }
            PathEl::Close => buf.push(4),
        }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("编码流越界".into());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32x(&mut self, n: usize) -> Result<Vec<f32>, String> {
        (0..n).map(|_| self.f32()).collect()
    }
    fn point(&mut self) -> Result<Vec2, String> {
        let v = self.f32x(2)?;
        Ok(Vec2::new(v[0], v[1]))
    }
    fn path(&mut self) -> Result<Path, String> {
        let count = self.u32()? as usize;
        let mut els = Vec::with_capacity(count);
        for _ in 0..count {
            let tag = self.take(1)?[0];
            els.push(match tag {
                0 => PathEl::MoveTo(self.point()?),
                1 => PathEl::LineTo(self.point()?),
                2 => {
                    let c = self.point()?;
                    PathEl::QuadTo(c, self.point()?)
                }
                3 => {
                    let c1 = self.point()?;
                    let c2 = self.point()?;
                    PathEl::CubicTo(c1, c2, self.point()?)
                }
                4 => PathEl::Close,
                _ => return Err(format!("未知路径元素 tag {tag}")),
            });
        }
        Ok(Path { els })
    }
}

fn put_transform(buf: &mut Vec<u8>, t: &Affine) {
    for v in [t.a, t.b, t.c, t.d, t.e, t.f] {
        put_f32(buf, v);
    }
}

fn read_transform(r: &mut Reader) -> Result<Affine, String> {
    let v = r.f32x(6)?;
    Ok(Affine { a: v[0], b: v[1], c: v[2], d: v[3], e: v[4], f: v[5] })
}

/// 场景编码为紧凑字节流。
pub fn encode_scene(scene: &Scene) -> Vec<u8> {
    let mut buf = Vec::with_capacity(256 + scene.draws.len() * 64);
    buf.extend_from_slice(STREAM_MAGIC);
    put_u32(&mut buf, STREAM_VERSION);
    put_u32(&mut buf, scene.width);
    put_u32(&mut buf, scene.height);
    put_u32(&mut buf, scene.clips.len() as u32);
    put_u32(&mut buf, scene.draws.len() as u32);
    for clip in &scene.clips {
        put_transform(&mut buf, &clip.transform);
        put_path(&mut buf, &clip.path);
    }
    for draw in &scene.draws {
        put_transform(&mut buf, &draw.transform);
        match &draw.paint {
            Paint::Solid(c) => {
                put_u32(&mut buf, 0); // paint 类型标签：0 = 实心
                put_color(&mut buf, c);
            }
            Paint::Gradient(g) => {
                put_u32(&mut buf, 1); // 1 = 渐变
                // 几何：kind 标签 + 参数
                match g.kind {
                    GradientKind::Linear { from, to } => {
                        put_u32(&mut buf, 0);
                        put_f32(&mut buf, from.x);
                        put_f32(&mut buf, from.y);
                        put_f32(&mut buf, to.x);
                        put_f32(&mut buf, to.y);
                    }
                    GradientKind::Radial { center, radius } => {
                        put_u32(&mut buf, 1);
                        put_f32(&mut buf, center.x);
                        put_f32(&mut buf, center.y);
                        put_f32(&mut buf, radius);
                        put_f32(&mut buf, 0.0); // 占位，保持定长便于解码
                    }
                    GradientKind::Sweep { center, start } => {
                        put_u32(&mut buf, 2);
                        put_f32(&mut buf, center.x);
                        put_f32(&mut buf, center.y);
                        put_f32(&mut buf, start);
                        put_f32(&mut buf, 0.0);
                    }
                }
                put_u32(&mut buf, u32::from(g.repeat));
                put_u32(&mut buf, g.stops.len() as u32);
                for s in &g.stops {
                    put_f32(&mut buf, s.t);
                    put_color(&mut buf, &s.color);
                }
            }
            Paint::Image { image, placement } => {
                // M6 现状：AMK1 流**不内嵌位图**（一张 717×692 的图就有约 2 MB，
                // 内嵌会让"场景流"退化成图像容器，失去轻量优势）。
                // 因此这里写入一个**显式的"含图像"标记**：解码方会明确报错，
                // 而不是静默丢掉图像导致画面悄悄缺一块。
                put_u32(&mut buf, 2); // 2 = 图像（当前不可序列化）
                put_u32(&mut buf, image.width);
                put_u32(&mut buf, image.height);
                put_f32(&mut buf, placement.origin.x);
                put_f32(&mut buf, placement.origin.y);
                put_f32(&mut buf, placement.u_axis.x);
                put_f32(&mut buf, placement.u_axis.y);
                put_f32(&mut buf, placement.v_axis.x);
                put_f32(&mut buf, placement.v_axis.y);
                put_u32(&mut buf, u32::from(placement.repeat));
            }
        }
        // v2：这个 u32 从"纯 blend id"升级为**标志字** —— bit 0..3 仍是 blend，
        // bit 12 是 fill_rule（0 = NonZero 默认，1 = EvenOdd）。
        // 布局**一字未改**（没有加字段、没有扩宽），v1 流里这一位恒为 0。
        put_u32(&mut buf, draw.blend.id() | (draw.fill_rule.bit() << FILL_RULE_BIT));
        put_u32(&mut buf, draw.clips.len() as u32);
        for &c in &draw.clips {
            put_u32(&mut buf, c);
        }
        put_path(&mut buf, &draw.path);
    }
    buf
}

/// 从字节流解码场景（格式校验失败返回 Err）。
pub fn decode_scene(bytes: &[u8]) -> Result<Scene, String> {
    let mut r = Reader { data: bytes, pos: 0 };
    if r.take(4)? != STREAM_MAGIC {
        return Err("编码流 magic 不符（需要 AMK1）".into());
    }
    let version = r.u32()?;
    // v1 与 v2 都接受；差别只在 blend 字的 bit 12 是否**定义**为 fill_rule。
    if version != STREAM_VERSION && version != STREAM_VERSION_V1 {
        return Err(format!("编码流版本不支持: {version}"));
    }
    let v2 = version >= 2;
    let width = r.u32()?;
    let height = r.u32()?;
    let n_clips = r.u32()?;
    let n_draws = r.u32()?;
    let mut scene = Scene::new(width, height);
    for _ in 0..n_clips {
        let t = read_transform(&mut r)?;
        let path = r.path()?;
        scene.push_clip(t, path);
    }
    for _ in 0..n_draws {
        let t = read_transform(&mut r)?;
        let paint = match r.u32()? {
            0 => {
                let c = r.f32x(4)?;
                Paint::Solid(Color { r: c[0], g: c[1], b: c[2], a: c[3] })
            }
            1 => {
                let kind_id = r.u32()?;
                let a = r.f32x(4)?;
                let kind = match kind_id {
                    0 => GradientKind::Linear {
                        from: Vec2::new(a[0], a[1]),
                        to: Vec2::new(a[2], a[3]),
                    },
                    1 => GradientKind::Radial { center: Vec2::new(a[0], a[1]), radius: a[2] },
                    2 => GradientKind::Sweep { center: Vec2::new(a[0], a[1]), start: a[2] },
                    other => return Err(format!("渐变类型不支持: {other}")),
                };
                let repeat = r.u32()? != 0;
                let n = r.u32()? as usize;
                let mut stops = Vec::with_capacity(n);
                for _ in 0..n {
                    let st = r.f32()?;
                    let c = r.f32x(4)?;
                    stops.push(GradientStop::new(
                        st,
                        Color { r: c[0], g: c[1], b: c[2], a: c[3] },
                    ));
                }
                if stops.len() < 2 {
                    return Err("渐变色标少于 2 个".into());
                }
                Paint::Gradient(std::sync::Arc::new(Gradient { kind, stops, repeat }))
            }
            2 => {
                // 图像：AMK1 不内嵌位图（见编码侧说明）。
                // 明确报错，而不是返回一个缺了图像的场景。
                let (w, h) = (r.u32()?, r.u32()?);
                return Err(format!(
                    "AMK1 流不内嵌位图，无法解码 {w}×{h} 的图像绘制；\
                     请在解码后重新插入图像"
                ));
            }
            other => return Err(format!("paint 类型不支持: {other}")),
        };        let flags = r.u32()?;
        // v2：bit 12 是 fill_rule；v1：该位保留，恒为 NonZero。
        let fill_rule = if v2 {
            FillRule::from_bit(flags >> FILL_RULE_BIT)
        } else {
            FillRule::NonZero
        };
        // blend id 只占低 4 位（0..15），掩掉标志位再查表。
        let blend_id = flags & 0x0FFF;
        let n_refs = r.u32()? as usize;
        let mut clips = Vec::with_capacity(n_refs);
        for _ in 0..n_refs {
            clips.push(r.u32()?);
        }
        let path = r.path()?;
        let blend = match blend_id {
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
            _ => return Err(format!("未知混合模式 id {blend_id}")),
        };
        scene.draws.push(Draw {
            transform: t,
            path,
            paint,
            fill_rule,
            blend,
            clips,
        });
    }
    Ok(scene)
}

// ============================================================================
// M4：damage region —— tile 级脏区模型
// ============================================================================
//
// 目的：让"与画面内容成比例"从**显存**扩展到 **CPU 时间与提交量**。
// 保留模式下，只重算被改动区域覆盖的 tile；未改动区域复用上一帧结果。
//
// 与分箱的一致性：tile 粒度必须等于分箱粒度，否则增量结果会与全量结果
// 边界不一致。`amiki-raster::TILE` / `amiki-gpu` 的分箱均取 16，此处同值。
// 注意这是**跨 crate 的约定**，若将来调整分箱 tile 尺寸，这里必须同步。

