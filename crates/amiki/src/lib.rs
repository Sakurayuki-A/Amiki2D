//! Amiki2D —— 下一代通用 2D 渲染引擎。
//!
//! 门面 API：上层只需构造 [`Canvas`]，产出 [`Scene`]，再交给 GPU / CPU 后端。
//! 详细架构见仓库 `docs/ARCHITECTURE.md`，算法推导见 `docs/RENDERING_STRATEGY.md`。
//!
//! # API 稳定性
//!
//! **本 crate 的公开项是稳定面**（0.3.x 内不破坏）。完整清单、实验面与内部面的
//! 划分见仓库 `docs/API.md`。
//!
//! `amiki::{gpu, math, raster, scene}` 是再导出，其中带诊断性质的项
//! （`FrameProfile`、`PipelineDump`、容量表等）属于**实验面** ——
//! 能跑、有测试，但不承诺兼容。

#![forbid(unsafe_code)]

#[cfg(feature = "gpu")]
pub use amiki_gpu as gpu;
pub use amiki_math as math;
pub use amiki_raster as raster;
pub use amiki_scene as scene;

use amiki_math::{Affine, Vec2};
use amiki_scene::{BlendMode, FillRule, Paint, Path, PathBuilder, Scene};

/// 立即模式画布：记录绘制指令，产出场景。
pub struct Canvas {
    pub scene: Scene,
    transform: Affine,
    blend: BlendMode,
    /// 当前填充规则（0.8.0）。状态式、粘性：设置后影响**后续**填充，
    /// 直到再次设置。只作用于填充，不影响描边（描边是轮廓化）。
    fill_rule: FillRule,
    clip_stack: Vec<u32>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Canvas {
        Canvas {
            scene: Scene::new(width, height),
            transform: Affine::IDENTITY,
            blend: BlendMode::Normal,
            fill_rule: FillRule::NonZero,
            clip_stack: Vec::new(),
        }
    }

    /// 设置**后续填充**的填充规则（0.8.0）。
    ///
    /// 状态式，与 [`Canvas::set_blend`] 一致：设置后一直生效，直到再次设置。
    /// 默认 [`FillRule::NonZero`] —— 不调用本方法的代码行为**逐字节不变**。
    ///
    /// ```ignore
    /// canvas.set_fill_rule(FillRule::EvenOdd);
    /// canvas.fill_path(&star, paint);   // 这条按奇偶填充
    /// canvas.set_fill_rule(FillRule::NonZero);
    /// ```
    pub fn set_fill_rule(&mut self, rule: FillRule) -> &mut Self {
        self.fill_rule = rule;
        self
    }

    /// 当前填充规则。
    pub fn fill_rule(&self) -> FillRule {
        self.fill_rule
    }

    /// 把当前填充规则写进**刚记录的那条** draw。
    ///
    /// 单独抽出来是因为填充有三个落点（`fill_path_with` 与两处文本），
    /// 漏掉任何一处都会表现为"某个入口的 even-odd 静默失效"。
    fn stamp_fill_rule(&mut self) {
        if let Some(d) = self.scene.draws.last_mut() {
            d.fill_rule = self.fill_rule;
        }
    }

    /// 设置后续绘制的混合模式。
    pub fn set_blend(&mut self, blend: BlendMode) -> &mut Self {
        self.blend = blend;
        self
    }

    /// 压入裁剪层：后续绘制被其裁剪（叠加到已有裁剪上，取交集）。
    pub fn push_clip(&mut self, path: &Path) -> &mut Self {
        let idx = self.scene.push_clip(self.transform, path.clone());
        self.clip_stack.push(idx);
        self
    }

    /// 弹出最近压入的裁剪层。
    pub fn pop_clip(&mut self) -> &mut Self {
        self.clip_stack.pop();
        self
    }

    pub fn set_transform(&mut self, t: Affine) -> &mut Self {
        self.transform = t;
        self
    }

    pub fn reset(&mut self) -> &mut Self {
        self.transform = Affine::IDENTITY;
        self
    }

    pub fn translate(&mut self, v: Vec2) -> &mut Self {
        self.transform = self.transform.then(Affine::translate(v));
        self
    }

    pub fn rotate(&mut self, theta: f32) -> &mut Self {
        self.transform = self.transform.then(Affine::rotate(theta));
        self
    }

    pub fn scale(&mut self, s: f32) -> &mut Self {
        self.transform = self.transform.then(Affine::scale(s));
        self
    }

    /// 绘制路径，并可指定**任意 paint**（含渐变/图像等非实心样式）。
    ///
    /// `fill_path` 是它的实心便捷形式。图像等样式需要它，因为
    /// `Paint::Image` 携带放置信息，不能由路径自身推导。
    pub fn fill_path_with(&mut self, path: &Path, paint: Paint) -> &mut Self {
        self.scene.push_blend(self.transform, path.clone(), paint, self.blend);
        self.stamp_fill_rule();
        if !self.clip_stack.is_empty() {
            self.scene.draws.last_mut().unwrap().clips = self.clip_stack.clone();
        }
        self
    }

    pub fn fill_path(&mut self, path: &Path, paint: Paint) -> &mut Self {
        self.fill_path_with(path, paint)
    }

    /// 描边一条路径，并可指定任意 paint（P0-1）。
    ///
    /// # 线宽是**用户单位**
    ///
    /// 描边在**用户空间**解析成填充轮廓，再由画布变换作用一次。
    /// 因此 `scale(3.0)` 会把线宽一起放大 3 倍、旋转与非等比缩放都正确，
    /// 而且**不会双重变换**（`stroke_path` 不接收变换，变换只在这里发生）。
    ///
    /// # 后端无感知
    ///
    /// 轮廓在提交前已退化成普通多边形，所以 CPU 与 GPU 两侧都**不需要**知道
    /// 描边的存在 —— 也就没有"两端各写一份描边逻辑"的漂移风险。
    ///
    /// # 精度
    ///
    /// 曲线先按 `eps_device / 变换尺度` 的容差压平（`eps_device` 取 0.1px），
    /// 于是设备空间的压平误差与填充路径同级。
    pub fn stroke_path_with(
        &mut self,
        path: &Path,
        style: amiki_scene::StrokeStyle,
        paint: Paint,
    ) -> &mut Self {
        const EPS_DEVICE: f32 = 0.1;
        let scale = self.transform.max_scale().max(1e-6);
        let outline = amiki_scene::stroke_path(path, &style, EPS_DEVICE / scale);
        self.fill_path_with(&outline, paint)
    }

    /// 描边的实心便捷形式（等价于 `stroke_path_with(path, style, Paint::Solid(color))`）。
    pub fn stroke_path(
        &mut self,
        path: &Path,
        style: amiki_scene::StrokeStyle,
        color: amiki_scene::Color,
    ) -> &mut Self {
        self.stroke_path_with(path, style, Paint::Solid(color))
    }

    /// 绘制一张位图（M6），放置到设备空间的矩形 `[x0,x1] × [y0,y1]`。
    ///
    /// 放置会经过画布当前变换，因此可以配合 `translate` / `rotate` / `scale`
    /// 自由摆放、旋转、缩放图像。
    ///
    /// 采样用**双线性**（缩放时不产生最近邻的锯齿与抖动）；
    /// 图像空间 y 向下，与引擎设备坐标一致，因此不需要翻转。
    pub fn image(
        &mut self,
        image: std::sync::Arc<amiki_scene::Image>,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) -> &mut Self {
        let place = amiki_scene::ImagePlacement::rect(x0, y0, x1, y1);
        self.image_with_placement(image, place)
    }

    /// 绘制位图并指定完整放置（含旋转/切变/平铺）。
    ///
    /// 放置的三个点会被画布当前变换作用 —— 与其它绘制保持一致。
    pub fn image_with_placement(
        &mut self,
        image: std::sync::Arc<amiki_scene::Image>,
        place: amiki_scene::ImagePlacement,
    ) -> &mut Self {
        let t = self.transform;
        let transformed = amiki_scene::ImagePlacement {
            origin: t.map_point(place.origin),
            u_axis: t.map_vec(place.u_axis),
            v_axis: t.map_vec(place.v_axis),
            repeat: place.repeat,
        };
        // 覆盖范围 = 放置四角的包围盒（供分箱与 damage region 使用）
        let mut pb = PathBuilder::new();
        let p = |u: f32, v: f32| {
            transformed.origin + transformed.u_axis * u + transformed.v_axis * v
        };
        let (a, b, c, d) = (p(0.0, 0.0), p(1.0, 0.0), p(1.0, 1.0), p(0.0, 1.0));
        let (mut x0, mut y0) = (a.x, a.y);
        let (mut x1, mut y1) = (a.x, a.y);
        for q in [b, c, d] {
            x0 = x0.min(q.x);
            y0 = y0.min(q.y);
            x1 = x1.max(q.x);
            y1 = y1.max(q.y);
        }
        pb.rect(x0, y0, x1, y1);
        self.fill_path_with(
            &pb.build(),
            amiki_scene::Paint::Image { image, placement: transformed },
        )
    }

    /// 推入一个**离屏层**：闭包里的所有绘制会先渲到独立缓冲，
    /// 整层套上 `filter`，再按 `opacity` 与 `blend` 合成回画布（M6）。
    ///
    /// 这是模糊/阴影的唯一实现途径 —— 它们要读取"周围像素"，
    /// 而周围像素只有整层画完之后才存在，无法逐图元计算。
    ///
    /// `bounds` 是**设备空间**的层范围，须由调用方给出：被模糊的层影响范围会
    /// 超出内容包围盒，引擎自行推导必须算准每个滤镜的外扩量，任何保守不足都会
    /// 让边缘被切掉且很难察觉。语义清楚比"少写一个参数"重要。
    ///
    /// ```ignore
    /// canvas.push_layer_filtered(
    ///     LayerBounds::new(40, 40, 200, 120),
    ///     Filter::Blur { sigma: 8.0 },
    ///     1.0,
    ///     BlendMode::Normal,
    ///     |c| { c.rect(40.0, 40.0, 240.0, 160.0, Paint::Solid(RED)); },
    /// );
    /// ```
    pub fn push_layer_filtered<F>(
        &mut self,
        bounds: amiki_scene::LayerBounds,
        filter: amiki_scene::Filter,
        opacity: f32,
        blend: amiki_scene::BlendMode,
        build: F,
    ) -> &mut Self
    where
        F: FnOnce(&mut Canvas),
    {
        // 层内的画布从"单位变换"开始：层的 bounds 已是设备空间，
        // 层内绘制用设备坐标即可（与父场景同一套坐标，减少一类坐标错误）。
        let mut sub = Canvas::new(self.scene.width, self.scene.height);
        sub.blend = blend;
        build(&mut sub);
        self.scene.push_layer(
            amiki_scene::Layer {
                bounds,
                scene: sub.scene,
                filter,
                opacity,
                blend,
            },
            |_| {},
        );
        self
    }

    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, paint: Paint) -> &mut Self {
        let mut pb = PathBuilder::new();
        pb.rect(x0, y0, x1, y1);
        self.fill_path(&pb.build(), paint)
    }

    pub fn circle(&mut self, c: Vec2, r: f32, paint: Paint) -> &mut Self {
        let mut pb = PathBuilder::new();
        pb.circle(c, r, false);
        self.fill_path(&pb.build(), paint)
    }

    pub fn pentagram(&mut self, c: Vec2, r: f32, paint: Paint) -> &mut Self {
        let mut pb = PathBuilder::new();
        pb.pentagram(c, r);
        self.fill_path(&pb.build(), paint)
    }

    /// 绘制一段**已 shaping** 的文本。
    ///
    /// # 为什么接收 `TextRun` 而不是字符串
    ///
    /// shaping（字符 → 带位置的字形序列）需要字体，那是 `amiki-text` 的职责。
    /// 门面若直接吃 `&str`，就不得不依赖字体栈，从而破坏核心的零依赖性质。
    /// 因此这里接收**排版结果**，字体相关的部分由调用方交给 `amiki-text`：
    ///
    /// ```ignore
    /// let run = amiki_text::shape(&font, "Hello", 48.0)?;   // 排版
    /// canvas.text(&run, Vec2::new(40.0, 120.0), paint, |id| font.outline(id).ok());  // 渲染
    /// ```
    ///
    /// 每个字形展开为**一条独立 draw**（而非合并成整行大路径），
    /// 这样每个字形可被分箱与 damage region 独立处理 —— 改一个字只脏一小块。
    ///
    /// `origin` 是**基线原点**（第一个字形的基线起点），不是包围盒左上角。
    ///
    /// `outline` 只在需要时按字形 id 取轮廓（配合调用方的缓存即可避免重复解析）。
    pub fn text<F>(
        &mut self,
        run: &amiki_scene::TextRun,
        origin: Vec2,
        paint: Paint,
        outline: F,
    ) -> &mut Self
    where
        F: FnMut(u32) -> Option<amiki_scene::GlyphOutline>,
    {
        // 文本自身的变换 = 画布当前变换 ∘ 基线平移
        let base = self.transform.then(Affine::translate(origin));
        let paths = amiki_scene::expand_text_run(run, outline);
        for (path, glyph_t) in paths {
            // `Paint` 非 Copy（渐变经 Arc 共享）；clone 只做一次原子加，可忽略。
            self.scene
                .push_blend(base.then(glyph_t), path, paint.clone(), self.blend);
            self.stamp_fill_rule();
            if !self.clip_stack.is_empty() {
                self.scene.draws.last_mut().unwrap().clips = self.clip_stack.clone();
            }
        }
        self
    }

    /// 绘制一段**已 shaping** 的文本，字形可按**字体下标**取轮廓（字体回退）。
    ///
    /// 与 [`Canvas::text`] 的唯一区别：`outline` 同时收到字形所属的**字体下标**
    /// （[`amiki_scene::PositionedGlyph::font_index`]）。多字体的一段文本里，
    /// 同一个字形 id 在不同字体中是**不同字形**，取轮廓必须带上它：
    ///
    /// ```ignore
    /// let run = amiki_text::shape_with_stack(&stack, "Hello 世界", 48.0, &opts)?;
    /// canvas.text_with_stack(&run, origin, paint, |fi, id| stack.get(fi)?.outline(id).ok());
    /// ```
    pub fn text_with_stack<F>(
        &mut self,
        run: &amiki_scene::TextRun,
        origin: Vec2,
        paint: Paint,
        outline: F,
    ) -> &mut Self
    where
        F: FnMut(usize, u32) -> Option<amiki_scene::GlyphOutline>,
    {
        let base = self.transform.then(Affine::translate(origin));
        let paths = amiki_scene::expand_text_run_with(run, outline);
        for (path, glyph_t) in paths {
            self.scene
                .push_blend(base.then(glyph_t), path, paint.clone(), self.blend);
            self.stamp_fill_rule();
            if !self.clip_stack.is_empty() {
                self.scene.draws.last_mut().unwrap().clips = self.clip_stack.clone();
            }
        }
        self
    }
}

/// 便捷入口：GPU 渲染为 RGBA8，GPU 不可用时自动落到 CPU 引擎。
#[cfg(feature = "gpu")]
pub fn render_to_rgba(canvas: &Canvas, flatten_eps: f32) -> Result<Vec<u8>, gpu::RenderError> {
    match pollster::block_on(gpu::GpuRenderer::new()) {
        Ok(r) => r.render_blocking(&canvas.scene, flatten_eps),
        Err(e) => {
            eprintln!("GPU 不可用（{e}），使用 CPU 引擎");
            Ok(render_cpu(canvas, flatten_eps))
        }
    }
}

/// 纯 CPU 渲染为 RGBA8：`gpu` feature 关掉时的唯一出口，也是
/// `render_to_rgba` 的 GPU 不可用回退路径。
///
/// 这条路径**在 WASM 上同样可用**，而且是一个好的验证基准 ——
/// 与 GPU 无关，所以原生与 wasm 的产物应当**逐字节一致**。
pub fn render_cpu(canvas: &Canvas, flatten_eps: f32) -> Vec<u8> {
    let binned = raster::bin(&canvas.scene, flatten_eps);
    raster::render(&binned)
}

/// 颜色快捷方式。
pub mod color {
    pub use amiki_scene::Color;
    pub const RED: Color = Color::rgb8(231, 76, 60);
    pub const ORANGE: Color = Color::rgb8(243, 156, 18);
    pub const GREEN: Color = Color::rgb8(46, 204, 113);
    pub const BLUE: Color = Color::rgb8(52, 152, 219);
    pub const PURPLE: Color = Color::rgb8(155, 89, 182);
    pub const INK: Color = Color::rgb8(16, 20, 24);
}
