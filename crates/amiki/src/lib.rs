//! Amiki2D —— 下一代通用 2D 渲染引擎。
//!
//! 门面 API：上层只需构造 [`Canvas`]，产出 [`Scene`]，再交给 GPU / CPU 后端。
//! 详细架构见仓库 `docs/ARCHITECTURE.md`，算法推导见 `docs/RENDERING_STRATEGY.md`。

#![forbid(unsafe_code)]

pub use amiki_gpu as gpu;
pub use amiki_math as math;
pub use amiki_raster as raster;
pub use amiki_scene as scene;

use amiki_math::{Affine, Vec2};
use amiki_scene::{BlendMode, Paint, Path, PathBuilder, Scene};

/// 立即模式画布：记录绘制指令，产出场景。
pub struct Canvas {
    pub scene: Scene,
    transform: Affine,
    blend: BlendMode,
    clip_stack: Vec<u32>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Canvas {
        Canvas {
            scene: Scene::new(width, height),
            transform: Affine::IDENTITY,
            blend: BlendMode::Normal,
            clip_stack: Vec::new(),
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
        if !self.clip_stack.is_empty() {
            self.scene.draws.last_mut().unwrap().clips = self.clip_stack.clone();
        }
        self
    }

    pub fn fill_path(&mut self, path: &Path, paint: Paint) -> &mut Self {
        self.fill_path_with(path, paint)
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
            if !self.clip_stack.is_empty() {
                self.scene.draws.last_mut().unwrap().clips = self.clip_stack.clone();
            }
        }
        self
    }
}

/// 便捷入口：GPU 渲染为 RGBA8，GPU 不可用时自动落到 CPU 引擎。
pub fn render_to_rgba(canvas: &Canvas, flatten_eps: f32) -> Result<Vec<u8>, gpu::RenderError> {
    match pollster::block_on(gpu::GpuRenderer::new()) {
        Ok(r) => r.render_blocking(&canvas.scene, flatten_eps),
        Err(e) => {
            eprintln!("GPU 不可用（{e}），使用 CPU 引擎");
            let binned = raster::bin(&canvas.scene, flatten_eps);
            Ok(raster::render(&binned))
        }
    }
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
