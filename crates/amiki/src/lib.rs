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

/// 状态栈的最大深度（0.9.0）。
///
/// # 为什么是 32
///
/// UI 框架的典型嵌套深度是 5–10 层（窗口 → 面板 → 分组 → 控件 → 文本），
/// 32 是 3–5 倍余量。**超过 32 说明调用方有 bug**（多半是漏了 `restore`）——
/// 这时显式报错比静默截断好：静默截断会让"少 restore 一次"这种 bug
/// 一直潜伏到某次重构之后才以别的方式爆出来。
pub const MAX_STACK_DEPTH: usize = 32;

/// 状态栈的错误。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanvasError {
    /// [`Canvas::save`] 时栈已满（≥ [`MAX_STACK_DEPTH`]）。
    StackOverflow,
    /// [`Canvas::restore`] 时栈空 —— 说明多 restore 了一次。
    StackUnderflow,
}

impl std::fmt::Display for CanvasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanvasError::StackOverflow => {
                write!(f, "状态栈已满（{MAX_STACK_DEPTH} 层）—— 多半是漏了 restore")
            }
            CanvasError::StackUnderflow => write!(f, "状态栈为空 —— restore 多于 save"),
        }
    }
}

impl std::error::Error for CanvasError {}

/// 一次 [`Canvas::save`] 记录下来的状态快照。
///
/// 四项：**变换、混合模式、裁剪栈、填充规则**。`restore` 用整份快照
/// **完全覆盖**当前状态（不是叠加）—— 这是 canvas 语义的标准定义。
#[derive(Clone, Debug)]
pub struct CanvasState {
    transform: Affine,
    blend: BlendMode,
    /// 裁剪栈的克隆（引用 `Scene::clips` 的下标）。
    clips: Vec<u32>,
    fill_rule: FillRule,
}

/// 立即模式画布：记录绘制指令，产出场景。
pub struct Canvas {
    pub scene: Scene,
    /// 当前状态的**唯一来源**：所有 setter 都改这里，`restore` 整体写回这里。
    state: CanvasState,
    /// 状态栈。`save` 压入当前状态，`restore` 弹出并覆盖。
    ///
    /// **不进 AMK1**：栈是录制期的概念。编码时 Draw 已带完整状态、栈已展开。
    state_stack: Vec<CanvasState>,
    /// 颜色矩阵滤镜（0.10.5）。**不在状态栈里**，见 `set_color_filter`。
    color_filter: Option<amiki_scene::ColorMatrix>,
}

impl Canvas {
    pub fn new(width: u32, height: u32) -> Canvas {
        Canvas {
            scene: Scene::new(width, height),
            state: CanvasState {
                transform: Affine::IDENTITY,
                blend: BlendMode::Normal,
                clips: Vec::new(),
                fill_rule: FillRule::NonZero,
            },
            state_stack: Vec::new(),
            color_filter: None,
        }
    }

    // ---------------------------------------------------------------- 状态栈

    /// 保存当前状态（变换 / 混合模式 / 裁剪栈 / 填充规则）到栈上。
    ///
    /// 与 [`Canvas::restore`] 配对使用，是 UI 框架的基本假设：
    /// 每个控件在自己的局部状态里画完就收回去，不必手工记账。
    ///
    /// 栈已满（≥ [`MAX_STACK_DEPTH`]）返回 [`CanvasError::StackOverflow`]，
    /// **不静默丢弃**。
    ///
    /// ```ignore
    /// canvas.save()?;
    /// canvas.set_transform(局部变换).set_blend(BlendMode::Multiply);
    /// canvas.fill_path(&path, paint);
    /// canvas.restore()?;          // 四项一起回到保存点
    /// ```
    pub fn save(&mut self) -> Result<(), CanvasError> {
        if self.state_stack.len() >= MAX_STACK_DEPTH {
            return Err(CanvasError::StackOverflow);
        }
        self.state_stack.push(self.state.clone());
        Ok(())
    }

    /// 恢复最近一次 [`Canvas::save`] 的状态。
    ///
    /// **整体覆盖**（不是叠加）：四项一起回到保存点。栈空返回
    /// [`CanvasError::StackUnderflow`]，**不静默 no-op** ——
    /// 多 restore 一次是实实在在的 bug，吞掉它只会让问题延后爆发。
    pub fn restore(&mut self) -> Result<(), CanvasError> {
        match self.state_stack.pop() {
            Some(saved) => {
                self.state = saved;
                Ok(())
            }
            None => Err(CanvasError::StackUnderflow),
        }
    }

    /// 当前状态栈深度（已 save 未 restore 的层数）。
    pub fn save_depth(&self) -> usize {
        self.state_stack.len()
    }

    /// 当前裁剪栈深度。
    ///
    /// C ABI 需要它：`restore` 会把裁剪栈一起回退，C 侧那本"clip 深度"账
    /// 必须跟着同步，否则 `amiki_canvas_pop_clip` 会误判下溢。
    pub fn clip_depth(&self) -> usize {
        self.state.clips.len()
    }

    /// 设置**后续填充**的填充规则（0.8.0）。
    ///
    /// 状态式，与 [`Canvas::set_blend`] 一致：设置后一直生效，直到再次设置
    /// 或被 [`Canvas::restore`] 回退。默认 [`FillRule::NonZero`] ——
    /// 不调用本方法的代码行为**逐字节不变**。
    pub fn set_fill_rule(&mut self, rule: FillRule) -> &mut Self {
        self.state.fill_rule = rule;
        self
    }

    /// 当前填充规则。
    pub fn fill_rule(&self) -> FillRule {
        self.state.fill_rule
    }

    /// 把当前填充规则写进**刚记录的那条** draw。
    ///
    /// 单独抽出来是因为填充有三个落点（`fill_path_with` 与两处文本），
    /// 漏掉任何一处都会表现为"某个入口的 even-odd 静默失效"。
    fn stamp_fill_rule(&mut self) {
        let rule = self.state.fill_rule;
        if let Some(d) = self.scene.draws.last_mut() {
            d.fill_rule = rule;
        }
    }

    /// 设置后续绘制的混合模式。
    pub fn set_blend(&mut self, blend: BlendMode) -> &mut Self {
        self.state.blend = blend;
        self
    }

    /// 压入裁剪层：后续绘制被其裁剪（叠加到已有裁剪上，取交集）。
    pub fn push_clip(&mut self, path: &Path) -> &mut Self {
        let idx = self.scene.push_clip(self.state.transform, path.clone());
        self.state.clips.push(idx);
        self
    }

    /// 弹出最近压入的裁剪层。
    pub fn pop_clip(&mut self) -> &mut Self {
        self.state.clips.pop();
        self
    }

    pub fn set_transform(&mut self, t: Affine) -> &mut Self {
        self.state.transform = t;
        self
    }

    pub fn reset(&mut self) -> &mut Self {
        self.state.transform = Affine::IDENTITY;
        self
    }

    pub fn translate(&mut self, v: Vec2) -> &mut Self {
        self.state.transform = self.state.transform.then(Affine::translate(v));
        self
    }

    pub fn rotate(&mut self, theta: f32) -> &mut Self {
        self.state.transform = self.state.transform.then(Affine::rotate(theta));
        self
    }

    pub fn scale(&mut self, s: f32) -> &mut Self {
        self.state.transform = self.state.transform.then(Affine::scale(s));
        self
    }

    /// 绘制路径，并可指定**任意 paint**（含渐变/图像等非实心样式）。
    ///
    /// `fill_path` 是它的实心便捷形式。图像等样式需要它，因为
    /// `Paint::Image` 携带放置信息，不能由路径自身推导。
    pub fn fill_path_with(&mut self, path: &Path, paint: Paint) -> &mut Self {
        let paint = self.filtered_paint(paint);
        self.scene.push_blend(self.state.transform, path.clone(), paint, self.state.blend);
        self.stamp_fill_rule();
        if !self.state.clips.is_empty() {
            self.scene.draws.last_mut().unwrap().clips = self.state.clips.clone();
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
        let scale = self.state.transform.max_scale().max(1e-6);
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

    /// **颜色矩阵滤镜**（0.10.5）：对**已解析的颜色**生效，不改任何着色管线。
    ///
    /// 传 `None` 关闭。作用范围与代价**都写在明处**：
    ///
    /// | 绘制类型 | 处理方式 | 代价 |
    /// |---|---|---|
    /// | `Paint::Solid` | 提交时变换颜色 | 无 |
    /// | `Paint::Gradient` | 提交时变换**每个色标** | 一份渐变定义的浅拷贝 |
    /// | `Paint::Image` | 提交时**逐像素**变换（**源图不被修改**） | 与像素数成正比 |
    ///
    /// 三类都处理是为了**不留静默缺口** —— 宁可付代价，也不出现"滤镜对图像不生效
    /// 但没人告诉你"。图像很大时请自行权衡（大图逐帧过滤是真实的开销）。
    ///
    /// **不在 `save` / `restore` 的状态栈里**（状态栈是四项：变换 / 混合 / 裁剪 /
    /// 填充规则）。这一点是**有意**的、也写在这里：需要"局部滤镜"请自行存取
    /// [`Canvas::color_filter`] 的返回值。
    ///
    /// 变换在**提交时**发生 ⇒ CPU / GPU 两侧看到的仍是普通 `Paint`，
    /// **不需要知道滤镜存在**（与描边轮廓化、九宫格同一个思路）。
    pub fn set_color_filter(&mut self, m: Option<amiki_scene::ColorMatrix>) -> &mut Self {
        self.color_filter = m;
        self
    }

    /// 当前颜色滤镜（`None` = 未设置）。
    pub fn color_filter(&self) -> Option<amiki_scene::ColorMatrix> {
        self.color_filter
    }

    /// 把滤镜应用到一份 `Paint`（无滤镜时原样返回，**零拷贝**）。
    fn filtered_paint(&self, paint: Paint) -> Paint {
        let Some(m) = self.color_filter else {
            return paint;
        };
        match paint {
            Paint::Solid(c) => Paint::Solid(m.apply(c)),
            Paint::Gradient(g) => {
                let mut g2 = (*g).clone();
                for s in &mut g2.stops {
                    s.color = m.apply(s.color);
                }
                Paint::Gradient(std::sync::Arc::new(g2))
            }
            Paint::Image { image, placement } => {
                // 逐像素变换；**克隆后改**，源图保持不变
                let mut img2 = (*image).clone();
                for px in img2.data.chunks_mut(4) {
                    if px.len() != 4 {
                        continue;
                    }
                    let c = amiki_scene::Color {
                        r: px[0] as f32 / 255.0,
                        g: px[1] as f32 / 255.0,
                        b: px[2] as f32 / 255.0,
                        a: px[3] as f32 / 255.0,
                    };
                    let o = m.apply(c);
                    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    px[0] = to_u8(o.r);
                    px[1] = to_u8(o.g);
                    px[2] = to_u8(o.b);
                    px[3] = to_u8(o.a);
                }
                Paint::Image {
                    image: std::sync::Arc::new(img2),
                    placement,
                }
            }
        }
    }

    /// **九宫格贴图**（0.10.3）：把源图按 `insets` 切成 9 块铺满 `dst`。
    ///
    /// 四角**保持源尺寸不缩放**、四边单向拉伸、中间双向拉伸 —— 与 Skia
    /// `drawImageNine` / Android 9-patch 同义。UI 里"一个圆角面板贴图适配任意
    /// 尺寸"就是靠它：圆角与边框装饰在放大时**不变形**。
    ///
    /// 实现方式：按 [`amiki_scene::nine_patch`] 算出的 9 组矩形，**取子图**后用
    /// 最普通的 [`Canvas::image`]（矩形放置）逐块画 —— 不涉及 uv 子区间的任何约定，
    /// 因此 CPU / GPU 两侧与 AMK1 编码都不需要知道九宫格存在。
    pub fn image_nine(
        &mut self,
        image: std::sync::Arc<amiki_scene::Image>,
        insets: amiki_scene::Insets,
        dst: (f32, f32, f32, f32),
    ) -> &mut Self {
        for slice in amiki_scene::nine_patch(image.width, image.height, insets, dst) {
            let Some(sub) = amiki_scene::sub_image(&image, slice.src) else {
                continue; // 越界块：跳过（显式，不画半张图）
            };
            let (x0, y0, x1, y1) = slice.dst;
            self.image(std::sync::Arc::new(sub), x0, y0, x1, y1);
        }
        self
    }

    /// 描边的**虚线版**（0.10.2）：`pattern` 是交替的「实 / 虚」长度。
    ///
    /// 与 [`Canvas::stroke_path_with`] 的关系是"先切成虚线、再走同一条描边路径" ——
    /// 端帽 / 连接 / miter limit 语义完全一致，只是每段是**开口**的，各自带端帽
    /// （与 SVG / Skia 一致）。**奇数个元素**视为把列表重复一遍。
    ///
    /// 返回 `None` = **虚线模式非法**（空 / 全 0 / 含负数或非有限值），
    /// 此时**什么都不画** —— 显式拒绝，不静默按实线画。
    ///
    /// ```ignore
    /// canvas.stroke_path_dashed(&rect, style, Color::rgb8(200,70,40), &[14.0, 8.0]);
    /// ```
    pub fn stroke_path_dashed(
        &mut self,
        path: &Path,
        style: amiki_scene::StrokeStyle,
        color: amiki_scene::Color,
        pattern: &[f32],
    ) -> Option<&mut Self> {
        const EPS_DEVICE: f32 = 0.1;
        let scale = self.state.transform.max_scale().max(1e-6);
        let outline =
            amiki_scene::stroke_path_dashed(path, &style, pattern, EPS_DEVICE / scale)?;
        self.fill_path(&outline, Paint::Solid(color));
        Some(self)
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
        let t = self.state.transform;
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
        sub.state.blend = blend;
        // **初始状态继承**：子画布继承父画布的**填充规则**（与 C ABI 的
        // `amiki_canvas_begin_layer` 一致 —— 0.8.0 只改了 ABI 那侧，门面漏了，
        // 于是"门面 + EvenOdd + 层"会静默退回 NonZero，本轮补齐）。
        //
        // 不继承的是：变换（层内一律从单位变换开始，见上）、裁剪栈
        // （层内容的裁剪由合成侧决定）、以及**状态栈本身** ——
        // 子画布的栈是空的，层内的 save/restore 只影响层内。
        sub.state.fill_rule = self.state.fill_rule;
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
        let base = self.state.transform.then(Affine::translate(origin));
        let paths = amiki_scene::expand_text_run(run, outline);
        for (path, glyph_t) in paths {
            // `Paint` 非 Copy（渐变经 Arc 共享）；clone 只做一次原子加，可忽略。
            self.scene
                .push_blend(base.then(glyph_t), path, self.filtered_paint(paint.clone()), self.state.blend);
            self.stamp_fill_rule();
            if !self.state.clips.is_empty() {
                self.scene.draws.last_mut().unwrap().clips = self.state.clips.clone();
            }
        }
        self
    }

    /// 与 [`Canvas::text`] 同样绘制，但**用调用方持有的 [`amiki_scene::TextCache`]** 复用字形轮廓。
    ///
    /// # 为什么缓存必须由调用方持有（0.10.0，见 `docs/KNOWN_ISSUES.md` 的 D8）
    ///
    /// 缓存键是 [`amiki_scene::GlyphKey`] = `{ font_id, 字形 id, 字号 }`。
    /// 门面的 `text()` 只拿得到"按字形 id 取轮廓"的闭包，**拿不到字体身份** ——
    /// 若在门面内部自建缓存，就只能把 `font_id` 填成 `FontId::UNKNOWN`，
    /// 于是"字体 A 的字形 5"与"字体 B 的字形 5"在**同字号**下会命中同一条 ⇒
    /// **静默画错**。所以缓存交给调用方：**谁知道字体身份，谁持有缓存**。
    ///
    /// 典型用法（UI 每帧重画同一批文字）：
    ///
    /// ```ignore
    /// let mut cache = TextCache::new();
    /// loop {
    ///     canvas.text_cached(&mut cache, &run, origin, paint.clone(), |id| font.outline(id).ok());
    /// }
    /// // cache.hits() 会随帧数增长，misses 只在首次或换字号时增加
    /// ```
    ///
    /// 输出与 [`Canvas::text`] **逐字段相同**（同一份展开逻辑，只是查了缓存）——
    /// 有测试断言两者的 `Draw` 完全相等。
    pub fn text_cached<F>(
        &mut self,
        cache: &mut amiki_scene::TextCache,
        run: &amiki_scene::TextRun,
        origin: Vec2,
        paint: Paint,
        outline: F,
    ) -> &mut Self
    where
        F: FnMut(u32) -> Option<amiki_scene::GlyphOutline>,
    {
        let base = self.state.transform.then(Affine::translate(origin));
        let paths = cache.expand(run, outline);
        for (path, glyph_t) in paths {
            self.scene
                .push_blend(base.then(glyph_t), path, self.filtered_paint(paint.clone()), self.state.blend);
            self.stamp_fill_rule();
            if !self.state.clips.is_empty() {
                self.scene.draws.last_mut().unwrap().clips = self.state.clips.clone();
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
        let base = self.state.transform.then(Affine::translate(origin));
        let paths = amiki_scene::expand_text_run_with(run, outline);
        for (path, glyph_t) in paths {
            self.scene
                .push_blend(base.then(glyph_t), path, self.filtered_paint(paint.clone()), self.state.blend);
            self.stamp_fill_rule();
            if !self.state.clips.is_empty() {
                self.scene.draws.last_mut().unwrap().clips = self.state.clips.clone();
            }
        }
        self
    }
}

/// 进程级共享的 GPU 渲染器（0.10.8）。
///
/// # 为什么必须共享（这是修一个真实的性能 bug）
///
/// 原先本函数**每次调用**都执行 `GpuRenderer::new_blocking()` —— 那是一次完整的
/// wgpu 设备/队列/适配器初始化。实测（`tests/gpu_fixed_cost.rs`）：
/// 64×64 两个矩形每次都要 **87–138 ms**，而 CPU 做同一件事只要 **0.15 ms**；
/// 且耗时**与尺寸内容完全无关** ⇒ 正是嵌入式反馈里"与尺寸无关的 ~20 ms 停顿"的形状。
///
/// 现在初始化只做一次；失败也**缓存**（`Err(String)`），避免在没有 GPU 的机器上
/// 每次调用都重试一次昂贵的初始化。
#[cfg(feature = "gpu")]
fn shared_gpu_renderer() -> Result<&'static std::sync::Mutex<gpu::GpuRenderer>, String> {
    static CELL: std::sync::OnceLock<Result<std::sync::Mutex<gpu::GpuRenderer>, String>> =
        std::sync::OnceLock::new();
    match CELL.get_or_init(|| {
        gpu::GpuRenderer::new_blocking()
            .map(std::sync::Mutex::new)
            .map_err(|e| format!("{e:?}"))
    }) {
        Ok(m) => Ok(m),
        Err(msg) => Err(msg.clone()),
    }
}

/// GPU 路径**已验证支持的最大边长**（0.10.8）。
///
/// 超过它的画布会**明确回落到 CPU**，而不是静默画错。理由（实测，见
/// `crates/amiki/tests/consistency_probe.rs`）：3840×2160 的画布，GPU 会在
/// **`y ≥ 2048` 的横带**上给出错误像素（最差像素差 255、`>16` 的通道 19.4 万个、
/// 错误聚集在第 128 行瓦片起 —— `2048 / TILE(16) = 128`）。
/// 在查清并修掉那个上限之前，**宁可少用 GPU，也不输出错的像素**。
#[cfg(feature = "gpu")]
pub const GPU_MAX_SUPPORTED_SIDE: u32 = 2048;

/// 便捷入口：GPU 渲染为 RGBA8；GPU 不可用、或画布超出**已验证范围**时回落 CPU 引擎。
///
/// 渲染器是**进程级共享**的（见私有函数 `shared_gpu_renderer`）；锁只在一帧渲染期间持有。
#[cfg(feature = "gpu")]
pub fn render_to_rgba(canvas: &Canvas, flatten_eps: f32) -> Result<Vec<u8>, gpu::RenderError> {
    let (w, h) = (canvas.scene.width, canvas.scene.height);
    if w > GPU_MAX_SUPPORTED_SIDE || h > GPU_MAX_SUPPORTED_SIDE {
        eprintln!(
            "画布 {w}×{h} 超出 GPU 路径已验证范围（≤ {GPU_MAX_SUPPORTED_SIDE}），\
             已知 y≥2048 会算错；使用 CPU 引擎"
        );
        return Ok(render_cpu(canvas, flatten_eps));
    }
    match shared_gpu_renderer() {
        Ok(cell) => {
            // 锁中毒时取回内部值继续，不 panic（与 C ABI 的 Poison 机制同一态度）
            let r = cell.lock().unwrap_or_else(|p| p.into_inner());
            r.render_blocking(&canvas.scene, flatten_eps)
        }
        Err(msg) => {
            eprintln!("GPU 不可用（{msg}），使用 CPU 引擎");
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
