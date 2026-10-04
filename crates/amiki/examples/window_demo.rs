//! 窗口动画示例：winit + wgpu 表面，每帧重建场景（旋转五角星 + 环绕圆）。
//!
//! ```bash
//! cargo run --release -p amiki --example window_demo
//! ```

use amiki::color::*;
use amiki::gpu::{Blitter, GpuRenderer};
use amiki::math::Vec2;
use amiki::scene::{edit_damage, BlendMode, Damage, Paint, PathBuilder};
use amiki::Canvas;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct FrameState {
    window: std::sync::Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    renderer: GpuRenderer,
    /// **跨帧复用的 `Frame`** —— 内部约 7 MB 缓冲不再每帧新建。
    ///
    /// 这是逐帧渲染的**正确用法**：`prepare()` 每次都会重新分配，
    /// 而大于 128 KB 的分配走 `mmap` + 缺页，实测同一批数据慢 **5.3 倍**
    /// （见 `GpuScene::reset_for_reuse` 的实测表）。
    frame: amiki::gpu::Frame,
    blitter: Option<Blitter>,
    scene_tex: Option<(wgpu::Texture, wgpu::TextureView)>,
    scene_size: (u32, u32),
    start: Instant,
    /// M4：上一帧的场景，用于推导脏区（`edit_damage` 做差分）
    prev_scene: Option<amiki::scene::Scene>,
    /// 最近一帧的耗时（ms）与脏 tile 占比，写进标题便于观察 M4 收益
    last_ms: f64,
    last_dirty_ratio: f64,
    /// M5：字体与字形轮廓缓存（轮廓解析一次后复用，避免每帧重新解析）。
    /// 用 `RefCell`：`build_scene` 是 `&self`，而缓存的填充需要内部可变性。
    ///
    /// 0.7.0 起是**有序字体栈**：拉丁走 Inter、汉字走 Noto Sans SC。
    /// 顺序即优先级（谁先压谁赢）。
    fonts: amiki_text::FontStack,
    /// shaping 选项：缺字画 `.notdef` 方框而不是让整段文字消失 ——
    /// 示例要能一眼看出"哪个字没字体"，而不是"文字不见了"。
    text_opts: amiki_text::TextOptions,
    text_cache: std::cell::RefCell<OutlineCache>,
    /// dogfooding 列表视图的状态
    show_list: bool,
    scroll: f32,
    selected: Option<usize>,
    cursor: Vec2,
}

/// 字形轮廓缓存：`(字体 ID, glyph id) → 轮廓`。
///
/// 为什么必须缓存：轮廓解析（含二次曲线还原）比拷贝一条路径贵得多，
/// 而同一个字形每帧都会被复用。缓存让"每帧重新 shaping"变得廉价。
/// 用 `Vec` 而非 `HashMap`：字形数量小（几十到几百），线性查找更快且无哈希依赖。
///
/// **键必须带字体**：同一个 glyph id 在 Inter 与 Noto 里是不同字形，
/// 只按 id 分键的话，遇到两边都有的码点（如数字、拉丁字母在 Noto 里也有）
/// 就会把甲字体的轮廓用在乙字体上 —— 画出来是"串形"的字。
/// 用**字体内部 ID** 而不是栈下标：下标会随栈的调整而变，内部 ID 不会。
#[derive(Default)]
struct OutlineCache {
    entries: Vec<((amiki_text::FontId, u32), amiki::scene::GlyphOutline)>,
}

impl OutlineCache {
    fn get_or_insert(
        &mut self,
        stack: &amiki_text::FontStack,
        font_index: usize,
        id: u32,
    ) -> Option<amiki::scene::GlyphOutline> {
        let font = stack.get(font_index)?;
        let key = (font.id(), id);
        if let Some((_, o)) = self.entries.iter().find(|(k, _)| *k == key) {
            return Some(o.clone());
        }
        let o = font.outline(id).ok()?;
        self.entries.push((key, o.clone()));
        Some(o)
    }
}

impl FrameState {
    fn new(el: &ActiveEventLoop) -> FrameState {
        let window = std::sync::Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Amiki2D — GPU 计算着色器 2D 渲染")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
            )
            .expect("创建窗口失败"),
        );
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("创建表面失败");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("无 GPU 适配器");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("amiki2d"),
            required_features: wgpu::Features::empty(),
            // **必须用引擎给的上限**：`Limits::default()` 只保证 8 个 storage
            // buffer/阶段，而引擎需要 9 个（第 9 个是图像放置参数）。
            // 这里曾经写成 `Limits::default()`，于是示例一启动就 panic ——
            // 而且因为示例不在测试覆盖里，一直没被发现（dogfooding 才跑出来）。
            required_limits: amiki::gpu::required_limits(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .expect("设备创建失败");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_cfg);

        let renderer = GpuRenderer::from_device(device, queue).expect("渲染器初始化失败");
        // 字体栈：**有序**（先压的先赢）。Inter 管拉丁，Noto Sans SC 管汉字
        // —— 这正是 0.7.0 的字体回退：一段文本里不同字符由不同字体渲染。
        // 缺字体时示例仍应能演示图形能力，所以这里是"能加载几个算几个"。
        let mut fonts = amiki_text::FontStack::new();
        for path in [
            "assets/fonts/Inter-Regular.ttf",
            "assets/fonts/NotoSansSC-Regular.ttf",
        ] {
            match amiki_text::Font::from_file(path) {
                Ok(f) => {
                    let idx = fonts.push(f);
                    let f = fonts.get(idx).expect("刚压入");
                    println!(
                        "已加载字体 #{idx}：{path}（{}），units_per_em = {}, 字形数 = {}",
                        f.id_name(),
                        f.units_per_em(),
                        f.glyph_count()
                    );
                }
                Err(e) => eprintln!("未加载字体 {path}（{e}）"),
            }
        }
        if fonts.is_empty() {
            eprintln!("一个字体都没加载成功；文本部分将跳过");
        }
        FrameState {
            window,
            surface,
            surface_cfg,
            renderer,
            frame: amiki::gpu::Frame::default(),
            blitter: None,
            scene_tex: None,
            scene_size: (0, 0),
            start: Instant::now(),
            prev_scene: None,
            last_ms: 0.0,
            last_dirty_ratio: 1.0,
            fonts,
            // 缺字画方框：示例里宁可看见"这里有个字没字体"，也不要整段消失
            text_opts: amiki_text::TextOptions::default()
                .with_missing_glyph(amiki_text::MissingGlyphPolicy::NotDef),
            text_cache: std::cell::RefCell::new(OutlineCache::default()),
            // 默认进**应用视图**（列表）；展示页用 Tab 切换。
            show_list: true,
            scroll: 0.0,
            selected: None,
            cursor: Vec2::ZERO,
        }
    }

    /// **dogfooding：一个真实的应用视图** —— 可滚动、可点击选中的长列表。
    ///
    /// 这一页的存在意义不是"再画点东西"，而是**用引擎真的做一个应用**，
    /// 从而暴露示例画面覆盖不到的问题。它同时压到了三条主线：
    ///
    /// - **文字**：每行一段 shaping（M5）；
    /// - **增量**：改一行只脏那一行，滚动才脏一大片（M4）；
    /// - **内容规模**：200 行只画可见的那些（与稀疏分箱同源的思想）。
    fn build_list(&self) -> amiki::scene::Scene {
        const ROW_H: f32 = 30.0;
        const ROWS: usize = 200;
        const HEADER: f32 = 44.0;

        let w = self.surface_cfg.width as f32;
        let h = self.surface_cfg.height as f32;
        let mut canvas = Canvas::new(self.surface_cfg.width, self.surface_cfg.height);
        canvas.rect(0.0, 0.0, w, h, Paint::Solid(INK));

        // 只画可见行：内容再多，绘制量也只与**窗口高度**有关。
        let first = (self.scroll / ROW_H).floor().max(0.0) as usize;
        let last = (((self.scroll + h) / ROW_H).ceil() as usize).min(ROWS);
        for i in first..last {
            let y = HEADER + i as f32 * ROW_H - self.scroll;
            if y > h || y + ROW_H < 0.0 {
                continue;
            }
            let sel = self.selected == Some(i);
            let bg = if sel {
                Color::rgba8(41, 128, 185, 255)
            } else if i % 2 == 0 {
                Color::rgba8(28, 34, 40, 255)
            } else {
                Color::rgba8(22, 27, 32, 255)
            };
            canvas.rect(0.0, y, w, y + ROW_H - 1.0, Paint::Solid(bg));
            // 左侧色条：颜色按行号变化，滚动时一眼能看出是否对齐
            let hue = (i as f32) / ROWS as f32;
            canvas.rect(
                12.0,
                y + 6.0,
                22.0,
                y + ROW_H - 7.0,
                Paint::Solid(Color { r: 0.2 + 0.7 * hue, g: 0.7 - 0.4 * hue, b: 0.9 - 0.5 * hue, a: 1.0 }),
            );
            if !self.fonts.is_empty() {
                // 中英混排：汉字由 Noto 渲染、拉丁与数字由 Inter 渲染
                let label = format!("第 {i} 行 · Amiki2D 渲染引擎 dogfooding");
                if let Ok(run) =
                    amiki_text::shape_with_stack(&self.fonts, &label, 15.0, &self.text_opts)
                {
                    canvas.text_with_stack(
                        &run,
                        Vec2::new(34.0, y + 20.0),
                        Paint::Solid(if sel {
                            Color::rgba8(250, 250, 250, 255)
                        } else {
                            Color::rgba8(200, 208, 214, 255)
                        }),
                        |fi, id| {
                            self.text_cache
                                .borrow_mut()
                                .get_or_insert(&self.fonts, fi, id)
                        },
                    );
                }
            }
        }

        // 固定表头：**不随滚动移动**，于是滚动时它是唯一"每帧相同"的部分，
        // 增量渲染会把它排除在脏区之外。
        canvas.rect(0.0, 0.0, w, HEADER, Paint::Solid(Color::rgba8(12, 15, 18, 255)));
        if !self.fonts.is_empty() {
            let label = format!(
                "Amiki2D 列表 · {ROWS} 行 · 滚动 {:.0} · Tab 切换视图 · 滚轮/↑↓ 滚动 · 点击选中",
                self.scroll
            );
            if let Ok(run) = amiki_text::shape_with_stack(&self.fonts, &label, 15.0, &self.text_opts)
            {
                canvas.text_with_stack(
                    &run,
                    Vec2::new(16.0, 28.0),
                    Paint::Solid(Color::rgba8(236, 240, 241, 255)),
                    |fi, id| {
                        self.text_cache
                            .borrow_mut()
                            .get_or_insert(&self.fonts, fi, id)
                    },
                );
            }
        }
        canvas.scene
    }

    /// 列表的滚动上限（内容高度超出窗口的部分）。
    fn max_scroll(&self) -> f32 {
        const ROW_H: f32 = 30.0;
        const ROWS: usize = 200;
        const HEADER: f32 = 44.0;
        (ROWS as f32 * ROW_H + HEADER - self.surface_cfg.height as f32).max(0.0)
    }

    fn scroll_by(&mut self, dy: f32) {
        let m = self.max_scroll();
        self.scroll = (self.scroll - dy).clamp(0.0, m);
    }

    /// 把窗口坐标换算成列表行号（考虑滚动与表头）。
    fn row_at(&self, p: Vec2) -> Option<usize> {
        const ROW_H: f32 = 30.0;
        const ROWS: usize = 200;
        const HEADER: f32 = 44.0;
        if p.y < HEADER {
            return None;
        }
        let idx = ((p.y + self.scroll - HEADER) / ROW_H).floor();
        if idx < 0.0 {
            None
        } else {
            let i = idx as usize;
            (i < ROWS).then_some(i)
        }
    }

    fn build_scene(&self, t: f32) -> amiki::scene::Scene {
        let w = self.surface_cfg.width as f32;
        let h = self.surface_cfg.height as f32;
        let mut canvas = Canvas::new(self.surface_cfg.width, self.surface_cfg.height);
        canvas.rect(0.0, 0.0, w, h, Paint::Solid(INK));

        // 1) 旋转五角星（左上，自相交 nonzero 填充）
        canvas.translate(Vec2::new(w * 0.2, h * 0.3));
        canvas.rotate(t * 0.6);
        canvas.pentagram(Vec2::ZERO, 150.0, Paint::Solid(RED));

        // 2) 环绕半透明圆（src-over 合成）
        canvas.reset();
        let cx = w * 0.5;
        let cy = h * 0.3;
        for i in 0..6 {
            let a = t * 1.2 + i as f32 * std::f32::consts::TAU / 6.0;
            canvas.circle(
                Vec2::new(cx + a.cos() * 260.0, cy + a.sin() * 150.0),
                34.0,
                Paint::Solid(Color { r: BLUE.r, g: BLUE.g, b: BLUE.b, a: 0.7 }),
            );
        }

        // 3) 圆环（反绕挖洞）
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(w * 0.84, h * 0.18), 62.0, false);
        pb.circle(Vec2::new(w * 0.84, h * 0.18), 30.0, true);
        canvas.fill_path(&pb.build(), Paint::Solid(GREEN));

        // 4) 裁剪展示：移动圆窗内的旋转星（M2d）
        let clip_c = Vec2::new(w * 0.18 + (t * 0.9).sin() * 70.0, h * 0.66);
        let mut pb = PathBuilder::new();
        pb.circle(clip_c, 100.0, false);
        canvas.push_clip(&pb.build());
        canvas.translate(clip_c);
        canvas.rotate(-t * 0.8);
        canvas.pentagram(
            Vec2::ZERO,
            140.0,
            Paint::Solid(Color { r: PURPLE.r, g: PURPLE.g, b: PURPLE.b, a: 0.95 }),
        );
        canvas.reset();
        canvas.pop_clip();
        // 裁剪窗边界（细环微光）
        let mut pb = PathBuilder::new();
        pb.circle(clip_c, 100.0, false);
        pb.circle(clip_c, 96.0, true);
        canvas.fill_path(&pb.build(), Paint::Solid(Color { r: 1.0, g: 1.0, b: 1.0, a: 0.12 }));

        // 5) 混合模式画廊（M2e）：彩色底条 + 8 种模式动态色块
        let gy = h * 0.82;
        let gh = h * 0.15;
        let cols = [RED, ORANGE, GREEN, BLUE, PURPLE];
        let bw = w * 0.9 / 5.0;
        for (i, c) in cols.iter().enumerate() {
            let x0 = w * 0.05 + i as f32 * bw;
            canvas.rect(x0, gy, x0 + bw, gy + gh, Paint::Solid(*c));
        }
        let modes = [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Overlay,
            BlendMode::Difference,
            BlendMode::Exclusion,
            BlendMode::Hue,
            BlendMode::Luminosity,
        ];
        let tw = w * 0.9 / modes.len() as f32;
        for (i, m) in modes.iter().enumerate() {
            let x0 = w * 0.05 + i as f32 * tw;
            let y0 = gy + (t * 2.0 + i as f32 * 0.8).sin() * 8.0;
            canvas.set_blend(*m);
            canvas.rect(
                x0 + tw * 0.12,
                y0 + gh * 0.15,
                x0 + tw * 0.88,
                y0 + gh * 0.85,
                Paint::Solid(Color { r: 0.92, g: 0.86, b: 0.25, a: 0.85 }),
            );
        }
        canvas.set_blend(BlendMode::Normal);

        // 6) 文本（M5）：字形轮廓直入解析覆盖率管线 —— 任意缩放不糊。
        //    每帧重新 shaping（很快），但轮廓只解析一次并缓存；
        //    文字随时间轻微缩放，用来直观展示"放大不糊"。
        if !self.fonts.is_empty() {
            let size = (h * 0.10) * (1.0 + 0.25 * (t * 0.8).sin());
            let label = "Amiki2D 渲染引擎";
            if let Ok(run) = amiki_text::shape_with_stack(&self.fonts, label, size, &self.text_opts) {
                let ox = w * 0.5 - size * 1.75;
                let oy = h * 0.46;
                canvas.text_with_stack(
                    &run,
                    Vec2::new(ox, oy),
                    Paint::Solid(Color { r: 0.96, g: 0.94, b: 0.90, a: 0.96 }),
                    |fi, id| {
                        self.text_cache
                            .borrow_mut()
                            .get_or_insert(&self.fonts, fi, id)
                    },
                );
            }
        }

        canvas.scene
    }

    fn frame(&mut self) {
        let t = self.start.elapsed().as_secs_f32();
        let w = self.surface_cfg.width;
        let h = self.surface_cfg.height;

        // 场景尺寸变化时重建输出纹理与 blit 管线
        if self.scene_size != (w, h) {
            let tex = self.renderer.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("amiki_scene_tex"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let blitter = Blitter::new(&self.renderer.device, &view, self.surface_cfg.format);
            self.scene_tex = Some((tex, view));
            self.blitter = Some(blitter);
            self.scene_size = (w, h);
        }

        // M4：只重画改动过的区域。
        // 场景每帧重建，但**静态部分**（背景、圆环、混合模式画廊）每帧构造结果相同，
        // 于是 `edit_damage` 只会把真正在动的那几个元素标脏 —— 这就是增量的收益来源。
        let t_frame = Instant::now();
        let scene = if self.show_list { self.build_list() } else { self.build_scene(t) };
        let damage = match &self.prev_scene {
            Some(prev) => edit_damage(prev, &scene),
            None => Damage::new(scene.width, scene.height), // 首帧全脏
        };
        self.last_dirty_ratio = damage.tile_count() as f64 / damage.total_tiles().max(1) as f64;

        // 复用 `self.frame`：缓冲跨帧保留容量，避免每帧 mmap + 缺页
        self.renderer
            .prepare_into(&scene, 0.1, &mut self.frame)
            .expect("prepare 失败");
        let scene_tex = self.scene_tex.as_ref().expect("场景纹理未初始化");
        let skipped = self
            .renderer
            .render_into_persistent(&scene_tex.0, &self.frame, Some(&damage))
            .expect("GPU 渲染失败");
        if skipped {
            // 无脏区：连一帧 GPU 工作都不做（静态界面即此情形）
            self.last_ms = 0.0;
        } else {
            // 等待本帧 GPU 完成，使标题里的耗时是真实帧成本而非排队时间
            let _ = self.renderer.device.poll(wgpu::PollType::Wait);
            self.last_ms = t_frame.elapsed().as_secs_f64() * 1000.0;
        }
        self.prev_scene = Some(scene);

        let surface_tex = self
            .surface
            .get_current_texture()
            .expect("获取表面纹理失败");
        let mut encoder = self
            .renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_frame") });
        if let Some(b) = &self.blitter {
            b.blit(&mut encoder, &surface_tex.texture.create_view(&Default::default()), w, h);
        }
        self.renderer.queue.submit(Some(encoder.finish()));
        surface_tex.present();

        // M4 收益可视化：帧耗时 + 脏 tile 占比。
        // 静态内容占比越高、动的元素越少，脏区占比就越小 —— 这就是增量渲染省下的工作。
        self.window.set_title(&format!(
            "Amiki2D [{}] — {:.2} ms/帧 · 脏 tile {:.1}%{}（M4 damage region）",
            if self.show_list { "列表" } else { "展示" },
            self.last_ms,
            self.last_dirty_ratio * 100.0,
            match self.selected {
                Some(i) => format!(" · 选中第 {i} 行 · 滚动 {:.0}", self.scroll),
                None => format!(" · 滚动 {:.0}", self.scroll),
            }
        ));
    }
}

#[derive(Default)]
struct App {
    state: Option<FrameState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        self.state = Some(FrameState::new(el));
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Escape) => el.exit(),
                        PhysicalKey::Code(KeyCode::Tab) => state.show_list = !state.show_list,
                        PhysicalKey::Code(KeyCode::ArrowDown) => state.scroll_by(-60.0),
                        PhysicalKey::Code(KeyCode::ArrowUp) => state.scroll_by(60.0),
                        _ => {}
                    }
                }
            }
            // ---- dogfooding：应用交互 ----
            WindowEvent::CursorMoved { position, .. } => {
                state.cursor = Vec2::new(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };
                state.scroll_by(dy);
            }
            WindowEvent::MouseInput { state: bs, button, .. } => {
                if bs == ElementState::Pressed && button == MouseButton::Left && state.show_list {
                    // 命中测试 → 切换选中。**只改一行**，因此脏区只有那一行 ——
                    // 这正是增量渲染要展示的效果。
                    match state.row_at(state.cursor) {
                        Some(i) if state.selected == Some(i) => state.selected = None,
                        Some(i) => state.selected = Some(i),
                        None => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                state.surface_cfg.width = size.width.max(1);
                state.surface_cfg.height = size.height.max(1);
                state.surface.configure(&state.renderer.device, &state.surface_cfg);
            }
            WindowEvent::RedrawRequested => state.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::builder().build().expect("事件循环创建失败");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("事件循环运行失败");
}
