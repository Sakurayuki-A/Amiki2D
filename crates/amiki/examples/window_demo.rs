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
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

struct FrameState {
    window: std::sync::Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    renderer: GpuRenderer,
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
    font: Option<amiki_text::Font>,
    text_cache: std::cell::RefCell<OutlineCache>,
}

/// 字形轮廓缓存：`glyph id → 轮廓`。
///
/// 为什么必须缓存：轮廓解析（含二次曲线还原）比拷贝一条路径贵得多，
/// 而同一个字形每帧都会被复用。缓存让"每帧重新 shaping"变得廉价。
/// 用 `Vec` 而非 `HashMap`：字形数量小（几十到几百），线性查找更快且无哈希依赖。
#[derive(Default)]
struct OutlineCache {
    entries: Vec<(u32, amiki::scene::GlyphOutline)>,
}

impl OutlineCache {
    fn get_or_insert(&mut self, font: &amiki_text::Font, id: u32) -> Option<amiki::scene::GlyphOutline> {
        if let Some((_, o)) = self.entries.iter().find(|(k, _)| *k == id) {
            return Some(o.clone());
        }
        let o = font.outline(id).ok()?;
        self.entries.push((id, o.clone()));
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
            required_limits: wgpu::Limits::default(),
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
        // 字体可选：缺失时示例仍应能演示图形能力
        let font = match amiki_text::Font::from_file("assets/fonts/Inter-Regular.ttf") {
            Ok(f) => {
                println!(
                    "已加载 Inter：units_per_em = {}, 字形数 = {}",
                    f.units_per_em(),
                    f.glyph_count()
                );
                Some(f)
            }
            Err(e) => {
                eprintln!("未加载字体（{e}）；文本部分将跳过");
                None
            }
        };
        FrameState {
            window,
            surface,
            surface_cfg,
            renderer,
            blitter: None,
            scene_tex: None,
            scene_size: (0, 0),
            start: Instant::now(),
            prev_scene: None,
            last_ms: 0.0,
            last_dirty_ratio: 1.0,
            font,
            text_cache: std::cell::RefCell::new(OutlineCache::default()),
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
        if let Some(font) = self.font.as_ref() {
            let size = (h * 0.10) * (1.0 + 0.25 * (t * 0.8).sin());
            if let Ok(run) = amiki_text::shape(font, "Amiki2D", size) {
                let ox = w * 0.5 - size * 1.75;
                let oy = h * 0.46;
                canvas.text(
                    &run,
                    Vec2::new(ox, oy),
                    Paint::Solid(Color { r: 0.96, g: 0.94, b: 0.90, a: 0.96 }),
                    |id| self.text_cache.borrow_mut().get_or_insert(font, id),
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
                format: wgpu::TextureFormat::Rgba8Unorm,
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
        let scene = self.build_scene(t);
        let damage = match &self.prev_scene {
            Some(prev) => edit_damage(prev, &scene),
            None => Damage::new(scene.width, scene.height), // 首帧全脏
        };
        self.last_dirty_ratio = damage.tile_count() as f64 / damage.total_tiles().max(1) as f64;

        let frame = self.renderer.prepare(&scene, 0.1).expect("prepare 失败");
        let scene_tex = self.scene_tex.as_ref().expect("场景纹理未初始化");
        let skipped = self
            .renderer
            .render_into_persistent(&scene_tex.0, &frame, Some(&damage))
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
            "Amiki2D — 增量渲染 {:.2} ms/帧 · 脏 tile {:.1}%（M4 damage region）",
            self.last_ms,
            self.last_dirty_ratio * 100.0
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
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                {
                    el.exit();
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
