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

    pub fn fill_path(&mut self, path: &Path, paint: Paint) -> &mut Self {
        self.scene.push_blend(self.transform, path.clone(), paint, self.blend);
        if !self.clip_stack.is_empty() {
            self.scene.draws.last_mut().unwrap().clips = self.clip_stack.clone();
        }
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
