//! 帧路径：prepare / render / 持久化与保护式回读、层合成。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

/// M6：把一个离屏层合成到根图像上。
///
/// `layer_img` 是层的**内容**（`Filter::Blur` 时已由 GPU 模糊过）；
/// `unblurred` 是**未模糊**的层内容，仅投影需要。
///
/// # 为什么投影要用未模糊的内容
///
/// CPU 侧的投影是"用 alpha 形状 + 固定阴影色先着色、再模糊"。
/// 若直接拿"模糊后的层内容"当阴影，语义变成"模糊内容再着色"，两者不同
/// （实测差 125/255）。因此阴影必须从**未模糊**的内容出发再单独模糊。
///
/// 阴影的滤波**复用 CPU 的 `apply_filter`**（GPU 侧已模糊过内容，故传 sigma=0
/// 只做"着色 + 偏移 + 置于下方"）—— 两份实现必然漂移，共用一份才对。
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_layer_gpu(
    root: &mut [[f32; 4]],
    root_w: u32,
    root_h: u32,
    layer: &amiki_scene::Layer,
    layer_img: &[[f32; 4]],
    layer_w: u32,
    layer_h: u32,
    shadow: Option<&[[f32; 4]]>,
) {
    use amiki_raster::{inflated_bounds, LayerBuffer};

    if layer.bounds.is_empty() {
        return;
    }
    let b = inflated_bounds(layer.bounds, layer.filter);
    let bx = b.x.max(0);
    let by = b.y.max(0);
    if bx >= root_w as i32 || by >= root_h as i32 {
        return;
    }
    let lw = b.width.min(root_w - bx as u32);
    let lh = b.height.min(root_h - by as u32);
    if lw == 0 || lh == 0 {
        return;
    }

    // 取层内容（直线色：滤波在颜色上做，与 CPU 的 LayerBuffer 语义一致）
    let mut buf = LayerBuffer::new(lw, lh);
    for y in 0..lh {
        for x in 0..lw {
            let sx = bx + x as i32;
            let sy = by + y as i32;
            let src = if sx >= 0 && sy >= 0 && (sx as u32) < layer_w && (sy as u32) < layer_h {
                layer_img[(sy as u32 * layer_w + sx as u32) as usize]
            } else {
                [0.0; 4]
            };
            buf.set(x, y, src);
        }
    }

    // 投影：从**未模糊**的层内容出发，用 CPU 的同一份实现做
    // "按阴影色着色 + 按 (dx,dy) 偏移 + 置于内容下方"。
    // GPU 已模糊过内容（Blur 情形），故这里对内容本身传 sigma=0；
    // 但阴影需要"先着色再模糊"，那一步由**主机侧调用方**在 GPU 上单独完成，
    // 结果经 `unblurred` 传入的 alpha 形状参与。
    // 投影：把主机侧已生成并模糊好的阴影按 (dx,dy) 偏移，叠到内容**下方**。
    if let (Some(sh), amiki_scene::Filter::DropShadow { dx, dy, .. }) = (shadow, layer.filter) {
        for y in 0..lh {
            for x in 0..lw {
                // 阴影在 (gx-dx, gy-dy) 处 —— 即"内容右下方出现的阴影"
                let gxs = bx + x as i32 - dx.round() as i32;
                let gys = by + y as i32 - dy.round() as i32;
                let sp = if gxs >= 0 && gys >= 0 && (gxs as u32) < root_w && (gys as u32) < root_h {
                    sh[(gys as u32 * root_w + gxs as u32) as usize]
                } else {
                    [0.0; 4]
                };
                let d = buf.get(x, y);
                let ao = d[3] + sp[3] * (1.0 - d[3]);
                let inv = if ao > 0.0 { 1.0 / ao } else { 0.0 };
                let co = [
                    (d[0] * d[3] + sp[0] * sp[3] * (1.0 - d[3])) * inv,
                    (d[1] * d[3] + sp[1] * sp[3] * (1.0 - d[3])) * inv,
                    (d[2] * d[3] + sp[2] * sp[3] * (1.0 - d[3])) * inv,
                ];
                buf.set(x, y, [co[0], co[1], co[2], ao]);
            }
        }
    }

    // 合成回根图像：与 CPU 侧同一套数学（预乘空间，见 amiki-raster 的说明）
    let opacity = layer.opacity.clamp(0.0, 1.0);
    for y in 0..lh {
        for x in 0..lw {
            let p = buf.get(x, y);
            let sa = p[3] * opacity;
            if sa <= 0.0 {
                continue;
            }
            let src_pm = [p[0] * sa, p[1] * sa, p[2] * sa, sa];
            let gx = (bx + x as i32) as u32;
            let gy = (by + y as i32) as u32;
            let o = (gy * root_w + gx) as usize;
            let d = root[o];
            let da = d[3];
            let dst_pm = [d[0] * da, d[1] * da, d[2] * da, da];
            let out_pm = amiki_raster::composite_premul(dst_pm, src_pm, layer.blend.id());
            let oa = out_pm[3];
            let inv = if oa > 0.0 { 1.0 / oa } else { 0.0 };
            // **不 clamp** —— >1.0 要留给后续的层与最终的量化
            root[o] = [out_pm[0] * inv, out_pm[1] * inv, out_pm[2] * inv, oa];
        }
    }
}

impl GpuRenderer {
    /// M4：**持久化渲染** —— 复用同一个输出纹理，只对脏 tile 派发 fine。
    ///
    /// 与 [`GpuRenderer::render`] 的区别：
    /// - `render` 每帧新建输出纹理并全量派发；
    /// - 本方法持有输出纹理与上一帧场景，**场景未变则一次派发都不做**，
    ///   只改了一小块则只派发覆盖脏区的 tile 子矩形。
    ///
    /// 正确性依据：fine 内核由 `global_invocation_id` 推导像素与 tile 坐标，
    /// 未派发的 tile **不写入**，纹理中保留的正是上一帧算出的同一结果
    /// （单 tile 的着色只依赖该 tile 的成员段与画家序，与其他 tile 的当前内容无关）。
    ///
    /// `damage` 必须与 `scene` 的画布尺寸匹配，否则内部的 `check_damage_bounds` 会断言失败
    /// ——那是在阻止"全脏被误当作局部脏区"从而静默残留陈旧像素。
    ///
    /// 返回 `true` 表示本帧完全跳过（未做任何 GPU 工作）。
    #[allow(clippy::too_many_arguments)]
    pub fn render_into_persistent(
        &self,
        tex: &wgpu::Texture,
        frame: &Frame,
        damage: Option<&amiki_scene::Damage>,
    ) -> Result<bool, RenderError> {
        if damage.is_some_and(|d| d.is_empty()) {
            return Ok(true); // 无脏区：完全跳过
        }
        check_damage_bounds(damage, frame.scene.tiles_x, frame.scene.tiles_y);
        let mut prof = FrameProfile::default();
        match self.path {
            RenderPath::Dense => self.run_pipeline_impl(
                &frame.scene,
                tex,
                false,
                &frame.clip_views,
                damage,
                None,
                None,
                &mut prof,
            )?,
            RenderPath::Sparse => self.run_pipeline_sparse(
                &frame.scene,
                tex,
                false,
                &frame.clip_views,
                damage,
                None,
                None,
                &mut prof,
            )?,
        }
        // P0-3：层在 GPU 上合成（不读回）。damage 限制写入范围，
        // 因此静止的层不会每帧重算 —— 与图元共用同一套脏区语义。
        self.composite_layers_gpu(&frame.scene, tex, damage, frame.eps)?;
        Ok(false)
    }

    /// 准备逐帧资源：GpuScene 提取 + clip 覆盖率纹理。
    ///
    /// **逐帧调用请改用 [`Self::prepare_into`]** —— 本函数每次返回新的 `Frame`，
    /// 而 `Frame` 里的缓冲每次都是新分配的（约 7 MB，实测差 5.3 倍，
    /// 见 `GpuScene::reset_for_reuse`）。保留它是为了一次性调用与兼容。
    pub fn prepare(&self, scene: &Scene, eps: f32) -> Result<Frame, RenderError> {
        let mut frame = Frame::default();
        self.prepare_into(scene, eps, &mut frame)?;
        Ok(frame)
    }

    /// [`Self::prepare`] 的带 mip bias 版本（B4）。
    pub fn prepare_with(&self, scene: &Scene, eps: f32, mip_bias: f32) -> Result<Frame, RenderError> {
        let mut frame = Frame::default();
        self.prepare_into_with(scene, eps, mip_bias, &mut frame)?;
        Ok(frame)
    }

    /// 准备逐帧资源，**复用调用方持有的 `Frame` 的缓冲**。
    ///
    /// ```no_run
    /// # use amiki_gpu::{GpuRenderer, Frame};
    /// # fn f(r: &GpuRenderer, scene: &amiki_scene::Scene) -> Result<(), amiki_gpu::RenderError> {
    /// let mut frame = Frame::default();
    /// loop {
    ///     r.prepare_into(scene, 0.05, &mut frame)?;   // 缓冲跨帧复用
    ///     // … render_into(&tex, &frame)?
    /// }
    /// # }
    /// ```
    pub fn prepare_into(
        &self,
        scene: &Scene,
        eps: f32,
        frame: &mut Frame,
    ) -> Result<(), RenderError> {
        self.prepare_into_with(scene, eps, 0.0, frame)
    }

    /// [`Self::prepare_into`] 的带 mip bias 版本（B4）。
    ///
    /// bias 影响每个图像 draw 打包进放置缓冲的 LOD，从而影响采样层级。
    /// 放在提取入口而不是渲染入口：LOD 是**打包期**常量（见
    /// `pack_image_place`），WGSL 只读不算。
    pub fn prepare_into_with(
        &self,
        scene: &Scene,
        eps: f32,
        mip_bias: f32,
        frame: &mut Frame,
    ) -> Result<(), RenderError> {
        use amiki_scene::{Color, Paint};
        // 注意：`prepare` 本身不拒绝层 —— `render` 会经它进来，而
        // `render_blocking` 才负责层合成。拒绝放在逐帧路径的两个入口
        // （`render_into` / `render_into_persistent`），它们才是真正会
        // 静默丢掉层的地方。
        build_gpu_scene_into(scene, eps, &mut frame.scene);
        frame.scene.mip_bias = mip_bias;
        let mut clip_texs = Vec::new();
        let mut clip_views = Vec::new();
        for clip in &scene.clips {
            let mut cs = Scene::new(scene.width, scene.height);
            cs.push(clip.transform, clip.path.clone(), Paint::Solid(Color::rgb8(255, 255, 255)));
            // 复用同一份 scratch：clip 覆盖纹理是**即刻渲染**的，用完即可覆盖
            build_gpu_scene_into(&cs, eps, &mut frame.clip_scratch);
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("amiki_clip_cov"),
                size: wgpu::Extent3d {
                    width: frame.clip_scratch.width,
                    height: frame.clip_scratch.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.run_pipeline(&frame.clip_scratch, &tex, true, &[], None)?;
            clip_views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            clip_texs.push(tex);
        }
        frame.clip_texs = clip_texs;
        frame.clip_views = clip_views;
        frame.eps = eps;
        Ok(())
    }

    pub fn render(&self, scene: &Scene, eps: f32) -> Result<wgpu::Texture, RenderError> {
        let frame = self.prepare(scene, eps)?;
        let tex = self.create_output_texture(&frame.scene);
        self.render_into(&tex, &frame)?;
        Ok(tex)
    }

    /// 渲染 Frame 到已存在的 rgba8unorm 纹理（窗口逐帧复用）。
    pub fn render_into(&self, tex: &wgpu::Texture, frame: &Frame) -> Result<(), RenderError> {
        self.run_pipeline(&frame.scene, tex, false, &frame.clip_views, None)?;
        // P0-3：无 damage 信息 ⇒ 各层按完整作用范围合成
        self.composite_layers_gpu(&frame.scene, tex, None, frame.eps)
    }

    /// 与 [`Self::render_blocking_with`] 相同，但用默认的 `ToneMap::Raw`。
    pub fn render_blocking(&self, scene: &Scene, eps: f32) -> Result<Vec<u8>, RenderError> {
        self.render_blocking_with(scene, eps, amiki_scene::ToneMap::Raw, 0.0)
    }

    /// 渲染并读回 8 位，按 `tone` 把 HDR 压到显示范围。
    ///
    /// `ToneMap::Raw`（默认）等价于旧行为：只裁剪。
    /// 完整说明见 `docs/COLOR.md`。
    pub fn render_blocking_with(
        &self,
        scene: &Scene,
        eps: f32,
        tone: amiki_scene::ToneMap,
        mip_bias: f32,
    ) -> Result<Vec<u8>, RenderError> {
        // 含层场景的参考实现：**层内容与模糊都在 GPU，合成在主机侧**。
        //
        // 为什么合成留在主机侧：着色与模糊（真正的大头）都在 GPU，而合成是每像素
        // 几次乘加；放主机侧使"层"与"图元"共用同一套混合数学，避免两份可能漂移的
        // 实现（见 `composite_layer_gpu`）。
        //
        // 这是**对拍基准**：GPU 帧路径的层合成（`composite_layers_gpu`，P0-3）
        // 必须与它一致 —— 两者共用同一份 WGSL 数学，差异只来自"读回 u8 →
        // 主机 f32 → 写回 u8"的往返，实测 ≤1/255。
        //
        // 直接走管线（不经过 `render_into` 的层合成路径），因为下面要拿未合成的
        // 根图像自己按层合成。
        let frame = self.prepare_with(scene, eps, mip_bias)?;
        let tex = self.create_output_texture(&frame.scene);
        self.run_pipeline(&frame.scene, &tex, false, &frame.clip_views, None)?;
        // 根图像保持 **f32 且不裁剪**：层之间要能累积 >1.0 的值，
        // 降到 8 位是最后一步（`rgba8_from_f32`）。
        let mut root = self.readback_f32(&tex)?;
        if scene.layers.is_empty() {
            return Ok(rgba8_from_f32_with(&root, tone));
        }

        // M6：逐层渲染 → GPU 模糊 → 合成回根图像。
        //
        // 每层的图元仍由**同一套细粒度管线**渲染（不复制一份着色逻辑），
        // 只是目标换成整画布尺寸的纹理。于是"层"几乎没有额外实现成本，
        // 也不可能出现两套着色语义漂移。
        let (root_w, root_h) = (tex.width(), tex.height());
        for layer in &scene.layers {
            if layer.bounds.is_empty() {
                continue;
            }
            let sub = {
                let mut sc = Scene::new(root_w, root_h);
                sc.draws = layer.scene.draws.clone();
                sc.clips = layer.scene.clips.clone();
                sc
            };
            let lframe = self.prepare_with(&sub, eps, mip_bias)?;
            let ltex = self.create_output_texture(&lframe.scene);
            self.run_pipeline(&lframe.scene, &ltex, false, &lframe.clip_views, None)?;
            let mut limg = self.readback_f32(&ltex)?;

            // 内容与阴影必须**分开**模糊：
            //
            // - `Filter::Blur`：只模糊内容。
            // - `Filter::DropShadow`：内容**不模糊**；阴影是"用层 alpha 形状 +
            //   固定阴影色先着色、再模糊"（与 CPU 一致）。若直接拿模糊后的内容
            //   当阴影，就变成"模糊内容再着色"，语义不同（实测差 125/255）。
            let mut shadow_img: Option<Vec<[f32; 4]>> = None;
            let mut shadow_img_buf: Vec<[f32; 4]> =
                vec![[0.0f32; 4]; (root_w * root_h) as usize];
            match layer.filter {
                amiki_scene::Filter::Blur { sigma } => {
                    limg = self.blur_rgba(&limg, root_w, root_h, sigma, 3)?;
                }
                amiki_scene::Filter::DropShadow { sigma, color, .. } => {
                    // 阴影遮罩在**主机侧**用与 CPU 完全相同的步骤生成：
                    // 1) RGB 覆盖为阴影色、alpha 乘 color.a
                    // 2) 盒式模糊（直线色语义，与 CPU 的 blur 一致）
                    //
                    // 为什么这里不走 GPU 模糊：投影需要的是"未模糊内容的 alpha 形状"，
                    // 而 GPU 模糊作用在整画布纹理上、坐标空间与层范围不同，
                    // 拼接时极易错位（实测差 125~162）。既然 CPU 已有这份数学，
                    // 直接复用最稳妥；GPU 仍承担**内容**的模糊（那才是大头）。
                    let mask: Vec<[f32; 4]> = limg
                        .iter()
                        .map(|p| [color.r, color.g, color.b, p[3] * color.a])
                        .collect();
                    let mut lb = amiki_raster::LayerBuffer::new(root_w, root_h);
                    lb.data.copy_from_slice(&mask);
                    amiki_raster::blur_layer(&mut lb, sigma);
                    // f32 直存：阴影同样不能在这里被裁，否则高光源的投影会失真
                    shadow_img_buf.copy_from_slice(&lb.data);
                    shadow_img = Some(std::mem::take(&mut shadow_img_buf));
                }
                amiki_scene::Filter::None => {}
            }
            composite_layer_gpu(
                &mut root,
                root_w,
                root_h,
                layer,
                &limg,
                root_w,
                root_h,
                shadow_img.as_deref(),
            );
        }
        // 唯一的 8 位出口：HDR 在这里降到显示范围（按调用方选的映射）
        Ok(rgba8_from_f32_with(&root, tone))
    }
}
