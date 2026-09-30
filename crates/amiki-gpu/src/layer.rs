//! P0-3：**离屏层的 GPU 合成**（帧路径用，不读回）。
//!
//! # 为什么是 compute pass 而不是 render pass
//!
//! 帧路径的输出纹理由**调用方**提供，其用途标志是
//! `STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC` —— 没有 `RENDER_ATTACHMENT`，
//! 也没有 `COPY_DST`。于是：
//!
//! - 不能用 render pass 写它（缺 `RENDER_ATTACHMENT`）
//! - 不能把结果拷回它（缺 `COPY_DST`）
//!
//! 而 compute **可以**写 storage texture，也只需要已有的 `COPY_SRC` 就能把
//! 目标先拷进 backdrop。这恰好与本项目"整条管线都是 compute"的架构一致。
//!
//! # 与 CPU 的关系
//!
//! 数学必须与 CPU 的 `amiki_raster::composite_layer` **镜像**，否则两端会漂移：
//! 都在**预乘空间**做 W3C 合成（直线色直接相乘会在模糊后的小 alpha 处放大
//! 量化噪声，实测边缘色偏 86/255）。混合函数 `blend_color` 来自 `fine_math.wgsl`
//! —— 那是与 CPU 同一份数学的 WGSL 版。
//!
//! # drop shadow 为什么不需要单独的"造遮罩"pass
//!
//! CPU 造的阴影遮罩是 `[阴影色, 内容alpha × 阴影alpha]`。因为**模糊是线性的**
//! 且遮罩的 RGB 处处相同，`blur([c, a·k]).rgb = c`、`.a = blur(a)·k`。
//! 所以直接模糊层内容，再在合成时用 `override_rgb` 换掉 RGB、用 `alpha_scale`
//! 乘上阴影 alpha，得到的结果**完全相同**，省掉一趟 pass。

use amiki_scene::Scene;

use crate::{blur, err, GpuScene, RenderError};

/// 合成参数（uniform；与 WGSL 的 `struct Params` 一一对应）。
///
/// 布局必须与 WGSL 对齐：每个 `vec4` 落在 16 字节边界上。
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerCompositeParams {
    /// **源纹理原点落在目标的哪个像素**（投影时即 `(dx, dy)`；内容为 `(0, 0)`）
    pub dst_origin: [f32; 2],
    /// 层的尺寸（设备像素）
    pub dst_size: [f32; 2],
    /// 源纹理尺寸（用于把目标像素映射回图层像素）
    pub src_size: [f32; 2],
    /// 目标纹理尺寸
    pub target_size: [f32; 2],
    /// 层不透明度
    pub opacity: f32,
    /// 混合模式 id
    pub mode: u32,
    /// 是否启用阴影（0 或 1）
    pub shadow_on: f32,
    /// 保留（与旧 `alpha_scale` 同位，便于布局对齐）
    pub _reserved: f32,
    /// 阴影色 RGBA（`.a` 为阴影 alpha 的缩放系数）
    pub shadow_rgb: [f32; 4],
    /// 只在这些像素上写入（damage 限制）：`x0, y0, x1, y1`
    pub rect: [f32; 4],
    /// 阴影相对内容的偏移 `(dx, dy)`（`.zw` 保留）
    pub shadow_offset: [f32; 4],
}

const COMPOSITE_WGSL: &str = r#"
struct Params {
    dst_origin: vec2<f32>,
    dst_size: vec2<f32>,
    src_size: vec2<f32>,
    target_size: vec2<f32>,
    opacity: f32,
    mode: u32,
    shadow_on: f32,
    _reserved: f32,
    shadow_rgb: vec4<f32>,
    rect: vec4<f32>,
    shadow_offset: vec4<f32>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var backdrop: texture_2d<f32>;
/// 层**内容**（投影时是未模糊的那份）
@group(0) @binding(2) var layer_tex: texture_2d<f32>;
/// 层内容**模糊后**的纹理（仅投影用；不启用阴影时绑同一个视图）
@group(0) @binding(3) var shadow_tex: texture_2d<f32>;
// 绑定名不用 `target`：**它是 WGSL 保留字**，用它会让着色器解析失败。
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba8unorm, write>;

/// 预乘空间下的 W3C 合成 —— 与 `amiki_raster::composite_premul` **逐式对应**。
fn composite_premul(dst: vec4<f32>, src: vec4<f32>, mode: u32) -> vec4<f32> {
    let sa = src.a;
    let da = dst.a;
    let out_a = sa + da * (1.0 - sa);
    let inv_d = select(0.0, 1.0 / da, da > 0.0);
    let inv_s = select(0.0, 1.0 / sa, sa > 0.0);
    let cb = dst.rgb * inv_d;
    let cs = src.rgb * inv_s;
    let b = select(blend_color(mode, cb, cs), cs, mode == 0u);
    let rgb = sa * (1.0 - da) * cs + sa * da * b + (1.0 - sa) * da * cb;
    return vec4<f32>(rgb, out_a);
}

@compute @workgroup_size(8, 8)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
    // **`gid` 是相对调度原点的偏移，不是绝对像素坐标**，必须加上 `rect.xy`。
    // 曾经直接拿 gid 当像素坐标 —— 合成被整体平移到左上角，层的实际位置几乎
    // 没被写到（与参考差 179/255，且差异范围恰好是内容矩形）。
    let base = vec2<i32>(i32(p.rect.x), i32(p.rect.y));
    let px = base + vec2<i32>(i32(gid.x), i32(gid.y));
    let tsize = vec2<i32>(i32(p.target_size.x), i32(p.target_size.y));
    if (px.x < 0 || px.y < 0 || px.x >= tsize.x || px.y >= tsize.y) {
        return;
    }
    let r = vec4<i32>(i32(p.rect.x), i32(p.rect.y), i32(p.rect.z), i32(p.rect.w));
    if (px.x < r.x || px.y < r.y || px.x >= r.z || px.y >= r.w) {
        return;
    }

    let dst = textureLoad(backdrop, px, 0);

    // `dst_origin` = **源纹理原点落在目标的哪个像素**（内容为 0）
    let o = vec2<i32>(i32(p.dst_origin.x), i32(p.dst_origin.y));
    let s = vec2<i32>(i32(p.dst_size.x), i32(p.dst_size.y));
    let sp = px - o;

    // 内容（越界视为全透明，与 CPU 的"缓冲外为 0"一致）
    var src = vec4<f32>(0.0);
    if (sp.x >= 0 && sp.y >= 0 && sp.x < s.x && sp.y < s.y) {
        src = textureLoad(layer_tex, sp, 0);
    }

    if (p.shadow_on > 0.5) {
        // 阴影：来自**已模糊**的层内容，按 (dx,dy) 偏移；
        // RGB 换成阴影色、alpha 乘阴影 alpha（与 CPU 的着色步骤等价）
        let off = vec2<i32>(i32(p.shadow_offset.x), i32(p.shadow_offset.y));
        let hp = px - off;
        var sh = vec4<f32>(0.0);
        if (hp.x >= 0 && hp.y >= 0 && hp.x < s.x && hp.y < s.y) {
            let t = textureLoad(shadow_tex, hp, 0);
            sh = vec4<f32>(p.shadow_rgb.rgb, t.a * p.shadow_rgb.a);
        }
        // 层内合成：**内容 over 阴影**（直线色 src-over）——
        // 与 CPU `apply_filter(DropShadow)` 内部那一步逐式对应
        let ao = src.a + sh.a * (1.0 - src.a);
        let inv = select(0.0, 1.0 / ao, ao > 0.0);
        src = vec4<f32>(
            (src.rgb * src.a + sh.rgb * sh.a * (1.0 - src.a)) * inv,
            ao,
        );
    }

    // 整体按不透明度合成（**只在这里施加 opacity**，与 CPU 一致）
    let sa = src.a * p.opacity;
    let da = dst.a;
    let dst_pm = vec4<f32>(dst.rgb * da, da);
    let src_pm = vec4<f32>(src.rgb * sa, sa);
    let out_pm = composite_premul(dst_pm, src_pm, p.mode);
    let oa = out_pm.a;
    let inv = select(0.0, 1.0 / oa, oa > 0.0);
    textureStore(out_tex, px, vec4<f32>(out_pm.rgb * inv, oa));
}
"#;

/// 层合成资源。
pub struct LayerCompositeResources {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

/// 创建合成管线。
pub fn create_resources(device: &wgpu::Device) -> LayerCompositeResources {
    // 与 fine_math 拼接：混合函数的唯一来源，不在这里再写一份
    let source = format!("{}\n{}", crate::FINE_MATH_WGSL, COMPOSITE_WGSL);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("amiki_layer_composite"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("amiki_layer_composite_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            tex_entry(1),
            tex_entry(2),
            tex_entry(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("amiki_layer_composite_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("amiki_layer_composite_pipeline"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("cs"),
        compilation_options: Default::default(),
        cache: None,
    });
    LayerCompositeResources { pipeline, bgl }
}

/// 一趟层合成：读 `backdrop` 与 `layer_tex`，写 `target`。
///
/// 调用方需先把 `target` 拷进 `backdrop`（因为同一纹理不能既读又写）。
/// 调度范围由 `range` 给出（`x0, y0, x1, y1`，闭开）。
#[allow(clippy::too_many_arguments)]
pub fn composite_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    res: &LayerCompositeResources,
    params_buf: &wgpu::Buffer,
    backdrop: &wgpu::TextureView,
    layer_tex: &wgpu::TextureView,
    shadow_tex: &wgpu::TextureView,
    target: &wgpu::TextureView,
    range: (u32, u32, u32, u32),
) -> Result<(), RenderError> {
    let (x0, y0, x1, y1) = range;
    if x0 >= x1 || y0 >= y1 {
        return Ok(());
    }
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("amiki_layer_composite_bind"),
        layout: &res.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(backdrop) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(layer_tex) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_tex) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(target) },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("amiki_layer_composite_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&res.pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.dispatch_workgroups((x1 - x0).div_ceil(8), (y1 - y0).div_ceil(8), 1);
    Ok(())
}

/// 创建一张可用于层合成的纹理（既是渲染目标，又能被拷贝/读取）。
pub fn create_layer_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> Result<(wgpu::Texture, wgpu::TextureView), RenderError> {
    if width == 0 || height == 0 {
        return err("层纹理尺寸为 0");
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        // 需要：被 compute 管线写（STORAGE）、被模糊 pass 写（RENDER_ATTACHMENT）、
        // 被合成 pass 读（TEXTURE_BINDING）、被拷贝（COPY_SRC/DST）
        usage: wgpu::TextureUsages::STORAGE_BINDING
            .union(wgpu::TextureUsages::TEXTURE_BINDING)
            .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
            .union(wgpu::TextureUsages::COPY_SRC)
            .union(wgpu::TextureUsages::COPY_DST),
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((tex, view))
}

// ============ 层的渲染与滤镜（`impl GpuRenderer`）============
//
// 从 `lib.rs` 拆出：这一簇全部是**层的模糊与合成**，与 `create_resources`
// 定义的 compute 管线同属一个关注点，放在一起便于对照"管线怎么建"与
// "管线怎么用"。

impl crate::GpuRenderer {
    /// M6：对一张**直线 alpha** 的 RGBA8 图做可分离模糊，返回直线 alpha 的像素。
    ///
    /// # 通道语义：与 CPU 保持一致的**直线色直接平均**
    ///
    /// CPU 的 `amiki_raster::filter::blur` 对 4 个通道直接做盒式平均
    /// （直线色语义）。GPU 必须完全相同，否则跨端对拍必然不一致。
    /// 因此这里**不做**预乘/反预乘 —— 详见函数末尾的说明。
    #[allow(dead_code)]
    pub(crate) fn blur_rgba(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
        sigma: f32,
        passes: u32,
    ) -> Result<Vec<u8>, RenderError> {
        self.blur_rgba_impl(src, width, height, sigma, passes, None)
    }

    /// `override_rgb` 为 `Some(c)` 时，先把 RGB 覆盖为 `c`（保留 alpha 形状）再模糊。
    ///
    /// 这是为**投影**准备的：CPU 侧的阴影是"用 alpha 形状 + 固定阴影色"先着色、
    /// 再模糊。若直接模糊层内容，就变成"模糊内容再当阴影"，语义不同
    /// （实测差 125/255）。
    pub(crate) fn blur_rgba_impl(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
        sigma: f32,
        passes: u32,
        override_rgb: Option<[f32; 3]>,
    ) -> Result<Vec<u8>, RenderError> {
        let box_w = blur::box_width_for(sigma, 3);
        // 先做可选的 RGB 覆盖（生成阴影遮罩）
        let mut owned;
        let src: &[u8] = match override_rgb {
            Some(c) => {
                owned = src.to_vec();
                for px in owned.as_chunks_mut::<4>().0.iter_mut() {
                    px[0] = (c[0] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    px[1] = (c[1] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    px[2] = (c[2] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
                &owned
            }
            None => src,
        };
        if sigma <= 0.0 || box_w <= 1 || width == 0 || height == 0 {
            return Ok(src.to_vec());
        }
        let (t_a, v_a) = blur::create_blur_texture(&self.device, "amiki_blur_a", width, height)?;
        let (_t_b, v_b) = blur::create_blur_texture(&self.device, "amiki_blur_b", width, height)?;

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &t_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            src,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        // **两个独立参数缓冲**（横 / 纵），而不是一个写两次。
        //
        // 这是实际踩到的坑：同一命令缓冲内多趟渲染若共用一个 uniform 缓冲，
        // 队列上"写参数1 → 写参数2 → 提交"会使两趟都读到第二次的值 ——
        // 表现为横向那趟从未执行。若测试数据恰好对该方向不敏感
        // （例如竖贯整幅的横条），就完全看不出异常，极难定位。
        let mk = |label: &str, vertical: bool| {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<blur::BlurParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(
                &b,
                0,
                bytemuck::bytes_of(&blur::BlurParams {
                    width: width as f32,
                    height: height as f32,
                    box_w: box_w as f32,
                    vertical: if vertical { 1.0 } else { 0.0 },
                }),
            );
            b
        };
        let params_h = mk("amiki_blur_h", false);
        let params_v = mk("amiki_blur_v", true);

        // 每遍：A --横--> B --纵--> A。每遍结束结果都回到 A。
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_blur") });
        for _ in 0..passes {
            blur::blur_pass(&self.device, &mut enc, &self.blur, &params_h, &v_a, &v_b);
            blur::blur_pass(&self.device, &mut enc, &self.blur, &params_v, &v_b, &v_a);
        }
        self.queue.submit(Some(enc.finish()));

        // **不做预乘/反预乘**：CPU 的 `amiki_raster::filter::blur` 是对 4 个通道
        // **直接做平均**（直线色语义），因此 GPU 也必须直接平均直线数据，
        // 两端才对得上。
        //
        // 我一度加了"预乘→模糊→反预乘"，理由是"模糊应在颜色上做、透明区 RGB
        // 未定义会洗暗"。那个理由对**正确性**而言是对的，但与 CPU 的**现行语义
        // 不一致** —— 结果是 `Filter::Blur` 的差异从 1 恶化到 64。
        // 既然对拍基准是 CPU 实现，就必须与它一致；若将来要把两端都改成
        // 预乘语义，那是**一次同时改两端**的独立决定，不能只改一侧。
        self.readback(&t_a)
    }

    /// P0-3：把场景的离屏层**在 GPU 上**合成进 `tex`（不读回；帧路径用）。
    ///
    /// # 与 CPU 的等价性
    ///
    /// 层内容纹理是**整画布尺寸**，所以采样直接发生在画布坐标下；写入范围限制在
    /// `inflated_bounds ∩ damage`。这正是 CPU 侧"把层图按 `inflated_bounds`
    /// 裁进缓冲再合成"的等价形式 —— CPU 裁掉的部分在 GPU 上被写入范围排除。
    ///
    /// 投影：CPU 造 `[阴影色, 内容alpha × 阴影alpha]` 遮罩后模糊。因为**模糊是
    /// 线性的**、且遮罩 RGB 处处相同，直接模糊层内容再在合成时换 RGB 并乘阴影
    /// alpha，得到完全相同的结果 —— 因此不需要单独一趟"造遮罩"pass。
    ///
    /// # 为什么先拷 backdrop
    ///
    /// 同一纹理不能既被读又被写，而 W3C 混合需要**背板**（目标当前内容）。
    /// 所以每趟合成前把目标的相关范围拷进 backdrop 纹理（目标有 `COPY_SRC`）。
    pub(crate) fn composite_layers_gpu(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        damage: Option<&amiki_scene::Damage>,
        eps: f32,
    ) -> Result<(), RenderError> {
        if gs.layers.is_empty() {
            return Ok(());
        }
        let (tw, th) = (tex.width(), tex.height());
        let tile = amiki_scene::DAMAGE_TILE;
        // damage 限制的设备矩形
        let (dx0, dy0, dx1, dy1) = match damage.and_then(|d| d.tiles()) {
            Some((tx0, ty0, tx1, ty1)) => (
                (tx0 * tile).min(tw),
                (ty0 * tile).min(th),
                (tx1 * tile).min(tw),
                (ty1 * tile).min(th),
            ),
            None => (0, 0, tw, th),
        };
        if dx0 >= dx1 || dy0 >= dy1 {
            return Ok(());
        }

        let target_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let backdrop = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("amiki_layer_backdrop"),
            size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let backdrop_view = backdrop.create_view(&wgpu::TextureViewDescriptor::default());
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_layer_params"),
            size: std::mem::size_of::<LayerCompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for l in &gs.layers {
            if l.bounds.is_empty() {
                continue;
            }
            // 该层的作用范围（经滤镜外扩）∩ 画布 ∩ damage
            let b = l.inflated_bounds();
            let lx0 = b.x.max(0) as u32;
            let ly0 = b.y.max(0) as u32;
            let lx1 = ((b.x + b.width as i32).max(0) as u32).min(tw);
            let ly1 = ((b.y + b.height as i32).max(0) as u32).min(th);
            let (rx0, ry0, rx1, ry1) =
                (lx0.max(dx0), ly0.max(dy0), lx1.min(dx1), ly1.min(dy1));
            if rx0 >= rx1 || ry0 >= ry1 {
                continue;
            }

            // ① 层内容 → 整画布尺寸的纹理（坐标即设备坐标）
            let sub = {
                let mut sc = Scene::new(tw, th);
                sc.draws = l.scene.draws.clone();
                sc.clips = l.scene.clips.clone();
                sc
            };
            let lframe = self.prepare(&sub, eps)?;
            let (ltex, lview) =
                create_layer_texture(&self.device, "amiki_layer_content", tw, th)?;
            self.run_pipeline(&lframe.scene, &ltex, false, &lframe.clip_views, None)?;

            // ② 模糊（产出**独立**的模糊纹理；投影同时需要未模糊的那份）
            let sigma = match l.filter {
                amiki_scene::Filter::Blur { sigma } => sigma,
                amiki_scene::Filter::DropShadow { sigma, .. } => sigma,
                amiki_scene::Filter::None => 0.0,
            };
            let blurred = self.blur_texture(&lview, tw, th, sigma)?;

            // ③ 合成：**一趟**完成"内容 over 阴影"再整体乘 opacity。
            //
            // 不能拆成"先阴影一趟、再内容一趟"：那样当 opacity<1 且内容盖住阴影时，
            // 阴影会从内容下面透出来，与 CPU 的"层内先合成、再整体合成"不符
            // （实测投影层差 8/255，且差异恰好落在阴影区）。
            let (layer_view, shadow_view, shadow_on, shadow_rgb, shadow_offset) = match l.filter {
                amiki_scene::Filter::DropShadow { dx, dy, color, .. } => {
                    let sv = blurred
                        .as_ref()
                        .map(|(_, v)| v)
                        .unwrap_or(&lview);
                    (
                        &lview,
                        sv,
                        1.0f32,
                        [color.r, color.g, color.b, color.a],
                        [dx.round(), dy.round(), 0.0, 0.0],
                    )
                }
                amiki_scene::Filter::Blur { .. } => {
                    // 模糊：内容本身就是模糊后的
                    let bv = blurred.as_ref().map(|(_, v)| v).unwrap_or(&lview);
                    (bv, bv, 0.0f32, [0.0; 4], [0.0; 4])
                }
                amiki_scene::Filter::None => (&lview, &lview, 0.0f32, [0.0; 4], [0.0; 4]),
            };
            let _ = &layer_view;
            {
                let p = LayerCompositeParams {
                    dst_origin: [0.0, 0.0],
                    dst_size: [tw as f32, th as f32],
                    src_size: [tw as f32, th as f32],
                    target_size: [tw as f32, th as f32],
                    opacity: l.opacity.clamp(0.0, 1.0),
                    mode: l.blend.id(),
                    shadow_on,
                    _reserved: 0.0,
                    shadow_rgb,
                    rect: [rx0 as f32, ry0 as f32, rx1 as f32, ry1 as f32],
                    shadow_offset,
                };
                self.queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&p));
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("amiki_layer_composite"),
                });
                enc.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: rx0, y: ry0, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &backdrop,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: rx0, y: ry0, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: rx1 - rx0,
                        height: ry1 - ry0,
                        depth_or_array_layers: 1,
                    },
                );
                composite_pass(
                    &self.device,
                    &mut enc,
                    &self.layer_composite,
                    &params_buf,
                    &backdrop_view,
                    // 参数顺序是 **(backdrop, layer_tex, shadow_tex, target)**。
                    // 这里曾经把两者写反 —— 于是"模糊后的内容"被当成图层内容合成、
                    // 而阴影拿到的是未模糊的内容。症状是层边缘出现一圈**不该有的淡影**
                    // （实测 7567 个像素不同，首个差异点 (58,17) 的值恰好等于模糊值）。
                    layer_view,
                    shadow_view,
                    &target_view,
                    (rx0, ry0, rx1, ry1),
                )?;
                self.queue.submit(Some(enc.finish()));
            }
        }
        Ok(())
    }

    /// 对一张纹理做可分离盒式模糊（3 遍），返回**新纹理**（源纹理不动）。
    ///
    /// 为什么**不改源纹理**：投影同时需要"未模糊的内容"（画内容本身）与
    /// "已模糊的内容"（画阴影），所以源必须留着。
    ///
    /// 乒乓用两张新纹理：`view → B → A`，之后两轮是 `A → B → A`。
    /// **必须让"上一轮的结果"成为下一轮的源** —— 曾经三轮都从 `view` 出发，
    /// 等于只做了一遍模糊（实测与参考差 45/255，差异范围恰好是模糊扩散区）。
    ///
    /// **横纵两趟必须用各自独立的参数缓冲**：同一个命令缓冲里共用一个缓冲并写
    /// 两次，队列上"写参数1 → 写参数2 → 提交"会使两趟都读到第二次的值 ——
    /// 横向那趟从未执行（这是 M6 实际踩到的坑，见 `blur.rs`）。
    pub(crate) fn blur_texture(
        &self,
        view: &wgpu::TextureView,
        w: u32,
        h: u32,
        sigma: f32,
    ) -> Result<Option<(wgpu::Texture, wgpu::TextureView)>, RenderError> {
        let box_w = blur::box_width_for(sigma, 3);
        if sigma <= 0.0 || box_w <= 1 || w == 0 || h == 0 {
            return Ok(None);
        }
        // 乒乓两张，**不动源纹理**（投影同时需要未模糊与已模糊两份）
        let (a_tex, a_view) =
            create_layer_texture(&self.device, "amiki_layer_blur_a", w, h)?;
        let (_b_tex, b_view) =
            create_layer_texture(&self.device, "amiki_layer_blur_b", w, h)?;
        let mk = |label: &str, vertical: bool| {
            let b = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<blur::BlurParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(
                &b,
                0,
                bytemuck::bytes_of(&blur::BlurParams {
                    width: w as f32,
                    height: h as f32,
                    box_w: box_w as f32,
                    vertical: if vertical { 1.0 } else { 0.0 },
                }),
            );
            b
        };
        let params_h = mk("amiki_layer_blur_h", false);
        let params_v = mk("amiki_layer_blur_v", true);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("amiki_layer_blur"),
        });
        // **乒乓必须让"上一轮的结果"成为下一轮的源**：
        // 第一对是 content → B → A，之后两对必须是 A → B → A。
        // 这里曾经写成三轮都从 `view`（原始内容）出发 —— 等于只做了一遍模糊
        // （实测模糊层与参考差 45/255，差异范围恰好是模糊扩散区）。
        blur::blur_pass(&self.device, &mut enc, &self.blur, &params_h, view, &b_view);
        blur::blur_pass(&self.device, &mut enc, &self.blur, &params_v, &b_view, &a_view);
        for _ in 0..2 {
            blur::blur_pass(&self.device, &mut enc, &self.blur, &params_h, &a_view, &b_view);
            blur::blur_pass(&self.device, &mut enc, &self.blur, &params_v, &b_view, &a_view);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(Some((a_tex, a_view)))
    }
}
