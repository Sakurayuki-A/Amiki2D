//! M6：GPU 可分离盒式模糊（近似高斯）。
//!
//! # 为什么用渲染 pass + fragment shader，而不是 compute + storage texture
//!
//! 最初我用 compute 着色器 + `textureLoad` 读源、`textureStore` 写目标。
//! 结果是 **dispatch 确实执行了（写常量可见），但读不到数据** ——
//! 输出与输入逐字节相同。
//!
//! 改用渲染 pass 后问题消失：fragment shader 的**采样器**是本项目已经用熟的
//! 路径（裁剪覆盖率 pass 就是这么做的），而"采样纹理 + 存储纹理"的绑定组合
//! 在我这里的用法下始终拿不到数据。
//!
//! 结论：**走已验证的模式，而不是走理论上更"现代"的模式。**
//!
//! # 为什么是可分离 + 三次盒式
//!
//! 真二维高斯每像素 O(r²)；**三次盒式**（横纵各 3 遍）在每像素固定采样数下
//! 即可良好逼近高斯，且**可分离**（先横后纵）。与 CPU 参考实现
//! （`amiki-raster::filter`）同算法、同盒宽公式，因此两端结果应当接近 ——
//! 这正是跨端对拍能成立的前提。
//!
//! 盒宽 `w ≈ sqrt(12σ²/n + 1)`（n = 3）并强制为奇数保证对称。

use crate::{err, RenderError};

/// 模糊参数（uniform；fragment 与 vertex 共用，故布局需与 WGSL 一致）。
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurParams {
    pub width: f32,
    pub height: f32,
    /// 盒宽（奇数）
    pub box_w: f32,
    /// 0 = 横向，1 = 纵向
    pub vertical: f32,
}

/// 顶点着色器：一个覆盖全屏的三角形（无需顶点缓冲）。
///
/// 用 3 个顶点覆盖整个裁剪空间，比 4 顶点四边形少一次三角形装配，
/// 且不需要 `@builtin(vertex_index)` 之外的任何输入。
const VS: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    // 覆盖 [-1,3] 的三角形：(0,0) (2,0) (0,2) 映射到 uv
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var o: VOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    // 转换到 [0,1] 纹理坐标
    o.uv = (p[i] + vec2<f32>(1.0, 1.0)) * 0.5;
    return o;
}
"#;

/// 片元着色器：沿一个方向做盒式平均。
const FS: &str = r#"
struct P {
    width: f32,
    height: f32,
    box_w: f32,
    vertical: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var src_smp: sampler;

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let r = i32(p.box_w * 0.5);
    let n = f32(2 * r + 1);
    // 纹素尺寸
    let texel = vec2<f32>(1.0 / p.width, 1.0 / p.height);
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = -r; i <= r; i = i + 1) {
        var off = vec2<f32>(0.0, 0.0);
        if (p.vertical < 0.5) {
            off = vec2<f32>(f32(i) * texel.x, 0.0);
        } else {
            off = vec2<f32>(0.0, f32(i) * texel.y);
        }
        // 钳到 [0,1] 避免采到边缘外
        let s = clamp(uv + off, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
        acc += textureSampleLevel(src_tex, src_smp, s, 0.0);
    }
    return acc / n;
}
"#;

/// 计算与 CPU 一致的盒宽（奇数）。
///
/// 与 `amiki-raster::filter` 的 `box_width_for` **必须是同一公式** ——
/// 否则两端模糊半径不同，跨端对拍就失去意义。
pub fn box_width_for(sigma: f32, n: u32) -> u32 {
    if sigma <= 0.0 {
        return 1;
    }
    let ideal = (12.0 * sigma * sigma / n as f32 + 1.0).sqrt();
    let mut w = ideal.round() as u32;
    if w.is_multiple_of(2) {
        w += 1;
    }
    w.max(1)
}

/// 模糊纹理用途：既要能作渲染目标，又要能作采样源。
pub const BLUR_TEXTURE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
    .union(wgpu::TextureUsages::TEXTURE_BINDING)
    .union(wgpu::TextureUsages::COPY_SRC)
    .union(wgpu::TextureUsages::COPY_DST);

/// 模糊所需的资源（渲染管线 + 布局 + 采样器）。
pub struct BlurResources {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
}

/// 创建模糊管线。格式固定 `Rgba8Unorm`（与本项目输出纹理一致）。
pub fn create_resources(device: &wgpu::Device) -> BlurResources {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("amiki_blur"),
        source: wgpu::ShaderSource::Wgsl(format!("{VS}\n{FS}").into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("amiki_blur_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("amiki_blur_pl"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("amiki_blur_pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("amiki_blur_sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    BlurResources { pipeline, bgl, sampler }
}

/// 一趟方向模糊：把 `src` 模糊后渲染进 `dst`。
///
/// 调用方需在此**之前**把含方向的 `BlurParams` 写进 `params_buf`。
pub fn blur_pass(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    res: &BlurResources,
    params_buf: &wgpu::Buffer,
    src: &wgpu::TextureView,
    dst: &wgpu::TextureView,
) {
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("amiki_blur_bind"),
        layout: &res.bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&res.sampler) },
        ],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("amiki_blur_pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: dst,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(&res.pipeline);
    pass.set_bind_group(0, &bind, &[]);
    pass.draw(0..3, 0..1);
}

/// 创建一张可用于模糊（可渲染 + 可采样 + 可读回）的纹理。
pub fn create_blur_texture(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> Result<(wgpu::Texture, wgpu::TextureView), RenderError> {
    if width == 0 || height == 0 {
        return err("模糊纹理尺寸为 0");
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: BLUR_TEXTURE_USAGE,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((tex, view))
}
