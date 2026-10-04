//! M6：GPU 可分离盒式模糊（近似高斯），以**渲染 pass + fragment 采样**实现。
//!
//! # 为什么是渲染 pass 而不是 compute + storage texture
//!
//! 最初用 compute 着色器 + `textureLoad`/`textureStore`：dispatch 确实执行
//! （换成写常量的着色器可见全屏变红），但**读不到数据**，输出与输入逐字节相同。
//! 换成渲染 pass + 采样器后问题消失 —— 这是本项目已经用熟的路径
//! （裁剪覆盖率 pass 就是这么做的）。
//!
//! 教训：**走已验证的模式，而不是理论上更"现代"的模式。**
//!
//! # 为什么是可分离 + 三次盒式
//!
//! 真二维高斯每像素 O(r²)；三次盒式（横纵各 3 遍）在每像素固定采样数下即可
//! 良好逼近高斯，且可分离。与 CPU 参考实现（`amiki-raster::filter`）同算法、
//! 同盒宽公式，因此两端结果应当接近 —— 这正是跨端对拍能成立的前提。
//!
//! 盒宽 `w ≈ sqrt(12σ²/n + 1)`（n = 3）并强制为奇数保证对称。

use crate::{err, RenderError};

/// 模糊参数（uniform；4 个 f32，与 WGSL 的 `struct P` 一一对应）。
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

const VS: &str = r#"
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    // 覆盖整个裁剪空间的三角形（无需顶点缓冲）
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var o: VOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.uv = (p[i] + vec2<f32>(1.0, 1.0)) * 0.5;
    return o;
}
"#;

const FS: &str = r#"
struct P {
    width: f32,
    height: f32,
    box_w: f32,
    vertical: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
// 仍保留采样器绑定：绑定布局与既有调用方保持一致。着色器已改为整数抽头，
// **不再使用它** —— 见 `fs` 里的说明。
@group(0) @binding(2) var src_smp: sampler;

@fragment
fn fs(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let r = i32(p.box_w * 0.5);
    let n = f32(2 * r + 1);
    let size = vec2<i32>(i32(p.width), i32(p.height));
    // 本 fragment 对应的像素（uv 落在像素中心，故 floor(uv*size) 即像素索引）
    let px = vec2<i32>(i32(floor(uv.x * p.width)), i32(floor(uv.y * p.height)));
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = -r; i <= r; i = i + 1) {
        var q = px;
        if (p.vertical < 0.5) {
            q.x = q.x + i;
        } else {
            q.y = q.y + i;
        }
        // **整数抽头 + 显式钳制**，而不是线性采样。
        //
        // 这里曾经用 `textureSampleLevel` + Linear 过滤：即使采样点落在纹素中心，
        // 浮点误差也会让相邻纹素被插值进来，于是在**真实模糊半径之外漏出微小数值**
        // （实测在远离内容处产生 ~2% 的假 alpha）。三遍累积后被放大成可见的假阴影
        // —— 投影层与 CPU 差 122/255，而差异恰好落在阴影实际不该出现的区域。
        //
        // 整数抽头与 CPU 的滑动窗口**逐抽头一致**，没有插值、也没有泄漏。
        q = clamp(q, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
        acc += textureLoad(src_tex, q, 0);
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

/// 模糊资源（渲染管线 + 绑定布局 + 采样器）。
pub struct BlurResources {
    pub pipeline: wgpu::RenderPipeline,
    pub bgl: wgpu::BindGroupLayout,
    pub sampler: wgpu::Sampler,
}

/// 创建模糊管线。目标格式 **`Rgba16Float`**（与层纹理、模糊纹理一致）：
/// HDR 要求 >1.0 的高光源能穿过模糊存活，8 位格式会在写入时裁到 1.0。
/// 8 位只出现在最终输出那一步（见 `docs/COLOR.md`）。
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
                format: wgpu::TextureFormat::Rgba16Float,
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
/// 调用方需在此**之前**把对应方向的 `BlurParams` 写进 `params_buf`。
///
/// **注意**：横纵两趟必须使用**各自独立的** `params_buf`。同一个命令缓冲内
/// 若共用一个缓冲并写两次，队列上"写参数1 → 写参数2 → 提交"会使两趟都读到
/// 第二次的值 —— 表现为"横向那趟从未执行"，且若测试数据恰好对该方向不敏感，
/// 就会看不出任何异常。
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
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&res.sampler),
            },
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
        format: wgpu::TextureFormat::Rgba16Float,
        usage: BLUR_TEXTURE_USAGE,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    Ok((tex, view))
}
