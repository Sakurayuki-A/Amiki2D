//! amiki-gpu —— wgpu 计算着色器渲染器（M3：稀疏桶表全 GPU 分箱）。
//!
//! CPU 每帧只做：场景 → 控制点/直线提取（不压平、不分箱）→ 上传。
//!
//! **稀疏路径（默认，M3b）**：显存与画面实际内容成比例
//! ```text
//! enc1a: clear strips → flatten+strip_extract → measure → 3×scan →
//!        tile 行活跃列表 → 逐 tile 桶表        〖读回 32B 规模〗
//! enc1b: clear_sparse → emit（成员计数 + carry 定点桶）〖读回 32B 成员数〗
//! enc2:  copy+scan(单元 CSR) → tile_fill → scatter → carry → fine
//! ```
//! **稠密路径（M2f，对拍基准）**：`(tile, draw, row)` 稠密网格，显存 ∝ tiles×draws；
//! 与稀疏路径渲染结果逐字节一致（见 `sparse_matches_dense` 测试）。
//!
//! 两条路径共用：GPU 均匀压平（闭式段数界）、通用三级前缀和、active tile 元数据
//! （[`PipelineDump`] 可逐级读出）、GPU 时间戳分析（[`PassTimings`]）。

//! API 稳定性：本 crate 的稳定面为 `GpuRenderer` / `RenderError` /
//! `RenderPath` / `Frame` / `Blitter` / `GpuScene` / `build_gpu_scene`；
//! `profile`、`consistency`、容量表与各内部模块属**实验面**。
//! 完整划分见仓库 `docs/API.md`。

#![forbid(unsafe_code)]

use amiki_scene::Scene;
use std::sync::Mutex;
use std::time::Instant;
use wgpu::util::DeviceExt;

const FINE_MATH_WGSL: &str = include_str!("fine_math.wgsl");
const FINE_WGSL: &str = include_str!("fine.wgsl");
const SPARSE_FINE_WGSL: &str = include_str!("sparse_fine.wgsl");

/// M6：GPU 离屏层滤镜（可分离盒式模糊）。
use amiki_scene::rgba8_from_f32_with;
pub mod blit;
pub mod blur;
pub mod consistency;
pub mod dense;
pub mod extract;
pub mod layer;
pub mod profile;
pub mod readback;
pub mod sparse;
pub use blit::Blitter;
pub use consistency::check_sparse_consistency;
pub use extract::{build_gpu_scene, build_gpu_scene_into, GpuScene};
pub(crate) use extract::{BinsParams, FineParams, FlattenParams, ScanParams};
pub(crate) use profile::TimestampRecorder;
pub use profile::{
    FrameProfile, PassTimings, PipelineDump, SceneKey, SparseCapacity, SparseCapacityTable,
    SparseStats,
};

/// `GpuScene` 各类图元的数量（诊断与容量换算共用）。
fn nq_len(gs: &GpuScene) -> usize {
    gs.quads.len()
}
fn nc_len(gs: &GpuScene) -> usize {
    gs.cubics.len()
}
fn nl_len(gs: &GpuScene) -> usize {
    gs.lines.len()
}

const SCAN_WGSL: &str = include_str!("scan.wgsl");
const FLATTEN_WGSL: &str = include_str!("flatten.wgsl");
const BINS_WGSL: &str = include_str!("bins.wgsl");
const STRIPS_WGSL: &str = include_str!("strips.wgsl");
const SPARSE_MEASURE_WGSL: &str = include_str!("sparse_measure.wgsl");
const SPARSE_TILES_WGSL: &str = include_str!("sparse_tiles.wgsl");
const SPARSE_EMIT_WGSL: &str = include_str!("sparse_emit.wgsl");
const SPARSE_SCATTER_WGSL: &str = include_str!("sparse_scatter.wgsl");
const SPARSE_CARRY_WGSL: &str = include_str!("sparse_carry.wgsl");

/// 渲染路径：稠密网格（M2f，正确性基准）或稀疏桶表（M3b，主线）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderPath {
    /// 稠密 (tile, draw, row) 网格；显存与 tiles×draws 成比例，仅限小场景
    Dense,
    /// 稀疏 strip/桶表；显存与画面实际内容成比例（M3c 验证后为默认）
    #[default]
    Sparse,
}

#[derive(Debug)]
pub struct RenderError(pub String);

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RenderError {}

fn err<T>(msg: impl Into<String>) -> Result<T, RenderError> {
    Err(RenderError(msg.into()))
}

fn uniform_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_bgl_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}



/// M6：把渐变定义打包成 `sparse_fine.wgsl` 期望的 `vec4<f32>` 布局。
///
/// **必须与 WGSL 侧 `GRAD_STRIDE` / `grad_color()` 严格一致**，
/// 否则着色会静默取错数据（这类错误不会 panic，只会画错）。
///
/// 布局（`S = 2 + MAX_GRAD_STOPS*2` 个 vec4）：
/// ```text
/// [0]                    (n_grad, 0, 0, 0)
/// [1 + g*S]              (kind, repeat, n_stops, MAX_GRAD_STOPS)
/// [1 + g*S + 1]          (x0, y0, x1, y1)     径向: (cx, cy, r, 0)
/// [1 + g*S + 2 + i*2]    (t_i, r, g, b)
/// [1 + g*S + 2 + i*2 + 1](0, 0, 0, 0)          对齐占位
/// ```
/// 色标超过 `MAX_GRAD_STOPS` 时**截断并保持端点色**：与其静默丢中间色造成
/// 肉眼可见的跳变，不如明确只保留前 N 个（调用方应自行控制色标数量）。
/// 返回的缓冲至少含头部 1 个 vec4（空场景也要能绑定）。
fn pack_gradients(grads: &[std::sync::Arc<amiki_scene::Gradient>]) -> Vec<[f32; 4]> {
    const MAX_STOPS: usize = 8;
    const STRIDE: usize = 2 + MAX_STOPS * 2;

    let mut out: Vec<[f32; 4]> = Vec::with_capacity(1 + grads.len() * STRIDE);
    out.push([grads.len() as f32, 0.0, 0.0, 0.0]);
    for g in grads {
        // kind 标签与几何一起由 match 求出，避免"先声明后赋值"的分支式写法
        let (kind_tag, geo) = match g.kind {
            amiki_scene::GradientKind::Linear { from, to } => {
                (0.0, [from.x, from.y, to.x, to.y])
            }
            amiki_scene::GradientKind::Radial { center, radius } => {
                (1.0, [center.x, center.y, radius, 0.0])
            }
            amiki_scene::GradientKind::Sweep { center, start } => {
                (2.0, [center.x, center.y, start, 0.0])
            }
        };
        let n = g.stops.len().min(MAX_STOPS);
        let head = [
            kind_tag,
            if g.repeat { 1.0 } else { 0.0 },
            n as f32,
            MAX_STOPS as f32,
        ];
        out.push(head);
        out.push(geo);
        for i in 0..MAX_STOPS {
            if i < n {
                let s = &g.stops[i];
                out.push([s.t, s.color.r, s.color.g, s.color.b]);
            } else {
                out.push([0.0; 4]);
            }
            out.push([0.0; 4]); // 对齐占位
        }
    }
    out
}
/// M6：把场景用到的位图上传为一个**纹理数组**（每张图一层）。
///
/// 用纹理数组而非逐图绑定：绑定槽位有限，而场景可能有很多张图；
/// 数组 + 层号让 draw 只需在放置参数里带一个整数层号。
///
/// 尺寸不一致时**居中左上对齐、越界部分透明**：不做缩放，
/// 因为缩放语义（拉伸/平铺/留白）应由调用方通过放置显式决定，
/// 引擎在这里静默缩放会让"图像显示成什么样"变得不可预测。
///
/// 无图像时创建一个 1×1 的占位层，保证绑定始终合法。
impl GpuRenderer {
    fn mk_image_texture_array(
        &self,
        images: &[(std::sync::Arc<amiki_scene::Image>, amiki_scene::ImagePlacement)],
    ) -> Result<(wgpu::TextureView, wgpu::Sampler, wgpu::Sampler), RenderError> {
        // **GL 后端上禁用位图（显式报错）**。
        //
        // 实测：`Backends::GL` 下 `amiki-gpu` 的 20 条测试里 19 条通过，
        // 唯独位图那条整张图**读不出来** —— 经采样器得到 `(0,0,0,1)`（不透明黑、
        // GL 对"不完整纹理"的规定返回值），直接 `textureLoad` 得到 `(0,0,0,0)`，
        // 也就是这张数组纹理在该后端上是空的。
        //
        // 已排除的原因：采样器 LOD 钳位、`textureSampleLevel` vs `textureSampleGrad`、
        // 辅助函数内联、`write_texture` 行距 256 对齐 —— 都不是。
        // 判断是 wgpu/naga 的 GL 后端在 `texture_2d_array` 上的问题，不在我方代码。
        //
        // 按本项目一贯约定：**不支持就显式报错，不静默画错**。
        // 走到这里说明调用方要画位图，而该后端给不出正确结果 ——
        // 与其画出空白，不如让调用方明确知道该换后端或走 CPU 引擎。
        // **必须是"场景里确实有位图"才拦**：本函数在没有位图时也会被调用来
        // 创建占位资源（绑定布局需要合法的非空绑定），那种情况与位图能力无关。
        if self.backend == Some(wgpu::Backend::Gl) && !images.is_empty() {
            return err(
                "GL 后端无法正确读取位图（`texture_2d_array` 在该后端上读回为空，实测采样恒为不透明黑）。请改用 Vulkan/DX12 后端，或使用 CPU 引擎（amiki_raster）；可用 AMIKI_BACKENDS 指定后端",
            );
        }
        let max_dim = self.limits.max_texture_dimension_2d;
        let w = images.iter().map(|(i, _)| i.width).max().unwrap_or(1).min(max_dim);
        let h = images.iter().map(|(i, _)| i.height).max().unwrap_or(1).min(max_dim);
        let layers = images.len().max(1) as u32;
        // 层数上限：不同后端不同，取保守值并在超出时明确报错
        let max_layers = self.limits.max_texture_array_layers;
        if layers > max_layers {
            return err(format!(
                "位图层数 {layers} 超出设备上限 {max_layers}；请合并图像或分帧绘制"
            ));
        }

        // B4：任一图像带 mip 链 => 数组纹理开 mip 层。链在 CPU 生成（见
        // `amiki_scene::mipmap`），这里只是上传 —— GPU 生成会引入第二份滤波
        // 实现，与"双端逐像素一致"冲突。
        // 层级数按**数组尺寸**取：所有层共享数组纹理，小于数组的图像
        // 用补齐版链（`generate_padded_mip_levels`，同一套滤波数学）。
        let any_mip = images.iter().any(|(i, _)| i.mip.is_some());
        let mip_count = if any_mip { amiki_scene::mip_level_count(w, h) } else { 1 };

        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("amiki_image_array"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 逐层逐级上传（上层若比底图小，剩余部分保持透明）
        const BPP: usize = 4;
        for (li, (img, _)) in images.iter().enumerate() {
            // 该层的链：数组尺寸恰好等于图像尺寸时直接用图像自己的链；
            // 否则现场生成补齐版链（确定性，与直接生成同一套滤波数学）
            let padded;
            let chain: &[amiki_scene::MipLevel] =
                if img.mip.is_some() && (img.width, img.height) != (w, h) {
                    padded = amiki_scene::generate_padded_mip_levels(
                        img.width, img.height, &img.data, w, h,
                    );
                    &padded
                } else {
                    img.mip.as_ref().map(|c| &c.levels[..]).unwrap_or(&[])
                };
            for level in 0..mip_count {
                let lw = (w >> level).max(1);
                let lh = (h >> level).max(1);
                // level 0 = 原图（补齐到数组尺寸）；更深 = 链级（同尺寸公式）
                let level_bytes: Vec<u8> = if level == 0 {
                    let mut buf = vec![0u8; (w * h * 4) as usize];
                    let cw = img.width.min(w) as usize;
                    let ch = img.height.min(h) as usize;
                    let stride = (w as usize) * BPP;
                    for y in 0..ch {
                        let src = (y * img.width as usize) * BPP;
                        let dst = y * stride;
                        buf[dst..dst + cw * BPP].copy_from_slice(&img.data[src..src + cw * BPP]);
                    }
                    buf
                } else {
                    match chain.get(level as usize - 1) {
                        Some(l) if (l.width, l.height) == (lw, lh) => l.data.clone(),
                        Some(l) => {
                            return err(format!(
                                "图像 {li} 的 mip 层级 {level} 尺寸 {}×{} 与数组级 {lw}×{lh} 不符",
                                l.width, l.height
                            ))
                        }
                        // 链到此为止（图像比数组小、层级更深）：保持透明。
                        // 该图像自己的 LOD 钳在链深处，采不到这些空层。
                        None => vec![0u8; (lw * lh * 4) as usize],
                    }
                };
                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: li as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &level_bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(lw * 4),
                        rows_per_image: Some(lh),
                    },
                    wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
                );
            }
        }

        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("amiki_image_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        // 采样器（binding 16）：clamp-to-edge + 线性过滤，供**非平铺**的图像采样。
        // 平铺在接缝半纹素处必须真回绕（与 CPU 的 `rem_euclid` 取纹素一致），
        // 那是另一个采样器的事 —— 见下方 `repeat_sampler` / binding 17。
        // clip 覆盖率纹理走 `textureLoad`，不经采样器，与本处无关
        // （0.6.0 之前只有一个采样器时，"它同时服务 clip 纹理"曾是它不能设
        // repeat 的理由；现在这个理由已不成立，故拆成两个）。
        //
        // **LOD 钳制**（GL 完整性问题的历史：wgpu `default()` 的
        // `lod_max_clamp` 是 32.0，GL 据 LOD 范围判纹理"不完整" ->
        // 采样恒为 (0,0,0,1)，实测整张位图变纯黑）：
        // - 无 mip：钳到 0（lod_max_clamp = 0.0），原因同上；
        // - 有 mip：放开钳制（超出实际层级数的 LOD 由硬件钳到最深一级），
        //   且 mip 过滤改 Linear —— `textureSampleLevel` 的显式 LOD
        //   （CPU 算好打包进放置缓冲）才能做层间插值。
        //   GL 后端有位图时本来就显式报错（本函数开头），不受影响。
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("amiki_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: if any_mip { 100.0 } else { 0.0 },
            mipmap_filter: if any_mip {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            ..Default::default()
        });
        // 平铺专用：除地址模式外与 clamp 版逐字段一致（同样的 LOD 钳制策略）。
        // GL 后端有位图时已在本函数开头显式报错，完整性问题与 clamp 版同。
        //
        // 为什么需要第二个采样器：WGSL 里的 `uv - floor(uv)` 只把 uv 归到 [0,1)，
        // 而双线性在每个 mip 层的**取纹素**还会越出一次半纹素 —— 那一步由采样器
        // 的地址模式决定。ClampToEdge 会钳到边缘，CPU 的 `rem_euclid` 会回绕到
        // 另一侧，于是 uv 落在 [0,1] 边缘约 1.6% 带内时两端不一致（mipmap 对拍
        // 首次暴露的潜伏 bug，此前平铺测试的像素中心恰好够不到那条带）。
        let repeat_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("amiki_image_sampler_repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: if any_mip { 100.0 } else { 0.0 },
            mipmap_filter: if any_mip {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            },
            ..Default::default()
        });
        Ok((view, sampler, repeat_sampler))
    }
}

/// M6：把每个 draw 的图像放置打包成 `sparse_fine.wgsl` 期望的布局。
///
/// **每 draw 三个 vec4**，与 WGSL 的 `image_color` / `image_index_of` 严格对应：
/// ```text
/// [d*3+0] = (origin.x, origin.y, u_axis.x, u_axis.y)
/// [d*3+1] = (v_axis.x, v_axis.y, tex_layer, 0)
/// [d*3+2] = (bitcast<f32>(image_index), repeat, lod, 0)
/// ```
/// 把 `image_index` 放进放置数据自身，使该缓冲区**自描述**：
/// "这个 draw 是不是图像"无需另占 `draw_meta` 分量，也不会与渐变下标争位置。
/// 非图像 draw 在 `[d*3+2].x` 写 `u32::MAX` 的位模式（WGSL 用 bitcast 读回）。
fn pack_image_place(gs: &GpuScene) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; gs.image_index.len() * 3];
    for (d, idx) in gs.image_index.iter().enumerate() {
        let slot = d * 3;
        if *idx == u32::MAX {
            out[slot + 2][0] = f32::from_bits(u32::MAX);
            continue;
        }
        let (img, place) = &gs.images[*idx as usize];
        out[slot] = [place.origin.x, place.origin.y, place.u_axis.x, place.u_axis.y];
        // 纹理层号 = 该图像在纹理数组中的层（与 gs.images 同序）
        out[slot + 1] = [place.v_axis.x, place.v_axis.y, *idx as f32, 0.0];
        // B4：e2.z 是该 draw 的 LOD（含 bias）。**在 CPU 上算好打包**，
        // WGSL 只读不算 —— `log2` 在 CPU libm 与 GPU 内建函数间可能差 1 ULP，
        // 采样层级对它敏感，不在着色器里算就从根上消掉这个漂移源。
        // 没有 mip 链的图像固定 0（采样器钳制 + 纹理只有 1 层，行为不变）。
        let lod = if img.mip.is_some() {
            amiki_scene::compute_lod(
                place.u_axis,
                place.v_axis,
                img.width,
                img.height,
                gs.mip_bias,
            )
        } else {
            0.0
        };
        out[slot + 2] =
            [f32::from_bits(*idx), if place.repeat { 1.0 } else { 0.0 }, lod, 0.0];
    }
    out
}

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
fn composite_layer_gpu(
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

/// 一帧的推进状态（P3 第 ② 步）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PollStatus {
    Pending,
    Ready,
}

/// **在飞的一帧**（P3 第 ② 步）：`begin` 之后它已在 GPU 上跑，`poll` 不阻塞。
///
/// 语义（写死在文档里，C ABI 也照此）：
/// - `poll()` 非阻塞、可反复调用（`Polling` 单次实测 0.071 ms）；
/// - `try_read()` **只取值、不推进**：回调还没跑过就返回 `None`，句柄留在原地，
///   调用方可以继续 `poll` —— 所以它**既不返回半成品，也不 panic**
///   （未映射的缓冲上取像素是 wgpu 明确会 panic 的用法）；
/// - **层场景直接同步算完**（层合成需要根图像，异步没有收益）⇒ `begin` 后第一次
///   `poll` 即 `Ready`。这是**有意**的，不是遗漏；
/// - CPU 后端不走这里（它同步），所以"首帧即就绪"对调用方是统一语义。
pub struct PendingFrame {
    gpu: std::sync::Arc<GpuRenderer>,
    /// `Some` = 已算完（层场景，或推进中出错）
    ready: Option<Result<Vec<u8>, RenderError>>,
    /// `Some` = 还在飞：在飞回读 + 最终要用的色调映射
    pending: Option<(readback::PendingRead, amiki_scene::ToneMap)>,
}

impl GpuRenderer {
    /// 提交一帧，**不等待**（等价 `begin_with(..., ToneMap::Raw, 0.0)`）。
    pub fn begin(
        self: &std::sync::Arc<Self>,
        scene: &Scene,
        eps: f32,
    ) -> Result<PendingFrame, RenderError> {
        self.begin_with(scene, eps, amiki_scene::ToneMap::Raw, 0.0)
    }

    /// 提交一帧，**不等待**；返回在飞句柄，何时 `poll` / `try_read` 由调用方决定。
    pub fn begin_with(
        self: &std::sync::Arc<Self>,
        scene: &Scene,
        eps: f32,
        tone: amiki_scene::ToneMap,
        mip_bias: f32,
    ) -> Result<PendingFrame, RenderError> {
        if !scene.layers.is_empty() {
            // 层合成需要**根图像**（每层要合成回它）⇒ 异步在这里没有收益，
            // 直接同步算完并放进 `ready`。文档已写明，不是静默降级。
            return Ok(PendingFrame {
                gpu: std::sync::Arc::clone(self),
                ready: Some(self.render_blocking_with(scene, eps, tone, mip_bias)),
                pending: None,
            });
        }
        let frame = self.prepare_with(scene, eps, mip_bias)?;
        let tex = self.create_output_texture(&frame.scene);
        self.run_pipeline(&frame.scene, &tex, false, &frame.clip_views, None)?;
        // 拷贝 + map_async 都已提交；这里**不等**。
        let (p, _) = self.begin_read_raw(&tex, 8)?;
        Ok(PendingFrame {
            gpu: std::sync::Arc::clone(self),
            ready: None,
            pending: Some((p, tone)),
        })
    }
}

impl PendingFrame {
    /// 非阻塞推进一次。
    pub fn poll(&mut self) -> PollStatus {
        if self.ready.is_some() {
            return PollStatus::Ready;
        }
        let Some((p, _)) = self.pending.as_mut() else {
            return PollStatus::Ready;
        };
        match self.gpu.poll_read_raw(p) {
            Ok(true) => PollStatus::Ready,
            Ok(false) => PollStatus::Pending,
            Err(e) => {
                // 出错也转 Ready，让 try_read 把错误原样报出去（不吞错）
                self.ready = Some(Err(e));
                self.pending = None;
                PollStatus::Ready
            }
        }
    }

    /// **就绪则取走**（`&mut self`）：未就绪返回 `None`，且**这一帧什么都不动**
    /// （还能继续 `poll`、再取）。
    ///
    /// 判据是 `device.poll` **有没有把 map 回调跑过**（`PendingRead::map_outcome`）：
    /// 未映射的缓冲上取像素是 wgpu 明确会 panic 的用法，所以"先问再取"不是防御性
    /// 代码，而是**唯一正确**的顺序。
    ///
    /// C ABI 的三入口走这条 —— 它需要"取不到不算数"（未就绪 ⇒ `AMIKI_ERROR_BUSY`，
    /// 帧还在，调用方可以接着问）。
    pub fn take_ready(&mut self) -> Option<Result<Vec<u8>, RenderError>> {
        if let Some(r) = self.ready.take() {
            return Some(r);
        }
        let (mut p, tone) = self.pending.take()?;
        match p.map_outcome() {
            Some(Ok(())) => {
                Some(self.gpu.finish_read_f32(p).map(|px| rgba8_from_f32_with(&px, tone)))
            }
            Some(Err(msg)) => Some(Err(RenderError(msg))),
            None => {
                // 还在飞：原样放回，**不消费**这次提交
                self.pending = Some((p, tone));
                None
            }
        }
    }

    /// 取值（**消费句柄**）：就绪则取走；未就绪返回 `None`，句柄连同在飞的回读
    /// 一起被丢弃 —— 调用方已经放弃它了。想要"取不到还能接着问"用 [`Self::take_ready`]。
    pub fn try_read(mut self) -> Option<Result<Vec<u8>, RenderError>> {
        self.take_ready()
    }
}

/// 校验脏区与画布 tile 尺寸一致。
///
/// `Damage` 用的是它自己的画布 tile 数，而派发范围按当前场景的 tile 数钳制。
/// 若两者不一致（例如调用方复用了为不同尺寸画布建的 `Damage`），
/// "全脏"会被误判成"只脏左上角"，从而**静默留下陈旧像素**。
/// 这类错误必须炸出来，不能靠钳制掩盖。
fn check_damage_bounds(damage: Option<&amiki_scene::Damage>, tiles_x: u32, tiles_y: u32) {
    if let Some(d) = damage {
        assert_eq!(
            (d.total_tiles()),
            tiles_x.max(1) * tiles_y.max(1),
            "damage 的 tile 数与当前画布不一致（damage 可能是为别的画布尺寸创建的）—— 否则「全脏」会被误当作局部脏区，导致陈旧像素残留"
        );
    }
}

/// fine 阶段的**脏 tile 索引表**（1D 派发用）：每项打包成 `tx | ty << 16`。
///
/// - `damage` 为 `None` 或全脏 ⇒ 整幅画布的 tile（row-major）
/// - 否则 ⇒ 脏矩形覆盖的 tile（row-major；矩形是 `Damage` 目前唯一的表达）
///
/// # 为什么把 (tx, ty) 放进缓冲，而不是让内核从 `global_invocation_id` 推
///
/// wgpu 的 `dispatch_workgroups` **没有起点偏移**。早先的实现把脏区换算成
/// `(x_start, x_count, y_start, y_count)`，却只用了后两个 —— 于是脏区不在
/// 左上角时，被重算的是 `[0,x_count) × [0,y_count)`，**真正的脏 tile 一个都没
/// 重算**，画面静默保留上一帧像素（`window_demo` 的滚动就会踩到）。
///
/// 查表之后，派发只与"要重算的 tile 数"有关，与画布尺寸、与脏区起点都无关；
/// 未来要支持非矩形脏区（多个小矩形）也只需要往这张表里多塞几项。
pub(crate) fn dirty_tile_list(
    damage: Option<&amiki_scene::Damage>,
    tiles_x: u32,
    tiles_y: u32,
) -> Vec<u32> {
    let (tx0, ty0, tx1, ty1) = match damage.and_then(|d| d.tiles()) {
        Some(r) => r,
        None => (0, 0, tiles_x, tiles_y),
    };
    let tx0 = tx0.min(tiles_x);
    let tx1 = tx1.min(tiles_x).max(tx0);
    let ty0 = ty0.min(tiles_y);
    let ty1 = ty1.min(tiles_y).max(ty0);
    debug_assert!(tx1 <= 0xFFFF && ty1 <= 0xFFFF, "tile 坐标超出一个 u16 的打包范围");
    let mut out = Vec::with_capacity(((tx1 - tx0) as usize) * ((ty1 - ty0) as usize));
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            out.push(tx | (ty << 16));
        }
    }
    out
}

/// fine 的 1D 派发维度：`(x_workgroups, z_workgroups)`，`y` 恒为 1。
///
/// 每个 workgroup = 一个 tile（`workgroup_size(16,16)`）。单维 workgroup 上限
/// 是 WebGPU 规范保证的 **65535**，所以用 z 维分批 —— 这样任意画布尺寸
/// （含 8K：480×270 = 129 600 tile）都在一张表里放得下。
pub(crate) fn fine_dispatch_dims(n_tiles: u32) -> (u32, u32) {
    /// WebGPU 规范保证的 `maxComputeWorkgroupsPerDimension` 下限。
    const MAX_PER_DIM: u32 = 65535;
    // `clamp` 而非 `min().max()`：后者在 clippy 眼里是手写的 clamp。
    // 下限 1 是因为 workgroup 数不能为 0（`n_tiles == 0` 时调用方本来就跳过派发）。
    let x = n_tiles.clamp(1, MAX_PER_DIM);
    let z = n_tiles.div_ceil(MAX_PER_DIM).max(1);
    (x, z)
}

/// 稀疏路径单维派发（64 线程/工作组）的维度：`(x_workgroups, y_workgroups)`。
///
/// `n_elements == 0` 返回 `(0, 0)`（调用方跳过派发，`dispatch_workgroups(0, ..)` 非法）。
///
/// # 与内核 `lin()` 的契约（**改一处必须改另一处**）
///
/// ```text
/// 行距 = 1024 个工作组 × 64 线程 = 65536 个元素
/// x    = min(⌈n/64⌉, 1024)      y = ⌈⌈n/64⌉ / 1024⌉
/// ```
///
/// 内核的 `lin(gid) = gid.x + gid.y * 65536u` 必须与这里的 `1024` 一致 ——
/// 两侧步长不一致就是这个式子最贵的错法：旧写法按 **65535 个工作组**推进、
/// 内核按 **65535 个元素**解码（差 64 倍），于是行间重叠 95%（`atomicAdd`
/// 的内核重复计数）、尾部元素**永不执行** ⇒ 4K 画布最下面两行瓦片整片空白。
/// 事故全记录见 `CHANGELOG.md` 的 0.10.9 ③。
/// 一致性由 `dispatch_dims_1d_covers_every_element_exactly_once`（纯算术，
/// 不需要 GPU）钉住：任何元素都必须**恰好被处理一次**。
pub(crate) fn dispatch_dims_1d(n_elements: u32) -> (u32, u32) {
    /// 与所有稀疏内核的 `@workgroup_size(64)` 一致。
    const WG: u32 = 64;
    /// 一行的工作组数（= 内核 `lin()` 的行距 65536 个元素）。
    const ROW_WG: u32 = 1024;
    if n_elements == 0 {
        return (0, 0);
    }
    let wg = n_elements.div_ceil(WG);
    let x = wg.min(ROW_WG);
    let y = wg.div_ceil(ROW_WG);
    // 元素上限 = `65535 行 × 65536 元素 ≈ 4.29e9`，而 `n_elements` 是 u32
    // （最大 4.29e9）：只有理论极值会超出 y 维上限，那时 wgpu 报派发维度非法
    // （**响亮地失败**，不是静默少算）。
    debug_assert!(y <= 65_535, "1D 派发的行数超出 WebGPU 的 y 维上限");
    (x, y)
}

fn ts_writes(
    rec: Option<&TimestampRecorder>,
    slot: Option<(u32, u32)>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    match (rec, slot) {
        (Some(rec), Some((b, e))) => Some(wgpu::ComputePassTimestampWrites {
            query_set: &rec.query_set,
            beginning_of_pass_write_index: Some(b),
            end_of_pass_write_index: Some(e),
        }),
        _ => None,
    }
}

/// 从**预分配**的槽位表里取第 `i` 个槽位的时间戳写入描述。
///
/// 刻意写成自由函数并接收 `slots` 的**副本**（`[(u32,u32); N]` 是 `Copy`）：
/// 闭包会捕获并保持对 `ts`/`ts_slots` 的借用，与 `run_scan(.., ts.as_deref_mut())`
/// 的可变借用冲突；自由函数每次只做一次瞬时再借用，故无冲突。
fn ts_slot<'a, const N: usize>(
    ts: Option<&'a TimestampRecorder>,
    slots: Option<[(u32, u32); N]>,
    i: usize,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    match (ts, slots) {
        (Some(r), Some(s)) => ts_writes(Some(r), Some(s[i])),
        _ => None,
    }
}

/// M3a：逐帧资源。主场景 + clip 覆盖率纹理视图。
#[derive(Default)]
pub struct Frame {
    scene: GpuScene,
    /// **clip 覆盖率纹理用的复用缓冲**。每个 clip 都要提取一次场景，
    /// 若每帧每个 clip 都新建缓冲，大块分配 + 缺页的代价会照常付
    /// （见 `GpuScene::reset_for_reuse` 的实测）。
    clip_scratch: GpuScene,
    /// 只用于**持有** clip 纹理的生命周期 —— `clip_views` 借用它们，
    /// 纹理一旦 drop，视图即失效。字段本身从不被读，所以需要 allow。
    #[allow(dead_code)]
    clip_texs: Vec<wgpu::Texture>,
    pub clip_views: Vec<wgpu::TextureView>,
    /// `prepare` 时用的压平容差。层内容的渲染要用**同一个**容差，
    /// 否则层内几何的压平精度会与主场景不一致。
    pub eps: f32,
}

/// GPU 渲染器（设备级状态，可跨帧复用）。
pub struct GpuRenderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// 分箱路径：稠密网格（M2f 对拍基准）或稀疏桶表（M3b 主线）
    pub path: RenderPath,
    /// 适配器名称/后端（基准报告用；`new` 填充，`from_device` 为 "unknown"）
    pub adapter_name: String,
    /// 实际使用的 wgpu 后端。`new` 填充；`from_device` 为 `None`（未知 ⇒ 不做后端相关判定）。
    pub backend: Option<wgpu::Backend>,
    /// 适配器是否支持 GPU 时间戳（`profile` 的前提）
    pub has_timestamps: bool,
    /// 时间戳 tick → 纳秒
    pub timestamp_period_ns: f32,
    /// 设备限制（稠密路径容量预检用：提前返回可读错误而非让 wgpu 校验层 panic）
    pub limits: wgpu::Limits,
    /// M3.6：MAP_READ 暂存缓冲池（读回复用，避免每次读回都新建 + 冷映射）。
    /// 用 `Mutex` 而非 `RefCell`：`&self` 方法需要内部可变性，且便于将来多线程提交。
    readback_pool: Mutex<Vec<wgpu::Buffer>>,
    /// M3.6：上一帧实测的稀疏容量（投机分配）。锁中毒时按"无容量"处理
    /// ——退回精确路径，只会更慢，不会更错。
    sparse_capacity: Mutex<SparseCapacityTable>,
    fine_pipeline: wgpu::ComputePipeline,
    coverage_pipeline: wgpu::ComputePipeline,
    fine_bgl: wgpu::BindGroupLayout,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scan_sums_pipeline: wgpu::ComputePipeline,
    scan_add_pipeline: wgpu::ComputePipeline,
    scan_bgl: wgpu::BindGroupLayout,
    flatten_pipeline: wgpu::ComputePipeline,
    flatten_bgl: wgpu::BindGroupLayout,
    bins_clear_pipeline: wgpu::ComputePipeline,
    bins_emit_pipeline: wgpu::ComputePipeline,
    bins_scatter_pipeline: wgpu::ComputePipeline,
    bins_carry_pipeline: wgpu::ComputePipeline,
    bins_bgl: wgpu::BindGroupLayout,
    strip_clear_pipeline: wgpu::ComputePipeline,
    strip_extract_pipeline: wgpu::ComputePipeline,
    strip_bgl: wgpu::BindGroupLayout,
    sparse_bgl: wgpu::BindGroupLayout,
    sparse_measure_pipeline: wgpu::ComputePipeline,
    sparse_tr_scatter_pipeline: wgpu::ComputePipeline,
    sparse_tr_offsets_pipeline: wgpu::ComputePipeline,
    sparse_tile_count_pipeline: wgpu::ComputePipeline,
    sparse_tile_fill_pipeline: wgpu::ComputePipeline,
    sparse_clear_pipeline: wgpu::ComputePipeline,
    sparse_emit_pipeline: wgpu::ComputePipeline,
    sparse_scatter_pipeline: wgpu::ComputePipeline,
    sparse_carry_pipeline: wgpu::ComputePipeline,
    sparse_fine_pipeline: wgpu::ComputePipeline,
    sparse_coverage_pipeline: wgpu::ComputePipeline,
    dummy_clip_view: wgpu::TextureView,
    _dummy_clip_tex: wgpu::Texture,
    /// M6：图像绑定的占位（稠密路径不用图像，但共用同一 bind group layout）
    dummy_img_view: wgpu::TextureView,
    _dummy_img_tex: wgpu::Texture,
    dummy_img_sampler: wgpu::Sampler,
    /// M6：放置参数的占位缓冲（layout 共用，需要合法非空 storage buffer）
    dummy_img_place: wgpu::Buffer,
    /// M6：离屏层滤镜资源（可分离盒式模糊）。`blur_rgba` 用它。
    blur: blur::BlurResources,
    /// P0-3：离屏层的 **GPU 合成**资源（帧路径用；compute pass，不读回）。
    layer_composite: layer::LayerCompositeResources,
}

/// 临时 bind group 构建期间的 16B 垫片（满足 vec4 数组的最小绑定尺寸）。
const PAD16: [u8; 16] = [0u8; 16];

#[doc(hidden)] // 仅基准与诊断用，见 docs/API.md「实验面」
/// 稠密网格（`(tile, draw, row)`）所需规模：`(元素数, 桶数)`。
///
/// 布局见 `bins.wgsl`：`b = ((tr * tiles_x + col) * n_draws + draw) * 16 + row_local`。
pub fn dense_grid_shape(tiles_x: u32, tiles_y: u32, n_draws: u32) -> (u64, u64) {
    let n_grid = (tiles_x as u64) * (tiles_y as u64) * (n_draws as u64) * 16;
    let n_bucket = (n_draws as u64) * (tiles_y as u64) * 16 * (tiles_x as u64 + 1);
    (n_grid, n_bucket)
}

#[doc(hidden)] // 仅基准与诊断用，见 docs/API.md「实验面」
/// 稠密路径容量预检：`Ok((n_grid, n_bucket))` 或带具体数字的可读错误。
///
/// # 为什么需要它
///
/// 稠密网格显存 ∝ `tiles × draws`，会**远早于其他限制**撞上
/// `max_storage_buffer_binding_size`（wgpu 默认 128 MiB）。若不预检，失败发生在
/// `create_bind_group` 内部，表现为 wgpu 校验层 panic——调用方拿不到可读原因，
/// 也无法在生产代码里优雅降级到稀疏路径。
///
/// # 两条天花板：**绑定尺寸墙先到**
///
/// - **绑定墙**：`grid_aux`（8 B/元素）与 `bucket_sums`（4 B/桶）各占一个 storage 绑定；
/// - **dispatch 墙**：`clear` 按 `⌈max(n_grid, n_bucket)/64⌉` 个工作组**线性 2D** 派发，
///   上限 `65535²`——极高，实际不是瓶颈。
///
/// 实测（RTX 3050 / Vulkan，默认 limits `max_storage_buffer_binding_size = 128 MiB`）：
/// **720p 约 292 draw、1080p 约 129 draw** 即撞绑定墙（由本函数公式逐 draw 复算）。
/// 注意 `benches/render.rs` 里打印的"稠密网格上限：draw 31/71"是**按单维
/// 65535 工作组**估算的保守值（见 `dense_draw_ceiling`），远小于线性 2D 的真实容量；
/// 两条口径不同，勿混用。
pub fn dense_capacity(
    limits: &wgpu::Limits,
    tiles_x: u32,
    tiles_y: u32,
    n_draws: u32,
) -> Result<(u64, u64), RenderError> {
    let (n_grid, n_bucket) = dense_grid_shape(tiles_x, tiles_y, n_draws);
    // 实际进 bind group 的 storage 缓冲（见 run_pipeline_impl 的分配）
    let grid_aux_bytes = n_grid * 8; // 绑定 4：counters + fill 前缀区
    let bucket_bytes = n_bucket * 4; // 绑定 5
    // 单缓冲上限取两者较小值（wgpu 校验两条都查）
    let binding_cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);

    for (name, bytes) in [
        ("grid_aux（counters+fill 前缀区）", grid_aux_bytes),
        ("bucket_sums", bucket_bytes),
    ] {
        if bytes > binding_cap {
            return err(format!(
                "稠密网格超限（RenderPath::Dense）：{name} 需 {:.1} MB，单缓冲上限 {:.1} MB（max_storage_buffer_binding_size / max_buffer_size 取小）—— 本场景 tiles={tiles_x}x{tiles_y} draws={n_draws}。请改用 RenderPath::Sparse（显存 ∝ 画面内容，无此上限）；稠密路径仅适用于小场景对拍与诊断。",
                bytes as f64 / 1e6,
                binding_cap as f64 / 1e6,
            ));
        }
    }

    // dispatch 墙：clear 的线性 2D 拆分上限为 65535² 个工作组
    let clear_wg = n_grid.max(n_bucket).div_ceil(64);
    if clear_wg > 65_535 * 65_535 {
        return err(format!(
            "稠密网格 clear dispatch 超限：需 {clear_wg} 个工作组，线性 2D 上限 {}（65535²）——请改用 RenderPath::Sparse。",
            65_535u64 * 65_535
        ));
    }
    Ok((n_grid, n_bucket))
}

/// 引擎需要的**设备上限**。自己创建 `wgpu::Device` 的调用方**必须**用它。
///
/// # 为什么需要这个函数（dogfooding 抓到的真问题）
///
/// 引擎的 `amiki_fine_bgl` 有 **10 个 storage buffer**（0.6.1 起；第 10 个是
/// 脏 tile 索引表，在此之前是 9 个、第 9 个是图像放置参数），
/// 而 `wgpu::Limits::default()` 只保证 **8** 个（WebGPU 基线）。
///
/// `GpuRenderer::new()` 会按适配器能力上调这个上限，但**自己建设备的调用方
/// 不会** —— 那样会在 `create_bind_group_layout` 里抛出 wgpu 校验错误：
///
/// ```text
/// In Device::create_bind_group_layout, label = 'amiki_fine_bgl'
///   Too many bindings of type StorageBuffers in Stage COMPUTE, limit is 8, count was 10
/// ```
///
/// 这个错误**深在 wgpu 内部**，调用方几乎无法从消息定位到"我该改 limits"。
/// 窗口示例（`window_demo`）就踩了这个坑，而且因为示例不在测试覆盖里，
/// **一直没被发现** —— 它是靠 dogfooding（真去跑那个应用）才暴露的。
///
/// 所以现在有两道防线：
/// 1. 本函数给出正确的上限，调用方直接用它请求设备；
/// 2. [`GpuRenderer::from_device`] **校验**传入的设备并**显式报错**，
///    不再把问题留给 wgpu 深处的校验失败。
pub fn required_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: REQUIRED_STORAGE_BUFFERS,
        ..wgpu::Limits::default()
    }
}

/// 引擎在单个着色阶段里需要的 storage buffer 数量（见 [`required_limits`]）。
///
/// 0.6.1 起是 **10**：fine 新增了**脏 tile 索引表**（binding 18，`array<u32>`）。
/// 这个数字是**对外契约**的一部分 —— 自行创建 `wgpu::Device` 的调用方必须用
/// [`required_limits`] 请求，否则 `from_device` 会显式报错。
pub const REQUIRED_STORAGE_BUFFERS: u32 = 10;

impl GpuRenderer {
    /// 独立初始化（无窗口）。
    pub async fn new() -> Result<GpuRenderer, RenderError> {
        // ---- 后端选择：**两段式** —— 先平台首选（快），拿不到适配器再全量回退 ----
        //
        // 自研的是**渲染器**，不是图形 API：Vulkan / DX12 / Metal 都是 `wgpu`
        // 在不同机器上落到的东西。所以默认值的第一要求是"哪台机器都能起来"，
        // 而不是最快的那条路；第二段的全量回退就是这条保证 ——
        // 它不把引擎绑到某个后端上（同一份 WGSL 在每个后端都跑通）。
        //
        // 启动开销几乎全在这一段。实测（RTX 3050 / Windows / release / 冷启动）：
        //
        // | 后端集合 | 冷 | 热 |
        // |---|---|---|
        // | `all()`（Vulkan + DX12 + GL） | 337–500 ms | 202 ms |
        // | 仅 Vulkan（本平台首选） | **154 ms** | **53 ms** |
        // | 仅 DX12 | 2272 ms | 2093 ms |
        //
        // 差在**后端枚举本身**：`Instance::new` + 请求适配器占 `all()` 的 86%
        // （其中 DX12 的探测约 170 ms），而全部 21 条管线只占 12.7 ms。
        // DX12 冷启动 2.3 s 是 FXC 在**运行期编译**着色器；DXC 会快得多，
        // 但要在构建期联网取 DLL，与"依赖轻量"冲突，故不采用。
        //
        // 平台首选：Windows / Linux → Vulkan（Windows 上 DX12 探测更贵）；macOS → Metal。
        // 想要别的组合有两条正路：`AMIKI_BACKENDS=vulkan|dx12|gl|all` 显式指定，
        // 或把 `new()`（本来就是 `async`）与窗口初始化**重叠**，别阻塞首帧。
        // （曾把默认值改成"只用 Vulkan"：启动快一倍，但削弱了"一套 WGSL 通吃"
        //  这个既定价值，已按此理由改回。）
        let forced = std::env::var("AMIKI_BACKENDS").ok();
        let override_set = match forced.as_deref() {
            Some("vulkan") => Some(wgpu::Backends::VULKAN),
            Some("dx12") => Some(wgpu::Backends::DX12),
            Some("gl") => Some(wgpu::Backends::GL),
            Some("all") => Some(wgpu::Backends::all()),
            Some(other) => {
                return err(format!(
                    "AMIKI_BACKENDS 取值无法识别：{other}（可用：vulkan / dx12 / gl / all）"
                ));
            }
            None => None,
        };
        let preferred = if cfg!(target_os = "macos") {
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::VULKAN
        };
        let attempts: Vec<wgpu::Backends> = match override_set {
            Some(b) => vec![b],
            // 首选与 all() 相同时不重复试一遍
            None if preferred == wgpu::Backends::all() => vec![wgpu::Backends::all()],
            None => vec![preferred, wgpu::Backends::all()],
        };
        let mut adapter = None;
        for (i, backends) in attempts.iter().enumerate() {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: *backends,
                ..Default::default()
            });
            match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
            {
                Ok(a) => {
                    if i > 0 {
                        eprintln!(
                            "amiki-gpu：首选后端 {preferred:?} 未找到适配器，已回退到 {backends:?} （启动会慢一些；可用 AMIKI_BACKENDS 指定后端）");

                    }
                    adapter = Some(a);
                    break;
                }
                Err(e) => {
                    if i + 1 == attempts.len() {
                        return err(format!(
                            "没有可用的 GPU 适配器（已尝试 {attempts:?}）：{e} —— 可用 AMIKI_BACKENDS 指定后端"));

                    }
                }
            }
        }
        let adapter =
            adapter.ok_or_else(|| RenderError("没有可用的 GPU 适配器".into()))?;
        let info = adapter.get_info();
        // fine 的 storage buffer 数（0.6.1 起 10 个：图像放置参数 + 脏 tile 表）
        // 超过了 wgpu `Limits::default()` 保证的 8 个（WebGPU 基线）。
        // 因此**按适配器能力上调**该上限，并在适配器不支持时明确报错，
        // 而不是让 create_bind_group 在运行时抛出一句难以定位的校验失败。
        let adapter_limits = adapter.limits();
        let want_sb = REQUIRED_STORAGE_BUFFERS;
        if adapter_limits.max_storage_buffers_per_shader_stage < want_sb {
            return err(format!(
                "适配器仅支持 {} 个 storage buffer/阶段，渲染管线需要 {want_sb} 个；请改用 CPU 引擎（amiki_raster）",
                adapter_limits.max_storage_buffers_per_shader_stage
            ));
        }
        let limits = required_limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("amiki2d"),
                // 基准所需：适配器支持时启用 GPU 时间戳（其余保持最小特性集）
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| RenderError(format!("设备创建失败: {e}")))?;
        let mut renderer = GpuRenderer::from_device(device, queue)?;
        renderer.adapter_name = format!("{} ({:?})", info.name, info.backend);
        renderer.backend = Some(info.backend);
        Ok(renderer)
    }

    /// 同步创建渲染器，**把"后端起不来"时的 panic 收敛成 `Err`**。
    ///
    /// # 为什么需要它（来自嵌入式集成反馈 P4，已在本机复现）
    ///
    /// 某些后端在**初始化期会内部 panic**，而不是返回错误：`AMIKI_BACKENDS=gl`
    /// 且进程无窗口时，wgpu 25 在创建 compute pipeline 那一步 panic
    /// （`wgpu_core.rs:1410`）—— GL 上下文没有 compute 能力时它不报错，直接 panic。
    ///
    /// 那个 panic 会一路冒到调用方：C ABI 的兜底捕获把它变成
    /// `AMIKI_ERROR_PANIC`（"引擎有 bug，请上报"），而契约是
    /// `AMIKI_ERROR_BACKEND_UNAVAILABLE`（"这个后端在这台机器上不可用"）。
    /// 对嵌入式宿主来说这两种说法指向完全不同的排查方向。
    ///
    /// # 只包初始化这一段
    ///
    /// 只捕获**创建渲染器**期间的 panic。渲染过程中的 panic 照旧冒出 ——
    /// 那是真的 bug，不该被这条吞掉：既不静默画错，也不把真 bug
    /// 伪装成"后端不可用"。
    pub fn new_blocking() -> Result<GpuRenderer, RenderError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pollster::block_on(Self::new())
        })) {
            Ok(r) => r,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "（panic 未提供信息）".to_string());
                Err(RenderError(format!(
                    "GPU 后端初始化失败，该后端在这台机器上不可用：{msg}"
                )))
            }
        }
    }

    pub fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<GpuRenderer, RenderError> {
        // **先校验设备上限，再动任何资源。** 否则失败会发生在
        // `create_bind_group_layout` 内部，报一句调用方无法定位的
        // wgpu 校验错误（窗口示例正是这样一直跑不起来的）。
        let d_limits = device.limits();
        if d_limits.max_storage_buffers_per_shader_stage < REQUIRED_STORAGE_BUFFERS {
            return err(format!(
                "传入的设备只支持 {} 个 storage buffer/阶段，引擎需要 {} 个。请用 `amiki_gpu::required_limits()` 构造 `DeviceDescriptor::required_limits` （或用 `GpuRenderer::new()` 让它自己请求）；也可改用 CPU 引擎（amiki_raster）",
                d_limits.max_storage_buffers_per_shader_stage, REQUIRED_STORAGE_BUFFERS
            ));
        }
        let fine_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_fine"),
            source: wgpu::ShaderSource::Wgsl(format!("{FINE_MATH_WGSL}\n{FINE_WGSL}").into()),
        });
        let sparse_fine_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_fine"),
            source: wgpu::ShaderSource::Wgsl(format!("{FINE_MATH_WGSL}\n{SPARSE_FINE_WGSL}").into()),
        });
        let scan_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_scan"),
            source: wgpu::ShaderSource::Wgsl(SCAN_WGSL.into()),
        });
        let flatten_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_flatten"),
            source: wgpu::ShaderSource::Wgsl(FLATTEN_WGSL.into()),
        });
        let bins_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_bins"),
            source: wgpu::ShaderSource::Wgsl(BINS_WGSL.into()),
        });
        let strips_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_strips"),
            source: wgpu::ShaderSource::Wgsl(STRIPS_WGSL.into()),
        });
        let sm_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_measure"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_MEASURE_WGSL.into()),
        });
        let st_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_tiles"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_TILES_WGSL.into()),
        });
        let se_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_emit"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_EMIT_WGSL.into()),
        });
        let ss_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_scatter"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_SCATTER_WGSL.into()),
        });
        let sc_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("amiki_sparse_carry"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_CARRY_WGSL.into()),
        });

        // fine：0 uniform，1..=8 storage ro（含 seg_refs），9 storage tex，10..=13 clip 纹理
        let mut fine_entries = vec![uniform_bgl_entry(0)];
        for b in 1..=7u32 {
            fine_entries.push(storage_bgl_entry(b, true));
        }
        fine_entries.push(storage_bgl_entry(8, true));
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                // **必须与着色器里的 `texture_storage_2d<...>` 格式一致** ——
                // 不一致会在创建管线时报 "shader requirements against the
                // pipeline"，而不是静默画错。HDR 要 >1.0 存得住，故为 f16。
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
        for b in 10..=13u32 {
            fine_entries.push(texture_bgl_entry(b));
        }
        // M6 图像：14 = 放置参数（与稠密路径共用，稠密永不用到），
        //          15 = 位图纹理数组，16 = 采样器
        fine_entries.push(storage_bgl_entry(14, true));
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 15,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        });
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 16,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        // B4：平铺（repeat）专用的 Repeat 地址模式采样器。原来只靠 WGSL 里
        // 的 `uv - floor(uv)` 预取模 + ClampToEdge 采样器，接缝的半纹素处
        // （uv 极接近 0/1 时跨层的 +1 取纹素）会被钳到边缘，而 CPU 是按
        // rem_euclid 真回绕 —— 双端在接缝带不一致（潜伏 bug，mipmap 对拍
        // 首次暴露）。真回绕必须发生在**采样器**上，每个 mip 层的取纹素
        // 才都会回绕。
        fine_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 17,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        // 0.6.1：**脏 tile 索引表**（1D 派发）。每项 `tx | ty << 16`，
        // 内核按下标查表得到自己要重算的 tile —— 这样派发就不需要起点偏移
        // （wgpu 没有），也不会再出现"起点被丢弃 → 重算错 tile"的静默残影。
        // 全量渲染时这张表就是整幅画布的 tile（row-major），因此只有一条代码路径。
        fine_entries.push(storage_bgl_entry(18, true));
        let fine_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_fine_bgl"),
            entries: &fine_entries,
        });

        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_scan_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, false),
                storage_bgl_entry(2, false),
                storage_bgl_entry(3, false),
            ],
        });

        // flatten：0 uniform，1 data ro，2 entry_draws ro，3 counts ro，4 seg_base ro，
        //          5 seg_ends rw，6 seg_draws rw，7 indirect rw
        let flatten_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_flatten_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, true),
                storage_bgl_entry(4, true),
                storage_bgl_entry(5, false),
                storage_bgl_entry(6, false),
                storage_bgl_entry(7, false),
            ],
        });

        // bins：0 uniform，1 seg_ends ro，2 seg_draws ro，3 counters rw，
        //       4 grid_aux rw，5 bucket_sums rw，6 carries rw，7 item_total rw，8 seg_refs rw
        // （binding 3 空缺：原 seg_total 已由 CPU 闭式段数取代）
        let bins_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_bins_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, false),
                storage_bgl_entry(4, false),
                storage_bgl_entry(5, false),
                storage_bgl_entry(6, false),
                storage_bgl_entry(7, false),
                storage_bgl_entry(8, false),
            ],
        });

        // strips（M3a）：0 uniform（复用 BinsParams 布局），1 seg_ends ro，2 seg_draws ro，
        //                3 活跃 strip 交错元数据 rw（4×u32/strip）
        let strip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_strip_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, false),
            ],
        });

        // M3b 稀疏管线：单一通用布局（uniform + 8×rw storage），各 kernel 自行填槽
        let sparse_entries: Vec<wgpu::BindGroupLayoutEntry> = std::iter::once(uniform_bgl_entry(0))
            .chain((1..=8u32).map(|b| storage_bgl_entry(b, false)))
            .collect();
        let sparse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("amiki_sparse_bgl"),
            entries: &sparse_entries,
        });

        let make_layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            })
        };
        let fine_pl = make_layout("amiki_fine_pl", &fine_bgl);
        let scan_pl = make_layout("amiki_scan_pl", &scan_bgl);
        let flatten_pl = make_layout("amiki_flatten_pl", &flatten_bgl);
        let bins_pl = make_layout("amiki_bins_pl", &bins_bgl);
        let strip_pl = make_layout("amiki_strip_pl", &strip_bgl);
        let sparse_pl = make_layout("amiki_sparse_pl", &sparse_bgl);

        let mk_compute = |label: &'static str, pl: &wgpu::PipelineLayout, module, entry: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fine_pipeline = mk_compute("amiki_fine_pipeline", &fine_pl, &fine_mod, "main");
        let coverage_pipeline = mk_compute("amiki_coverage_pipeline", &fine_pl, &fine_mod, "main_coverage");
        let scan_blocks_pipeline = mk_compute("amiki_scan_blocks", &scan_pl, &scan_mod, "scan_blocks");
        let scan_sums_pipeline = mk_compute("amiki_scan_sums", &scan_pl, &scan_mod, "scan_sums");
        let scan_add_pipeline = mk_compute("amiki_scan_add", &scan_pl, &scan_mod, "add_offsets");
        let flatten_pipeline = mk_compute("amiki_flatten_pipeline", &flatten_pl, &flatten_mod, "flatten");
        let bins_clear_pipeline = mk_compute("amiki_bins_clear", &bins_pl, &bins_mod, "clear");
        let bins_emit_pipeline = mk_compute("amiki_bins_emit", &bins_pl, &bins_mod, "emit");
        let bins_scatter_pipeline = mk_compute("amiki_bins_scatter", &bins_pl, &bins_mod, "scatter");
        let bins_carry_pipeline = mk_compute("amiki_bins_carry", &bins_pl, &bins_mod, "carry");
        let strip_clear_pipeline = mk_compute("amiki_strip_clear", &strip_pl, &strips_mod, "strip_clear");
        let strip_extract_pipeline = mk_compute("amiki_strip_extract", &strip_pl, &strips_mod, "strip_extract");
        // M3b 稀疏管线（统一布局）
        let sparse_measure_pipeline = mk_compute("amiki_sparse_measure", &sparse_pl, &sm_mod, "measure");
        let sparse_tr_scatter_pipeline = mk_compute("amiki_sparse_tr_scatter", &sparse_pl, &sm_mod, "tr_scatter");
        let sparse_tr_offsets_pipeline = mk_compute("amiki_sparse_tr_offsets", &sparse_pl, &sm_mod, "tr_offsets_fn");
        let sparse_tile_count_pipeline = mk_compute("amiki_sparse_tile_count", &sparse_pl, &st_mod, "tile_count");
        let sparse_tile_fill_pipeline = mk_compute("amiki_sparse_tile_fill", &sparse_pl, &st_mod, "tile_fill");
        let sparse_clear_pipeline = mk_compute("amiki_sparse_clear", &sparse_pl, &se_mod, "clear_sparse");
        let sparse_emit_pipeline = mk_compute("amiki_sparse_emit", &sparse_pl, &se_mod, "emit");
        let sparse_scatter_pipeline = mk_compute("amiki_sparse_scatter", &sparse_pl, &ss_mod, "scatter");
        let sparse_carry_pipeline = mk_compute("amiki_sparse_carry", &sparse_pl, &sc_mod, "carry");
        // 稀疏 fine（与稠密 fine 共用 fine_bgl/fine_pl）
        let sparse_fine_pipeline = mk_compute("amiki_sparse_fine_pipeline", &fine_pl, &sparse_fine_mod, "sparse_main");
        let sparse_coverage_pipeline = mk_compute("amiki_sparse_coverage_pipeline", &fine_pl, &sparse_fine_mod, "sparse_coverage");

        let dummy_clip_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("amiki_dummy_clip"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let dummy_clip_view = dummy_clip_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // M6：图像绑定的占位（稠密路径与本引擎的 clip-only 绑定需要合法资源）。
        // 1×1 透明层，作为 D2Array 视图 + clamp 采样器。
        let dummy_img_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("amiki_dummy_image"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[0, 0, 0, 0],
        );
        let dummy_img_view = dummy_img_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("amiki_dummy_image_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let dummy_img_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("amiki_dummy_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // M6：稠密路径的放置参数占位（layout 共用，需合法非空缓冲）
        let dummy_img_place = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_dummy_imgplace"),
            size: 3 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&dummy_img_place, 0, bytemuck::cast_slice(&[[0.0f32; 4]; 3]));

        // M6：模糊管线（离屏层滤镜）
        let blur = blur::create_resources(&device);
        // P0-3：离屏层的 GPU 合成管线
        let layer_composite = layer::create_resources(&device);

        let timestamp_period_ns = queue.get_timestamp_period();
        let limits = device.limits();
        Ok(GpuRenderer {
            path: RenderPath::default(),
            adapter_name: "unknown".to_string(),
            backend: None,
            has_timestamps: device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
                && timestamp_period_ns > 0.0,
            timestamp_period_ns,
            limits,
            readback_pool: Mutex::new(Vec::new()),
            sparse_capacity: Mutex::new(SparseCapacityTable::default()),
            device,
            queue,
            fine_pipeline,
            coverage_pipeline,
            fine_bgl,
            scan_blocks_pipeline,
            scan_sums_pipeline,
            scan_add_pipeline,
            scan_bgl,
            flatten_pipeline,
            flatten_bgl,
            bins_clear_pipeline,
            bins_emit_pipeline,
            bins_scatter_pipeline,
            bins_carry_pipeline,
            bins_bgl,
            strip_clear_pipeline,
            strip_extract_pipeline,
            strip_bgl,
            sparse_bgl,
            sparse_measure_pipeline,
            sparse_tr_scatter_pipeline,
            sparse_tr_offsets_pipeline,
            sparse_tile_count_pipeline,
            sparse_tile_fill_pipeline,
            sparse_clear_pipeline,
            sparse_emit_pipeline,
            sparse_scatter_pipeline,
            sparse_carry_pipeline,
            sparse_fine_pipeline,
            sparse_coverage_pipeline,
            dummy_clip_view,
            _dummy_clip_tex: dummy_clip_tex,
            dummy_img_view,
            _dummy_img_tex: dummy_img_tex,
            dummy_img_sampler,
            dummy_img_place,
            blur,
            layer_composite,
        })
    }

    /// 通用 u32 exclusive 前缀和：data 原位变为 exclusive 前缀。
    fn run_scan(
        &self,
        enc: &mut wgpu::CommandEncoder,
        data_buf: &wgpu::Buffer,
        n: u32,
        tw: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let n_blocks = n.div_ceil(256).max(1);
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_scan_params"),
            contents: bytemuck::bytes_of(&ScanParams { n, _pad0: 0, _pad1: 0, _pad2: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let block_sums = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_scan_sums"),
            size: (n_blocks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let block_offsets = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_scan_offsets"),
            size: (n_blocks as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_scan_bind"),
            layout: &self.scan_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: block_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: block_offsets.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("amiki_scan_pass"),
            timestamp_writes: tw,
        });
        pass.set_pipeline(&self.scan_blocks_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(n_blocks, 1, 1);
        pass.set_pipeline(&self.scan_sums_pipeline);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.scan_add_pipeline);
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
    }

    /// M2f 核心：全 GPU 管线（压平 + 分箱 + fine 光栅化）。
    /// 诊断/高级用途：直接运行管线（常规渲染走 render/render_into）。
    pub fn run_pipeline(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        dump: Option<&mut PipelineDump>,
    ) -> Result<(), RenderError> {
        match self.path {
            RenderPath::Dense => {
                let mut prof = FrameProfile::default();
                self.run_pipeline_impl(gs, tex, coverage, clip_views, None, dump, None, &mut prof)
            }
            RenderPath::Sparse => {
                let mut prof = FrameProfile::default();
                self.run_pipeline_sparse(gs, tex, coverage, clip_views, None, dump, None, &mut prof)
            }
        }
    }

    /// M3c：渲染一帧并报告稀疏分箱的内容规模（显存与内容成比例的直接量化）。
    /// 与 `render` 走同一条管线，只多一次 32B 读回；稠密路径或空画布返回 `None`。
    pub fn sparse_stats(&self, scene: &Scene, eps: f32) -> Result<Option<SparseStats>, RenderError> {
        if self.path != RenderPath::Sparse {
            return Ok(None);
        }
        let frame = self.prepare(scene, eps)?;
        if frame.scene.colors.is_empty() {
            return Ok(None);
        }
        let tex = self.create_output_texture(&frame.scene);
        let mut dump = PipelineDump { sparse_stats: Some([0; 8]), ..Default::default() };
        let mut prof = FrameProfile::default();
        self.run_pipeline_sparse(
            &frame.scene,
            &tex,
            false,
            &frame.clip_views,
            None,
            Some(&mut dump),
            None,
            &mut prof,
        )?;
        let s = dump.sparse_stats.unwrap_or([0; 8]);
        Ok(Some(SparseStats {
            n_cells: s[0],
            n_buckets: s[1],
            n_tile_entries: s[2],
            n_items: s[3],
            n_strips: s[4],
        }))
    }

    /// M3.6：取某场景的投机容量。`None` 表示该场景应走**精确测量路径**
    /// （从未见过，或已因反复溢出放弃投机）。
    ///
    /// **运行时报告**：测试与诊断据此断言"这一帧究竟走没走投机路径"，
    /// 而不是靠注释描述。
    pub fn sparse_capacity(&self, key: SceneKey) -> Option<SparseCapacity> {
        self.sparse_capacity.lock().ok().and_then(|mut t| t.get(key))
    }

    /// 累计溢出次数（只增）。测试用它断言"溢出确实发生并被处理"。
    pub fn capacity_overflow_total(&self) -> u64 {
        self.sparse_capacity
            .lock()
            .map(|t| t.overflow_total())
            .unwrap_or(0)
    }

    /// 诊断：容量表的项数（长跑里不该无限增长）。
    pub fn capacity_table_len(&self) -> usize {
        self.sparse_capacity.lock().map(|t| t.len()).unwrap_or(0)
    }

    /// 诊断：某场景的容量是否因反复溢出而被放弃投机。
    pub fn capacity_gave_up(&self, key: SceneKey) -> bool {
        self.sparse_capacity.lock().is_ok_and(|t| t.gave_up(key))
    }

    /// 诊断：列出容量表的全部表项（key 与容量）。
    ///
    /// 测试用 —— 避免在测试里硬编码场景特征（例如几何桶的具体取值），
    /// 那些属于实现的内部细节。
    pub fn capacity_entries(&self) -> Vec<(SceneKey, SparseCapacity)> {
        self.sparse_capacity
            .lock()
            .map(|t| {
                t.entries
                    .iter()
                    .map(|(k, e)| (*k, e.cap))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// M3.6：记录该场景的实测用量（`overflowed` 表示这一帧因容量不足作废）。
    ///
    /// 锁中毒时静默放弃 —— 容量只是加速手段，丢了只会退回精确测量路径
    /// （更慢，但结果不变）。
    fn store_sparse_capacity(&self, key: SceneKey, usage: SparseCapacity, overflowed: bool) {
        if let Ok(mut t) = self.sparse_capacity.lock() {
            t.record(key, usage, overflowed);
        }
    }

    /// **完整分析**：渲染一帧并同时返回 ①主机侧分阶段耗时（[`FrameProfile`]）
    /// 与 ②逐 compute pass 的 GPU 时长（[`FrameProfile::timestamps`]）。
    ///
    /// 适配器不支持 `TIMESTAMP_QUERY` 时 `timestamps.supported = false`，
    /// 但**主机侧计时仍然有效**——这正是必须两者都给的原因：GPU 计时不可用时，
    /// 至少还能回答"CPU 花在哪"。
    ///
    /// 注意：本方法会为时间戳额外提交一次 resolve + 读回，故其结果**略高于**
    /// 常规渲染路径的真实耗时；做墙钟基准请用 `render_into` + `poll`。
    pub fn profile(&self, scene: &Scene, eps: f32) -> Result<FrameProfile, RenderError> {
        // 每次新建 `Frame` —— 于是 `extract_ms` 里含**每帧全新分配**的代价。
        // 逐帧复用场景的真实提取成本请用 [`Self::profile_into`]。
        let mut frame = Frame::default();
        self.profile_into(scene, eps, &mut frame)
    }

    /// 同 [`Self::profile`]，但**复用调用方持有的 `Frame`**。
    ///
    /// 两者的 `extract_ms` 会明显不同：全新分配约 1.0 ms，复用约 0.56 ms
    /// （实测，见 `GpuScene::reset_for_reuse`）。逐帧路径用的是复用那一档，
    /// 所以**评估提取成本应当看这个方法**。
    pub fn profile_into(
        &self,
        scene: &Scene,
        eps: f32,
        frame: &mut Frame,
    ) -> Result<FrameProfile, RenderError> {
        let mut prof = FrameProfile {
            timestamps: PassTimings {
                supported: false,
                timestamp_period_ns: self.timestamp_period_ns,
                passes: Vec::new(),
            },
            ..Default::default()
        };
        let t_extract = Instant::now();
        self.prepare_into(scene, eps, frame)?;
        prof.extract_ms = t_extract.elapsed().as_secs_f64() * 1000.0;
        let tex = self.create_output_texture(&frame.scene);
        if !self.has_timestamps {
            // GPU 计时不可用：仍跑一次管线，让主机侧数字有意义
            match self.path {
                RenderPath::Dense => {
                    self.run_pipeline_impl(
                        &frame.scene, &tex, false, &frame.clip_views, None, None, None, &mut prof,
                    )?;
                }
                RenderPath::Sparse => {
                    self.run_pipeline_sparse(
                        &frame.scene, &tex, false, &frame.clip_views, None, None, None, &mut prof,
                    )?;
                }
            }
            return Ok(prof);
        }
        let mut rec = TimestampRecorder::new(&self.device);
        match self.path {
            RenderPath::Dense => {
                self.run_pipeline_impl(
                    &frame.scene,
                    &tex,
                    false,
                    &frame.clip_views,
                    None,
                    None,
                    Some(&mut rec),
                    &mut prof,
                )?;
            }
            RenderPath::Sparse => {
                self.run_pipeline_sparse(
                    &frame.scene,
                    &tex,
                    false,
                    &frame.clip_views,
                    None,
                    None,
                    Some(&mut rec),
                    &mut prof,
                )?;
            }
        }

        // resolve 全部时间戳 → 回读 → tick 换算为 ns
        let t_ts = Instant::now();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_ts_resolve") });
        enc.resolve_query_set(&rec.query_set, 0..rec.query_count(), &rec.resolve_buf, 0);
        self.queue.submit(Some(enc.finish()));
        let bytes = self.read_buffer(&rec.resolve_buf, rec.query_count() as usize * 8)?;
        let raw: &[u64] = bytemuck::cast_slice(&bytes);
        let period = self.timestamp_period_ns as f64;
        for (i, label) in rec.labels.iter().enumerate() {
            let (b, e) = (raw[2 * i], raw[2 * i + 1]);
            let ns = e.saturating_sub(b) as f64 * period;
            prof.timestamps.passes.push((label, ns));
        }
        prof.timestamps.supported = true;
        prof.resolve_ts_ms = Some(t_ts.elapsed().as_secs_f64() * 1000.0);
        Ok(prof)
    }

    /// 场景尺寸对应的输出纹理（storage + 读回）。
    fn create_output_texture(&self, gs: &GpuScene) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("amiki_out"),
            size: wgpu::Extent3d {
                width: gs.width.max(1),
                height: gs.height.max(1),
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
        })
    }

    /// M3b：稀疏管线 —— strip 归并 → tile 行列表 → 逐 tile 桶表 → 稀疏单元 → fine。
    ///
    /// 显存只与画面实际内容成比例：空画布下除 (draw × tile 行) 的条带元数据外
    /// 不分配任何缓冲。每帧 3 次 submit、2 次小读回（32B + 32B）：
    /// ```text
    /// enc1a: clear strips → flatten+strip_extract → measure → 3×scan →
    ///        tr_scatter → tr_offsets → tile_count → copy+scan(tile_start)  〖读回 stats〗
    /// enc1b: clear_sparse → emit（成员计数 + carry 定点桶）                 〖读回 stats〗
    /// enc2:  copy+scan(cell CSR) → tile_fill → scatter → carry → fine
    /// ```
    /// M3.6：稀疏帧路径的**投机分配外壳**。
    ///
    /// 内部 [`Self::run_pipeline_sparse_once`] 在容量不足时会报告溢出
    /// （返回 `true`）——此时那一帧的画面是**残缺的，必须重做**，绝不能交出去。
    ///
    /// 重做方式：清空容量缓存（`_once` 见容量无效即走精确测量路径），再跑一次。
    /// 测量路径按实测值分配，因此第二次必然成功，所以这里只重试一次。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pipeline_sparse(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        damage: Option<&amiki_scene::Damage>,
        dump: Option<&mut PipelineDump>,
        ts: Option<&mut TimestampRecorder>,
        prof: &mut FrameProfile,
    ) -> Result<(), RenderError> {
        let overflow = self.run_pipeline_sparse_once(
            gs, tex, coverage, clip_views, damage, dump, ts, prof,
        )?;
        if !overflow {
            return Ok(());
        }

        // 这一帧的产出不可用（有写入被丢弃）→ **必须重做**。
        //
        // 不必清空容量：`_once` 在返回溢出前已把实测用量记进该场景的表项
        // （并给该 key 的溢出计数 +1），而实测用量正是本帧需要的格数，
        // 故第二次必然容得下。若该 key 反复溢出，表会放弃它的投机并
        // 永久退回测量路径 —— 最坏情况也只是回到没有投机时的性能。
        //
        // 诊断与时间戳只在第一次尝试时收集（第二次为的是正确性，不是测量）。
        prof.capacity_overflows += 1;
        self.run_pipeline_sparse_once(gs, tex, coverage, clip_views, damage, None, None, prof)?;
        Ok(())
    }

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



#[cfg(test)]
mod tests;
