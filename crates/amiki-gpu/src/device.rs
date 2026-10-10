//! 设备与管线：adapter/device 选择、绑定组布局、着色器模块与 21 条管线、纹理与采样器。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

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
pub(crate) fn pack_gradients(grads: &[std::sync::Arc<amiki_scene::Gradient>]) -> Vec<[f32; 4]> {
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
    pub(crate) fn mk_image_texture_array(
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
pub(crate) fn pack_image_place(gs: &GpuScene) -> Vec<[f32; 4]> {
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

}
