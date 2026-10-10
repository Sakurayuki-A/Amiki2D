//! 执行：scan / 稠密管线 / 稀疏管线与输出纹理。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

impl GpuRenderer {
    /// 通用 u32 exclusive 前缀和：data 原位变为 exclusive 前缀。
    pub(crate) fn run_scan(
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

}
impl GpuRenderer {
    /// 场景尺寸对应的输出纹理（storage + 读回）。
    pub(crate) fn create_output_texture(&self, gs: &GpuScene) -> wgpu::Texture {
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

}
