//! 稠密网格路径的 GPU 实现（`run_pipeline_impl`）。
//!
//! 从 `lib.rs` 拆出。M3b 起**主线走稀疏桶表**（见 `sparse.rs`），
//! 本文件是稠密网格入口 —— 它保留下来有明确用途：
//!
//! - **对拍基准**：两套分箱在同一场景上必须给出逐字节相同的画面
//!   （`run_pipeline` 会在两条路径间切换，测试同时覆盖）；
//! - **诊断路径**：稀疏结构出问题时，用它判断问题在分箱还是在着色。

use crate::*;

impl GpuRenderer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pipeline_impl(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        damage: Option<&amiki_scene::Damage>,
        mut dump: Option<&mut PipelineDump>,
        mut ts: Option<&mut TimestampRecorder>,
        prof: &mut FrameProfile,
    ) -> Result<(), RenderError> {
        let w = tex.width();
        let h = tex.height();
        if w != gs.width || h != gs.height {
            return err("纹理尺寸与场景尺寸不一致");
        }
        let total_entries = gs.counts.len() as u32;
        let n_draws = gs.colors.len() as u32;
        if n_draws == 0 || total_entries == 0 || gs.total_segments == 0 {
            return Ok(()); // 空场景：纹理保持零初始化（透明）
        }
        // 容量预检：超限时返回可读错误，而非让 wgpu 校验层在 create_bind_group 处 panic
        let (n_grid_u64, n_bucket_u64) =
            dense_capacity(&self.limits, gs.tiles_x, gs.tiles_y, n_draws)?;
        let n_grid = n_grid_u64 as usize;
        let n_bucket = n_bucket_u64 as usize;

        // 时间戳槽位（仅分析路径分配；常规路径零开销）
        let ts_slots: Option<[(u32, u32); 5]> = match ts.as_mut() {
            Some(r) => Some([
                r.slot("clear")?,
                r.slot("scan:seg_base")?,
                r.slot("flatten+strips+emit")?,
                r.slot("scan:bucket")?,
                r.slot("scatter+carry+fine")?,
            ]),
            None => None,
        };
        let slot = |i: usize| ts_slots.map(|s| s[i]);

        let mk_storage = |label: &str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mk_init = |label: &str, data: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let pad_ro = mk_init("amiki_pad_ro", &PAD16);
        let pad_rw1 = mk_init("amiki_pad_rw1", &PAD16);
        let pad_rw2 = mk_init("amiki_pad_rw2", &PAD16);
        let pad_rw3 = mk_init("amiki_pad_rw3", &PAD16);

        // 上传：合并数据布局（quad 2×vec4 | cubic 2×vec4 | line 1×vec4）
        let mut data: Vec<[f32; 4]> = Vec::with_capacity((nq_len(gs) + nc_len(gs)) * 2 + nl_len(gs));
        for q in &gs.quads {
            data.push(q[0]);
            data.push(q[1]);
        }
        for c in &gs.cubics {
            data.push(c[0]);
            data.push(c[1]);
        }
        for l in &gs.lines {
            data.push(*l);
        }
        let mut entry_draws: Vec<u32> = Vec::with_capacity(data.len());
        entry_draws.extend_from_slice(&gs.quad_draws);
        entry_draws.extend_from_slice(&gs.cubic_draws);
        entry_draws.extend_from_slice(&gs.line_draws);

        let data_buf = mk_init("amiki_data", bytemuck::cast_slice(&data));
        let entry_draws_buf = mk_init("amiki_edraws", bytemuck::cast_slice(&entry_draws));
        let colors_buf = mk_init("amiki_colors", bytemuck::cast_slice(&gs.colors));
        // 稠密路径的 WGSL 用 `array<u32>` 的 flags 与 `array<vec4<u32>>` 的 clips，
        // 而宿主的 `GpuScene` 现在只存合并后的 `draw_meta`（见 `extract.rs`）。
        // **稠密仅用于对拍基准与诊断**，不在热路径上，所以在这里拆开即可。
        let draw_flags: Vec<u32> = gs.draw_meta.iter().map(|m| m[0]).collect();
        let draw_clips: Vec<[u32; 4]> = gs
            .draw_meta
            .iter()
            .map(|m| {
                let (a, b) = (m[1], m[2]);
                [a & 0xFF, (a >> 8) & 0xFF, b & 0xFF, (b >> 8) & 0xFF]
            })
            .collect();
        let flags_buf = mk_init("amiki_flags", bytemuck::cast_slice(&draw_flags));
        let clips_buf = mk_init("amiki_clips", bytemuck::cast_slice(&draw_clips));
        let counts_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_counts"),
            contents: bytemuck::cast_slice(&gs.counts),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let seg_base_buf = mk_storage("amiki_seg_base", (total_entries as u64) * 4);
        let seg_ends_buf = mk_storage("amiki_seg_ends", (gs.total_segments as u64) * 16);
        let seg_draws_buf = mk_storage("amiki_seg_draws", (gs.total_segments as u64) * 4);
        let indirect_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_indirect"),
            contents: bytemuck::bytes_of(&gs.indirect),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });
        let counters_buf = mk_storage("amiki_counters", (n_grid as u64) * 4);
        let grid_aux_buf = mk_storage("amiki_grid_aux", (n_grid as u64) * 8); // [0,n)=base  [n,2n)=fill
        let bucket_buf = mk_storage("amiki_bucket", (n_bucket as u64) * 4);
        let carries_buf = mk_storage("amiki_carries", (n_grid as u64) * 4);
        let item_total_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_item_total"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        // M3a 活跃 tile 元数据：每 (draw, tile 行) 4×u32（lo, hi, y0, y1），转置稠密
        let n_strips = (n_draws as u64) * (gs.tiles_y as u64);
        let strip_meta_buf = mk_storage("amiki_strip_meta", n_strips * 16);

        let flatten_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_flatten_params"),
            contents: bytemuck::bytes_of(&FlattenParams {
                n_quads: nq_len(gs) as u32,
                n_cubics: nc_len(gs) as u32,
                n_lines: nl_len(gs) as u32,
                eps: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bins_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_bins_params"),
            contents: bytemuck::bytes_of(&BinsParams {
                tiles_x: gs.tiles_x,
                n_draws,
                tiles_y: gs.tiles_y,
                height: gs.height,
                total_segments: gs.total_segments,
                n_cells: 0,
                n_buckets: 0,
                n_tile_entries: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let fine_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_fine_params"),
            contents: bytemuck::bytes_of(&FineParams {
                width: gs.width,
                height: gs.height,
                tiles_x: gs.tiles_x,
                n_draws,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let flatten_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_flatten_bind"),
            layout: &self.flatten_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: flatten_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: entry_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counts_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: seg_base_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: indirect_buf.as_entire_binding() },
            ],
        });
        let bins_bind_clear = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_bins_bind_clear"),
            layout: &self.bins_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: pad_ro.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: pad_ro.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bucket_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pad_rw2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pad_rw3.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: pad_rw1.as_entire_binding() },
            ],
        });
        let bins_bind_emit = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_bins_bind_emit"),
            layout: &self.bins_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bucket_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pad_rw1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: item_total_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: pad_rw2.as_entire_binding() },
            ],
        });
        let strip_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_strip_bind"),
            layout: &self.strip_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: strip_meta_buf.as_entire_binding() },
            ],
        });

        // —— enc1：pass(clear) → copy(counts→seg_base) → pass(scan) → pass(flatten+strips → emit) ——
        let mut enc1 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_enc1") });
        {
            let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_pass_clear"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(0)),
            });
            pass.set_pipeline(&self.bins_clear_pipeline);
            pass.set_bind_group(0, &bins_bind_clear, &[]);
            pass.dispatch_workgroups(n_grid.max(n_bucket).div_ceil(64) as u32, 1, 1);
            pass.set_pipeline(&self.strip_clear_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            pass.dispatch_workgroups(n_strips.div_ceil(64) as u32, 1, 1);
        }
        enc1.copy_buffer_to_buffer(&counts_buf, 0, &seg_base_buf, 0, (total_entries as u64) * 4);
        self.run_scan(&mut enc1, &seg_base_buf, total_entries, ts_writes(ts.as_deref(), slot(1)));
        {
            let mut pass = enc1.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_pass_flatten_emit"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(2)),
            });
            pass.set_pipeline(&self.flatten_pipeline);
            pass.set_bind_group(0, &flatten_bind, &[]);
            pass.dispatch_workgroups(total_entries.div_ceil(64), 1, 1);
            // M3a：活跃 tile 提取（读 flatten 刚写出的 seg_ends，同 pass 内隐式屏障）
            pass.set_pipeline(&self.strip_extract_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            pass.dispatch_workgroups(gs.total_segments.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.bins_emit_pipeline);
            pass.set_bind_group(0, &bins_bind_emit, &[]);
            pass.dispatch_workgroups_indirect(&indirect_buf, 0);
        }
        self.queue.submit(Some(enc1.finish()));

        // 诊断转储（enc1 后：压平与计数结果）
        if let Some(d) = dump.as_deref_mut() {
            if d.seg_ends.is_some() {
                let b = self.read_buffer(&seg_ends_buf, (gs.total_segments as usize) * 16)?;
                d.seg_ends = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.seg_draws.is_some() {
                let b = self.read_buffer(&seg_draws_buf, (gs.total_segments as usize) * 4)?;
                d.seg_draws = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.counters.is_some() {
                let b = self.read_buffer(&counters_buf, n_grid * 4)?;
                d.counters = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.item_total.is_some() {
                let b = self.read_buffer(&item_total_buf, 4)?;
                d.item_total = Some(u32::from_le_bytes(b.try_into().unwrap()));
            }
            if d.bucket_sums.is_some() {
                let b = self.read_buffer(&bucket_buf, n_bucket * 4)?;
                d.bucket_sums = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.strip_meta.is_some() {
                let b = self.read_buffer(&strip_meta_buf, n_strips as usize * 16)?;
                d.strip_meta = Some(bytemuck::cast_slice(&b).to_vec());
            }
        }

        // 读回成员项总数（4B）
        let t_rbd = Instant::now();
        let item_bytes = self.read_buffer(&item_total_buf, 4)?;
        prof.readback_dense_ms = Some(t_rbd.elapsed().as_secs_f64() * 1000.0);
        let items = u32::from_le_bytes(item_bytes.try_into().unwrap()) as usize;
        if items == 0 {
            return Ok(());
        }
        let seg_refs_buf = mk_storage("amiki_seg_refs", (items as u64) * 4);

        // —— enc2：copy → scan(桶前缀) → scatter → carry → fine ——
        let mut enc2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_enc2") });
        enc2.copy_buffer_to_buffer(&counters_buf, 0, &grid_aux_buf, 0, (n_grid as u64) * 4);
        self.run_scan(&mut enc2, &grid_aux_buf, n_grid as u32, ts_writes(ts.as_deref(), slot(3)));
        let bins_bind_scatter = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_bins_bind_scatter"),
            layout: &self.bins_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bins_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: bucket_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: carries_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pad_ro.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: seg_refs_buf.as_entire_binding() },
            ],
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut views: Vec<&wgpu::TextureView> = (0..4).map(|_| &self.dummy_clip_view).collect();
        for (i, v) in clip_views.iter().take(4).enumerate() {
            views[i] = v;
        }
        let fine_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_fine_bind"),
            layout: &self.fine_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fine_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grid_aux_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counters_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: carries_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: colors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: clips_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: seg_refs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&tex_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(views[0]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(views[1]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(views[2]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(views[3]) },
                // M6：稠密路径不做图像着色，但 bind group layout 是共用的，
                // 必须提供合法的 14/15/16（占位资源），否则创建绑定即校验失败。
                wgpu::BindGroupEntry { binding: 14, resource: self.dummy_img_place.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&self.dummy_img_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Sampler(&self.dummy_img_sampler),
                },
            ],
        });
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_pass_scatter_carry_fine"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(4)),
            });
            pass.set_pipeline(&self.bins_scatter_pipeline);
            pass.set_bind_group(0, &bins_bind_scatter, &[]);
            pass.dispatch_workgroups_indirect(&indirect_buf, 0);
            pass.set_pipeline(&self.bins_carry_pipeline);
            pass.set_bind_group(0, &bins_bind_scatter, &[]);
            pass.dispatch_workgroups((n_draws * gs.tiles_y * 16).div_ceil(64), 1, 1);
            if coverage {
                pass.set_pipeline(&self.coverage_pipeline);
            } else {
                pass.set_pipeline(&self.fine_pipeline);
            }
            pass.set_bind_group(0, &fine_bind, &[]);
            // M4：只派发脏 tile 子矩形。fine 内核用 global_invocation_id 推导
            // 像素与 tile 坐标，因此 workgroup 数减少 = 只重写那些 tile 的像素，
            // 未派发区域的纹理内容原样保留（这正是复用上一帧的机制）。
            let (_fx, fw, _fy, fh) = fine_dispatch_range(damage, gs.tiles_x, gs.tiles_y);
            pass.dispatch_workgroups(fw, fh, 1);
        }
        self.queue.submit(Some(enc2.finish()));

        // 诊断转储（enc2 后：scan_base / carries / seg_refs 终值）
        // dump 在此按值取出（Option<&mut _> 的 move）；此后不再使用，故无需 as_deref_mut
        // 再借用，也无需 #[allow]——两处转储点都按值取出即可。
        if let Some(d) = dump {
            if d.scan_base.is_some() {
                let b = self.read_buffer(&grid_aux_buf, n_grid * 4)?;
                d.scan_base = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.carries.is_some() {
                let b = self.read_buffer(&carries_buf, n_grid * 4)?;
                d.carries = Some(bytemuck::cast_slice(&b).to_vec());
            }
            if d.seg_refs.is_some() {
                let items = d.item_total.unwrap_or(0) as usize;
                if items > 0 {
                    let b = self.read_buffer(&seg_refs_buf, items * 4)?;
                    d.seg_refs = Some(bytemuck::cast_slice(&b).to_vec());
                }
            }
        }
        Ok(())
    }

}
