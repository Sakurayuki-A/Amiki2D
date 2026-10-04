//! 稀疏分箱路径的 GPU 实现。
//!
//! 从 `lib.rs` 拆出：这是**主线渲染路径**（M3b 起稀疏桶表取代稠密网格），
//! 也是整个 crate 里最长的一段（约 600 行）。它同时是显存与画面内容成比例的
//! 直接来源 —— 单独成文件后，"稀疏到底怎么走"有了明确的阅读入口。
//!
//! 稠密入口（`run_pipeline_impl`）保留在 `lib.rs`，作为对拍基准与诊断路径。

use crate::*;

impl GpuRenderer {
    /// M3.6：稀疏帧路径**主体**。返回值 `true` 表示本帧容量不足（已溢出），
    /// 调用方须作废本帧重做 —— 见 [`Self::run_pipeline_sparse`]。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_pipeline_sparse_once(
        &self,
        gs: &GpuScene,
        tex: &wgpu::Texture,
        coverage: bool,
        clip_views: &[wgpu::TextureView],
        damage: Option<&amiki_scene::Damage>,
        mut dump: Option<&mut PipelineDump>,
        mut ts: Option<&mut TimestampRecorder>,
        prof: &mut FrameProfile,
    ) -> Result<bool, RenderError> {
        let w = tex.width();
        let h = tex.height();
        if w != gs.width || h != gs.height {
            return err("纹理尺寸与场景尺寸不一致");
        }
        let total_entries = gs.counts.len() as u32;
        let n_draws = gs.colors.len() as u32;
        if n_draws == 0 || total_entries == 0 || gs.total_segments == 0 {
            return Ok(false); // 空场景：纹理保持零初始化（透明）
        }
        // M3.6：投机分配的容量缓存只由**主色逐帧路径**维护。
        // - clip 覆盖率（`coverage`）是辅助 pass，规模与主场景无关：若它也读/写
        //   这份缓存，两帧之间容量会被反复改写，主场景每帧都要重做；
        // - 诊断转储（`dump`）要的正是**精确**规模，测量本身就是它的目的。
        let learns_capacity = !coverage && dump.is_none();
        // M3.6：本场景的特征键 —— 容量按它分表，避免大小场景互相踩。
        let key = SceneKey::of(gs);
        let (tiles_x, tiles_y) = (gs.tiles_x, gs.tiles_y);
        let n_tiles = (tiles_x as u64) * (tiles_y as u64);
        let n_strips = (n_draws as u64) * (tiles_y as u64);
        let total_segments = gs.total_segments;

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
        let pad16 = mk_init("amiki_sparse_pad", &PAD16);
        // M3.5：上传阶段计时起点（到数据缓冲创建完成为止）
        let t_upload = Instant::now();

        // —— 上传（与稠密路径同构）——
        // **用 `extend_from_slice` 一次 memcpy，而不是逐元素 push。**
        //
        // 这里原本是三个 `push` 循环：一条圆场景的三次曲线是 4 万条 →
        // 光这一段就是 **8 万次 push**（加上 quad 的 2 万次，共约 10 万次），
        // 与提取本身同量级。而它被记在"上传"名下（`upload_ms` 0.74 ms），
        // 所以一直没被当成 push 问题 —— **名字掩盖了它**。
        //
        // `[[f32;4];2]` 与 `[f32;4]` 都是 `Pod`，用 `cast_slice` 展平后
        // `extend_from_slice` 就是一段连续内存拷贝。
        let mut data: Vec<[f32; 4]> = Vec::with_capacity((nq_len(gs) + nc_len(gs)) * 2 + nl_len(gs));
        data.extend_from_slice(bytemuck::cast_slice::<[[f32; 4]; 2], [f32; 4]>(&gs.quads));
        data.extend_from_slice(bytemuck::cast_slice::<[[f32; 4]; 2], [f32; 4]>(&gs.cubics));
        data.extend_from_slice(&gs.lines);
        let mut entry_draws: Vec<u32> = Vec::with_capacity(data.len());
        entry_draws.extend_from_slice(&gs.quad_draws);
        entry_draws.extend_from_slice(&gs.cubic_draws);
        entry_draws.extend_from_slice(&gs.line_draws);
        // draw_meta 已由 `build_gpu_scene` **直接拼好**
        // （`(flags, clip0|clip1<<8, clip2|clip3<<8, grad_index)`；
        //  每裁剪索引 8 bit，w 为 M6 渐变下标，u32::MAX 表示实心色）。
        //
        // 这里曾经是"把三个数组 zip 成 draw_meta"的装配 pass —— P1-1 实测它是
        // 提取的 82% 时间（`Vec::push`）的一部分，已改为在提取循环里直接生成。
        let draw_meta: &[[u32; 4]] = &gs.draw_meta;

        // M6：渐变数据打包（定长条目，布局与 sparse_fine.wgsl 的 grad_color 对应）
        let grad_vec = pack_gradients(&gs.gradients);
        let grad_buf = mk_init("amiki_sparse_grad", bytemuck::cast_slice(&grad_vec));

        // M6：图像放置参数 + 位图纹理数组（每 draw 3 个 vec4，自描述）
        let place_vec = pack_image_place(gs);
        let place_buf = mk_init("amiki_sparse_imgplace", bytemuck::cast_slice(&place_vec));
        let (img_tex_view, img_sampler, img_sampler_repeat) =
            self.mk_image_texture_array(&gs.images)?;

        let data_buf = mk_init("amiki_sparse_data", bytemuck::cast_slice(&data));
        let entry_draws_buf = mk_init("amiki_sparse_edraws", bytemuck::cast_slice(&entry_draws));
        let colors_buf = mk_init("amiki_sparse_colors", bytemuck::cast_slice(&gs.colors));
        let draw_meta_buf = mk_init("amiki_sparse_draw_meta", bytemuck::cast_slice(draw_meta));
        let counts_buf = mk_storage("amiki_sparse_counts", (total_entries as u64) * 4);
        // counts 需要 CPU 数据 → 用初始化版本
        self.queue
            .write_buffer(&counts_buf, 0, bytemuck::cast_slice(&gs.counts));
        let seg_base_buf = mk_storage("amiki_sparse_seg_base", (total_entries as u64) * 4);
        let seg_ends_buf = mk_storage("amiki_sparse_seg_ends", (total_segments as u64) * 16);
        let seg_draws_buf = mk_storage("amiki_sparse_seg_draws", (total_segments as u64) * 4);
        let indirect_buf = mk_init("amiki_sparse_indirect", bytemuck::bytes_of(&gs.indirect));
        prof.upload_ms += t_upload.elapsed().as_secs_f64() * 1000.0;
        // M3.5：分箱结构分配阶段计时
        let t_alloc = Instant::now();

        // —— 分箱结构（规模 = draws × tile 行）——
        let strip_meta_buf = mk_storage("amiki_sparse_strip_meta", n_strips * 16);
        let strip_cell_off_buf = mk_storage("amiki_sparse_strip_cell_off", n_strips * 4);
        let strip_bucket_off_buf = mk_storage("amiki_sparse_strip_bucket_off", n_strips * 4);
        let tr_prefix_buf = mk_storage("amiki_sparse_tr_prefix", n_strips * 4);
        let tr_list_buf = mk_storage("amiki_sparse_tr_list", n_strips * 4);
        let tr_offsets_buf = mk_storage("amiki_sparse_tr_offsets", (tiles_y as u64) * 8);
        let tile_count_buf = mk_storage("amiki_sparse_tile_count", n_tiles * 4);
        let tile_start_buf = mk_storage("amiki_sparse_tile_start", n_tiles * 4);
        let stats_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_sparse_stats"),
            contents: &[0u8; 32],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        prof.alloc_ms += t_alloc.elapsed().as_secs_f64() * 1000.0;
        let params = |n_cells: u32, n_buckets: u32, n_tile_entries: u32| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("amiki_sparse_params"),
                contents: bytemuck::bytes_of(&BinsParams {
                    tiles_x,
                    n_draws,
                    tiles_y,
                    height: gs.height,
                    total_segments,
                    n_cells,
                    n_buckets,
                    n_tile_entries,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let params_v1 = params(0, 0, 0);

        // 稀疏 kernel 的统一 bind group：未用槽位绑 16B 垫片
        let sparse_bind = |params_buf: &wgpu::Buffer, entries: &[(u32, &wgpu::Buffer)]| {
            let mut all: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(9);
            all.push(wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() });
            for b in 1..=8u32 {
                let buf = entries
                    .iter()
                    .find(|(bb, _)| *bb == b)
                    .map(|(_, bf)| *bf)
                    .unwrap_or(&pad16);
                all.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
            }
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("amiki_sparse_bind"),
                layout: &self.sparse_bgl,
                entries: &all,
            })
        };
        /// 单维派发（入参为**元素数**；自动按 65535 拆分到 2D，规避工作组维度上限）
        fn dispatch_1d(pass: &mut wgpu::ComputePass<'_>, n_elements: u32) {
            if n_elements == 0 {
                return;
            }
            let wg = n_elements.div_ceil(64);
            let x = wg.min(65_535);
            let y = wg.div_ceil(65_535);
            pass.dispatch_workgroups(x, y, 1);
        }

        // —— 时间戳槽位（仅分析路径）——
        // 原实现只有 5 个粗粒度槽位，把中间 5 次 scan 全藏在黑盒里；现在为每次 scan
        // 单独分配槽位，并补齐原先被省略的 pass，使稀疏全链路可逐段归因。
        let ts_slots: Option<[(u32, u32); 10]> = match ts.as_mut() {
            Some(r) => Some([
                r.slot("clear_strips")?,
                r.slot("scan:seg_base")?,
                r.slot("flatten+strip_extract+measure")?,
                r.slot("scan:strip_cell_off")?,
                r.slot("scan:strip_bucket_off")?,
                r.slot("scan:tr_prefix")?,
                r.slot("tr_scatter+tr_offsets+tile_count")?,
                r.slot("scan:tile_start")?,
                r.slot("clear_sparse+emit")?,
                r.slot("scan:cell_csr+tile_fill+scatter+carry+fine")?,
            ]),
            None => None,
        };
        let slot = |i: usize| ts_slots.map(|s| s[i]);

        // strip/extract 的专用 bind group（strip_bgl）
        let strip_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_sparse_strip_bind"),
            layout: &self.strip_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_v1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: seg_ends_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seg_draws_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: strip_meta_buf.as_entire_binding() },
            ],
        });
        let b_bins = sparse_bind(
            &params_v1,
            &[
                (1, &strip_meta_buf),
                (2, &strip_cell_off_buf),
                (3, &strip_bucket_off_buf),
                (4, &tr_prefix_buf),
                (5, &stats_buf),
                (6, &tr_list_buf),
                (7, &tr_offsets_buf),
            ],
        );
        let b_tile_count = sparse_bind(
            &params_v1,
            &[
                (1, &strip_meta_buf),
                (2, &tr_offsets_buf),
                (3, &tr_list_buf),
                (4, &tile_count_buf),
                (5, &stats_buf),
            ],
        );
        let flatten_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_sparse_flatten_params"),
            contents: bytemuck::bytes_of(&FlattenParams {
                n_quads: nq_len(gs) as u32,
                n_cubics: nc_len(gs) as u32,
                n_lines: nl_len(gs) as u32,
                eps: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let flatten_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_sparse_flatten_bind"),
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

        // ———— enc1a：分箱结构 ————
        let t_enc1 = Instant::now();
        let mut enc1a = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_sparse_enc1a") });
        {
            let mut pass = enc1a.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_clear_strips"),
                timestamp_writes: ts_writes(ts.as_deref(), slot(0)),
            });
            pass.set_pipeline(&self.strip_clear_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            dispatch_1d(&mut pass, n_strips as u32);
        }
        enc1a.copy_buffer_to_buffer(&counts_buf, 0, &seg_base_buf, 0, (total_entries as u64) * 4);
        self.run_scan(
            &mut enc1a,
            &seg_base_buf,
            total_entries,
            ts_slot(ts.as_deref(), ts_slots, 1),
        );
        {
            let mut pass = enc1a.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_flatten_strips"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 2),
            });
            pass.set_pipeline(&self.flatten_pipeline);
            pass.set_bind_group(0, &flatten_bind, &[]);
            pass.dispatch_workgroups(total_entries.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.strip_extract_pipeline);
            pass.set_bind_group(0, &strip_bind, &[]);
            dispatch_1d(&mut pass, total_segments);
            pass.set_pipeline(&self.sparse_measure_pipeline);
            pass.set_bind_group(0, &b_bins, &[]);
            dispatch_1d(&mut pass, n_strips as u32);
        }
        self.run_scan(&mut enc1a, &strip_cell_off_buf, n_strips as u32, ts_slot(ts.as_deref(), ts_slots, 3));
        self.run_scan(&mut enc1a, &strip_bucket_off_buf, n_strips as u32, ts_slot(ts.as_deref(), ts_slots, 4));
        self.run_scan(&mut enc1a, &tr_prefix_buf, n_strips as u32, ts_slot(ts.as_deref(), ts_slots, 5));
        {
            let mut pass = enc1a.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_tr_tiles"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 6),
            });
            pass.set_pipeline(&self.sparse_tr_scatter_pipeline);
            pass.set_bind_group(0, &b_bins, &[]);
            dispatch_1d(&mut pass, n_strips as u32);
            pass.set_pipeline(&self.sparse_tr_offsets_pipeline);
            pass.set_bind_group(0, &b_bins, &[]);
            dispatch_1d(&mut pass, tiles_y);
            pass.set_pipeline(&self.sparse_tile_count_pipeline);
            pass.set_bind_group(0, &b_tile_count, &[]);
            dispatch_1d(&mut pass, n_tiles as u32);
        }
        enc1a.copy_buffer_to_buffer(&tile_count_buf, 0, &tile_start_buf, 0, n_tiles * 4);
        self.run_scan(&mut enc1a, &tile_start_buf, n_tiles as u32, ts_slot(ts.as_deref(), ts_slots, 7));
        self.queue.submit(Some(enc1a.finish()));
        prof.encode1a_ms = t_enc1.elapsed().as_secs_f64() * 1000.0;

        // 〖容量 1〗投机分配：该场景容量已知则**直接采用，不再停顿**
        //
        // 原实现必须在此读回规模，而那是一次同步停顿（实测 2.35 ms，其中
        // 约 1.4 ms 是纯 fence/map 延迟）。改为用**该场景**上一帧学到的容量
        // 直接分配；只有首帧、或该场景已放弃投机时才真的读一次。
        let cap = if learns_capacity { self.sparse_capacity(key) } else { None };
        let speculative = cap.is_some();
        prof.speculative = speculative;

        let (n_cells, n_buckets, n_tile_entries) = match cap {
            Some(c) => (c.cells, c.buckets, c.tile_entries),
            None => {
                let t_rb1 = Instant::now();
                let stats_bytes = self.read_buffer(&stats_buf, 32)?;
                prof.readback1_ms = Some(t_rb1.elapsed().as_secs_f64() * 1000.0);
                let st: [u32; 8] = bytemuck::pod_read_unaligned(&stats_bytes);
                if let Some(d) = dump.as_deref_mut() {
                    if d.sparse_stats.is_some() {
                        d.sparse_stats = Some(st);
                    }
                }
                if st[0] == 0 {
                    return Ok(false); // 无任何内容覆盖画布
                }
                (st[0], st[1], st[2])
            }
        };
        let cell_aux_buf = mk_storage("amiki_sparse_cell_aux", (n_cells as u64) * 8);
        let cell_csr_buf = mk_storage("amiki_sparse_cell_csr", ((n_cells + 1) as u64) * 4);
        let buckets_buf = mk_storage("amiki_sparse_buckets", (n_buckets as u64) * 4);
        let cell_meta_buf = mk_storage("amiki_sparse_cell_meta", (n_cells as u64) * 16);
        let tile_list_buf = mk_storage("amiki_sparse_tile_list", (n_tile_entries as u64) * 16);
        let params_v2 = params(n_cells, n_buckets, n_tile_entries);

        // ———— enc1b：稀疏清零 + emit ————
        let t_enc1b = Instant::now();
        let b_clear = sparse_bind(&params_v2, &[(6, &cell_aux_buf), (7, &buckets_buf)]);
        let b_emit = sparse_bind(
            &params_v2,
            &[
                (1, &seg_ends_buf),
                (2, &seg_draws_buf),
                (3, &strip_meta_buf),
                (4, &strip_cell_off_buf),
                (5, &strip_bucket_off_buf),
                (6, &cell_aux_buf),
                (7, &buckets_buf),
                (8, &stats_buf),
            ],
        );
        let mut enc1b = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_sparse_enc1b") });
        {
            let mut pass = enc1b.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_clear_emit"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 8),
            });
            pass.set_pipeline(&self.sparse_clear_pipeline);
            pass.set_bind_group(0, &b_clear, &[]);
            dispatch_1d(&mut pass, n_cells.max(n_buckets));
            pass.set_pipeline(&self.sparse_emit_pipeline);
            pass.set_bind_group(0, &b_emit, &[]);
            dispatch_1d(&mut pass, total_segments);
        }
        self.queue.submit(Some(enc1b.finish()));
        prof.encode1b_ms = t_enc1b.elapsed().as_secs_f64() * 1000.0;

        // 〖容量 2〗成员总数：投机时同样来自该场景的容量（理由见「容量 1」）
        let (n_items, measured2) = match cap {
            Some(c) => (c.items, None),
            None => {
                let t_rb2 = Instant::now();
                let stats_bytes = self.read_buffer(&stats_buf, 32)?;
                prof.readback2_ms = Some(t_rb2.elapsed().as_secs_f64() * 1000.0);
                let st2: [u32; 8] = bytemuck::pod_read_unaligned(&stats_bytes);
                (st2[3], Some(st2))
            }
        };
        let cell_segs_buf = mk_storage("amiki_sparse_cell_segs", (n_items.max(1) as u64) * 16);

        // ———— enc2：单元 CSR → tile 桶表 → scatter → carry → fine ————
        let b_tile_fill = sparse_bind(
            &params_v2,
            &[
                (1, &strip_meta_buf),
                (2, &tr_offsets_buf),
                (3, &tr_list_buf),
                (6, &strip_cell_off_buf),
                (7, &tile_start_buf),
                (8, &tile_list_buf),
            ],
        );
        let b_scatter = sparse_bind(
            &params_v2,
            &[
                (1, &seg_ends_buf),
                (2, &seg_draws_buf),
                (3, &strip_meta_buf),
                (4, &strip_cell_off_buf),
                (5, &cell_csr_buf),
                (6, &cell_aux_buf),
                (7, &cell_segs_buf),
            ],
        );
        let b_carry = sparse_bind(
            &params_v2,
            &[
                (1, &strip_meta_buf),
                (2, &strip_cell_off_buf),
                (3, &strip_bucket_off_buf),
                (4, &buckets_buf),
                (5, &cell_csr_buf),
                (6, &cell_meta_buf),
            ],
        );
        // 0.6.0：fine 参数（宽高 / tile 列数 / draw 数）
        let fine_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("amiki_sparse_fine_params"),
            contents: bytemuck::bytes_of(&FineParams {
                width: gs.width,
                height: gs.height,
                tiles_x,
                n_draws,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut views: Vec<&wgpu::TextureView> = (0..4).map(|_| &self.dummy_clip_view).collect();
        for (i, v) in clip_views.iter().take(4).enumerate() {
            views[i] = v;
        }
        let b_fine = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("amiki_sparse_fine_bind"),
            layout: &self.fine_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fine_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_segs_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: cell_meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: colors_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: draw_meta_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: tile_count_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: tile_start_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: tile_list_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: grad_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&tex_view) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(views[0]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(views[1]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(views[2]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(views[3]) },
                // M6 图像：放置参数、位图纹理数组、采样器
                wgpu::BindGroupEntry { binding: 14, resource: place_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&img_tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::Sampler(&img_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::Sampler(&img_sampler_repeat),
                },
            ],
        });
        let t_enc2 = Instant::now();
        let mut enc2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_sparse_enc2") });
        enc2.copy_buffer_to_buffer(&cell_aux_buf, 0, &cell_csr_buf, 0, (n_cells as u64) * 4);
        self.run_scan(&mut enc2, &cell_csr_buf, n_cells + 1, ts_slot(ts.as_deref(), ts_slots, 9));
        {
            let mut pass = enc2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("amiki_sparse_scatter_carry_fine"),
                timestamp_writes: ts_slot(ts.as_deref(), ts_slots, 9),
            });
            pass.set_pipeline(&self.sparse_tile_fill_pipeline);
            pass.set_bind_group(0, &b_tile_fill, &[]);
            dispatch_1d(&mut pass, n_tiles as u32);
            pass.set_pipeline(&self.sparse_scatter_pipeline);
            pass.set_bind_group(0, &b_scatter, &[]);
            dispatch_1d(&mut pass, total_segments);
            pass.set_pipeline(&self.sparse_carry_pipeline);
            pass.set_bind_group(0, &b_carry, &[]);
            dispatch_1d(&mut pass, (n_strips as u32).saturating_mul(16));
            if coverage {
                pass.set_pipeline(&self.sparse_coverage_pipeline);
            } else {
                pass.set_pipeline(&self.sparse_fine_pipeline);
            }
            pass.set_bind_group(0, &b_fine, &[]);
            // M4：只派发脏 tile 子矩形（理由见 run_pipeline_impl 处注释）
            let (_fx, fw, _fy, fh) = fine_dispatch_range(damage, tiles_x, tiles_y);
            pass.dispatch_workgroups(fw, fh, 1);
        }
        prof.encode2_ms = t_enc2.elapsed().as_secs_f64() * 1000.0;
        let t_sub2 = Instant::now();
        self.queue.submit(Some(enc2.finish()));
        prof.submit2_ms = t_sub2.elapsed().as_secs_f64() * 1000.0;

        // M3.6：本帧 GPU 工作已全部提交。确定**实际用量**并记进该场景的容量表项。
        //
        // 测量路径下分配值本身就来自实测，用量即它，不可能溢出。
        // 投机路径下分配值只是容量，**必须读回实际用量**才能判断是否溢出
        // —— 溢出的那一帧画面残缺，由外层作废重做（见 `run_pipeline_sparse`）。
        let allocated = SparseCapacity {
            cells: n_cells,
            buckets: n_buckets,
            tile_entries: n_tile_entries,
            items: n_items,
            overflows: 0,
        };
        let (usage, overflow) = if speculative {
            let t_use = Instant::now();
            let bytes = self.read_buffer(&stats_buf, 32)?;
            prof.usage_readback_ms = Some(t_use.elapsed().as_secs_f64() * 1000.0);
            let u: [u32; 8] = bytemuck::pod_read_unaligned(&bytes);
            let usage = SparseCapacity {
                cells: u[0],
                buckets: u[1],
                tile_entries: u[2],
                items: u[3],
                overflows: 0,
            };
            let ovf = allocated.exceeds(&usage);
            (usage, ovf)
        } else {
            (allocated, false)
        };
        if learns_capacity {
            self.store_sparse_capacity(key, usage, overflow);
        }
        if overflow {
            return Ok(true);
        }

        // —— 诊断转储（M3.5：全字段无条件转储，供 check_sparse_consistency 使用）——
        let t_dump = Instant::now();
        if let Some(d) = dump {
            // 诊断路径必然走测量（`learns_capacity` 要求 `dump.is_none()`），
            // 因此这里一定拿得到实测统计。用 `if let` 而非 `expect`：
            // 万一将来约束被改动，也只是少一份诊断，而不是 panic 掉渲染。
            if let Some(st2) = measured2 {
                d.sparse_stats = Some(st2);
            }
            let b = self.read_buffer(&strip_meta_buf, n_strips as usize * 16)?;
            d.strip_meta = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tr_offsets_buf, tiles_y as usize * 8)?;
            d.tr_offsets = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tr_list_buf, n_strips as usize * 4)?;
            d.tr_list = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tile_count_buf, n_tiles as usize * 4)?;
            d.tile_count = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tile_start_buf, n_tiles as usize * 4)?;
            d.tile_start = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&tile_list_buf, n_tile_entries as usize * 16)?;
            d.tile_list = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_meta_buf, n_cells as usize * 16)?;
            d.cell_meta = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_segs_buf, n_items as usize * 16)?;
            d.cell_segs = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&strip_cell_off_buf, n_strips as usize * 4)?;
            d.strip_cell_off = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&strip_bucket_off_buf, n_strips as usize * 4)?;
            d.strip_bucket_off = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_aux_buf, n_cells as usize * 8)?;
            d.cell_aux = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&cell_csr_buf, (n_cells as usize + 1) * 4)?;
            d.cell_csr_off = Some(bytemuck::cast_slice(&b).to_vec());
            let b = self.read_buffer(&buckets_buf, n_buckets as usize * 4)?;
            d.sparse_buckets = Some(bytemuck::cast_slice(&b).to_vec());
        }
        prof.dump_ms = Some(t_dump.elapsed().as_secs_f64() * 1000.0);
        Ok(false)
    }

    /// M3.5：公开的稀疏诊断入口——跑一次稀疏管线并转储**全部**稀疏结构，
    /// 供 [`check_sparse_consistency`] 与 `PipelineDump` 消费。
    ///
    /// 与 [`GpuRenderer::sparse_stats`] 的区别：后者只读回 32B 规模统计（性能敏感，
    /// 可用于基准循环），本方法会读回全部稀疏结构（条带/桶表/单元/段端点），
    /// **仅供诊断与测试**，勿放进热路径。
    pub fn dump_sparse(
        &self,
        scene: &Scene,
        eps: f32,
    ) -> Result<(GpuScene, PipelineDump), RenderError> {
        let gs = build_gpu_scene(scene, eps);
        let tex = self.create_output_texture(&gs);
        let mut dump = PipelineDump::default();
        let mut prof = FrameProfile::default();
        self.run_pipeline_sparse(&gs, &tex, false, &[], None, Some(&mut dump), None, &mut prof)?;
        Ok((gs, dump))
    }
}
