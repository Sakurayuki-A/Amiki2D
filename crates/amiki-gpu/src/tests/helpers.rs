//! 共享的测试装置（一致性判据、场景构造、dump 工具）。
//!
//! 从 `src/tests.rs` 分出（只搬代码，未改逻辑）。

use super::*;

// ===================== M3a：活跃 tile 表（sparse strips 底座） =====================

/// 运行一次管线并转储中间数据（M3a 验证用：固定走稠密参照路径）。
pub(super) fn dump_pipeline(renderer: &GpuRenderer, scene: &Scene, eps: f32) -> (GpuScene, PipelineDump) {
    let gs = build_gpu_scene(scene, eps);
    let tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("amiki_m3a_out"),
        size: wgpu::Extent3d { width: gs.width, height: gs.height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let mut dump = PipelineDump {
        counters: Some(Vec::new()),
        carries: Some(Vec::new()),
        strip_meta: Some(Vec::new()),
        ..Default::default()
    };
    let mut prof = FrameProfile::default();
    renderer
        .run_pipeline_impl(&gs, &tex, false, &[], None, Some(&mut dump), None, &mut prof)
        .expect("管线运行失败");
    (gs, dump)
}

/// 稠密网格活动单元 ↔ strip 区间一致性：
/// · 覆盖性（恒成立）：任何活动单元（成员计数>0 / 真实 carry≠0）必落在对应 strip 内；
/// · 无幻影：完全无活动的 (draw, tr) 不应有 strip；
/// · `exact = true` 时额外要求区间与活动包围盒逐格相等——画布内轴对齐几何
///   应精确成立；越界内容会按稠密语义保守扩张左/右界（见 strips.wgsl）。
pub(super) fn check_strips(gs: &GpuScene, dump: &PipelineDump, exact: bool) {
    /// 低于此视为闭合路径求和的浮点残差（约 1e-7），非真实 carry。
    const CARRY_EPS: f32 = 1e-4;
    let counters = dump.counters.as_ref().expect("counters 未转储");
    let carries = dump.carries.as_ref().expect("carries 未转储");
    let meta = dump.strip_meta.as_ref().expect("strip_meta 未转储");
    let n_draws = gs.colors.len() as u32;
    let (mut n_active, mut n_covered) = (0usize, 0usize);
    for d in 0..n_draws {
        for tr in 0..gs.tiles_y {
            let si = (tr * n_draws + d) as usize; // 转置布局：si = tr * n_draws + d
            let (lo_s, hi_s, y0_s, y1_s) =
                (meta[si * 4], meta[si * 4 + 1], meta[si * 4 + 2], meta[si * 4 + 3]);
            let (mut mn_c, mut mx_c, mut mn_r, mut mx_r) = (u32::MAX, 0u32, u32::MAX, 0u32);
            for col in 0..gs.tiles_x {
                for row in 0..16 {
                    let b = (((tr * gs.tiles_x + col) * n_draws + d) * 16 + row) as usize;
                    if counters[b] > 0 || carries[b].abs() > CARRY_EPS {
                        mn_c = mn_c.min(col);
                        mx_c = mx_c.max(col);
                        mn_r = mn_r.min(row);
                        mx_r = mx_r.max(row);
                    }
                }
            }
            if mn_c == u32::MAX {
                assert_eq!(lo_s, gs.tiles_x, "无活动的 (draw={d}, tr={tr}) 不应有 strip");
                continue;
            }
            n_active += ((mx_c - mn_c + 1) as usize) * ((mx_r - mn_r + 1) as usize);
            assert!(lo_s != gs.tiles_x, "有活动但 strip 缺失 (draw={d}, tr={tr})");
            // 覆盖性：strip 必须框住全部活动单元
            assert!(
                lo_s <= mn_c && mx_c <= hi_s && y0_s <= mn_r && mx_r <= y1_s,
                "strip 未覆盖活动单元 (draw={d}, tr={tr}): strip=({lo_s},{hi_s},{y0_s},{y1_s}) 活动=({mn_c},{mx_c},{mn_r},{mx_r})"
            );
            n_covered += ((hi_s - lo_s + 1) as usize) * ((y1_s - y0_s + 1) as usize);
            if exact {
                assert_eq!(
                    (lo_s, hi_s, y0_s, y1_s),
                    (mn_c, mx_c, mn_r, mx_r),
                    "strip 必须与稠密活动包围盒逐格相等 (draw={d}, tr={tr})"
                );
            }
        }
    }
    eprintln!("M3a 对拍：活动单元 {n_active}，strip 覆盖单元 {n_covered}");
}

// ===================== M3.5：稀疏自洽检查 =====================

/// 覆盖各类稀疏结构的场景集（矩形 / 曲线+洞 / 自相交 / 越界 / 裁剪+混合）。
pub(super) fn consistency_scenes() -> Vec<(&'static str, Scene)> {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{BlendMode, Color, Paint, PathBuilder};
    let mut out: Vec<(&'static str, Scene)> = Vec::new();
    {
        // 轴对齐 + 跨 tile 边界（tile 区间与局部行区间最严格）
        let mut s = Scene::new(72, 72);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 72.0, 72.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(16.0, 16.0, 33.0, 49.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(40, 90, 160, 200)));
        out.push(("rects", s));
    }
    {
        // 曲线 + 反绕洞 + 旋转（斜边跨带，carry 非零多）
        let mut s = Scene::new(96, 96);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        let t = Affine::translate(Vec2::new(48.0, 48.0))
            .then(Affine::rotate(0.3))
            .then(Affine::translate(Vec2::new(-48.0, -48.0)));
        s.push(t, pb.build(), Paint::Solid(Color::rgb8(0, 200, 100)));
        out.push(("donut_rot", s));
    }
    {
        // 自相交（W=2）+ 越出画布四缘（strip 保守扩张路径）
        let mut s = Scene::new(80, 64);
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(40.0, 30.0), 34.0);
        s.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(230, 120, 60, 200)),
            BlendMode::Multiply,
        );
        let mut pb = PathBuilder::new();
        pb.rect(-30.0, 4.0, 30.0, 40.0); // 跨左缘（桶 0 钳制）
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(30, 160, 90)));
        let mut pb = PathBuilder::new();
        pb.rect(40.0, 8.0, 140.0, 48.0); // 跨右缘（桶 tiles_x 排除）
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgba8(200, 90, 30, 200)));
        out.push(("pentagram_offscreen", s));
    }
    {
        // 裁剪 + 非可分离混合
        let mut s = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        s.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
        let c0 = s.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(32.0, 32.0), 26.0);
        s.push_blend(
            Affine::IDENTITY,
            pb.build(),
            Paint::Solid(Color::rgba8(40, 90, 160, 220)),
            BlendMode::Hue,
        );
        s.draws.last_mut().unwrap().clips = vec![c0];
        out.push(("clip_blend", s));
    }
    out
}

/// 深拷贝转储（破坏性测试用）。
pub(super) fn clone_dump(d: &PipelineDump) -> PipelineDump {
    PipelineDump {
        seg_ends: d.seg_ends.clone(),
        seg_draws: d.seg_draws.clone(),
        counters: d.counters.clone(),
        scan_base: d.scan_base.clone(),
        carries: d.carries.clone(),
        item_total: d.item_total,
        seg_refs: d.seg_refs.clone(),
        bucket_sums: d.bucket_sums.clone(),
        strip_meta: d.strip_meta.clone(),
        sparse_stats: d.sparse_stats,
        tr_offsets: d.tr_offsets.clone(),
        tr_list: d.tr_list.clone(),
        tile_count: d.tile_count.clone(),
        tile_start: d.tile_start.clone(),
        tile_list: d.tile_list.clone(),
        cell_meta: d.cell_meta.clone(),
        cell_segs: d.cell_segs.clone(),
        strip_cell_off: d.strip_cell_off.clone(),
        strip_bucket_off: d.strip_bucket_off.clone(),
        cell_aux: d.cell_aux.clone(),
        cell_csr_off: d.cell_csr_off.clone(),
        sparse_buckets: d.sparse_buckets.clone(),
    }
}

// ============================================================ even-odd（0.8.0）

/// 造一个 even-odd 场景：同向双环（NonZero 实心 / EvenOdd 镂空）+ 自相交五角星。
pub(super) fn even_odd_scene(w: u32, h: u32) -> Scene {
    use amiki_math::{Affine, Vec2};
    use amiki_scene::{Color, FillRule, Paint, PathBuilder};

    let mut scene = Scene::new(w, h);
    let mut pb = PathBuilder::new();
    pb.rect(0.0, 0.0, w as f32, h as f32);
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(255, 255, 255)),
    );

    // 同向双环（非整数对齐，保证有部分覆盖像素）
    let mut pb = PathBuilder::new();
    pb.rect(3.3, 3.1, 28.7, 28.9);
    pb.rect(11.2, 11.3, 21.8, 20.7);
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(0, 0, 0)),
    );
    scene.draws.last_mut().unwrap().fill_rule = FillRule::EvenOdd;

    // `{5/2}` 五角星（真自相交：中心区绕数为 2）
    let mut pb = PathBuilder::new();
    let (cx, cy, r) = (44.0f32, 44.0f32, 17.3f32);
    for i in 0..5 {
        let a = 2.0 * std::f32::consts::PI * (i as f32) * 2.0 / 5.0 - std::f32::consts::FRAC_PI_2;
        let p = Vec2::new(cx + r * a.cos(), cy + r * a.sin());
        if i == 0 {
            pb.move_to(p);
        } else {
            pb.line_to(p);
        }
    }
    pb.close();
    scene.push(
        Affine::IDENTITY,
        pb.build(),
        Paint::Solid(Color::rgb8(20, 60, 200)),
    );
    scene.draws.last_mut().unwrap().fill_rule = FillRule::EvenOdd;
    scene
}

