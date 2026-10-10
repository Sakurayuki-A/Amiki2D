//! model（从 `lib.rs` 分出，未改逻辑）。


use crate::*;

    #[test]
    fn damage_defaults_to_full() {
        let d = Damage::new(128, 64);
        assert!(d.is_full(), "新建 Damage 必须是全脏（无可复用内容）");
        assert!(!d.is_empty());
        assert_eq!(d.tiles(), None, "全脏用 None 表达");
        assert_eq!(d.total_tiles(), 8 * 4);
        assert_eq!(d.tile_count(), d.total_tiles());
    }

    #[test]
    fn damage_clear_means_skip() {
        let mut d = Damage::new(128, 64);
        d.clear();
        assert!(d.is_empty(), "clear 后应无脏区（静态画面可完全跳过重绘）");
        assert!(!d.is_full());
        assert_eq!(d.tile_count(), 0);
    }

    #[test]
    fn damage_rect_quantizes_outward() {
        let mut d = Damage::new(128, 64);
        d.clear();
        // 1 像素的改动必须把**覆盖它的那个 tile** 整个标记为脏
        d.invalidate_rect(20.0, 20.0, 21.0, 21.0);
        assert_eq!(d.tiles(), Some((1, 1, 2, 2)), "1px 改动应量化到 tile(1,1)");
        assert_eq!(d.tile_count(), 1);
    }

    #[test]
    fn damage_union_of_rects() {
        let mut d = Damage::new(256, 256);
        d.clear();
        d.invalidate_rect(10.0, 10.0, 20.0, 20.0);   // tile (0,0)
        d.invalidate_rect(200.0, 200.0, 210.0, 210.0); // tile (12,12)
        // 并集是外接包围盒：覆盖 0..14 共 14x14 个 tile（保守，但保证不漏）
        let (tx0, ty0, tx1, ty1) = d.tiles().unwrap();
        assert_eq!((tx0, ty0), (0, 0));
        assert_eq!((tx1, ty1), (14, 14));
        assert_eq!(d.tile_count(), 14 * 14);
        // 保守性的关键断言：两块脏区所在的 tile 都必须落在范围内
        // （注意 tx0/ty0 是 u32，用 == 而非 <= 0，否则断言恒真、毫无价值）
        assert_eq!((tx0, ty0), (0, 0));
        assert!(tx1 > 13 && ty1 > 13, "并集必须覆盖到第二块脏区所在的 tile(13,13)");
    }

    #[test]
    fn damage_clamps_to_canvas() {
        let mut d = Damage::new(70, 40); // tiles_x = ceil(70/16) = 5, tiles_y = 3
        d.clear();
        // 越出画布右/下的改动必须被钳到画布 tile 数内
        d.invalidate_rect(60.0, 30.0, 9999.0, 9999.0);
        let (tx0, ty0, tx1, ty1) = d.tiles().unwrap();
        assert_eq!((tx0, ty0), (3, 1));
        assert_eq!((tx1, ty1), (5, 3), "必须钳到画布 tile 边界");
        // 完全在画布外的改动不产生脏区
        let mut d2 = Damage::new(70, 40);
        d2.clear();
        d2.invalidate_rect(-100.0, -100.0, -50.0, -50.0);
        assert_eq!(d2.tile_count(), 0, "画布外的改动不应产生脏 tile");
    }

    #[test]
    fn damage_invalidate_all_wins() {        let mut d = Damage::new(64, 64);
        d.clear();
        d.invalidate_rect(0.0, 0.0, 5.0, 5.0);
        assert!(!d.is_full());
        d.invalidate_all();
        assert!(d.is_full(), "invalidate_all 后必须回到全脏");
    }

    #[test]
    fn stream_round_trip() {
        let mut scene = Scene::new(800, 600);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(100.0, 100.0), 60.0, false);
        pb.circle(Vec2::new(100.0, 100.0), 30.0, true);
        let clip = scene.push_clip(Affine::rotate(0.3), pb.build());
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(400.0, 300.0), 120.0);
        scene.push_blend(
            Affine::translate(Vec2::new(10.0, 20.0)),
            pb.build(),
            Paint::Solid(Color::rgba8(255, 0, 0, 200)),
            BlendMode::Multiply,
        );
        scene.draws.last_mut().unwrap().clips = vec![clip];
        let mut pb = PathBuilder::new();
        pb.quad_to(Vec2::new(0.0, 100.0), Vec2::new(100.0, 0.0));
        pb.cubic_to(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0), Vec2::new(5.0, 6.0));
        pb.close();
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::TRANSPARENT));

        let bytes = encode_scene(&scene);
        let back = decode_scene(&bytes).expect("解码失败");
        assert_eq!(back.width, 800);
        assert_eq!(back.height, 600);
        assert_eq!(back.clips.len(), 1);
        assert_eq!(back.clips[0].path, scene.clips[0].path);
        assert_eq!(back.draws.len(), 2);
        assert_eq!(back.draws[0].path, scene.draws[0].path);
        assert_eq!(back.draws[0].blend, BlendMode::Multiply);
        assert_eq!(back.draws[0].clips, vec![clip]);
        assert_eq!(back.draws[1].path, scene.draws[1].path);
        assert_eq!(back.draws[1].paint, scene.draws[1].paint);
    }

    #[test]
    fn stream_rejects_garbage() {
        assert!(decode_scene(b"XXXX").is_err());
        let mut bytes = encode_scene(&Scene::new(10, 10));
        bytes[4] = 99; // 版本破坏
        assert!(decode_scene(&bytes).is_err());
    }

    /// AMK1 v2（0.8.0）：`fill_rule` 必须往返，且**混排顺序**不变。
    #[test]
    fn stream_v2_round_trip_keeps_fill_rule() {
        let mut scene = Scene::new(64, 64);
        for (i, rule) in [FillRule::NonZero, FillRule::EvenOdd, FillRule::NonZero, FillRule::EvenOdd]
            .into_iter()
            .enumerate()
        {
            let mut pb = PathBuilder::new();
            pb.rect(i as f32 * 8.0, 2.1, i as f32 * 8.0 + 6.0, 8.3);
            scene.push(
                Affine::IDENTITY,
                pb.build(),
                Paint::Solid(Color::rgb8(0, 0, 0)),
            );
            scene.draws.last_mut().unwrap().fill_rule = rule;
        }
        // 顺带放一条非 Normal 的混合模式，验证标志位没污染 blend
        scene.draws[1].blend = BlendMode::Multiply;

        let bytes = encode_scene(&scene);
        assert_eq!(&bytes[0..4], STREAM_MAGIC);
        assert_eq!(
            u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            STREAM_VERSION,
            "编码器必须写 v2"
        );

        let back = decode_scene(&bytes).expect("v2 解码失败");
        assert_eq!(back.draws.len(), 4);
        for (i, r) in [FillRule::NonZero, FillRule::EvenOdd, FillRule::NonZero, FillRule::EvenOdd]
            .into_iter()
            .enumerate()
        {
            assert_eq!(back.draws[i].fill_rule, r, "第 {i} 条 draw 的填充规则应往返一致");
        }
        assert_eq!(back.draws[1].blend, BlendMode::Multiply, "blend 不应被标志位污染");
        assert_eq!(back.draws[0].blend, BlendMode::Normal);
        eprintln!("AMK1 v2：4 条 draw（含 2 条 EvenOdd）往返一致，blend 未被污染");
    }

    /// **v1 只解不编**：把版本改回 1 后，即使字节里 bit 12 是 1，也必须解成 NonZero。
    #[test]
    fn stream_v1_bytes_decode_as_nonzero() {
        let mut scene = Scene::new(32, 32);
        let mut pb = PathBuilder::new();
        pb.rect(2.3, 2.1, 28.7, 28.9);
        pb.rect(11.2, 11.3, 21.8, 20.7);
        scene.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(0, 0, 0)));
        scene.draws.last_mut().unwrap().fill_rule = FillRule::EvenOdd;

        let mut bytes = encode_scene(&scene);
        // 版本字段改回 1（布局相同，所以这就是一份合法的 v1 流）；
        // bit 12 仍然留着 1 —— v1 里它是保留位，必须被忽略。
        bytes[4..8].copy_from_slice(&STREAM_VERSION_V1.to_le_bytes());
        let back = decode_scene(&bytes).expect("v1 必须能解");
        assert_eq!(
            back.draws[0].fill_rule,
            FillRule::NonZero,
            "v1 的 bit 12 是保留位 ⇒ 必须解成 NonZero（不静默错解）"
        );
        eprintln!("AMK1 v1：同一份字节按 v1 解出 NonZero（保留位被忽略）");
    }

    /// 未知版本必须**显式报错**，而不是静默按某个版本解读。
    #[test]
    fn stream_rejects_unknown_version() {
        let mut bytes = encode_scene(&Scene::new(8, 8));
        bytes[4..8].copy_from_slice(&3u32.to_le_bytes());
        let e = decode_scene(&bytes).expect_err("版本 3 必须被拒绝");
        assert!(e.contains("版本"), "错误信息应说明版本问题：{e}");
        eprintln!("AMK1：未知版本被显式拒绝（{e}）");
    }

    // ===================== M5：文本（字形轮廓即路径） =====================

    /// 一个矩形字形：em 单位下覆盖 [0,0.1]×[0,0.6]（模拟一个"竖条"字形）。
    fn bar_glyph() -> GlyphOutline {
        GlyphOutline::new(vec![
            PathEl::MoveTo(Vec2::new(0.0, 0.0)),
            PathEl::LineTo(Vec2::new(0.1, 0.0)),
            PathEl::LineTo(Vec2::new(0.1, 0.6)),
            PathEl::LineTo(Vec2::new(0.0, 0.6)),
            PathEl::Close,
        ])
    }

    /// 一个方形字形（不同占位，用于验证多字形各自独立展开）。
    fn box_glyph() -> GlyphOutline {
        GlyphOutline::new(vec![
            PathEl::MoveTo(Vec2::new(0.0, 0.0)),
            PathEl::LineTo(Vec2::new(0.5, 0.0)),
            PathEl::LineTo(Vec2::new(0.5, 0.6)),
            PathEl::LineTo(Vec2::new(0.0, 0.6)),
            PathEl::Close,
        ])
    }

    #[test]
    fn text_run_expands_to_one_path_per_glyph() {
        let mut run = TextRun::new(20.0);
        run.push_glyph(1, 0.0, 0.0);
        run.push_glyph(2, 0.8, 0.0);
        run.push_glyph(1, 1.6, 0.0);

        let paths = expand_text_run(&run, |id| match id {
            1 => Some(bar_glyph()),
            2 => Some(box_glyph()),
            _ => None,
        });
        assert_eq!(paths.len(), 3, "三个字形应展开成三条独立路径");

        // 第 3 个字形与第 1 个同形，但位置应为 1.6 em × 20px = 32px
        let p0 = paths[0].1.map_point(Vec2::ZERO);
        let p2 = paths[2].1.map_point(Vec2::ZERO);
        assert!((p0.x - 0.0).abs() < 1e-4, "第 1 个字形应在 x=0，实际 {}", p0.x);
        assert!((p2.x - 32.0).abs() < 1e-4, "第 3 个字形应在 x=32，实际 {}", p2.x);

        // 字号应折进坐标：字形高度 0.6 em × 20 px = 12 px
        let top = paths[0].1.map_point(Vec2::new(0.0, 0.6));
        assert!((top.y - 12.0).abs() < 1e-4, "字形高度应为 12px，实际 {}", top.y);
    }

    #[test]
    fn missing_glyph_is_skipped_not_panicking() {
        let mut run = TextRun::new(16.0);
        run.push_glyph(1, 0.0, 0.0);
        run.push_glyph(99, 1.0, 0.0); // 轮廓缺失
        run.push_glyph(2, 2.0, 0.0);
        let paths = expand_text_run(&run, |id| match id {
            1 => Some(bar_glyph()),
            2 => Some(box_glyph()),
            _ => None,
        });
        assert_eq!(paths.len(), 2, "缺失轮廓的字形应被跳过而不是 panic");
    }

    /// 字体回退（0.7.0）：展开时要按**字体下标**取轮廓 ——
    /// 同一个 glyph id 在不同字体里是不同字形。
    #[test]
    fn expand_text_run_with_passes_font_index_to_lookup() {
        let mut run = TextRun::new(20.0);
        run.push_glyph_in(0, 1, 0.0, 0.0, 0); // 字体 0 的字形 1
        run.push_glyph_in(1, 1, 1.0, 0.0, 3); // 字体 1 的**同一个**字形 id

        let mut seen: Vec<(usize, u32)> = Vec::new();
        let paths = expand_text_run_with(&run, |fi, id| {
            seen.push((fi, id));
            // 两个字体对同一个 id 给出**不同**轮廓
            match fi {
                0 => Some(bar_glyph()),
                _ => Some(box_glyph()),
            }
        });
        assert_eq!(seen, vec![(0, 1), (1, 1)], "lookup 必须收到 (字体下标, 字形 id)");
        assert_eq!(paths.len(), 2);
        assert_ne!(
            paths[0].0.els, paths[1].0.els,
            "同一 glyph id 在两个字体下必须得到不同轮廓（否则等于没做字体回退）"
        );
    }

    /// 缓存必须按**字体**分键：同一个 glyph id 换了字体不能命中旧条目。
    #[test]
    fn text_cache_separates_fonts() {
        let mut cache = TextCache::new();
        let mut run = TextRun::new(20.0);
        run.push_glyph_in(0, 1, 0.0, 0.0, 0);
        run.push_glyph_in(1, 1, 1.0, 0.0, 0);

        let fonts = [FontId::from_bits(0xAAAA), FontId::from_bits(0xBBBB)];
        let mut lookups = 0;
        let out = cache.expand_with(
            &run,
            |fi| fonts[fi],
            |fi, _id| {
                lookups += 1;
                if fi == 0 {
                    Some(bar_glyph())
                } else {
                    Some(box_glyph())
                }
            },
        );
        assert_eq!(out.len(), 2);
        assert_eq!(lookups, 2, "不同字体必须各提取一次，不能共用缓存");
        assert_eq!(cache.misses(), 2);
        assert_eq!(cache.hits(), 0);
        assert_ne!(out[0].0.els, out[1].0.els, "两次展开的轮廓必须不同");

        // 同一个字体再展开一次：应当命中
        let again = cache.expand_with(&run, |fi| fonts[fi], |_fi, _id| {
            lookups += 1;
            Some(bar_glyph())
        });
        assert_eq!(again.len(), 2);
        assert_eq!(lookups, 2, "同一字体集合再展开不应重新提取");
        assert_eq!(cache.hits(), 2);
    }

    /// **零破坏**：老入口 `GlyphKey::new` 不带字体身份，且量化规则不变。
    #[test]
    fn glyph_key_new_keeps_legacy_semantics() {
        let k = GlyphKey::new(7, 10.0);
        assert_eq!(k.font_id, FontId::UNKNOWN);
        assert_eq!(k.id, 7);
        assert_eq!(k.size_q, 160, "10.0 × 16 = 160");
        // 带字体身份的键与老键**必须不同**，否则字体回退会跨字体误命中
        let with = GlyphKey::with_font(FontId::from_bits(42), 7, 10.0);
        assert_ne!(k, with);
        assert_eq!(with.size_q, k.size_q);
    }

    #[test]
    fn text_cache_hits_on_repeated_glyphs() {
        let mut cache = TextCache::new();
        let mut run = TextRun::new(20.0);
        for i in 0..10 {
            run.push_glyph(1, i as f32 * 0.5, 0.0);
        }
        let mut lookups = 0;
        let a = cache.expand(&run, |id| {
            lookups += 1;
            if id == 1 { Some(bar_glyph()) } else { None }
        });
        assert_eq!(a.len(), 10);
        assert_eq!(lookups, 1, "10 个相同字形只应提取一次轮廓");
        assert_eq!(cache.misses(), 1);
        assert_eq!(cache.hits(), 9);

        // 第二次展开同一字号：全部命中
        let b = cache.expand(&run, |id| {
            lookups += 1;
            if id == 1 { Some(bar_glyph()) } else { None }
        });
        assert_eq!(b.len(), 10);
        assert_eq!(lookups, 1, "再次展开不应再提取轮廓");
        assert!(cache.hit_rate() > 0.9, "命中率应 > 0.9，实际 {}", cache.hit_rate());

        // 输出必须与未缓存路径**几何等价**（缓存不得改变渲染结果）。
        // 注意两者的表示不同：缓存把字号折进坐标、变换只含平移；
        // 直接展开则把缩放留在变换里。因此比较的是**变换后的顶点**，而非结构。
        let direct = expand_text_run(&run, |id| if id == 1 { Some(bar_glyph()) } else { None });
        assert_eq!(b.len(), direct.len());
        for (gi, (cached, plain)) in b.iter().zip(direct.iter()).enumerate() {
            assert_eq!(cached.0.els.len(), plain.0.els.len());
            let cp = el_points_all(&cached.0.els, &cached.1);
            let pp = el_points_all(&plain.0.els, &plain.1);
            for (pi, (a, c)) in cp.iter().zip(pp.iter()).enumerate() {
                assert!(
                    (a.x - c.x).abs() < 1e-4 && (a.y - c.y).abs() < 1e-4,
                    "字形 {gi} 点 {pi}：缓存展开 {a:?} 与直接展开 {c:?} 不一致\
                     （缓存不得改变几何；曾因此抓出位置未随字号缩放的 bug）"
                );
            }
        }
    }

    /// 把路径元素全部映射到最终坐标。
    fn el_points_all(els: &[PathEl], t: &Affine) -> Vec<Vec2> {
        els.iter()
            .flat_map(el_points)
            .map(|p| t.map_point(p))
            .collect()
    }

    /// 取出一个路径元素的控制点（用于几何比较，忽略表示差异）。
    fn el_points(el: &PathEl) -> Vec<Vec2> {
        match *el {
            PathEl::MoveTo(p) | PathEl::LineTo(p) => vec![p],
            PathEl::QuadTo(c, p) => vec![c, p],
            PathEl::CubicTo(a, b, p) => vec![a, b, p],
            PathEl::Close => vec![],
        }
    }

    #[test]
    fn text_cache_separates_quantized_sizes() {
        let mut cache = TextCache::new();
        let mk = |size: f32| {
            let mut r = TextRun::new(size);
            r.push_glyph(1, 0.0, 0.0);
            r
        };
        cache.expand(&mk(20.0), |_| Some(bar_glyph()));
        cache.expand(&mk(40.0), |_| Some(bar_glyph()));
        assert_eq!(cache.misses(), 2, "不同字号必须是不同缓存项");
        // 同一量化桶（20.0 与 20.01 → 1/16 粒度）应视为同一项
        cache.expand(&mk(20.0), |_| Some(bar_glyph()));
        assert_eq!(cache.misses(), 2, "同字号复现不应新增缓存项");
        assert_eq!(cache.len(), 2);
    }
