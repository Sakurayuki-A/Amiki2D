//! layer_damage（从 `lib.rs` 分出，未改逻辑）。


use crate::*;

    fn layered(dx: f32, sigma: f32) -> Scene {
        let mut s = Scene::new(200, 160);
        s.push_layer(
            Layer {
                bounds: LayerBounds::new(40, 30, 60, 60),
                scene: Scene::new(200, 160),
                filter: Filter::Blur { sigma },
                opacity: 1.0,
                blend: BlendMode::Normal,
            },
            |sc| {
                let mut pb = PathBuilder::new();
                pb.rect(50.0 + dx, 40.0, 90.0 + dx, 80.0);
                sc.push(Affine::IDENTITY, pb.build(), Paint::Solid(Color::rgb8(255, 0, 0)));
            },
        );
        s
    }

    /// **层内容变化必须产生脏区。**
    ///
    /// 这条曾经是缺失的：`edit_damage` 只遍历 `draws`，而层内绘制不在根 draw 列表里，
    /// 于是层内改动产生**零脏区** —— 增量渲染永远看不到它，画面停在旧内容上，
    /// 而且不报错。这是典型的静默错误。
    #[test]
    fn layer_content_change_dirties_its_inflated_region() {
        let a = layered(0.0, 4.0);
        let b = layered(20.0, 4.0);
        let d = edit_damage(&a, &b);
        assert!(!d.is_full(), "层内改动不应退化为全脏（那说明没算，只是兜底）");
        let (tx0, ty0, tx1, ty1) = d.tiles().expect("应有脏区");
        // 作用范围 = bounds(40,30,60,60) 外扩 3σ=12 → (28,18,112,102)
        // 转成 tile（16px）：tx 1..7，ty 1..7
        assert_eq!(
            (tx0, ty0, tx1, ty1),
            (1, 1, 7, 7),
            "脏区应等于层的作用范围（bounds 外扩 3σ）"
        );
    }

    /// 层内容未变 ⇒ 不产生脏区（否则"静态界面"就拿不到收益）。
    #[test]
    fn unchanged_layer_dirties_nothing() {
        let a = layered(0.0, 4.0);
        let d = edit_damage(&a, &a.clone());
        assert!(d.tiles().is_some(), "全清空也是有效的脏区表达");
        let mut probe = Damage::new(200, 160);
        probe.clear();
        assert_eq!(
            d.tiles(),
            probe.tiles(),
            "层与图元都没变时不应标脏（这是静态界面零开销的前提）"
        );
    }

    /// 只有 sigma 变化（层内场景相同）也要标脏 —— 滤镜改了，作用范围与结果都变。
    #[test]
    fn filter_change_dirties_layer() {
        let a = layered(0.0, 2.0);
        let b = layered(0.0, 9.0);
        let d = edit_damage(&a, &b);
        assert!(!d.is_full(), "滤镜变化不应退化为全脏");
        let (tx0, ty0, tx1, ty1) = d.tiles().expect("应有脏区");
        // 3σ 从 6 涨到 27 → 作用范围 (13,3,127,117) → tile 0..8
        assert_eq!((tx0, ty0), (0, 0), "应覆盖更大的外扩范围");
        assert_eq!((tx1, ty1), (8, 8), "应覆盖更大的外扩范围");
    }

    /// 层被删除 ⇒ 旧层的作用范围要标脏（否则旧画面的层会永远残留）。
    #[test]
    fn removed_layer_dirties_old_region() {
        let a = layered(0.0, 4.0);
        let mut b = Scene::new(200, 160);
        b.draws = a.draws.clone(); // 图元相同，只少了层
        let d = edit_damage(&a, &b);
        assert!(!d.is_full(), "删除层不应退化为全脏");
        assert_eq!(
            d.tiles(),
            Some((1, 1, 7, 7)),
            "应标脏被删层的作用范围"
        );
    }

