//! M3.5：**渲染输出快照**——挡住"参照与被测同时被改错"的静默漂移。
//!
//! # 为什么需要它
//!
//! 现有对拍分两类：
//! - GPU vs CPU（`amiki-gpu` 的 `gpu_matches_cpu` 等）：验证两个**实现**一致；
//! - 稀疏 vs 稠密（`sparse_matches_dense`）：验证两条**管线**一致。
//!
//! 两类都是**差分**测试：若 `fine_math.wgsl` 与 `amiki-raster` 的同一处数学被改错，
//! 或稀疏与稠密的同一处判定一起改错，两者仍然"一致"——测试通过，画面却错了。
//! 本文件用固定 hash 钉住输出字节，任何非预期的像素变化都会失败。
//!
//! # 怎么用
//!
//! hash 不匹配时测试会打印每个场景的实测 hash。**若这次变化是有意为之**：
//! 把打印值抄进下面的 `SNAPSHOTS` 并在 `docs/RENDERING_STRATEGY.md` 记一笔
//! （有意变更必须留痕）；若不清楚为何变化，那就是 bug，别更新 hash。
//!
//! 快照取 **CPU 参考引擎**（`amiki-raster`）：它是零依赖的纯 Rust 路径，不受 GPU
//! 驱动/wgpu 版本影响，最适合当"字节级锚点"。

use amiki_math::{Affine, Vec2};
use amiki_raster::{bin, render};
use amiki_scene::{BlendMode, Color, Paint, PathBuilder, Scene};

/// 每场景的期望 hash（FNV-1a 64 位，按 RGBA 字节序）。
const SNAPSHOTS: &[(&str, u64)] = &[
    ("bg_rect", 0xb21c_d2d7_13f1_8f25),
    ("circle_aa", 0x9fce_2a0b_473e_bc35),
    ("donut_hole", 0x8b92_130a_e0e2_be30),
    ("pentagram_nonzero", 0x7a15_eccf_3367_cc25),
    ("offscreen_clip", 0xa81b_4b2a_2285_e77e),
    ("blend16", 0x9571_60e3_0404_487f),
    ("rotated_stack", 0xa95d_aabe_402b_8a99),
    ("empty_canvas", 0xb93a_0c83_ce3b_6325),
];

const EPS: f32 = 0.05;

/// FNV-1a 64 位：零依赖、确定性，足够捕捉"某个像素变了"。
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn solid(c: Color) -> Paint {
    Paint::Solid(c)
}

/// 8 个覆盖不同管线的锚点场景。
fn anchor_scenes() -> Vec<(&'static str, Scene)> {
    let mut out: Vec<(&'static str, Scene)> = Vec::new();

    {
        let mut s = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(6.0, 6.0, 58.0, 40.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(240, 220, 200)));
        out.push(("bg_rect", s));
    }
    {
        // 解析覆盖率 AA 锚点：圆的边缘 α 必须恒定
        let mut s = Scene::new(80, 80);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(40.0, 40.0), 28.0, false);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 128, 0)));
        out.push(("circle_aa", s));
    }
    {
        // 反绕挖洞（nonzero 绕数抵消）
        let mut s = Scene::new(96, 96);
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(48.0, 48.0), 40.0, false);
        pb.circle(Vec2::new(48.0, 48.0), 20.0, true);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(0, 200, 100)));
        out.push(("donut_hole", s));
    }
    {
        // 自相交：W=2 区域被 clamp 到 1（§1.2 已知近似，快照钉住其现状）
        let mut s = Scene::new(96, 96);
        let mut pb = PathBuilder::new();
        pb.pentagram(Vec2::new(48.0, 48.0), 40.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(200, 0, 200)));
        out.push(("pentagram_nonzero", s));
    }
    {
        // 越界 + 裁剪：tile 桶钳制与 clip 覆盖率纹理
        let mut s = Scene::new(72, 64);
        let mut pb = PathBuilder::new();
        pb.rect(-24.0, -12.0, 20.0, 30.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(230, 120, 60)));
        let mut pb = PathBuilder::new();
        pb.rect(40.0, 8.0, 140.0, 48.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(30, 160, 90)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 56.0, 56.0);
        let clip = s.push_clip(Affine::IDENTITY, pb.build());
        let mut pb = PathBuilder::new();
        pb.circle(Vec2::new(32.0, 32.0), 26.0, false);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(0, 120, 255)));
        s.draws.last_mut().unwrap().clips = vec![clip];
        out.push(("offscreen_clip", s));
    }
    {
        // 16 种 W3C 混合模式（含 HSL 非可分离四模式）
        let modes = [
            BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay,
            BlendMode::Darken, BlendMode::Lighten, BlendMode::ColorDodge, BlendMode::ColorBurn,
            BlendMode::HardLight, BlendMode::SoftLight, BlendMode::Difference, BlendMode::Exclusion,
            BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity,
        ];
        let mut s = Scene::new(64, 64);
        let mut pb = PathBuilder::new();
        pb.rect(0.0, 0.0, 64.0, 64.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(240, 220, 200)));
        let mut pb = PathBuilder::new();
        pb.rect(8.0, 8.0, 56.0, 56.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(40, 90, 160)));
        for (i, mode) in modes.iter().enumerate() {
            let mut pb = PathBuilder::new();
            let c = Vec2::new(16.0 + (i % 4) as f32 * 10.0, 16.0 + (i / 4) as f32 * 10.0);
            pb.circle(c, 14.0, false);
            s.push_blend(Affine::IDENTITY, pb.build(), solid(Color::rgba8(230, 120, 60, 200)), *mode);
        }
        out.push(("blend16", s));
    }
    {
        // 旋转叠加 + 半透明：画家序与 alpha 合成
        let mut s = Scene::new(100, 100);
        for i in 0..5 {
            let mut pb = PathBuilder::new();
            pb.rect(-30.0, -10.0, 30.0, 10.0);
            let t = Affine::translate(Vec2::new(50.0, 50.0))
                .then(Affine::rotate(0.3 * i as f32))
                .then(Affine::scale(1.0 + 0.1 * i as f32));
            s.push(t, pb.build(), solid(Color::rgba8(40 + i * 40, 120, 200 - i * 30, 180)));
        }
        out.push(("rotated_stack", s));
    }
    {
        // 空画布：完全出画布的内容不得产生任何像素（稀疏路径零分配路径的锚点）
        let mut s = Scene::new(32, 32);
        let mut pb = PathBuilder::new();
        pb.rect(100.0, 100.0, 120.0, 120.0);
        s.push(Affine::IDENTITY, pb.build(), solid(Color::rgb8(255, 0, 0)));
        out.push(("empty_canvas", s));
    }

    out
}

#[test]
fn render_output_matches_snapshot() {
    let scenes = anchor_scenes();
    assert_eq!(
        scenes.len(),
        SNAPSHOTS.len(),
        "锚点场景数与 SNAPSHOTS 长度不一致（新增场景请同步补 hash）"
    );

    let mut mismatches: Vec<String> = Vec::new();
    for (name, scene) in &scenes {
        let img = render(&bin(scene, EPS));
        let got = fnv1a64(&img);
        let want = SNAPSHOTS
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("SNAPSHOTS 缺少场景 {name}"))
            .1;
        if want == 0 {
            mismatches.push(format!("(\"{name}\", {got:#018x}),  // 首次生成"));
        } else if want != got {
            mismatches.push(format!("(\"{name}\", {got:#018x}),  // 期望 {want:#018x}"));
        }
    }

    if !mismatches.is_empty() {
        panic!(
            "渲染输出与快照不符（{} / {} 个场景）。\n\
             若该变化**是有意为之**：把下面各行抄回 SNAPSHOTS，并在 \
             docs/RENDERING_STRATEGY.md 记一笔；\n\
             若不清楚为何变化：那是 bug，不要更新快照，先查改动。\n\n{}",
            mismatches.len(),
            scenes.len(),
            mismatches.join("\n")
        );
    }
    eprintln!("输出快照 ✓ {} 个锚点场景逐字节一致", scenes.len());
}
