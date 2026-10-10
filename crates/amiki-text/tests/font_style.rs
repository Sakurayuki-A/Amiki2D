//! 字体样式属性（0.10.7）：**先测出真实值，再断言**。
use amiki_text::{Font, FontStyle};

fn font(name: &str) -> Option<Font> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/fonts")
        .join(name);
    Font::from_file(p.to_string_lossy().as_ref()).ok()
}

/// 从随包字体读出真实样式：实测两者都是 400 / 非斜体。
#[test]
fn reads_style_metadata_from_shipped_fonts() {
    let mut seen = 0;
    for name in ["Inter-Regular.ttf", "NotoSansSC-Regular.ttf"] {
        let Some(f) = font(name) else {
            eprintln!("{name} 不存在，跳过");
            continue;
        };
        seen += 1;
        let s = f.style();
        eprintln!("{name}: weight={} italic={}", s.weight, s.italic);
        assert_eq!(s.weight, 400, "{name} 应声明 400");
        assert!(!s.italic, "{name} 不应声明斜体");
        assert_eq!(f.weight(), s.weight, "便捷方法与 style() 必须一致");
        assert_eq!(f.is_italic(), s.italic);
    }
    assert!(seen > 0, "至少要能读到一个随包字体");
}

/// `distance` 的排序语义：命中为 0；**斜体不匹配比字重差更严重**。
#[test]
fn distance_ordering_matches_the_documented_tradeoff() {
    let want = FontStyle::ITALIC;
    let semi_bold_italic = FontStyle { weight: 600, italic: true };
    let exact_weight_upright = FontStyle { weight: 400, italic: false };
    assert_eq!(FontStyle::ITALIC.distance(want), 0);
    assert_eq!(semi_bold_italic.distance(want), 200);
    assert_eq!(exact_weight_upright.distance(want), 1000);
    assert!(semi_bold_italic.is_closer_than(exact_weight_upright, want));
    assert!(FontStyle::REGULAR.is_closer_than(FontStyle::BOLD, FontStyle::REGULAR));
    eprintln!("distance：命中 0 / 斜体命中重 200 / 斜体不匹配 1000；排序符合文档取舍");
}

/// 极端值：同等斜体性下的最大字重差（800）仍小于斜体惩罚（1000）。
///
/// 我第一版把候选自身的斜体惩罚漏算了（断言 500 < 1000）—— 实测打回，改成现在这条。
#[test]
fn weight_gap_never_outweighs_italic_mismatch() {
    let heaviest = FontStyle { weight: 900, italic: true };
    let lightest = FontStyle { weight: 100, italic: true };
    assert_eq!(heaviest.distance(lightest), 800, "900 对 100 ⇒ 800（字重差上限）");
    let want = FontStyle::ITALIC;
    assert!(
        heaviest.distance(want) < FontStyle::REGULAR.distance(want),
        "斜体 + 最大字重差 应优于 等重但非斜体"
    );
    eprintln!(
        "极端值：字重差上限 800 < 斜体惩罚 1000 ⇒ 斜体（哪怕 900）优于等重正体（{} vs {}）",
        heaviest.distance(want),
        FontStyle::REGULAR.distance(want)
    );
}
