//! M4：**帧间持久化** —— 把 damage region 从"证明正确"变成"真的省掉工作"。
//!
//! # 为什么需要它
//!
//! 前面的 `bin` + `render` 每次调用都从零开始：重新压平、重新分箱、重新着色全部 tile。
//! 于是 damage region 虽然已经被证明正确，却**没有省下任何工作**。
//!
//! 本模块持有跨帧状态（上一帧的场景、上一帧的已渲染像素、上一帧的分箱结果），
//! 因此可以：
//!
//! - **场景未变** ⇒ 一次着色都不做，直接复用上一帧像素（静态画面零开销）；
//! - **只改了一小块** ⇒ 只重算被改动区域覆盖的 tile；
//!
//! # 正确性依据
//!
//! 单个 tile 的着色只依赖**该 tile 的成员段与画家序**，与其他 tile 的当前内容无关，
//! 也与"这是第几帧"无关（见 `shade_tile` 的说明）。因此未脏 tile 保留上一帧结果
//! 与重新计算得到的是**同一结果**。
//!
//! # 复用分箱的前提
//!
//! 分箱结果（`BinnedScene`）依赖整个场景，不能局部更新，所以只在**场景无变化**时
//! 复用它。这是保守但正确的选择：一旦有 draw 变化就整体重新分箱。
//! （局部增量分箱是后续优化，需要 strip/tile 表支持就地更新。）

use amiki_scene::{edit_damage, Damage, Scene};

use crate::{bin, clear_tile, render, render_damage, shade_tile, BinnedScene};

/// 跨帧持久的 CPU 光栅器。
///
/// 用法：`let mut r = PersistentRenderer::new(0.05);` 然后每帧
/// `r.update(&scene)`，再读 `r.image()`。
#[derive(Debug)]
pub struct PersistentRenderer {
    flatten_eps: f32,
    width: u32,
    height: u32,
    /// 上一帧的场景（用于推导脏区）
    prev_scene: Option<Scene>,
    /// 上一帧的像素（增量渲染的基底，也是对外输出）
    image: Vec<u8>,
    /// 上一帧的分箱结果（仅当场景未变时复用）
    prev_binned: Option<BinnedScene>,
    /// 最近一次 update 的统计
    last: FrameStats,
}

/// 一次 `update` 的统计（用于验证"真的省了工作"，也给调用方做诊断）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// 是否完全跳过（场景未变 → 零着色）
    pub skipped: bool,
    /// 本帧是否重新分箱
    pub rebinned: bool,
    /// 本帧着色的 tile 数
    pub shaded_tiles: u32,
    /// 画布总 tile 数
    pub total_tiles: u32,
}

impl FrameStats {
    /// 省下的着色工作量比例（0.0 = 全画，1.0 = 一点没画）。
    pub fn saved_ratio(&self) -> f64 {
        if self.total_tiles == 0 {
            return 1.0;
        }
        1.0 - (self.shaded_tiles as f64 / self.total_tiles as f64)
    }
}

impl PersistentRenderer {
    /// 创建持久光栅器。`flatten_eps` 为压平误差界（设备像素）。
    pub fn new(flatten_eps: f32) -> PersistentRenderer {
        PersistentRenderer {
            flatten_eps,
            width: 0,
            height: 0,
            prev_scene: None,
            image: Vec::new(),
            prev_binned: None,
            last: FrameStats::default(),
        }
    }

    /// 当前像素（未预乘 RGBA8），尺寸为 `width * height * 4`。
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// 最近一次 `update` 的统计。
    pub fn last_stats(&self) -> FrameStats {
        self.last
    }

    /// 提交一帧场景，按需重算。
    ///
    /// - 画布尺寸变化 ⇒ 全量重建；
    /// - 场景与上一帧完全相同 ⇒ **零着色**；
    /// - 否则 ⇒ 重新分箱 + 只重算脏 tile。
    pub fn update(&mut self, scene: &Scene) {
        let w = scene.width.max(1);
        let h = scene.height.max(1);
        let total = w.div_ceil(16) * h.div_ceil(16);

        // 尺寸变化：整幅重建（分箱、像素全部作废）
        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            let binned = bin(scene, self.flatten_eps);
            self.image = render(&binned);
            self.prev_binned = Some(binned);
            self.prev_scene = Some(scene.clone());
            self.last = FrameStats {
                skipped: false,
                rebinned: true,
                shaded_tiles: total,
                total_tiles: total,
            };
            return;
        }

        // 场景未变：一次着色都不做
        if self.prev_scene.as_ref() == Some(scene) {
            self.last = FrameStats {
                skipped: true,
                rebinned: false,
                shaded_tiles: 0,
                total_tiles: total,
            };
            return;
        }

        // 变化：重新分箱，但只重算脏 tile
        let damage = match &self.prev_scene {
            Some(prev) => edit_damage(prev, scene),
            None => Damage::new(w, h),
        };
        let binned = bin(scene, self.flatten_eps);
        let shaded = match damage.tiles() {
            // 全脏
            None => {
                let n = binned.tiles_x * binned.tiles_y;
                for ty in 0..binned.tiles_y as usize {
                    for tx in 0..binned.tiles_x as usize {
                        clear_tile(&binned, &mut self.image, tx, ty);
                        shade_tile(&binned, &mut self.image, tx, ty);
                    }
                }
                n
            }
            Some((tx0, ty0, tx1, ty1)) => {
                let mut n = 0;
                for ty in ty0.min(binned.tiles_y)..ty1.min(binned.tiles_y) {
                    for tx in tx0.min(binned.tiles_x)..tx1.min(binned.tiles_x) {
                        let (tx, ty) = (tx as usize, ty as usize);
                        clear_tile(&binned, &mut self.image, tx, ty);
                        shade_tile(&binned, &mut self.image, tx, ty);
                        n += 1;
                    }
                }
                n
            }
        };
        self.prev_binned = Some(binned);
        self.prev_scene = Some(scene.clone());
        self.last = FrameStats {
            skipped: false,
            rebinned: true,
            shaded_tiles: shaded,
            total_tiles: total,
        };
    }

    /// 强制全量重建（例如外部直接改了像素、或要消除任何累积误差疑虑时）。
    pub fn invalidate_all(&mut self) {
        self.prev_scene = None;
        self.prev_binned = None;
        self.image.clear();
        self.width = 0;
        self.height = 0;
    }

    /// 上一帧的分箱结果（诊断用）。
    pub fn binned(&self) -> Option<&BinnedScene> {
        self.prev_binned.as_ref()
    }

    /// 用已渲染的上一帧像素 + 脏区更新的等价入口，供测试对照
    /// （与 `update` 的内部路径保持一致，避免测试与实际实现漂移）。
    pub fn apply_damage_for_test(&mut self, scene: &Scene) {
        let binned = bin(scene, self.flatten_eps);
        let damage = match &self.prev_scene {
            Some(prev) => edit_damage(prev, scene),
            None => Damage::new(scene.width, scene.height),
        };
        render_damage(&binned, &mut self.image, &damage);
    }
}
