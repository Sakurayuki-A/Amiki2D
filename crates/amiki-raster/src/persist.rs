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
//! - **含层场景** ⇒ 层缓冲跨帧复用，内容没变的层**不重新渲染、不重新滤波**。
//!
//! # 三个缓冲的分工（含层场景的关键）
//!
//! ```text
//! base   仅图元（不含任何层）
//! image  base + 按序合成各层 = 对外输出
//! layers 每层**已滤波**的缓冲（跨帧缓存）
//! ```
//!
//! 脏 tile 的更新是：`image[tile] ← base[tile]`，再按**原顺序**把覆盖该 tile 的
//! 层逐层合成上去。因为 `render()` 全量路径也是"图元 → 逐层合成"，
//! 两条路径的结果必然一致。
//!
//! # 曾经的静默错误（值得记住）
//!
//! 本模块早先只维护一个 `image`：首帧走全量（正确），**第二帧起增量分支只清/着色
//! tile，从不合成层** —— 于是层整个消失，而且不报错。同时 `edit_damage` 只遍历
//! `draws`，**层内改动产生零脏区**。两个洞合起来的效果是"有滤镜的界面在第二帧
//! 之后悄悄丢掉滤镜"。
//!
//! 现在：`edit_damage` 会标脏层的作用范围；本模块拆出 `base`，
//! 并用 `amiki-raster` 的 `layer_buffer_from_content` / `blend_layer_rect`
//! 让全量与增量**共用同一份数学**。
//!
//! # 正确性依据
//!
//! 单个 tile 的着色只依赖**该 tile 的成员段与画家序**，与其他 tile 的当前内容无关，
//! 也与"这是第几帧"无关（见 `shade_tile` 的说明）。层的合成只依赖
//! **该层的缓冲内容**与它覆盖的像素，同样与帧号无关。因此未脏 tile 保留上一帧
//! 结果与重新计算得到的是**同一结果**。
//!
//! # 复用分箱的前提
//!
//! 分箱结果（`BinnedScene`）依赖整个场景，不能局部更新，所以只在**场景无变化**时
//! 复用它。这是保守但正确的选择：一旦有 draw 变化就整体重新分箱。
//! （局部增量分箱是后续优化，需要 strip/tile 表支持就地更新。）

use amiki_scene::{edit_damage, BlendMode, Damage, Filter, Layer, LayerBounds, Scene};

use crate::filter::LayerBuffer;
use crate::{
    bin, blend_layer_rect, clear_tile, layer_buffer_from_content_f32, render_inner_f32, render_tiles, shade_tile, BinnedScene,
};

/// tile 边长（与分箱、damage 一致；三处必须相同）。
const TILE_SIZE: u32 = amiki_scene::DAMAGE_TILE;

/// 跨帧持久化的 CPU 光栅器。
///
/// 用法：`let mut r = PersistentRenderer::new(0.05);` 然后每帧
/// `r.update(&scene)`，再读 `r.image()`。
#[derive(Debug)]
pub struct PersistentRenderer {
    flatten_eps: f32,
    /// B4：mip 采样偏移（与全量路径的 `BinnedScene::mip_bias` 同语义；
    /// damage 契约要求增量与全量逐字节一致，故增量路径也必须用同一 bias）。
    ///
    /// **可以直接改**：`update` 会与上一帧实际用过的值比较，一旦不同就整幅重建。
    /// 缓存的像素与层缓冲都是按旧 bias 渲染的，只改字段而不重建会**静默**留下
    /// 旧画面 —— 所以不需要、也不应该要求调用方自己记得调 [`Self::invalidate_all`]。
    pub mip_bias: f32,
    /// 上一帧**实际用过**的 bias，用于检测调用方改了 [`Self::mip_bias`]。
    last_mip_bias: f32,
    width: u32,
    height: u32,
    /// 上一帧的场景（用于推导脏区）
    prev_scene: Option<Scene>,
    /// **仅图元**的像素（不含层）。层的合成都从它出发，因此可以反复合成而不叠加。
    base: Vec<u8>,
    /// 上一帧的像素（= `base` + 层，对外输出）
    image: Vec<u8>,
    /// 每层**已滤波**的缓冲，跨帧复用
    layers: Vec<CachedLayer>,
    /// 上一帧的分箱结果（仅当场景未变时复用）
    prev_binned: Option<BinnedScene>,
    /// 最近一次 update 的统计
    last: FrameStats,
}

/// 缓存的一层。
///
/// `key` 是**内容键**（范围 / 层内场景 / 滤镜）—— 只有它变了才需要重新渲染与滤波。
/// `opacity` / `blend` 是合成参数：改了它们**不必重新滤波**，但仍需在覆盖范围内
/// 重新合成。
#[derive(Debug, Clone)]
struct CachedLayer {
    bounds: LayerBounds,
    scene: Scene,
    filter: Filter,
    /// 已滤波的缓冲（直线色）
    buf: LayerBuffer,
    /// 缓冲左上角在设备空间的坐标
    x: i32,
    y: i32,
    opacity: f32,
    blend: BlendMode,
}

impl CachedLayer {
    /// 内容键是否与给定层一致（决定能否复用缓冲）。
    fn content_matches(&self, layer: &Layer) -> bool {
        self.bounds == layer.bounds && self.scene == layer.scene && self.filter == layer.filter
    }

    /// 合成参数是否一致（决定能否跳过重新合成）。
    fn composite_matches(&self, layer: &Layer) -> bool {
        self.opacity == layer.opacity && self.blend == layer.blend
    }

    /// 该层覆盖的 tile 范围（闭开区间）。
    fn tile_range(&self, tiles_x: usize, tiles_y: usize) -> (usize, usize, usize, usize) {
        let b = self.bounds.inflate_by_filter(self.filter);
        rect_tile_range(b.x, b.y, b.width, b.height, tiles_x, tiles_y)
    }
}

/// 一次 `update` 的统计（用于验证"真的省了工作"，也给调用方做诊断）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// 是否完全跳过（场景未变 → 零着色）
    pub skipped: bool,
    /// 本帧是否重新分箱
    pub rebinned: bool,
    /// 本帧**着色图元**的 tile 数
    pub shaded_tiles: u32,
    /// 本帧**重新合成层**的 tile 数（无层时为 0）
    pub composited_tiles: u32,
    /// 画布总 tile 数
    pub total_tiles: u32,
}

impl FrameStats {
    /// 省下的着色工作量比例（0.0 = 全画，1.0 = 一点没画）。
    ///
    /// 以"图元着色 + 层合成"里**做过工作的 tile 数**为准 —— 含层场景里
    /// 合成往往是主要开销，只看 `shaded_tiles` 会高估收益。
    pub fn saved_ratio(&self) -> f64 {
        if self.total_tiles == 0 {
            return 1.0;
        }
        let work = self.shaded_tiles.max(self.composited_tiles) as f64;
        1.0 - (work / self.total_tiles as f64)
    }
}

/// 把 `(x, y, w, h)` 的设备矩形转成 tile 的闭开范围，并裁剪到画布内。
fn rect_tile_range(
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    tiles_x: usize,
    tiles_y: usize,
) -> (usize, usize, usize, usize) {
    let ts = TILE_SIZE as i32;
    let x1 = x + w as i32;
    let y1 = y + h as i32;
    let tx0 = x.div_euclid(ts).max(0) as usize;
    let ty0 = y.div_euclid(ts).max(0) as usize;
    let tx1 = (x1.div_euclid(ts) + 1).max(0) as usize;
    let ty1 = (y1.div_euclid(ts) + 1).max(0) as usize;
    (
        tx0.min(tiles_x),
        ty0.min(tiles_y),
        tx1.min(tiles_x),
        ty1.min(tiles_y),
    )
}

impl PersistentRenderer {
    /// 创建持久光栅器。`flatten_eps` 为压平误差界（设备像素）。
    pub fn new(flatten_eps: f32) -> PersistentRenderer {
        PersistentRenderer {
            flatten_eps,
            mip_bias: 0.0,
            last_mip_bias: 0.0,
            width: 0,
            height: 0,
            prev_scene: None,
            base: Vec::new(),
            image: Vec::new(),
            layers: Vec::new(),
            prev_binned: None,
            last: FrameStats::default(),
        }
    }

    /// 当前像素（未预乘 RGBA8），尺寸为 `width * height * 4`。
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// 仅图元的像素（诊断用：便于把"图元错了"与"层合成错了"分开）。
    pub fn base_image(&self) -> &[u8] {
        &self.base
    }

    /// 最近一次 `update` 的统计。
    pub fn last_stats(&self) -> FrameStats {
        self.last
    }

    /// 缓存的层数（诊断用）。
    pub fn cached_layer_count(&self) -> usize {
        self.layers.len()
    }

    /// 提交一帧场景，按需重算。
    ///
    /// - 画布尺寸变化 ⇒ 全量重建；
    /// - [`Self::mip_bias`] 与上一帧不同 ⇒ 全量重建（缓存是按旧 bias 渲的）；
    /// - 场景与上一帧完全相同 ⇒ **零工作**；
    /// - 否则 ⇒ 重新分箱 + 只重算脏 tile（含层的作用范围）。
    pub fn update(&mut self, scene: &Scene) {
        let w = scene.width.max(1);
        let h = scene.height.max(1);
        let total = w.div_ceil(TILE_SIZE) * h.div_ceil(TILE_SIZE);

        // 尺寸变化、或 bias 变化：整幅重建（分箱、像素、层缓存全部作废）
        //
        // bias 必须在这里判：缓存的像素与层缓冲都是按**旧 bias** 渲染的，
        // 只改字段而不重建就会让"增量"与"全量"不一致（静默的旧画面）。
        if w != self.width || h != self.height || self.mip_bias != self.last_mip_bias {
            self.width = w;
            self.height = h;
            let mut binned = bin(scene, self.flatten_eps);
            // 主画布的像素也必须用调用方的 bias。此前只有层内容用到了它
            // （经 `render_inner_f32` 的入参），主画布一直是 0 —— 于是含位图的
            // 场景下增量与全量静默不一致。
            binned.mip_bias = self.mip_bias;
            self.base = render_tiles(&binned, None);
            self.layers.clear();
            let (tx_n, ty_n) = (binned.tiles_x as usize, binned.tiles_y as usize);
            let mut all = vec![true; tx_n * ty_n];
            self.sync_layers(scene, &mut all, tx_n, ty_n);
            self.recomposite_all();
            self.prev_binned = Some(binned);
            self.prev_scene = Some(scene.clone());
            self.last_mip_bias = self.mip_bias;
            self.last = FrameStats {
                skipped: false,
                rebinned: true,
                shaded_tiles: total,
                composited_tiles: total,
                total_tiles: total,
            };
            return;
        }

        // 场景未变：一次工作都不做
        if self.prev_scene.as_ref() == Some(scene) {
            self.last = FrameStats {
                skipped: true,
                rebinned: false,
                shaded_tiles: 0,
                composited_tiles: 0,
                total_tiles: total,
            };
            return;
        }

        let damage = match &self.prev_scene {
            Some(prev) => edit_damage(prev, scene),
            None => Damage::new(w, h),
        };
        let mut binned = bin(scene, self.flatten_eps);
        binned.mip_bias = self.mip_bias; // 主画布与层内容必须同一个 bias
        let (tx_n, ty_n) = (binned.tiles_x as usize, binned.tiles_y as usize);
        let mut dirty = vec![false; tx_n * ty_n];

        // ① 图元的脏 tile
        match damage.tiles() {
            // 全脏
            None => dirty.fill(true),
            Some((tx0, ty0, tx1, ty1)) => {
                for ty in (ty0 as usize).min(ty_n)..(ty1 as usize).min(ty_n) {
                    for tx in (tx0 as usize).min(tx_n)..(tx1 as usize).min(tx_n) {
                        dirty[ty * tx_n + tx] = true;
                    }
                }
            }
        }

        // ② 图元：只重算脏 tile（写进 `base`）
        let mut shaded = 0u32;
        for ty in 0..ty_n {
            for tx in 0..tx_n {
                if dirty[ty * tx_n + tx] {
                    clear_tile(&binned, &mut self.base, tx, ty);
                    shade_tile(&binned, &mut self.base, tx, ty);
                    shaded += 1;
                }
            }
        }

        // ③ 层：内容变了就重建缓冲；**变了或合成参数变了**就并入脏区
        self.sync_layers(scene, &mut dirty, tx_n, ty_n);

        // ④ 重新合成脏 tile
        let mut composited = 0u32;
        for ty in 0..ty_n {
            for tx in 0..tx_n {
                if dirty[ty * tx_n + tx] {
                    self.recomposite_tile(tx, ty);
                    composited += 1;
                }
            }
        }

        self.prev_binned = Some(binned);
        self.prev_scene = Some(scene.clone());
        self.last = FrameStats {
            skipped: false,
            rebinned: true,
            shaded_tiles: shaded,
            composited_tiles: composited,
            total_tiles: total,
        };
    }

    /// 对齐层缓存：能复用的复用，不能的重新渲染 + 滤波；并把需要重新合成的范围并入脏区。
    fn sync_layers(&mut self, scene: &Scene, dirty: &mut [bool], tx_n: usize, ty_n: usize) {
        // 先把旧缓存取出来独占，便于按内容键配对（**不能按下标配对**：
        // 层被插入/删除/重排后下标会错位，那会复用错层的缓冲）。
        let mut old = std::mem::take(&mut self.layers);
        let mut next: Vec<CachedLayer> = Vec::with_capacity(scene.layers.len());

        for layer in &scene.layers {
            let pos = old.iter().position(|c| c.content_matches(layer));
            let (cached, needs_marks) = match pos {
                Some(p) => {
                    let mut c = old.remove(p);
                    // 内容没变、合成参数也没变 ⇒ 这一层完全没动，不必重新合成
                    let unchanged = c.composite_matches(layer);
                    c.opacity = layer.opacity;
                    c.blend = layer.blend;
                    (c, !unchanged)
                }
                None => {
                    // 重建该层
                    // **必须与全量路径（composite_layer）用同一条 f32 路径**：
                    // 增量若还用 8 位，两条路会在 HDR 值上漂移 ——
                    // layer_incremental_matches_full_random_sequences 立刻就会红
                    // （改这一行之前它确实红了）。
                    let layer_img =
                        render_inner_f32(self.width, self.height, &layer.scene, self.flatten_eps, self.mip_bias);
                    let new = layer_buffer_from_content_f32(&layer_img, self.width, self.height, layer)
                        .map(|(x, y, buf)| CachedLayer {
                            bounds: layer.bounds,
                            scene: layer.scene.clone(),
                            filter: layer.filter,
                            buf,
                            x,
                            y,
                            opacity: layer.opacity,
                            blend: layer.blend,
                        });
                    match new {
                        Some(c) => (c, true),
                        // 层与画布不相交：不缓存，但**仍要标脏它的范围**
                        // （上一步它可能还与画布相交，现在移出去了要擦掉）
                        None => {
                            mark_layer_range(dirty, tx_n, ty_n, layer);
                            continue;
                        }
                    }
                }
            };
            if needs_marks {
                mark_layer_range(dirty, tx_n, ty_n, layer);
            }
            next.push(cached);
        }

        // 剩下的旧缓存对应**已被删除的层**：它们原先覆盖的范围要重新合成（擦掉）
        for gone in &old {
            let (tx0, ty0, tx1, ty1) = gone.tile_range(tx_n, ty_n);
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    dirty[ty * tx_n + tx] = true;
                }
            }
        }

        self.layers = next;
    }

    /// 用 `base` + 全部层重算**整个**图像（层顺序与 `render()` 一致）。
    fn recomposite_all(&mut self) {
        self.image.clear();
        self.image.extend_from_slice(&self.base);
        for c in &self.layers {
            blend_layer_rect(
                &mut self.image,
                self.width,
                self.height,
                &c.buf,
                c.x,
                c.y,
                c.opacity,
                c.blend.id(),
                (0, 0, self.width, self.height),
            );
        }
    }

    /// 重算单个 tile：从 `base` 复位，再按序把覆盖它的层合成上去。
    fn recomposite_tile(&mut self, tx: usize, ty: usize) {
        let w = self.width as usize;
        let ts = TILE_SIZE as usize;
        let x0 = tx * ts;
        let y0 = ty * ts;
        let x1 = (x0 + ts).min(self.width as usize);
        let y1 = (y0 + ts).min(self.height as usize);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        // ① 复位为"仅图元"
        for y in y0..y1 {
            let o = (y * w + x0) * 4;
            let n = (x1 - x0) * 4;
            self.image[o..o + n].copy_from_slice(&self.base[o..o + n]);
        }
        // ② 按序合成覆盖该 tile 的层
        //
        // 顺序必须与 `render()` 一致（层依次叠加），否则重叠层的结果会不同。
        for c in &self.layers {
            blend_layer_rect(
                &mut self.image,
                self.width,
                self.height,
                &c.buf,
                c.x,
                c.y,
                c.opacity,
                c.blend.id(),
                (x0 as u32, y0 as u32, x1 as u32, y1 as u32),
            );
        }
    }

    /// 强制全量重建（例如外部直接改了像素、或要消除任何累积误差疑虑时）。
    pub fn invalidate_all(&mut self) {
        self.prev_scene = None;
        self.prev_binned = None;
        self.layers.clear();
        self.base.clear();
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
        let mut binned = bin(scene, self.flatten_eps);
        // 与 `update` 的内部路径保持一致（含 bias）—— 否则测试与实际实现会漂移，
        // 而这条测试的全部价值就在于"两条路径逐字节相同"。
        binned.mip_bias = self.mip_bias;
        let damage = match &self.prev_scene {
            Some(prev) => edit_damage(prev, scene),
            None => Damage::new(scene.width, scene.height),
        };
        crate::render_damage(&binned, &mut self.image, &damage);
    }
}

/// 把一层的作用范围（经滤镜）并入脏 tile 集合。
fn mark_layer_range(dirty: &mut [bool], tx_n: usize, ty_n: usize, layer: &Layer) {
    let b = layer.inflated_bounds();
    if b.is_empty() {
        return;
    }
    let (tx0, ty0, tx1, ty1) = rect_tile_range(b.x, b.y, b.width, b.height, tx_n, ty_n);
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            dirty[ty * tx_n + tx] = true;
        }
    }
}
