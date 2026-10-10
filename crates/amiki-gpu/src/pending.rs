//! 异步帧：`PendingFrame` / `PollStatus` 与 begin 入口。
//!
//! 从 `lib.rs` 分出（只搬代码，未改逻辑）。

use crate::*;

/// 一帧的推进状态（P3 第 ② 步）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PollStatus {
    Pending,
    Ready,
}

/// **在飞的一帧**（P3 第 ② 步）：`begin` 之后它已在 GPU 上跑，`poll` 不阻塞。
///
/// 语义（写死在文档里，C ABI 也照此）：
/// - `poll()` 非阻塞、可反复调用（`Polling` 单次实测 0.071 ms）；
/// - `try_read()` **只取值、不推进**：回调还没跑过就返回 `None`，句柄留在原地，
///   调用方可以继续 `poll` —— 所以它**既不返回半成品，也不 panic**
///   （未映射的缓冲上取像素是 wgpu 明确会 panic 的用法）；
/// - **层场景直接同步算完**（层合成需要根图像，异步没有收益）⇒ `begin` 后第一次
///   `poll` 即 `Ready`。这是**有意**的，不是遗漏；
/// - CPU 后端不走这里（它同步），所以"首帧即就绪"对调用方是统一语义。
pub struct PendingFrame {
    gpu: std::sync::Arc<GpuRenderer>,
    /// `Some` = 已算完（层场景，或推进中出错）
    ready: Option<Result<Vec<u8>, RenderError>>,
    /// `Some` = 还在飞：在飞回读 + 最终要用的色调映射
    pending: Option<(readback::PendingRead, amiki_scene::ToneMap)>,
}

impl GpuRenderer {
    /// 提交一帧，**不等待**（等价 `begin_with(..., ToneMap::Raw, 0.0)`）。
    pub fn begin(
        self: &std::sync::Arc<Self>,
        scene: &Scene,
        eps: f32,
    ) -> Result<PendingFrame, RenderError> {
        self.begin_with(scene, eps, amiki_scene::ToneMap::Raw, 0.0)
    }

    /// 提交一帧，**不等待**；返回在飞句柄，何时 `poll` / `try_read` 由调用方决定。
    pub fn begin_with(
        self: &std::sync::Arc<Self>,
        scene: &Scene,
        eps: f32,
        tone: amiki_scene::ToneMap,
        mip_bias: f32,
    ) -> Result<PendingFrame, RenderError> {
        if !scene.layers.is_empty() {
            // 层合成需要**根图像**（每层要合成回它）⇒ 异步在这里没有收益，
            // 直接同步算完并放进 `ready`。文档已写明，不是静默降级。
            return Ok(PendingFrame {
                gpu: std::sync::Arc::clone(self),
                ready: Some(self.render_blocking_with(scene, eps, tone, mip_bias)),
                pending: None,
            });
        }
        let frame = self.prepare_with(scene, eps, mip_bias)?;
        let tex = self.create_output_texture(&frame.scene);
        self.run_pipeline(&frame.scene, &tex, false, &frame.clip_views, None)?;
        // 拷贝 + map_async 都已提交；这里**不等**。
        let (p, _) = self.begin_read_raw(&tex, 8)?;
        Ok(PendingFrame {
            gpu: std::sync::Arc::clone(self),
            ready: None,
            pending: Some((p, tone)),
        })
    }
}

impl PendingFrame {
    /// 非阻塞推进一次。
    pub fn poll(&mut self) -> PollStatus {
        if self.ready.is_some() {
            return PollStatus::Ready;
        }
        let Some((p, _)) = self.pending.as_mut() else {
            return PollStatus::Ready;
        };
        match self.gpu.poll_read_raw(p) {
            Ok(true) => PollStatus::Ready,
            Ok(false) => PollStatus::Pending,
            Err(e) => {
                // 出错也转 Ready，让 try_read 把错误原样报出去（不吞错）
                self.ready = Some(Err(e));
                self.pending = None;
                PollStatus::Ready
            }
        }
    }

    /// **就绪则取走**（`&mut self`）：未就绪返回 `None`，且**这一帧什么都不动**
    /// （还能继续 `poll`、再取）。
    ///
    /// 判据是 `device.poll` **有没有把 map 回调跑过**（`PendingRead::map_outcome`）：
    /// 未映射的缓冲上取像素是 wgpu 明确会 panic 的用法，所以"先问再取"不是防御性
    /// 代码，而是**唯一正确**的顺序。
    ///
    /// C ABI 的三入口走这条 —— 它需要"取不到不算数"（未就绪 ⇒ `AMIKI_ERROR_BUSY`，
    /// 帧还在，调用方可以接着问）。
    pub fn take_ready(&mut self) -> Option<Result<Vec<u8>, RenderError>> {
        if let Some(r) = self.ready.take() {
            return Some(r);
        }
        let (mut p, tone) = self.pending.take()?;
        match p.map_outcome() {
            Some(Ok(())) => {
                Some(self.gpu.finish_read_f32(p).map(|px| rgba8_from_f32_with(&px, tone)))
            }
            Some(Err(msg)) => Some(Err(RenderError(msg))),
            None => {
                // 还在飞：原样放回，**不消费**这次提交
                self.pending = Some((p, tone));
                None
            }
        }
    }

    /// 取值（**消费句柄**）：就绪则取走；未就绪返回 `None`，句柄连同在飞的回读
    /// 一起被丢弃 —— 调用方已经放弃它了。想要"取不到还能接着问"用 [`Self::take_ready`]。
    pub fn try_read(mut self) -> Option<Result<Vec<u8>, RenderError>> {
        self.take_ready()
    }
}

