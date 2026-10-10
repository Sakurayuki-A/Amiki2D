//! 输出：同步渲染 + 异步三入口（0.10.9）。
//!
//! 从 `api_obj.rs` 分出（只搬代码，未改逻辑）。

use crate::api::ffi;
use crate::objects::*;
use crate::types::*;


/// CPU 后端渲染（带画布自己的色调映射与 mip bias）。
///
/// 同步入口与异步入口**共用这一份实现** —— 否则"两条入口逐字节相同"没有根据。
fn render_cpu_with_tone(
    scene: &amiki_scene::Scene,
    eps: f32,
    tone: amiki_scene::ToneMap,
    bias: f32,
) -> Vec<u8> {
    let mut binned = amiki_raster::bin(scene, eps);
    binned.mip_bias = bias;
    amiki_raster::render_with_tone(&binned, tone)
}

/// GPU 错误 → C 错误码。
///
/// 后端的"不支持"要单独报出来（B1：GL 遇到位图），让调用方能区分
/// "引擎坏了"与"这个后端做不了"。
fn render_error_to_i32(msg: &str) -> i32 {
    if msg.contains("不支持") || msg.contains("无法") {
        AMIKI_ERROR_NOT_SUPPORTED
    } else {
        AMIKI_ERROR_UNKNOWN
    }
}

#[no_mangle]
pub extern "C" fn amiki_canvas_render_rgba8(
    canvas: *mut AmikiCanvas,
    out: *mut u8,
    out_len: usize,
) -> i32 {
    ffi!(canvas, |c| {
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let need = (c.width as usize) * (c.height as usize) * 4;
        if out_len < need {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // 打开中的层没配对的 end —— 直接渲染会**静默丢掉**那些绘制。
        // 这正是本项目最不能接受的一类错误，所以显式拒绝。
        if !c.layers.is_empty() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let eps = c.eps;
        let scene = &c.root.scene;
        let ctx = c.ctx;
        if ctx.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: context 必须比 canvas 活得久（header 契约）
        let ctx = unsafe { &*ctx };
        if let Err(r) = ctx.poison.check() {
            return r;
        }
        // 色调映射作用在**出口量化**这一步，两条后端路径都要用上它；
        // mip bias（B4）同理，两条路径共用同一设置
        let tone = c.tone_map;
        let bias = c.mip_bias;
        let rendered: Result<Vec<u8>, String> = match &ctx.backend {
            #[cfg(feature = "gpu")]
            Backend::Gpu(r) => {
                r.render_blocking_with(scene, eps, tone, bias).map_err(|e| e.to_string())
            }
            // CPU 路径与 GPU 无关 —— 这也是 WASM 构建下唯一的路径
            Backend::Cpu => Ok(render_cpu_with_tone(scene, eps, tone, bias)),
        };
        match rendered {
            Ok(px) if px.len() >= need => {
                // SAFETY: need <= out_len，且 out 非空
                unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), out, need) };
                AMIKI_OK
            }
            Ok(_) => AMIKI_ERROR_UNKNOWN,
            Err(msg) => render_error_to_i32(&msg),
        }
    })
}

// ---- 输出（异步三入口，P3 第 ③b 步）----
//
// 同步入口 `amiki_canvas_render_rgba8` **一行都没变**（除了上面那条超范围回落）：
// 异步是**另加**的能力，不是替换。两条入口共用同一套前置判断与同一份 CPU 实现，
// 所以产物逐字节相同（`tests.rs` 里按字节比对）。

/// **提交一帧、立即返回**（P3 第 ③b 步）。
///
/// 之后用 `amiki_canvas_poll_render_rgba8` 问就绪、`amiki_canvas_try_read_render_rgba8` 取值。
/// 在飞帧是**提交那一刻场景的快照**：提交后再改画布不影响它。
///
/// - **重复 begin 是允许的**：上一个还没取走的帧被丢弃（它只是"一次还没算完的计算"）。
///   这是**有意**的 —— 否则调用方一旦不想等了就再无合法出路，只能再加一个 cancel 符号。
/// - 打开中的层没配对的 `end` ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`（与同步入口同一条理由：
///   直接渲染会静默丢掉那些绘制）。
/// - 提交阶段就失败的（准备 / 派发）**立刻**报错，不装作"在飞"再让 `try_read` 报。
#[no_mangle]
pub extern "C" fn amiki_canvas_begin_render_rgba8(canvas: *mut AmikiCanvas) -> i32 {
    ffi!(canvas, |c| {
        if !c.layers.is_empty() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let ctx = c.ctx;
        if ctx.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: context 必须比 canvas 活得久（header 契约）
        let ctx = unsafe { &*ctx };
        if let Err(r) = ctx.poison.check() {
            return r;
        }
        let (eps, tone, bias) = (c.eps, c.tone_map, c.mip_bias);
        let scene = &c.root.scene;
        let pending = match &ctx.backend {
            #[cfg(feature = "gpu")]
            Backend::Gpu(r) => match r.begin_with(scene, eps, tone, bias) {
                Ok(p) => PendingCanvasRender::InFlight(p),
                // 提交阶段的失败（准备 / 派发）是同步可见的 —— 立刻报出去，
                // 不让它装作"在飞"然后再由 try_read 报
                Err(e) => return render_error_to_i32(&e.to_string()),
            },
            Backend::Cpu => PendingCanvasRender::Ready(render_cpu_with_tone(scene, eps, tone, bias)),
        };
        c.pending = Some(pending);
        AMIKI_OK
    })
}

/// **非阻塞地问一次：在飞的那一帧算完了吗**（P3 第 ③b 步）。
///
/// 写入 `*out_ready`：非 0 = 可以 `try_read` 了；0 = 还没好。**0 不是错误** ——
/// 它是"过一会儿再问"，这正是异步渲染存在的理由（提交与读回分到不同帧）。
///
/// 失败时**不写** `*out_ready`（含"没 begin 就 poll" ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`：
/// 与 `restore` 多于 `save` 同类，是调用方的 bug，**不假装已就绪**）。
#[no_mangle]
pub extern "C" fn amiki_canvas_poll_render_rgba8(
    canvas: *mut AmikiCanvas,
    out_ready: *mut i32,
) -> i32 {
    ffi!(canvas, |c| {
        if out_ready.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let Some(p) = c.pending.as_mut() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        let ready = match p {
            PendingCanvasRender::Ready(_) => true,
            #[cfg(feature = "gpu")]
            PendingCanvasRender::InFlight(f) => f.poll() == amiki_gpu::PollStatus::Ready,
        };
        // SAFETY: 已判空
        unsafe { *out_ready = if ready { 1 } else { 0 } };
        AMIKI_OK
    })
}

/// **就绪则取值**：写入 `out`（长度至少 `width * height * 4`）并消费掉这一帧
/// （P3 第 ③b 步）。行序与 alpha 约定与同步入口完全相同。
///
/// - 还没算完 ⇒ `AMIKI_ERROR_BUSY`，**不写入任何字节**，且这一帧**留在原地**
///   （可以继续 `poll` / 再 `try_read`）；
/// - `out_len` 不足 ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`，且这一帧**不被消费**
///   （长度不够是调用方的 bug，不该顺手毁掉在飞的那一帧，也不写任何字节）；
/// - 没 begin 就调用 ⇒ `AMIKI_ERROR_INVALID_ARGUMENT`；
/// - 取值阶段的 GPU 错误 ⇒ 帧被消费，并按与同步入口**同一套规则**报出来
///   （"不支持 / 无法" ⇒ `AMIKI_ERROR_NOT_SUPPORTED`，其余 ⇒ `AMIKI_ERROR_UNKNOWN`）——
///   错误是终局的，要重来就重新 `begin`。
#[no_mangle]
pub extern "C" fn amiki_canvas_try_read_render_rgba8(
    canvas: *mut AmikiCanvas,
    out: *mut u8,
    out_len: usize,
) -> i32 {
    ffi!(canvas, |c| {
        if out.is_null() {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let need = (c.width as usize) * (c.height as usize) * 4;
        if out_len < need {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        }
        let Some(p) = c.pending.as_mut() else {
            return AMIKI_ERROR_INVALID_ARGUMENT;
        };
        // 未就绪时**提前返回**：这一帧一个字节都没动过，调用方可以继续问。
        let got: Result<Vec<u8>, i32> = match p {
            // `mem::take` 而不是克隆：CPU 分支的结果可能有好几 MB
            PendingCanvasRender::Ready(v) => Ok(std::mem::take(v)),
            #[cfg(feature = "gpu")]
            PendingCanvasRender::InFlight(f) => match f.take_ready() {
                Some(Ok(v)) => Ok(v),
                Some(Err(e)) => Err(render_error_to_i32(&e.to_string())),
                // `take_ready` 的承诺：未就绪时**什么都不动**（还能继续 poll）
                None => return AMIKI_ERROR_BUSY,
            },
        };
        c.pending = None;
        match got {
            Ok(px) if px.len() >= need => {
                // SAFETY: need <= out_len，且 out 非空
                unsafe { std::ptr::copy_nonoverlapping(px.as_ptr(), out, need) };
                AMIKI_OK
            }
            // CPU 分支的像素数由画布尺寸决定，长度不符只可能是引擎侧算错；
            // 报 UNKNOWN 而不是写出半张图
            Ok(_) => AMIKI_ERROR_UNKNOWN,
            Err(code) => code,
        }
    })
}

