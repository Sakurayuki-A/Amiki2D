//! 读回：GPU 纹理 → 主机内存。
//!
//! 从 `lib.rs` 拆出。这一簇是**同步点**所在：M3.6 实测读回曾是主机侧瓶颈
//! （1080p×10k 时"读回 1" 2.35 ms），因此这里做了**缓冲池化**与
//! `map_and_take` 的同步处理。单独成文件便于对照"哪一步在等 GPU"。

use crate::*;

impl GpuRenderer {
    /// 取一个可用的 MAP_READ 暂存缓冲：优先复用池中容量足够的，否则新建并**精确**定容。
    ///
    /// M3.6：读回是墙钟瓶颈（实测读回 1 + 读回 2 占主机侧绝大部分），原实现每次读回都
    /// `create_buffer` + 冷映射，代价被重复支付。池化后同一尺寸的读回复用同一缓冲。
    ///
    /// 从池中取用的缓冲必须处于**未映射**状态（`read_buffer` 负责用完即 unmap）。
    fn take_readback_buffer(&self, byte_size: usize) -> wgpu::Buffer {
        let want = byte_size.max(4) as u64;
        if let Ok(mut pool) = self.readback_pool.lock() {
            if let Some(i) = pool.iter().position(|b| b.size() == want) {
                return pool.swap_remove(i);
            }
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("amiki_readback_pool"),
            size: want,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// 把缓冲归还池中（仅在确认已 unmap 后调用）。
    fn give_readback_buffer(&self, buf: wgpu::Buffer) {
        if let Ok(mut pool) = self.readback_pool.lock() {
            if pool.len() < 8 {
                pool.push(buf);
            }
        }
    }

    /// 映射 + 等待 + 拷贝出 `byte_size` 字节，然后 unmap 并归还缓冲。
    ///
    /// 失败路径**不归还**缓冲：若 `map_async` 失败则缓冲会保持已映射状态，
    /// 再次映射会报错，故直接丢弃（池会按需补建）。
    fn map_and_take(
        &self,
        out: wgpu::Buffer,
        byte_size: usize,
    ) -> Result<Vec<u8>, RenderError> {
        let (tx, rx) = std::sync::mpsc::channel();
        let slice = out.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let polled = self.device.poll(wgpu::PollType::Wait);
        let mapped = rx
            .recv()
            .map_err(|_| RenderError("map 回调通道关闭".into()))
            .and_then(|r| r.map_err(|e| RenderError(format!("map 失败: {e}"))));
        // 先把 poll 的错误报出去（保持与原实现一致的错误优先级）
        polled.map_err(|e| RenderError(format!("poll 失败: {e}")))?;
        mapped?;
        let v = {
            let data = slice.get_mapped_range();
            let v = data[..byte_size.min(data.len())].to_vec();
            drop(data);
            v
        };
        out.unmap();
        self.give_readback_buffer(out);
        Ok(v)
    }

    /// 读回 storage buffer 内容（通用小数据读回）。
    /// 诊断：读回 storage buffer。
    pub fn read_buffer(&self, buf: &wgpu::Buffer, byte_size: usize) -> Result<Vec<u8>, RenderError> {
        let out = self.take_readback_buffer(byte_size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_buf_copy") });
        encoder.copy_buffer_to_buffer(buf, 0, &out, 0, byte_size as u64);
        self.queue.submit(Some(encoder.finish()));
        self.map_and_take(out, byte_size)
    }

    /// 读回纹理像素（直线 RGBA，**f32 且不裁剪**）。
    ///
    /// `tex` 必须是 `Rgba16Float` —— 引擎的渲染输出与层/模糊中间缓冲都是这个格式。
    /// 不在这里降到 8 位：**HDR 的值必须原样回到主机侧**，否则层合成与最终量化
    /// 都拿不到 >1.0 的信息。要 8 位见 [`Self::readback_rgba8`]。
    ///
    /// M3.6：暂存缓冲取自本模块的 `take_readback_buffer` 池（同一画布尺寸逐帧复用）。
    #[allow(dead_code)]
    pub fn readback_f32(&self, tex: &wgpu::Texture) -> Result<Vec<[f32; 4]>, RenderError> {
        let raw = self.read_raw(tex, 8)?;
        let n = (tex.width() * tex.height()) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * 8;
            out.push([
                f16_to_f32(u16::from_le_bytes([raw[o], raw[o + 1]])),
                f16_to_f32(u16::from_le_bytes([raw[o + 2], raw[o + 3]])),
                f16_to_f32(u16::from_le_bytes([raw[o + 4], raw[o + 5]])),
                f16_to_f32(u16::from_le_bytes([raw[o + 6], raw[o + 7]])),
            ]);
        }
        Ok(out)
    }

    /// 读回并降到 **8 位直线 RGBA**。给"只要一张能看的图"的调用方与测试用。
    ///
    /// `clamp` 在这里发生 —— 那是 HDR 降到显示范围的**最后一步**。
    #[allow(dead_code)]
    pub fn readback_rgba8(&self, tex: &wgpu::Texture) -> Result<Vec<u8>, RenderError> {
        Ok(rgba8_from_f32(&self.readback_f32(tex)?))
    }

    /// 读回纹理像素（直线 RGBA8）。仅用于 8 位纹理（用户位图、覆盖图等）。
    ///
    /// M3.6：暂存缓冲取自本模块的 `take_readback_buffer` 池（同一画布尺寸逐帧复用）。
    pub fn readback(&self, tex: &wgpu::Texture) -> Result<Vec<u8>, RenderError> {
        self.read_raw(tex, 4)
    }

    /// 把纹理读成裸字节（每像素 `bpp` 字节），按行去掉 padding。
    fn read_raw(&self, tex: &wgpu::Texture, bpp: usize) -> Result<Vec<u8>, RenderError> {
        let w = tex.width();
        let h = tex.height();
        let bytes_per_row = (w as usize * bpp).div_ceil(256) * 256;
        let buf_size = bytes_per_row * h as usize;
        let buf = self.take_readback_buffer(buf_size);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("amiki_copy") });
        encoder.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let polled = self.device.poll(wgpu::PollType::Wait);
        let mapped = rx
            .recv()
            .map_err(|_| RenderError("map 回调通道关闭".into()))
            .and_then(|r| r.map_err(|e| RenderError(format!("map 失败: {e}"))));
        // 失败路径不归还缓冲（可能仍处于已映射状态）
        polled.map_err(|e| RenderError(format!("poll 失败: {e}")))?;
        mapped?;

        let mut out = Vec::with_capacity(w as usize * h as usize * bpp);
        {
            let data = slice.get_mapped_range();
            for row in 0..h as usize {
                let start = row * bytes_per_row;
                out.extend_from_slice(&data[start..start + w as usize * bpp]);
            }
        }
        buf.unmap();
        self.give_readback_buffer(buf);
        Ok(out)
    }
}

/// IEEE 754 binary16 → f32。
///
/// # 为什么需要它
///
/// HDR 要求渲染输出能存住 >1.0 的值，而 8 位格式在写入时就裁到 1.0，
/// 所以输出格式要换成 `Rgba16Float` —— 回读时硬件给的是 f16，必须自己解。
///
/// 不引 `half` crate：整个项目对第三方依赖很克制，而这段是标准算法、
/// 有单元测试守着（含次正规数、±0、±∞、NaN）。
///
/// 这个函数**当前还没有调用方** —— 它是为"把渲染输出换成 `Rgba16Float`"
/// 准备的。那件事牵动 GPU 的模糊上传/回读格式与主机侧合成（约 100 行），
/// 见 `docs/KNOWN_ISSUES.md` 的 B12。
#[allow(dead_code)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let man = h & 0x3FF;
    let v = if exp == 0 {
        // 次正规数（含 ±0）：没有隐含的前导 1，指数固定 2^-14
        (man as f32) * 2.0f32.powi(-24)
    } else if exp == 0x1F {
        if man == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + (man as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -v } else { v }
}

/// f32 → IEEE 754 binary16 位模式（`f16_to_f32` 的逆）。
///
/// # 为什么需要它
///
/// 层/模糊的中间纹理是 `Rgba16Float`（HDR 要求 >1.0 存得住），而
/// `queue.write_texture` 上传时要的是**目标格式的字节** —— 所以 f32 的像素
/// 必须先编码成 f16。
///
/// 用 round-to-nearest-even，与硬件一致；溢出给 ±Inf，下溢给 ±0。
/// 正确性由 `f16_round_trips_f32` 与 `f16_rounds_like_the_spec` 两条测试守着。
pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32;
    let man = x & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf / NaN（保留 NaN 的非零尾数位，避免把 NaN 变成 Inf）
        return sign | 0x7C00 | if man != 0 { 0x0200 } else { 0 };
    }

    let e = exp - 112; // -127 + 15
    if e >= 0x1F {
        return sign | 0x7C00; // 溢出 → Inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // 远小于最小次正规数 → ±0
        }
        // 次正规：补上隐含的前导 1，再右移到目标位置
        let m = (man | 0x0080_0000) >> (1 - e);
        let rounded = (m + 0x1000 - 1 + ((m >> 13) & 1)) >> 13;
        return sign | (rounded as u16 & 0x03FF);
    }

    // 正规：加上舍入偏置（0x0FFF + round-to-even 的粘位）后截断低 13 位
    let m = man + 0x0FFF + ((man >> 13) & 1);
    let mut e16 = e as u16;
    let mut m16 = (m >> 13) as u16;
    if m16 & 0x0400 != 0 {
        // 尾数进位 → 指数 +1
        m16 = 0;
        e16 += 1;
        if e16 >= 0x1F {
            return sign | 0x7C00;
        }
    }
    sign | (e16 << 10) | (m16 & 0x03FF)
}

/// f32 直线 RGBA → 8 位直线 RGBA。
///
/// **这里是 HDR 降到显示范围的地方**，也是唯一该做 `clamp` 的地方。
/// 用 `+ 0.5` 四舍五入，与 CPU 光栅器的量化方式一致。
///
/// 与 [`f16_to_f32`] 同样是"输出换 `Rgba16Float`"的准备工作，当前无调用方。
#[allow(dead_code)]
pub fn rgba8_from_f32(px: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(px.len() * 4);
    for p in px {
        for c in p {
            out.push((c * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn f16_decodes_the_textbook_values() {
        // 这些位模式是 IEEE 754 binary16 的定义值，不是"跑出来的结果"
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // 最大正规数
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14)); // 最小正规数
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24)); // 最小次正规数
    }

    #[test]
    fn f16_covers_the_range_the_renderer_writes() {
        // 引擎写出的颜色值落在 [0, 几十] 这个量级。f16 有 10 位尾数，
        // 相对误差约 2^-11，对这些量级足够。
        for &v in &[0.0f32, 0.001, 0.25, 1.0, 4.0, 16.0, 100.0] {
            let back = f16_to_f32(f32_to_f16_bits(v));
            let rel = if v == 0.0 { 0.0 } else { ((back - v) / v).abs() };
            assert!(rel < 1e-3, "f16 往返 {v} → {back}，相对误差 {rel}");
        }
    }

    /// 仅测试用：f32 → binary16 位模式（用于验证解码）。
    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let man = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 { sign } else if exp >= 31 { sign | 0x7C00 } else { sign | ((exp as u16) << 10) | man }
    }
    #[test]
    fn f16_round_trips_f32() {
        // 编码再解码 —— 只要编码器与解码器互逆，就说明位级处理是对的。
        for &v in &[
            0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, 4.0, 16.0, 100.0, 65504.0, 2.0f32.powi(-14),
            0.001, 0.3333, 1e-4,
        ] {
            let back = super::f16_to_f32(super::f32_to_f16(v));
            if v == 0.0 {
                assert_eq!(back, v);
                continue;
            }
            let rel = ((back - v) / v).abs();
            assert!(rel < 2e-3, "往返 {v} → {back}，相对误差 {rel}");
        }
    }

    #[test]
    fn f16_rounds_like_the_spec() {
        // 逐位对照定义值：确认编码不是"能自洽"而是"真的符合 binary16"
        assert_eq!(super::f32_to_f16(0.0), 0x0000);
        assert_eq!(super::f32_to_f16(-0.0), 0x8000);
        assert_eq!(super::f32_to_f16(1.0), 0x3C00);
        assert_eq!(super::f32_to_f16(-1.0), 0xBC00);
        assert_eq!(super::f32_to_f16(2.0), 0x4000);
        assert_eq!(super::f32_to_f16(0.5), 0x3800);
        assert_eq!(super::f32_to_f16(65504.0), 0x7BFF);
        assert_eq!(super::f32_to_f16(2.0f32.powi(-14)), 0x0400);
        assert_eq!(super::f32_to_f16(2.0f32.powi(-24)), 0x0001);
        // 溢出 → Inf；太小的正数 → +0
        assert_eq!(super::f32_to_f16(1e30), 0x7C00);
        assert_eq!(super::f32_to_f16(1e-30), 0x0000);
        assert_eq!(super::f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(super::f32_to_f16(f32::NEG_INFINITY), 0xFC00);
        assert!(super::f16_to_f32(super::f32_to_f16(f32::NAN)).is_nan());
        // 4.0 是 HDR 场景里的典型值
        assert_eq!(super::f32_to_f16(4.0), 0x4400);
    }
}
