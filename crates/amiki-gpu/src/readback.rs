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

    /// 读回纹理像素（直线 RGBA8）。
    ///
    /// M3.6：暂存缓冲取自 [`Self::take_readback_buffer`] 池（同一画布尺寸逐帧复用）。
    pub fn readback(&self, tex: &wgpu::Texture) -> Result<Vec<u8>, RenderError> {
        let w = tex.width();
        let h = tex.height();
        let bytes_per_row = ((w * 4) as usize).div_ceil(256) * 256;
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

        let mut out = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = slice.get_mapped_range();
            for row in 0..h as usize {
                let start = row * bytes_per_row;
                out.extend_from_slice(&data[start..start + (w as usize) * 4]);
            }
        }
        buf.unmap();
        self.give_readback_buffer(buf);
        Ok(out)
    }


}
