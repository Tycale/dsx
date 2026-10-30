//! CUDA rank-1 delta: y += scale * B * (A * x), applied before mHC post.
use crate::gpu::{Arg, Buf, Gpu, Kernel};
use anyhow::Result;

pub struct RankOne {
    a: Buf,
    b: Buf,
    dot: Buf,
    dot_kernel: Kernel,
    add_kernel: Kernel,
    rows: usize,
    cols: usize,
    max_batch: usize,
    scale: f32,
}

impl RankOne {
    pub fn new(g: &Gpu, a: &dsx_core::lora::RankOne, scale: f32, max_batch: usize) -> Result<Self> {
        anyhow::ensure!(max_batch > 0 && scale.is_finite(), "invalid LoRA allocation");
        Ok(Self {
            a: g.upload(&a.a)?,
            b: g.upload(&a.b)?,
            dot: g.alloc_zeroed(max_batch * 4)?,
            dot_kernel: g.kernel("lora_rank1_dot")?,
            add_kernel: g.kernel("lora_rank1_add")?,
            rows: a.b.len(),
            cols: a.a.len(),
            max_batch,
            scale,
        })
    }
    pub fn apply(&self, g: &Gpu, x: u64, y: u64, n: usize) -> Result<()> {
        anyhow::ensure!(n > 0 && n <= self.max_batch, "LoRA batch out of range");
        if self.scale == 0.0 {
            return Ok(());
        }
        g.launch(
            self.dot_kernel,
            (n as u32, 1, 1),
            256,
            0,
            &[Arg::P(self.a.ptr), Arg::P(x), Arg::P(self.dot.ptr), Arg::I(self.cols as i32)],
        )?;
        g.launch(
            self.add_kernel,
            ((self.rows as u32).div_ceil(256), n as u32, 1),
            256,
            0,
            &[Arg::P(self.b.ptr), Arg::P(self.dot.ptr), Arg::P(y), Arg::I(self.rows as i32), Arg::F(self.scale)],
        )
    }
}
