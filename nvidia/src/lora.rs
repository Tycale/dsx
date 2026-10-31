//! CUDA rank-1 delta: y += scale * B * (A * x), applied before mHC post.
use crate::gpu::{Arg, Buf, Gpu, Kernel};
use anyhow::Result;

pub struct RankOne {
    a: Buf,
    b: Buf,
    dot: Buf,
    dot_kernel: Kernel,
    add_kernel: Kernel,
    fused_kernel: Kernel,
    fused: bool,
    fused_blocks: u32,
    rows: usize,
    cols: usize,
    max_batch: usize,
    scale: f32,
}

impl RankOne {
    pub fn new(g: &Gpu, a: &dsx_core::lora::RankOne, scale: f32, max_batch: usize) -> Result<Self> {
        anyhow::ensure!(max_batch > 0 && scale.is_finite(), "invalid LoRA allocation");
        let fused = std::env::var("DSX_LORA_FUSED").is_ok_and(|v| v == "1");
        let fused_blocks = if fused {
            std::env::var("DSX_LORA_FUSED_BLOCKS").map_or(Ok(4), |v| v.parse::<u32>())?
        } else {
            4
        };
        anyhow::ensure!((1..=32).contains(&fused_blocks), "DSX_LORA_FUSED_BLOCKS must be in 1..=32");
        Ok(Self {
            a: g.upload(&a.a)?,
            b: g.upload(&a.b)?,
            dot: g.alloc_zeroed(max_batch * 4)?,
            dot_kernel: g.kernel("lora_rank1_dot")?,
            add_kernel: g.kernel("lora_rank1_add")?,
            fused_kernel: g.kernel("lora_rank1_fused")?,
            fused,
            fused_blocks,
            rows: a.b.len(),
            cols: a.a.len(),
            max_batch,
            scale,
        })
    }
    /// Override the startup selection for numerical checks and paired benchmarks.
    pub fn with_fused(mut self, enabled: bool) -> Self {
        self.fused = enabled;
        self
    }
    /// Select the number of identical dot reductions / output partitions for checks.
    pub fn with_fused_blocks(mut self, blocks: u32) -> Result<Self> {
        anyhow::ensure!((1..=32).contains(&blocks), "LoRA fused blocks must be in 1..=32");
        self.fused_blocks = blocks;
        Ok(self)
    }
    pub fn apply(&self, g: &Gpu, x: u64, y: u64, n: usize) -> Result<()> {
        anyhow::ensure!(n > 0 && n <= self.max_batch, "LoRA batch out of range");
        if self.scale == 0.0 {
            return Ok(());
        }
        // Keep the original path for prefill: fewer launches need not offset the
        // redundant dot reductions or the reduced output-store parallelism.
        if self.fused && n == 1 {
            return g.launch(
                self.fused_kernel,
                (self.fused_blocks, 1, 1),
                256,
                0,
                &[
                    Arg::P(self.a.ptr),
                    Arg::P(self.b.ptr),
                    Arg::P(x),
                    Arg::P(y),
                    Arg::I(self.cols as i32),
                    Arg::I(self.rows as i32),
                    Arg::F(self.scale),
                ],
            );
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
