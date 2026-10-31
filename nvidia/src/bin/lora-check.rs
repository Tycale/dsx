//! CUDA adapter checks: FP64 oracle, bitwise fused/unfused agreement and timing.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    use dsx::gpu::Gpu;
    #[derive(Parser)]
    struct Args {
        /// Optionally validate the downloaded adapter with the production parser.
        adapter: Option<std::path::PathBuf>,
        /// Run paired CUDA-event timings after the numerical checks.
        #[arg(long)]
        bench: bool,
    }
    let args = Args::parse();
    let g = Gpu::new(0)?;
    for (k, d, batches) in [(8192, 5120, &[1, 7, 256, 257, 1024, 4096][..]), (259, 513, &[1, 3][..]), (17, 31, &[1, 7][..])] {
        check(&g, k, d, batches)?;
    }
    if args.bench {
        bench(&g)?;
    }
    // Validate the exact downloaded adapter using the same production parser.
    if let Some(dir) = args.adapter {
        let a = dsx_core::lora::Adapter::load(&dir, 40, 5120, 8192, 1.)?;
        anyhow::ensure!(a.layers.len() == 38, "expected 38 layers");
        println!("PASS downloaded adapter: {} layers", a.layers.len());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn check(g: &dsx::gpu::Gpu, k: usize, d: usize, batches: &[usize]) -> anyhow::Result<()> {
    let a: Vec<f32> = (0..k).map(|i| ((i * 13 % 107) as f32 - 53.) / 107.).collect();
    let b: Vec<f32> = (0..d).map(|i| ((i * 7 % 97) as f32 - 48.) / 193.).collect();
    let adapter = dsx_core::lora::RankOne { a: a.clone(), b: b.clone() };
    for &n in batches {
        for scale in [0., 0.5, 1., -1.] {
            let plain = dsx::lora::RankOne::new(g, &adapter, scale, n)?.with_fused(false);
            let x: Vec<f32> = (0..n * k).map(|i| ((i * 19 % 113) as f32 - 56.) / 113.).collect();
            // Guard rows detect writes beyond the selected output range (prefill row offset).
            let y: Vec<f32> = (0..(n + 2) * d).map(|i| ((i * 11 % 89) as f32 - 44.) / 89.).collect();
            let gx = g.upload(&x)?;
            let gy = g.upload(&y)?;
            plain.apply(g, gx.ptr, gy.ptr + (d * 4) as u64, n)?;
            let got: Vec<f32> = g.download(gy.ptr, y.len())?;
            let partitions: &[u32] = if n == 1 { &[1, 2, 4, 8, 20] } else { &[4] };
            for &blocks in partitions {
                let fused = dsx::lora::RankOne::new(g, &adapter, scale, n)?.with_fused(true).with_fused_blocks(blocks)?;
                let gf = g.upload(&y)?;
                fused.apply(g, gx.ptr, gf.ptr + (d * 4) as u64, n)?;
                let fused_got: Vec<f32> = g.download(gf.ptr, y.len())?;
                anyhow::ensure!(
                    got.iter().zip(&fused_got).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "fused/unfused bitwise mismatch k={k} d={d} n={n} scale={scale} blocks={blocks}"
                );
            }
            anyhow::ensure!(got[..d] == y[..d] && got[(n + 1) * d..] == y[(n + 1) * d..], "guard overwritten");
            let mut err = 0f64;
            for t in 0..n {
                let dot: f64 = (0..k).map(|j| a[j] as f64 * x[t * k + j] as f64).sum();
                for j in 0..d {
                    let p = (t + 1) * d + j;
                    let expected = y[p] as f64 + scale as f64 * b[j] as f64 * dot;
                    err = err.max((got[p] as f64 - expected).abs());
                    anyhow::ensure!(
                        got[p].is_finite() && (got[p] as f64 - expected).abs() <= 2e-5 * (1. + expected.abs()),
                        "CUDA/CPU mismatch n={n} scale={scale} row={t} col={j}"
                    );
                }
            }
            if scale == 0. {
                anyhow::ensure!(got == y, "scale zero is not identity");
            }
            println!("PASS k={k} d={d} n={n} scale={scale} partitions={partitions:?} bitwise_equal=true max_abs_error={err:.9}");
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bench(g: &dsx::gpu::Gpu) -> anyhow::Result<()> {
    use serde_json::json;
    let (k, d) = (8192, 5120);
    let adapter = dsx_core::lora::RankOne {
        a: vec![0.001; k],
        b: vec![0.002; d],
    };
    let n = 1;
    for blocks in [1, 2, 4, 8, 20] {
        let plain = dsx::lora::RankOne::new(g, &adapter, 1., n)?.with_fused(false);
        let fused = dsx::lora::RankOne::new(g, &adapter, 1., n)?.with_fused(true).with_fused_blocks(blocks)?;
        let x = g.upload(&vec![0.01f32; n * k])?;
        let y = g.alloc_zeroed(n * d * 4)?;
        for _ in 0..20 {
            plain.apply(g, x.ptr, y.ptr, n)?;
            fused.apply(g, x.ptr, y.ptr, n)?;
        }
        g.sync()?;
        let iterations = 4000;
        let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
        let start = g.timed_event()?;
        let end = g.timed_event()?;
        for round in 0..7 {
            // Alternate order to reduce warm-up / clock bias between variants.
            for index in [round % 2, 1 - round % 2] {
                let layer = if index == 0 { &plain } else { &fused };
                g.record(start, false)?;
                for _ in 0..iterations {
                    layer.apply(g, x.ptr, y.ptr, n)?;
                }
                g.record(end, false)?;
                g.event_sync(end)?;
                let us = g.elapsed_ms(start, end)? as f64 * 1000. / iterations as f64;
                samples[index].push(us);
            }
        }
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "{}",
            json!({"event":"lora_bench","n":n,"k":k,"d":d,"blocks":blocks,"iterations":iterations,"rounds":7,
                "unfused_median_us":samples[0][3],"fused_median_us":samples[1][3],
                "speedup":samples[0][3]/samples[1][3],"unfused_samples_us":samples[0],"fused_samples_us":samples[1]})
        );
    }
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    panic!("requires cuda feature");
}
