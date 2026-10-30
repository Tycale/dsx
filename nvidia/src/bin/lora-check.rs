//! Standalone numerical check: actual CUDA adapter kernels vs FP64 CPU reference.
#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use dsx::gpu::Gpu;
    let g = Gpu::new(0)?;
    let (k, d) = (8192, 5120);
    let a: Vec<f32> = (0..k).map(|i| ((i * 13 % 107) as f32 - 53.) / 107.).collect();
    let b: Vec<f32> = (0..d).map(|i| ((i * 7 % 97) as f32 - 48.) / 193.).collect();
    let adapter = dsx_core::lora::RankOne { a: a.clone(), b: b.clone() };
    for n in [1, 7, 256, 257, 1024] {
        for scale in [0., 0.5, 1.] {
            let l = dsx::lora::RankOne::new(&g, &adapter, scale, n)?;
            let x: Vec<f32> = (0..n * k).map(|i| ((i * 19 % 113) as f32 - 56.) / 113.).collect();
            // Guard rows detect writes beyond the selected output range (prefill row offset).
            let y: Vec<f32> = (0..(n + 2) * d).map(|i| ((i * 11 % 89) as f32 - 44.) / 89.).collect();
            let gx = g.upload(&x)?;
            let gy = g.upload(&y)?;
            l.apply(&g, gx.ptr, gy.ptr + (d * 4) as u64, n)?;
            let got: Vec<f32> = g.download(gy.ptr, y.len())?;
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
            println!("PASS n={n} scale={scale} max_abs_error={err:.9}");
        }
    }
    // Validate the exact downloaded adapter using the same production parser.
    if let Some(dir) = std::env::args().nth(1) {
        let a = dsx_core::lora::Adapter::load(std::path::Path::new(&dir), 40, 5120, 8192, 1.)?;
        anyhow::ensure!(a.layers.len() == 38, "expected 38 layers");
        println!("PASS downloaded adapter: {} layers", a.layers.len());
    }
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    panic!("requires cuda feature");
}
