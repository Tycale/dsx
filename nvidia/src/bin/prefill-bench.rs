//! Full-prompt prefill and forced-continuation timing without reused KV prefixes.
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use dsx::{
    gpumodel::{GpuModel, Placement},
    model::Model,
};
use serde_json::json;
use std::{
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    rank: PathBuf,
    /// Comma- or whitespace-separated prompt token IDs.
    #[arg(long, conflicts_with = "text", required_unless_present = "text")]
    ids: Option<PathBuf>,
    /// Raw prompt text, encoded with the tokenizer embedded in the GGUF (no chat template).
    #[arg(long, conflicts_with = "ids")]
    text: Option<PathBuf>,
    /// Keep exactly this many prompt tokens; fail if the input is shorter.
    #[arg(long)]
    prompt_tokens: Option<usize>,
    /// Fixed continuation token IDs, fed to step() without sampling.
    #[arg(long, conflicts_with = "decode_text")]
    decode_ids: Option<PathBuf>,
    /// Raw continuation text, encoded separately with the embedded tokenizer.
    #[arg(long, conflicts_with = "decode_ids")]
    decode_text: Option<PathBuf>,
    /// Keep exactly this many forced continuation tokens; fail if the input is shorter.
    #[arg(long)]
    decode_tokens: Option<usize>,
    /// Linux/GGUF only: evict only continuation Engram pages after each prefill.
    /// Require verified nonresident pages before timing; model files remain read-only.
    #[arg(long)]
    cold_engram: bool,
    #[arg(long)]
    lora: Option<PathBuf>,
    #[arg(long, default_value_t = 1.0)]
    lora_scale: f32,
    #[arg(long, default_value_t = 262144)]
    ctx: usize,
    #[arg(long, default_value_t = 8192)]
    chunk: usize,
    #[arg(long, default_value_t = 110.)]
    arena_gb: f64,
    /// Main device ordinal; -1 chooses automatically. Pin this for controlled comparisons.
    #[arg(long, default_value_t = -1, allow_negative_numbers = true)]
    main_gpu: i32,
    #[arg(long, default_value_t=-1.5, allow_negative_numbers=true)]
    main_gb: f64,
    #[arg(long, default_value_t=-1., allow_negative_numbers=true)]
    aux_gb: f64,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Last round's complete final-prefill-token logits, little-endian F32.
    #[arg(long)]
    dump_logits: Option<PathBuf>,
    /// Last round's complete logits after each forced token, row-major little-endian F32.
    /// The decode_bench record gives the row count and vocabulary width.
    #[arg(long)]
    dump_decode_logits: Option<PathBuf>,
}

fn read_ids(path: &Path) -> Result<Vec<u32>> {
    std::fs::read_to_string(path)?
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(Into::into))
        .collect()
}

fn input_ids(ids: Option<&Path>, text: Option<&Path>, tok: Option<&Tokenizer>) -> Result<Vec<u32>> {
    if let Some(path) = ids {
        read_ids(path)
    } else if let Some(path) = text {
        let text = std::fs::read_to_string(path)?;
        Ok(tok
            .context("text input requires a tokenizer")?
            .encode(text, false)
            .map_err(|e| anyhow!("tokenization: {e}"))?
            .get_ids()
            .to_vec())
    } else {
        Ok(vec![])
    }
}

fn limit_ids(ids: &mut Vec<u32>, count: Option<usize>, name: &str) -> Result<()> {
    if let Some(count) = count {
        ensure!(count > 0 && ids.len() >= count, "{name} needs {count} tokens; input has {}", ids.len());
        ids.truncate(count);
    }
    Ok(())
}

fn fingerprint(bytes: impl Iterator<Item = u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn logit_fingerprint(logits: &[f32]) -> String {
    fingerprint(logits.iter().flat_map(|v| v.to_le_bytes()))
}

fn dump_rows<'a>(path: &Path, rows: impl IntoIterator<Item = &'a [f32]>) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for row in rows {
        for value in row {
            f.write_all(&value.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

#[cfg(target_os = "linux")]
mod cold_engram {
    use super::*;
    use dsx::model::{EngramTables, Source};
    use std::{collections::BTreeSet, ffi::c_void, fs::File, os::fd::AsRawFd};

    unsafe extern "C" {
        fn getpagesize() -> i32;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> i32;
    }
    const MADV_DONTNEED: i32 = 4;
    const POSIX_FADV_DONTNEED: i32 = 4;

    /// Borrowing the actual model mapping keeps every address valid through eviction.
    /// Ranges are page-aligned file offsets, entirely inside the Engram tensor.
    pub struct Pages<'a> {
        map: &'a memmap2::Mmap,
        file: File,
        ranges: Vec<(usize, usize)>,
        page_size: usize,
        page_count: usize,
    }

    impl<'a> Pages<'a> {
        pub fn new(m: &'a Model, prompt: &[u32], decode: &[u32]) -> Result<Self> {
            ensure!(!decode.is_empty(), "--cold-engram requires a forced continuation");
            let Source::Gguf(g) = &m.src else {
                anyhow::bail!("--cold-engram requires a GGUF model")
            };
            let tabs = m.engram.as_ref().context("--cold-engram requires an Engram table")?;
            let EngramTables::Table { mat, _file: None, .. } = tabs else {
                anyhow::bail!("--cold-engram requires the Engram table inside the model GGUF")
            };
            let t = g.get("per_layer_token_embd.weight")?;
            ensure!(mat.name == t.name, "Engram table does not match the GGUF tensor");
            let map = g.maps.get(t.shard).context("Engram shard mapping is missing")?;
            let tensor_start = usize::try_from(t.offset)?;
            let tensor_end = tensor_start.checked_add(usize::try_from(t.bytes)?).context("Engram tensor range overflow")?;
            ensure!(tensor_end <= map.len(), "Engram tensor exceeds its mapping");
            let page_size = unsafe { getpagesize() };
            ensure!(page_size > 0, "invalid system page size");
            let page_size = page_size as usize;
            let base = map.as_ptr() as usize;
            ensure!(base % page_size == 0, "GGUF mapping is not page-aligned");
            let mut hist = prompt
                .iter()
                .map(|&token| m.cfg.engram_token_map.get(token as usize).map(|&v| v as i64).context("token has no Engram mapping"))
                .collect::<Result<Vec<_>>>()?;
            let mut pages = BTreeSet::new();
            for &token in decode {
                hist.push(*m.cfg.engram_token_map.get(token as usize).context("token has no Engram mapping")? as i64);
                for li in 0..m.cfg.engram_layers.len() {
                    let tab = tabs.layer(li, m.cfg.engram_dim)?;
                    for row in m.engram_hashes_for(&hist, li) {
                        ensure!(row < m.cfg.engram_rows[li], "Engram hash is outside the table");
                        // Creating the row slice and reading its pointer/length does not fault in its contents.
                        let bytes = tab.row_bytes(usize::try_from(row)?);
                        let start = (bytes.as_ptr() as usize).checked_sub(base).context("Engram row is outside its mapping")?;
                        let end = start.checked_add(bytes.len()).context("Engram row range overflow")?;
                        ensure!(!bytes.is_empty() && start >= tensor_start && end <= tensor_end, "invalid Engram row bounds");
                        let first = start / page_size * page_size;
                        let last = (end - 1) / page_size * page_size;
                        let page_end = last.checked_add(page_size).context("Engram page range overflow")?;
                        // Refuse to evict a boundary page shared with another tensor rather than broaden the scope.
                        ensure!(
                            first >= tensor_start && page_end <= tensor_end,
                            "target Engram row shares a boundary page with another tensor"
                        );
                        pages.extend((first..page_end).step_by(page_size));
                    }
                }
            }
            ensure!(!pages.is_empty(), "no continuation Engram pages found");
            let page_count = pages.len();
            let mut ranges: Vec<(usize, usize)> = vec![];
            for offset in pages {
                if let Some(last) = ranges.last_mut() {
                    if last.0 + last.1 == offset {
                        last.1 += page_size;
                        continue;
                    }
                }
                ranges.push((offset, page_size));
            }
            // The source is opened read-only; neither syscall changes file contents.
            let file = File::open(g.paths.get(t.shard).context("Engram shard path is missing")?)?;
            ensure!(file.metadata()?.len() == map.len() as u64, "GGUF shard size changed since mapping");
            Ok(Self {
                map,
                file,
                ranges,
                page_size,
                page_count,
            })
        }

        fn resident_pages(&self) -> Result<usize> {
            let mut resident = 0;
            for &(offset, len) in &self.ranges {
                let mut residency = vec![0u8; len / self.page_size];
                // Constructor checked that each aligned range is contained in this live mapping.
                let addr = unsafe { self.map.as_ptr().add(offset) } as *mut c_void;
                ensure!(unsafe { mincore(addr, len, residency.as_mut_ptr()) } == 0, "mincore: {}", std::io::Error::last_os_error());
                resident += residency.iter().filter(|&&v| v & 1 != 0).count();
            }
            Ok(resident)
        }

        pub fn evict(&self, round: usize) -> Result<usize> {
            let before = self.resident_pages()?;
            // Prefill and its scoped prefetch threads have finished. Drop this process's PTEs first:
            // fadvise alone cannot reliably reclaim pages that remain mapped into its resident set.
            for &(offset, len) in &self.ranges {
                let addr = unsafe { self.map.as_ptr().add(offset) } as *mut c_void;
                ensure!(unsafe { madvise(addr, len, MADV_DONTNEED) } == 0, "madvise DONTNEED: {}", std::io::Error::last_os_error());
            }
            for &(offset, len) in &self.ranges {
                let rc = unsafe { posix_fadvise(self.file.as_raw_fd(), i64::try_from(offset)?, i64::try_from(len)?, POSIX_FADV_DONTNEED) };
                // Unlike madvise, posix_fadvise returns the error number directly.
                ensure!(rc == 0, "posix_fadvise DONTNEED: {}", std::io::Error::from_raw_os_error(rc));
            }
            let after = self.resident_pages()?;
            let cold = self.page_count - after;
            println!(
                "{}",
                json!({"event":"cold_engram","round":round,"target_pages":self.page_count,"page_bytes":self.page_size,"resident_before_eviction":before,"resident_before_decode":after,"cold_pages_before_decode":cold})
            );
            ensure!(
                after == 0,
                "Engram eviction incomplete: {after}/{} target pages remain resident; refusing cold timing",
                self.page_count
            );
            Ok(cold)
        }
    }
}

fn storage_read_bytes() -> Result<Option<u64>> {
    #[cfg(target_os = "linux")]
    {
        let io = std::fs::read_to_string("/proc/self/io")?;
        let bytes = io.lines().find_map(|line| line.strip_prefix("read_bytes:")).context("/proc/self/io has no read_bytes")?;
        Ok(Some(bytes.trim().parse()?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(None)
    }
}

fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.rounds >= 2 && a.chunk > 0, "at least two rounds and a nonzero chunk are required");
    ensure!(std::env::var("DSX_ADAPT").is_ok_and(|v| v == "0"), "set DSX_ADAPT=0 to freeze placement");
    ensure!(!a.cold_engram || cfg!(target_os = "linux"), "--cold-engram is supported only on Linux");
    ensure!(
        a.dump_logits.is_none() || a.dump_logits != a.dump_decode_logits,
        "prefill and decode dumps need different paths"
    );
    let tok = if a.text.is_some() || a.decode_text.is_some() {
        let g = dsx::gguf::Gguf::open(&a.model)?;
        let s =
            g.kv.get("dsx.tokenizer_json")
                .and_then(|v| v.as_str())
                .context("GGUF has no dsx.tokenizer_json; supply token IDs instead")?;
        Some(s.parse::<Tokenizer>().map_err(|e| anyhow!("embedded tokenizer: {e}"))?)
    } else {
        None
    };
    let mut ids = input_ids(a.ids.as_deref(), a.text.as_deref(), tok.as_ref())?;
    let mut decode = input_ids(a.decode_ids.as_deref(), a.decode_text.as_deref(), tok.as_ref())?;
    limit_ids(&mut ids, a.prompt_tokens, "prompt")?;
    limit_ids(&mut decode, a.decode_tokens, "continuation")?;
    ensure!(
        !ids.is_empty() && ids.len().checked_add(decode.len()).is_some_and(|n| n < a.ctx),
        "prompt and continuation must fit context"
    );
    ensure!(a.dump_decode_logits.is_none() || !decode.is_empty(), "decode dump requires a continuation");
    ensure!(!a.cold_engram || !decode.is_empty(), "--cold-engram requires a forced continuation");
    let m = Model::load(&a.model, None)?;
    ensure!(
        ids.iter().chain(&decode).all(|&id| (id as usize) < m.cfg.vocab),
        "input contains out-of-vocabulary token IDs"
    );
    #[cfg(target_os = "linux")]
    let cold = if a.cold_engram { Some(cold_engram::Pages::new(&m, &ids, &decode)?) } else { None };
    let adapter = a
        .lora
        .as_ref()
        .map(|p| dsx_core::lora::Adapter::load(p, m.cfg.n_layers, m.cfg.dim, m.cfg.o_groups * m.cfg.o_lora, a.lora_scale))
        .transpose()?;
    let mut gm = GpuModel::with_placement(
        &m,
        a.main_gpu,
        a.ctx,
        Placement {
            gpu0_gb: a.main_gb,
            aux_gb: a.aux_gb,
            arena_gb: a.arena_gb,
            prefill_chunk: a.chunk,
            rank: Some(a.rank),
        },
    )?;
    if let Some(adapter) = adapter {
        gm.install_lora(&adapter, a.chunk)?;
    }
    let pb = gm.pb.take().expect("prefill buffers");
    println!(
        "{}",
        json!({"event":"inputs","prompt_tokens":ids.len(),"decode_tokens":decode.len(),"prompt_fingerprint":fingerprint(ids.iter().flat_map(|v| v.to_le_bytes())),"decode_fingerprint":fingerprint(decode.iter().flat_map(|v| v.to_le_bytes())),"ctx":a.ctx,"chunk":a.chunk,"lora":a.lora.is_some(),"lora_scale":a.lora_scale,"vocab":m.cfg.vocab,"cold_engram":a.cold_engram})
    );
    let mut elapsed = 0.;
    let mut decode_elapsed = 0.;
    for round in 0..a.rounds {
        gm.reset();
        gm.g.sync()?;
        let start = Instant::now();
        let logits = gm.prefill(&pb, &ids)?;
        gm.g.sync()?;
        let seconds = start.elapsed().as_secs_f64();
        ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite prefill output");
        println!(
            "{}",
            json!({"event":"prefill_bench","round":round,"warm":round>0,"new_tokens":ids.len(),"cached_tokens":0,"chunk":a.chunk,"ctx":a.ctx,"elapsed_s":seconds,"tok_s":ids.len() as f64/seconds,"logit_fingerprint":logit_fingerprint(&logits)})
        );
        if round > 0 {
            elapsed += seconds;
        }
        // Store outputs until timing is finished: hashing, finite checks, JSON and disk I/O
        // are excluded from both compute intervals. No sampling divergence can change the work.
        let mut decode_logits = Vec::with_capacity(decode.len());
        let mut step_seconds = Vec::with_capacity(decode.len());
        #[cfg(target_os = "linux")]
        let cold_pages = cold.as_ref().map(|pages| pages.evict(round)).transpose()?;
        #[cfg(not(target_os = "linux"))]
        let cold_pages: Option<usize> = None;
        let read_bytes_before = if decode.is_empty() { None } else { storage_read_bytes()? };
        let start = Instant::now();
        for &token in &decode {
            let step_start = Instant::now();
            decode_logits.push(gm.step(token)?);
            step_seconds.push(step_start.elapsed().as_secs_f64());
        }
        gm.g.sync()?;
        let seconds = start.elapsed().as_secs_f64();
        let read_bytes_after = if decode.is_empty() { None } else { storage_read_bytes()? };
        let read_bytes_delta = read_bytes_before
            .zip(read_bytes_after)
            .map(|(before, after)| after.checked_sub(before).context("storage read counter moved backwards"))
            .transpose()?;
        if !decode.is_empty() {
            ensure!(decode_logits.iter().flatten().all(|v| v.is_finite()), "nonfinite decode output");
            let hashes: Vec<_> = decode_logits.iter().map(|row| logit_fingerprint(row)).collect();
            println!(
                "{}",
                json!({"event":"decode_bench","round":round,"warm":round>0,"forced_tokens":decode.len(),"vocab":m.cfg.vocab,"elapsed_s":seconds,"tok_s":decode.len() as f64/seconds,"step_elapsed_s":step_seconds,"logit_fingerprints":hashes,"cold_engram":a.cold_engram,"cold_pages_before_decode":cold_pages,"read_bytes_before":read_bytes_before,"read_bytes_after":read_bytes_after,"read_bytes_delta":read_bytes_delta})
            );
            if round > 0 {
                decode_elapsed += seconds;
            }
        }
        if round + 1 == a.rounds {
            if let Some(path) = &a.dump_logits {
                dump_rows(path, [logits.as_slice()])?;
            }
            if let Some(path) = &a.dump_decode_logits {
                dump_rows(path, decode_logits.iter().map(Vec::as_slice))?;
            }
        }
    }
    println!(
        "{}",
        json!({"event":"summary","warm_rounds":a.rounds-1,"new_tokens":ids.len()*(a.rounds-1),"elapsed_s":elapsed,"warm_tok_s":(ids.len()*(a.rounds-1)) as f64/elapsed,"forced_decode_tokens":decode.len()*(a.rounds-1),"decode_elapsed_s":decode_elapsed,"warm_decode_tok_s":if decode.is_empty(){None}else{Some((decode.len()*(a.rounds-1)) as f64/decode_elapsed)},"cold_engram":a.cold_engram})
    );
    Ok(())
}
