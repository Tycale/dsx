# Fork additions

This branch adds a Ryzen 9 CPU-affinity layout and experimental startup-only
rank-1 PEFT adapter support to the NVIDIA server. Benchmark numbers elsewhere in
the repository are upstream measurements, not measurements of this fork.

## CPU affinity

The layout in `nvidia/src/cpu/pool.rs` separates the inference thread, the expert
coordinator and helper threads. `DSX_CPU_LIST` selects the handoff worker pool.
The CPU benchmark uses the same worker ordering for all of its modes.

Logical CPU numbering depends on the host, firmware and allowed CPU set. This is
a specific layout, not automatic topology detection or a universal Ryzen 9
default. Inspect `lscpu -e=CPU,CORE,SOCKET` and the process's allowed CPUs before
using it on another machine. Keep expert workers off the coordinator's CPU.

## Experimental adapter support

The `dsx-server` binary accepts `--lora /path/to/adapter` and an optional
`--lora-scale` multiplier, which defaults to `1.0`. The directory must contain
`adapter_config.json` and `adapter_model.safetensors`.

The supported PEFT subset is deliberately narrow:

- Rank 1, alpha 1, targeting only the attention `wo_b` projection.
- FP32 A/B tensors with shapes matching the loaded model and explicitly selected
  layers. The accepted tensor names are validated by `core/src/lora.rs`.
- Inference mode, no dropout or bias, no DoRA, RS-LoRA or per-layer rank overrides.
- A finite, non-negative runtime scale.

The adapter adds `scale * B * (A * x)` before the attention mHC post step in both
prefill and decode. Base model files are unchanged. The fused output-projection
shortcut is disabled on adapted layers, and unsupported FP8 activation paths are
rejected at startup. This can add runtime overhead.

`DSX_LORA_FUSED=1` opts into a single-kernel delta for batches of one token,
including decode and single-token prefill fragments. Larger batches always use
the original two-kernel path, which remains the default for all batch sizes.
`DSX_LORA_FUSED_BLOCKS` selects 1–32 output partitions (default 4), each repeating
the identical dot reduction. `lora-check` requires bitwise agreement with the
original path; `lora-check --bench` adds paired timings for several partition
counts. End-to-end gains depend on the workload.

An adapter can be installed only once, before inference starts. Restart the
server to change it: existing prefix/KV caches must never be reused with different
adapter weights. Omitting `--lora` keeps the base-model path. This feature does
not add adapter support to the `dsx gen` or `dsx eval` commands.

No adapter weights are included in the repository. Use an adapter intended for
your model and check its license separately. Shape validation alone does not
establish model compatibility or output quality.

## Validation

The parser tests cover rank-one arithmetic, configuration/shape rejection,
truncated data and non-finite weights:

```sh
cargo test --locked --release -p dsx-core --lib lora::
```

The CUDA checker compares the rank-one kernels with an FP64 CPU reference,
checks output guard rows and verifies that zero scale leaves the output unchanged:

```sh
cargo build --locked --release -p dsx --features cuda --bin dsx-server --bin lora-check
./target/release/lora-check
```

Its optional adapter-directory argument checks the specific adapter fixture it
was written for; it is not a general model-compatibility validator. These checks
validate kernel arithmetic, not end-to-end equivalence to a PEFT framework.

## Controlled model measurements

`prefill-bench` accepts the same `--lora` and `--lora-scale` options, plus raw
`--text` and `--decode-text` fixtures encoded with the GGUF tokenizer. The text
is not wrapped in a chat template. `--prompt-tokens` and `--decode-tokens` select
exact input lengths. Forced continuation avoids sampling divergence between
variants, and `--dump-logits` / `--dump-decode-logits` save complete F32 outputs.

Use `DSX_ADAPT=0`, a fixed `--main-gpu`, positive identical `--main-gb` and
`--aux-gb` budgets, and verify identical resident-expert counts when comparing
numerics. Negative memory budgets can change placement as buffer sizes change.
Changing the chunk can also change transient expert scheduling and reduction
order even when the resident-expert counts match; compare the actual logits.
Compare production settings separately to include the benefit of freed memory.
The benchmark retains continuation logits, so keep continuations short when
measuring memory pressure. It requires exclusive use of the inference devices.

On Linux, `--cold-engram` evicts only the model-file pages needed by the forced
continuation, once after each prefill. It checks that every target page is
nonresident before timing and fails if eviction was incomplete. It does not
modify the model or drop the system-wide cache. Subsequent tokens may reuse
pages loaded earlier in that continuation. The `warm` field still identifies
rounds after initial model/kernel warm-up; `cold_engram` and the page counters
describe storage-cache state. `read_bytes_delta` includes all process reads,
including readahead, rather than identifying Engram traffic alone.
