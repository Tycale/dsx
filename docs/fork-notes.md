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
