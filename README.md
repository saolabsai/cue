# Cue

Streaming turn detection for voice agents, from SAO Labs.

Cue combines a Rust audio worker, pretrained Smart Turn v3.2 and Silero VAD,
and an optional asyncio/Pipecat bridge. It emits `end`, `wait`, `backchannel`,
and `interrupt` decisions so an application can decide when to respond.

Cue also supports an opt-in joint speech and turn model, with a training and
ONNX export recipe. See [the joint model guide](docs/JOINT_MODEL.md) for its
architecture, mathematics, usage, and paired evaluation.

**Research status:** the new trained candidates are experimental and have not
passed the conversation safety checks. No trained Cue checkpoint is bundled
or made the default by this change. The default remains Smart Turn plus
Silero; the local research cascade in the [research results](docs/CURRENT_RESEARCH.md)
is a separately configured system. Safe delivered decisions below 50 or
100 ms, natural Hinglish generalization, and superiority over Silero VAD
have not been established.

The context policy adds experimental English rules for explicit waits,
backchannels, interruption commands, and incomplete digit sequences. Both
benchmark policies share the same pretrained acoustic inference. No accuracy
advantage over Smart Turn has been established.

## Quick start

Install Rust 1.85+ and a linker, then run:

```console
git clone https://github.com/saolabsai/cue.git
cd cue
cargo build --release --locked
cargo test --release --locked
cargo run --release --locked -- replay examples/policy-demo.jsonl
cargo run --release --locked -- compare examples/paired-demo.jsonl
```

These examples use fabricated observations to exercise the policy. Default
tests use the included synthetic frontend fixtures and require no model download.
The executable remains `sao-turn` and the library remains `sao_turn` for
compatibility with existing integrations.

## Audio inference

Download and verify the pinned models with Python 3.11+:

```console
python scripts/prepare_models.py
python scripts/prepare_models.py --check
```

Install the native CPU ONNX Runtime library for your platform. Version 1.24.4
is tested. One way to obtain it is through a separate Python environment:

```console
python -m venv .venv
.venv/bin/python -m pip install onnxruntime==1.24.4
```

On Windows, use `.venv\Scripts\python.exe`. Find the native library in
`site-packages/onnxruntime/capi`: `onnxruntime.dll` on Windows,
`libonnxruntime.so*` on Linux, or `libonnxruntime*.dylib` on macOS. Set
`ORT_DYLIB_PATH` to that file, or pass its absolute path with `--runtime`.
The Rust worker loads the library directly; it does not run Python inference.

```console
cargo run --release --locked -- --runtime /absolute/path/to/onnxruntime-library wav input.wav --trace observations.jsonl
cargo run --release --locked -- replay observations.jsonl
```

Input must be mono 16 kHz PCM16 or normalized float32 WAV. Cue does not
resample, capture a microphone, or append silence. It reports an incomplete
final 512-sample frame as `unprocessed_tail_samples`.

An on-demand `--model` may expose two scalar ONNX outputs named `completion`
and `early_completion`, in that order. Cue uses `early_completion` at the
optional early silence gate and `completion` at the regular gate. The `score`
command uses `completion`. Single-output models use the same probability at
both gates.

For continuous audio, start `sao-turn stream` and send JSONL packets:

```json
{"samples":[0,15,-20,0],"context":{"mode":"conversation","agent_speaking":false},"reset":false}
```

Each packet gets one response containing `frames` and `buffered_samples`.
Frames are 512 samples / 32 ms. Supply continuous audio, including silence.
See the [streaming contract](docs/STREAMING.md) for context updates, reset
semantics, the Rust API, and policy settings. For Python and Pipecat, see the
[Python bridge](python/README.md).

## Benchmarks

```console
cargo run --release --locked -- benchmark manifest.jsonl --results paired.jsonl
```

The benchmark replays annotated WAV segments through both policies and compares
their first endpoints. Follow the [evaluation protocol](docs/EVALUATION.md) to
measure premature cuts, misses, endpoint delay, and compute cost on held-out
conversations. Synthetic examples validate behavior; no held-out conversational
quality dataset is bundled.

## Project layout

| Directory | Purpose |
| --- | --- |
| `src/` | Rust frontend, ONNX inference, policy, stream framing, CLI, evaluation |
| `tests/` | CLI and numerical frontend parity tests |
| `python/` | Async subprocess client and optional Pipecat adapter |
| `scripts/` | Pinned pretrained model downloader, joint-model data/training/export, paired replay |
| `assets/frontend/` | Deterministic synthetic fixtures used by default tests |
| `configs/`, `examples/` | Policy settings and replay examples |
| `docs/` | Streaming API, development, and evaluation guides |
| `vendor/` | Reference frontend code, fixture generator, and upstream notices |

## Limits

Cue detects turns; it does not transcribe speech or generate replies. The
application handles playback, interruptions, and stale STT results. Each
detector owns one stream and runs synchronous inference. The optional text
rules are English heuristics and do not establish Hindi/Hinglish semantic
understanding. Validate thresholds and behavior on your own audio before use.

Rust callers that construct `Config` or `FrameResult` with explicit struct
literals must account for the new public fields; `Config` can use
`..Default::default()` for unspecified settings. Legacy JSON and Python
invocations remain supported. `joint_inference_ms` is omitted from legacy
backend JSON responses.

## Contributing and licensing

For the complete build, native model tests and Python/Pipecat checks, follow
the [development guide](docs/DEVELOPMENT.md) and run `python tools/verify.py`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[MIT License](LICENSE) for first-party source terms. Smart Turn, Silero, and the
reference frontend retain their upstream licenses; sources and model hashes
are listed in [vendor/PROVENANCE.md](vendor/PROVENANCE.md).
