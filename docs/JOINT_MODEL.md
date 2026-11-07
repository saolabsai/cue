# Cue joint speech and turn model

This guide covers the **optional train-from-scratch backend implemented in
this repository**. It is a small recurrent model that produces speech and
turn-completion scores together. It is a research backend; its VAD targets
are weak energy proxies and its quality does not establish a replacement
for Silero or Smart Turn.

The default CLI continues to use pretrained Smart Turn v3.2 and Silero VAD.
The later encoder-adaptation experiments in [CURRENT_RESEARCH.md](CURRENT_RESEARCH.md)
use a separate local research cascade. Their results must not be attributed
to the small joint model or the default CLI. Trained Cue weights and research
datasets are not bundled with this checkout.

## Architecture and mathematics

```text
continuous mono 16 kHz PCM
  -> 512-sample frames (32 ms)
  -> fixed spectral frontend: 32 log-power bands + energy
  -> 48-unit causal GRU, with state owned by one stream
  -> speech probability + completion probability
  -> Cue policy: speech/silence gates, context holds, timeout
  -> end / wait / backchannel / interrupt
```

For frame $x_t$, features $f_t$, recurrent state $h_t$, and output $p_t$:

$$
f_t = F(x_t), \qquad
h_t = \operatorname{GRU}(f_t,h_{t-1}), \qquad
p_t = \sigma(W h_t+b).
$$

The two entries of $p_t$ are speech activity and turn completion. The default
network has **12,050 trainable parameters**. Its fixed spectral constants
are included in the ONNX graph but are not trainable parameters. The state
starts at zero and resets with the session. The graph receives only the
current frame and previous state; it has no access to future audio.

The trainer minimizes separately normalized, masked binary cross entropy:

$$
\mathcal L =
0.5\,\frac{\sum_{i,t}m^{s}_{it}\,\operatorname{BCE}(z^{s}_{it},y^{s}_{it})}
                 {\sum_{i,t}m^{s}_{it}}
+\frac{\sum_{i,t}m^{e}_{it}\,\operatorname{BCE}(z^{e}_{it},y^{e}_{it})}
          {\sum_{i,t}m^{e}_{it}}.
$$

Here $z$ is a logit, $y$ a target, and $m$ a supervision weight. Batch padding
has zero weight. Speech targets come from an RMS threshold, not human VAD
annotations. Completion targets combine the corpus clip label with an
identical 1.28-second observation tail for both classes. By default, complete
clips become positive three frames after the estimated boundary; the tail
has more endpoint weight than earlier audio. These are training assumptions,
not a measured response-time guarantee.

The first endpoint is the first accepted score under the configured policy,
including any forced timeout. A fast neural call alone cannot establish a
fast, correct conversation ending. See [EVALUATION.md](EVALUATION.md) for
first-endpoint metrics and latency definitions.

## Train and export

Use Python 3.11+ in a separate environment. The optional trainer requires
PyTorch, NumPy, ONNX, SciPy, SoundFile, and Requests. The Rust runtime does
not need these Python training packages. Record the installed versions with
your experiment; this recipe is not a locked training environment.

```console
python -m venv .venv-joint
.venv-joint/bin/python -m pip install torch numpy onnx scipy soundfile requests
.venv-joint/bin/python scripts/prepare_joint_data.py --output output/joint-data --train 600 --dev 120 --test 180
.venv-joint/bin/python scripts/train_joint.py train output/joint-data/manifest.jsonl --output output/joint-training/cue-joint.pt --epochs 24
.venv-joint/bin/python scripts/train_joint.py export output/joint-training/cue-joint.pt --output assets/models/cue-joint.onnx
```

On Windows, replace `.venv-joint/bin/python` with
`.venv-joint\Scripts\python.exe`. Run these commands from the repository root.
Use fresh output paths. The downloader caps retained WAV storage with
`--max-bytes` (250,000,000 by default); network transfer can be larger because
source files, metadata, rejected clips, and duplicates are also downloaded.
Requested split sizes are quotas, so inspect the saved manifest and
preparation report before training.
It does not need a GPU. `train --device cuda` opts into an available GPU;
there is no automatic AWS provisioning.

The downloader samples human English clips from the Pipecat
[training collection](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
and [test collection](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test).
It records upstream revisions, CC BY 4.0 attribution, source IDs, hashes, and
audio changes; retained audio is mono 16 kHz with a bounded clip tail. It
keeps the official test collection separate and checks exact duplicate PCM.
Speaker overlap, near duplicates, and exposure in upstream pretrained models
are not resolved by these checks. Dataset licenses remain separate from the
repository's MIT source license.

Each manifest line needs a unique `id`, an audio path relative to the manifest
directory, `split` (`train`, `dev`, or `test`), and Boolean `complete`:

```json
{"id":"example-001","audio":"audio/example-001.wav","split":"train","complete":false,"language":"en"}
```

Training and development must each contain both classes. WAV input to the
trainer must be mono PCM16 at 16 kHz. A false completion label means a known
incomplete clip, not missing annotation. Preserve provenance and enforce
conversation/speaker separation when supplying your own data.

The trainer uses AdamW and gradient clipping, selects weights by development
loss, and writes a checkpoint plus JSON metadata. A separate development
clip-tail threshold sweep charges false endings twice as much as misses.
That threshold is diagnostic; it does not automatically update the native
policy or establish a conversation operating point.

### Optional training flags

| Flag | Effect and limitation |
| --- | --- |
| `--supervise-pauses` | Emphasizes quiet runs bounded by speech inside a clip. The nominal 200 ms minimum becomes 224 ms on the 32 ms frame clock. |
| `--early-endpoint` | Starts positive tail labels on the first quiet frame, emphasizes the first three frames, and supervises internal pauses of at least 64 ms. Mutually exclusive with `--supervise-pauses`; not a latency guarantee. |
| `--balance-languages` | Requires `language` on every TRAIN/DEV row. Samples languages equally and averages development loss by language; inference still has no language input. |
| `--prosody` | Adds 24 periodicity features, a projection, and nonlinear output head. At hidden size 48 this has 21,762 trainable parameters. |
| `--frame-jitter` | Varies alignment within the 32 ms frame in augmented training clips. |
| `--annotated-cutoffs` | Uses supplied cutoff metadata where available; annotation quality remains the caller's responsibility. |
| `--initialize-from` | Loads compatible architecture/hidden-size weights. Starts a new optimizer run; it is not full training-state resume. |
| `--hidden-size`, `--learning-rate` | Change capacity and optimization explicitly; hidden size must be 1–256. |

See `python scripts/train_joint.py train --help` for all flags. Keep TEST
unopened until model, policy settings, and acceptance criteria are fixed.

## Run the model

Build the worker and install a native ONNX Runtime library as described in
the [README](../README.md#audio-inference). Once you have exported a model:

```console
cargo build --release --locked
cargo run --release --locked -- --runtime /absolute/path/to/onnxruntime-library --joint-model assets/models/cue-joint.onnx wav input.wav
cargo run --release --locked -- --runtime /absolute/path/to/onnxruntime-library --joint-model assets/models/cue-joint.onnx stream
```

`--joint-model` is supported by `wav`, `stream`, and `benchmark`. The joint
path loads only the supplied joint graph. It scores once per 32 ms frame;
the policy consumes the already computed completion probability.

| Tensor | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `pcm` | float32 | `[1,512]` | Current normalized PCM frame |
| `state` | float32 | `[1,1,H]` | Previous recurrent state |
| `probabilities` | float32 | `[1,2]` | Speech, then completion |
| `next_state` | float32 | `[1,1,H]` | State for the next frame |

`H=48` by default. The loader checks the ABI and matching fixed state widths
in 1–256. One detector owns one stream. Send continuous audio including
silence, and reset state between sessions. Arbitrary input chunks are
buffered into frames; a large packet can delay delivery of a decision made
inside it. The worker does not capture or resample audio.

For Python, pass `joint_model=...` to `SaoTurnClient` instead of `model` and
`vad_model`. That client also works with the optional `SaoTurnProcessor`.
See the [Python bridge](../python/README.md) and [streaming contract](STREAMING.md).
The application remains responsible for playback and downstream dispatch.

## Paired evaluation

Prepare the development replay first, choose settings there, and freeze them
before the test replay. The following commands illustrate the test stage:

```console
python scripts/train_joint.py prepare-eval output/joint-data/manifest.jsonl --split test --output output/joint-eval
python scripts/compare_joint.py output/joint-eval/test.jsonl --runtime /absolute/path/to/onnxruntime-library --joint-model assets/models/cue-joint.onnx --output output/joint-eval/native-compare
```

If training used `--annotated-cutoffs`, also pass `--annotated-cutoffs` to
`prepare-eval` so both stages preserve the same original clip boundary where
the required source cutoff metadata exists. Without that flag, evaluation
preparation uses energy-proxy trimming. The flag does not verify annotation
quality.

Download the pinned baseline models with `python scripts/prepare_models.py`
first. The comparison runs two independent processes on identical audio,
with text-context rules disabled. `--config` supplies shared policy settings;
`--joint-config` can supply a separately frozen joint setting.

Report premature cuts, missed complete turns, false endings on incomplete
clips, correct endpoint delay, and aggregate acoustic compute per audio
second. Per-call speed is insufficient: the joint graph runs every frame,
while Smart Turn runs at candidate pauses. Track startup separately.

By default, the prepared reference is an energy-estimated speech boundary;
annotated-cutoff mode uses the supplied cutoff. Equal added tails create a
controlled clip test; they do not represent natural dialogue
silence, overlaps, or queue latency. Weak speech targets do not independently
validate VAD quality. Use human labels on representative, reserved continuous
conversations for deployment decisions. Follow [EVALUATION.md](EVALUATION.md)
for grouping, paired uncertainty, and first-ending measurements.

## Recipe checks

From the repository root, run the downloader and resume-integrity checks:

```console
python -B tools/run_tests.py models
```

With the scientific dependencies installed, run the small synthetic
waveform/reference checks as well:

```console
python -B -m unittest discover -s tests/research -p test_training.py -v
```

These checks validate implementation behavior without a training run. They
are not audio-quality or conversational-accuracy benchmarks.

## Research references

These works motivated design questions; Cue does not claim to reproduce their
architectures, datasets, or reported improvements:

- [Raju et al., Two-pass Endpoint Detection for Speech Recognition](https://arxiv.org/abs/2401.08916): verify an initial endpoint proposal to study cutoff/latency trade-offs.
- [Udupa et al., Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training](https://arxiv.org/abs/2506.07081): causal endpointing and the effect of target timing. Cue's small joint frontend is spectral, not a neural audio codec.
- [Ekstedt and Skantze, Voice Activity Projection](https://arxiv.org/abs/2205.09812): model future speech activity as a related task; physical activity is not automatically semantic turn completion.
- [Ekstedt and Skantze, How Much Does Prosody Help Turn-taking?](https://aclanthology.org/2022.sigdial-1.51/): evidence that prosody can inform turn-taking. Cue's explicit periodicity option requires its own evaluation.
