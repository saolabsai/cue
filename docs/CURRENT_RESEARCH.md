# Cue research status — 8 November 2026

**The latest trained candidates were rejected for conversation safety
regressions.** They are not production replacements. Warm CPU scoring is
fast, but safe delivered turn decisions below 50 or 100 ms and overall
superiority over Smart Turn or Silero have not been established.

This is a summary of completed local experiments, including negative results.
Raw audio, trained Cue checkpoints, cloud execution packages, and generated
reports are not distributed in this checkout. The public
[joint-model recipe](JOINT_MODEL.md) is usable separately; it does not reproduce
the later encoder experiments below without their additional artifacts.

## What is being compared?

| System | Meaning |
| --- | --- |
| Default public CLI | Pretrained Smart Turn v3.2 plus Silero VAD. Default minimum silence is 200 ms; the optional early gate is disabled. |
| Optional joint backend | The small causal GRU trained by `scripts/train_joint.py`, loaded explicitly with `--joint-model`. No trained checkpoint is bundled. |
| Research Cue cascade | A separately configured local system with a frame proposer, dual-score early verification, a Recall late scorer at nominal 600 ms, and inherited timeout behavior. This is the baseline called “Research Cue” below; it is not the bare CLI default. |
| New encoder candidates | Replace only the research cascade's late scorer. They retain its original early decisions, timeout rules, and fixed 0.5 late cutoff. |
| Tuned Smart Turn comparator | Smart Turn inside the matched replay harness with previously fixed settings. This is not an independent reproduction of the entire Pipecat application. |

Speech activity and turn completion are different tasks. Silero estimates
whether speech is present; an endpointer estimates whether the speaker has
finished. Silence alone is insufficient evidence of a completed turn.

## Latest experiment: adding natural Hindi audio

Two matched training arms compared the same conservative adaptation recipe
with and without **1,393 natural-audio Hindi clips** from 70 calls / 68 known
components. Of these, 268 are pause-cut variants. Their labels were produced
by the publisher using Gemini, not adjudicated by human listeners. Accuracy
against these labels measures teacher agreement.

The older mixture contains 3,902 examples: human English and Tamil plus
synthetic Hindi and Arabic. The augmented arm changes English from **80% to
65% of the total sampling mixture**, assigns **15% to natural Hindi**, and
leaves the remaining 20% unchanged. It does not add Hinglish data.

Both arms use the same initialization and coupled sampling draws. Five
source-held adaptation folds plus one full fit per arm give 12 fits, each
with a predeclared 200 updates (2,400 updates total). Encoder and head learning
rates are respectively $10^{-6}$ and $10^{-5}$; dropout is 0.25, AdamW weight
decay is 0.003, and gradient norm is clipped at 1.0. An L2 penalty anchors the
encoder to its initial weights:

$$
\mathcal L = \frac{1}{B}\sum_{i=1}^{B}\operatorname{BCE}(z_i,y_i)
+\frac{1}{2}\sum_{j\in\text{encoder}}(\theta_j-\theta_{0,j})^2.
$$

Sampling supplies the mixture weights; BCE is not weighted a second time.
The penalty sums over encoder parameters, not pooling/classifier parameters.
The fixed step-200 checkpoint and 0.5 cutoff were not selected using native
development outcomes. Known call/speaker components remain inside one fold.
Earlier backbone exposure, unknown cross-corpus speaker identities, and
possible recoded/cropped duplicates limit independence.

The encoder receives an 80×800 mel window from a rolling eight-second
frontend, then internally keeps the latest 320 frames: approximately
**3.2 seconds of effective encoder context**. Inference has no language or
transcript input; that does not guarantee generalization to unseen languages.

### Source-held adaptation results

| Cohort | Control correct | Hindi-added correct | Control false END on HOLD | Hindi-added false END on HOLD |
| --- | ---: | ---: | ---: | ---: |
| Older mixture, 3,902 examples | 3,387 | 3,373 | 285 | 300 |
| Natural Hindi pseudo-labels, 1,393 clips | 960 | 1,022 | 136 | 140 |

The Hindi addition improves agreement with its teacher by 62 examples but
adds four false endings there and 15 on the older mixture. On the existing
120 synthetic Hindi examples, both arms get 108 correct / 11 false endings,
versus 110 / 8 for the unadapted FP32 reference. These are conditional
adaptation measurements, not a new human conversation benchmark.

### Primary native replay: both candidates rejected

Both standalone FP32 exports passed a fixed 32-TRAIN numerical gate, with
maximum cloud/Torch/ONNX error below $10^{-5}$ and no 0.5 decision flips. All
64 candidate native TRAIN scores then matched the saved ONNX reference bits.
The primary replay completed 2,060 native development calls and reconstructed
first endpoints under the unchanged early route and nominal 600 ms late gate.

| System | Correct clips / 600 | Correct full views / 209 | Early full cuts | Severe full cuts | Missed full views | Correct full endings below 100 ms of audio time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Research Cue | 390 | 149 | 32 | 31 | 28 | 5 |
| New control | 376 | 152 | 39 | 38 | 18 | 5 |
| New Hindi-added model | 374 | 155 | 39 | 38 | 15 | 5 |
| Tuned Smart Turn | 386 | 148 | 42 | 41 | 19 | 0 |

Both candidates miss fewer endings but introduce more early cuts. Clip false
endings rise from 104 to 130/131. Each fails six acceptance checks; neither
is promoted. The predeclared checks protect clip/full early, severe, missed,
and false-ending counts, existing fast endpoints, and overall correct counts.
A higher total of correct full views cannot compensate for these regressions.

The Hindi-specific result is worse, despite the new Hindi training data:

| Slice | Research Cue correct | Control correct | Hindi-added correct | Important change |
| --- | ---: | ---: | ---: | --- |
| Natural Hindi full views | 43/48 | 41/48 | 41/48 | Early cuts increase from 4 to 7 in each candidate. |
| Hindi LiveKit clips | 36/48 | 29/48 | 29/48 | False endings increase from 12 to 19. |
| English full views | 12/21 | 10/21 | 11/21 | No improvement in this slice. |
| Arabic full views | 57/80 | 61/80 | 64/80 | Most of the aggregate full-view gain comes from Arabic. |
| Synthetic Hinglish clips | 19/20 | 19/20 | 19/20 | Twenty synthetic clips do not establish natural Hinglish quality. |

These 809 views are **already consumed development cases**; clip/full views
can share recordings. Some references are structural or timing proxies,
not human semantic judgments. Counts should not be presented as independent
conversation samples or fresh test-set accuracy. Severe errors and all first
endpoints remain in the result, including those followed by later recovery.

### Separate 64 ms routing diagnostic: every arm rejected

A separately frozen diagnostic added a trigger after two quiet 32 ms frames
to the original cascade. Each of three scorers evaluated the same 2,102
eligible prefixes, for 6,306 calls. The earliest accepted trigger wins.
This changes routing; it is separate from the primary 600 ms experiment.

| Policy on 209 full views | Correct | Early cuts | Severe cuts | Missed | Correct endings below 100 ms of audio time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Research Cue cascade | 149 | 32 | 31 | 28 | 5 |
| Cascade + original scorer at 64 ms | 64 | 127 | 93 | 18 | 34 |
| Cascade + control scorer at 64 ms | 65 | 126 | 91 | 18 | 38 |
| Cascade + Hindi-added scorer at 64 ms | 64 | 129 | 95 | 16 | 34 |

All three added-trigger variants fail ten acceptance checks. Clip correctness
falls from 390/600 to 266, 252, and 254; clip false endings increase from 104
to 163, 168, and 169. More fast correct endings come with many more interruptions.
Two quiet frames were not sufficient evidence of completion in this experiment.
None establishes safe or delivered 64 ms endpointing.

## Compute time is not delivered latency

All four captures used the same Windows CPU, one ONNX thread, and the same
32 TRAIN clips, with three measured calls per clip after one same-input
warmup in each fresh process. Percentiles below are over the 32 per-clip
medians. They use linear interpolation, unlike the generic Rust evaluator's
nearest-rank percentiles. Capture windows were separate.

| Scorer | Encoder context | Precision | Warm p50 | Warm p95 |
| --- | --- | --- | ---: | ---: |
| Original research Cue scorer | 3.2 s | Dynamic INT8 | 15.74 ms | 16.88 ms |
| Smart Turn v3.2 | 8 s | QDQ INT8 | 47.60 ms | 51.34 ms |
| New control | 3.2 s | FP32 | 29.36 ms | 30.79 ms |
| New Hindi-added scorer | 3.2 s | FP32 | 29.11 ms | 31.04 ms |

These measurements include native feature extraction plus ONNX execution.
They exclude model/audio loading, startup, IPC, waiting for audio/silence,
and delivery. Different precision and context prevent an equal-architecture
speed comparison; system load and thermal conditions can also vary. The
new FP32 scorers are slower than the original INT8 scorer on this measurement.
The full turn detector's delivered latency is not given by this table.

An earlier paced check on 20 selected, consumed DEV views (12 complete,
eight incomplete) produced **zero correct delivered endings below 100 ms for
either system**. It counted arrival scheduling, worker communication, and
queue arbitration, but excluded microphone, network, and downstream response
generation. Its small sample and different correct subsets prevent a general
latency claim. No new paced delivery test was performed in the latest round.

## Earlier experiments and why the latest result matters

| Experiment | Result and interpretation |
| --- | --- |
| Small joint GRU | Cheap frame inference, but the first 180-clip English replay missed 70/90 complete clips at its chosen operating point. It remains an experimental backend. |
| Internal-pause and multilingual joint training | Some controlled English and synthetic-language slices improved; missed turns, language regressions, proxy boundaries, and repeated development use prevented a general superiority claim. |
| Physical-activity auxiliary prediction | Improved some held-call probability metrics, but physical continuation/turn exchange is not human semantic END/HOLD gold. Native first-ending regressions remained. |
| Semantic residual and TRAIN cutoff calibration | Better original-clip scores did not transfer to safe chronological endings. Both fixed and calibrated variants were rejected. |
| First AWS encoder/head comparison | At fixed epoch 5, head/encoder produced 157/150 correct full views versus 149, but 41/42 early cuts versus 32; clip correctness fell to 376/370 versus 390. Both were rejected, including later TRAIN-only cutoff controls. |
| Latest Hindi addition | Better pseudo-label agreement, worse natural-Hindi replay, and safety regressions. Additional data alone did not solve the objective mismatch. |

The two Cue AWS runs used 254 and 259 billable seconds on one managed L40S
instance each, approximately **$0.399 total training compute** at the captured
$2.80/hour rate. The instances terminated; this estimate excludes retained
storage, requests, logs, taxes, and other services. It is not an AWS invoice
or a production capacity forecast.

## What evidence is still needed?

1. **Human judgments at real decision opportunities.** Label chronological
   pauses as END, CONTINUE, or UNSURE, with adjudication and the audio available
   at that moment. A clip label must not be copied onto every earlier prefix.
   Existing reviews and source audits do not supply complete first-ending gold.
2. **Fresh conversation and speaker separation.** Reserve real English, Hindi,
   and Hinglish calls, overlaps, repairs, hesitations, accents, and environments.
   Unknown or restrictive source licenses must be resolved before distribution.
   Do not turn the consumed development replay into another tuning target.
3. **A first-ending objective and frozen safety budget.** Account for interruption
   severity, missed turns, and deadline coverage; record all scheduled opportunities
   even after a rejected proposal. Follow the [evaluation protocol](EVALUATION.md).
4. **Separate VAD evidence.** The audited candidate VAD cohort had no eligible
   verified quiet examples. We have not established independent speech/silence
   superiority over Silero. Turn-completion scores cannot substitute for that test.
5. **End-to-end latency measurement.** Measure audio acquisition, framing, queueing,
   compute, delivery, and downstream work separately on representative hardware.
   Report latency alongside correct endings and interruption/miss rates.

No candidate is promoted by these results. A production decision needs the
missing evidence, not another headline accuracy or isolated inference number.

## Evidence identity and reproducibility scope

The following SHA-256 values identify the locally retained, verified analysis
artifacts summarized here. They are provenance identifiers, not download links
or evidence that the unbundled experiments can be rerun from this checkout.
No cloud account, private storage URI, audio, or trained Cue weight is required
to read this summary.

| Artifact | SHA-256 |
| --- | --- |
| Hindi adaptation training plan | `6cf71b035a93e9d80fc897618abc8dcbee74a547fde7f5bb2d2f29e8f02af37b` |
| Source-held adaptation analysis | `80810f896d592082f8b70f6944a3289f7a45a7309fbb61af43073f5cec594ae5` |
| FP32 export/parity receipt | `9256c873d851017e4aeffc4efb453e9344e1a8dc3d0a0ab895ca722704f654cb` |
| Primary native replay | `21fe5b4f48df9359419c7a2661a16b7511c45ff65ba267516cbad6d51dcda364` |
| Separate 64 ms diagnostic | `ba2be88ea5d0c86192c6fc3547a7c92f52c7add4f9b6476fca13160ccd1706bb` |

For the available public implementation, start with the [README](../README.md),
[joint architecture and recipe](JOINT_MODEL.md), [streaming API](STREAMING.md),
and [evaluation protocol](EVALUATION.md). Research references and their relation
to the implementation are listed in the joint-model guide.
