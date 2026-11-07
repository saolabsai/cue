"""Train Cue's causal speech/activity model from a local labeled audio manifest.

No pretrained weights or model scores enter training. Per-frame speech targets
are energy-based proxies. Completion supervision uses the provided clip labels
and can emphasize within-clip pauses.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import wave

import numpy as np
import torch
from torch import nn

FRAME = 512
RATE = 16_000
HIDDEN = 48
TAIL = 40
MAX_AUDIO_FRAMES = 250
PAUSE_FRAMES = (RATE * 200 + FRAME * 1000 - 1) // (FRAME * 1000)


class Frontend(nn.Module):
    """Causal 32-band log-power features plus frame energy, exportable to ONNX."""

    def __init__(self):
        super().__init__()
        frequencies = np.fft.rfftfreq(FRAME, 1 / RATE)
        phase = 2 * np.pi * np.arange(FRAME)[:, None] * frequencies / RATE
        window = np.hanning(FRAME)[:, None] / (FRAME / 2)
        mel = 2595 * np.log10(1 + frequencies / 700)
        bounds = np.linspace(mel[2], mel[-1], 34)
        weights = np.maximum(
            0,
            np.minimum(
                (mel[:, None] - bounds[:-2]) / (bounds[1:-1] - bounds[:-2]),
                (bounds[2:] - mel[:, None]) / (bounds[2:] - bounds[1:-1]),
            ),
        )
        weights /= np.maximum(weights.sum(axis=0), 1e-9)
        self.register_buffer(
            "real", torch.tensor(window * np.cos(phase), dtype=torch.float32)
        )
        self.register_buffer(
            "imag", torch.tensor(window * np.sin(phase), dtype=torch.float32)
        )
        self.register_buffer("bands", torch.tensor(weights, dtype=torch.float32))

    def power_spectrum(self, pcm):
        return (pcm @ self.real).square() + (pcm @ self.imag).square()

    def spectral_features(self, pcm, power):
        bands = power @ self.bands
        energy = pcm.square().mean(dim=-1, keepdim=True)
        return (
            (torch.cat((bands, energy), dim=-1).clamp_min(1e-10).log() + 11.5) / 4
        ).clamp(-3, 3)

    def forward(self, pcm):
        return self.spectral_features(pcm, self.power_spectrum(pcm))


class ProsodyFrontend(Frontend):
    """Include normalized periodicity at lags spanning typical vocal pitch."""

    LAG_COUNT = 24

    def __init__(self):
        super().__init__()
        lags = np.geomspace(32, 240, self.LAG_COUNT)
        bins = np.arange(FRAME // 2 + 1)
        # Real Fourier bins have conjugate partners except at DC and Nyquist.
        weights = np.ones(len(bins))
        weights[[0, -1]] = 0.5
        periodicity = weights[:, None] * np.cos(
            2 * np.pi * bins[:, None] * lags / FRAME
        )
        self.register_buffer(
            "periodicity", torch.tensor(periodicity, dtype=torch.float32)
        )
        self.register_buffer(
            "power_weights", torch.tensor(weights[:, None], dtype=torch.float32)
        )

    def forward(self, pcm):
        power = self.power_spectrum(pcm)
        features = self.spectral_features(pcm, power)
        correlation = (power @ self.periodicity) / (
            power @ self.power_weights
        ).clamp_min(1e-8)
        return torch.cat((features, correlation), dim=-1)


class JointModel(nn.Module):
    def __init__(self, *, prosody=False, hidden=HIDDEN):
        super().__init__()
        if prosody:
            self.frontend = ProsodyFrontend()
            self.projection = nn.Sequential(
                nn.Linear(33 + self.frontend.LAG_COUNT, 64), nn.SiLU()
            )
            self.encoder = nn.GRU(64, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 2)
            )
        else:
            self.frontend = Frontend()
            self.projection = nn.Identity()
            self.encoder = nn.GRU(33, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)

    def score(self, features, state=None):
        hidden, next_state = self.encoder(self.projection(features), state)
        return self.head(hidden), next_state

    def forward(self, pcm, state):
        logits, next_state = self.score(self.frontend(pcm).unsqueeze(1), state)
        return logits[:, 0].sigmoid(), next_state


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"Expected mono PCM16 at 16 kHz: {path}")
        raw = wav.readframes(wav.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def prepare_clip(
    path,
    complete,
    rng=None,
    *,
    early_endpoint=False,
    frame_jitter=False,
    cutoff_samples=None,
):
    """Give both classes the same observation tail and record its estimated start."""
    audio = read_wav(path)
    if not len(audio):
        raise ValueError(f"Empty audio: {path}")
    if cutoff_samples is not None and abs(cutoff_samples - len(audio)) > 1:
        raise ValueError(f"Annotated cutoff does not match the end of {path}")
    if rng is not None and frame_jitter:
        audio = np.pad(audio, (int(rng.integers(FRAME)), 0))
    frames = np.pad(audio, (0, (-len(audio)) % FRAME)).reshape(-1, FRAME)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    speech = rms >= max(0.001, 0.04 * float(np.percentile(rms, 95)))
    active = np.flatnonzero(speech)
    if len(active) == 0:
        raise ValueError(f"No energy above the speech proxy threshold: {path}")
    end = len(frames) if cutoff_samples is not None else int(active[-1]) + 1
    start = max(0, end - MAX_AUDIO_FRAMES)
    frames, speech = frames[start:end], speech[start:end]
    if rng is not None:
        lead = int(rng.integers(0, 7))
        gain = float(rng.uniform(0.55, 1.15))
    else:
        lead, gain = 3, 1.0
    audio = np.pad(frames * gain, ((lead, TAIL), (0, 0)))
    if rng is not None:
        noise_rms = float(rng.choice([0, 0.0004, 0.001, 0.002, 0.004]))
        audio += rng.normal(0, noise_rms, audio.shape).astype(np.float32)
    audio = np.clip(audio, -1, 32767 / 32768)
    speech = np.pad(speech, (lead, TAIL))
    endpoint = np.zeros(len(audio), dtype=np.float32)
    boundary = len(audio) - TAIL
    if complete:
        first_positive = boundary if early_endpoint else boundary + 3
        endpoint[first_positive:] = 1
    # Emphasize both classes' tails, where silence alone is insufficient.
    weight = np.full(len(audio), 0.20, dtype=np.float32)
    weight[boundary:] = 1.0
    if early_endpoint:
        # The first three quiet frames cover the under-100 ms decision window.
        weight[boundary : boundary + 3] = 4.0
    if cutoff_samples is not None:
        # Supervise the real quiet frames leading up to the annotated pause.
        speech_end = lead + max(0, int(active[-1]) + 1 - start)
        weight[speech_end:boundary] = 1.0
    return audio, speech, endpoint, weight, boundary


def load_manifest(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate manifest ID: {row['id']}")
        if (
            row["split"] not in {"train", "dev", "test"}
            or type(row["complete"]) is not bool
        ):
            raise ValueError(
                "Expected a train/dev/test split and Boolean completion label"
            )
        audio = (path.parent / row["audio"]).resolve()
        if not audio.is_relative_to(path.parent.resolve()):
            raise ValueError("Audio must stay within the manifest directory")
        row["path"] = audio
        ids.add(row["id"])
    return rows


def emphasize_internal_pauses(weight, speech, boundary, min_frames=PAUSE_FRAMES):
    """Give quiet runs bounded by speech the same weight as the tail."""
    voiced = np.flatnonzero(speech[:boundary])
    quiet_frames = np.diff(voiced) - 1
    eligible = quiet_frames >= min_frames
    starts = voiced[:-1][eligible] + 1
    ends = voiced[1:][eligible]
    for start, end in zip(starts, ends, strict=True):
        weight[start:end] = 1.0


def cutoff_for(row, annotated_cutoffs):
    """Use the same optional source cutoff for training and replay preparation."""
    if annotated_cutoffs and "source_silence_start_s" in row:
        return (
            int(row["source_silence_start_s"] * RATE)
            - row["source_prefix_offset_samples"]
        )
    return None


def features_for(
    rows,
    frontend,
    seed,
    variants,
    *,
    supervise_pauses=False,
    early_endpoint=False,
    frame_jitter=False,
    annotated_cutoffs=False,
):
    rng = np.random.default_rng(seed)
    samples = []
    with torch.inference_mode():
        for row in rows:
            cutoff_samples = cutoff_for(row, annotated_cutoffs)
            for variant in range(variants):
                augmentation = rng if variant else None
                audio, speech, endpoint, weight, boundary = prepare_clip(
                    row["path"],
                    row["complete"],
                    augmentation,
                    early_endpoint=early_endpoint,
                    frame_jitter=frame_jitter,
                    cutoff_samples=cutoff_samples,
                )
                if early_endpoint:
                    emphasize_internal_pauses(weight, speech, boundary, min_frames=2)
                elif supervise_pauses:
                    emphasize_internal_pauses(weight, speech, boundary)
                samples.append(
                    (
                        frontend(torch.from_numpy(audio)),
                        speech,
                        endpoint,
                        weight,
                        boundary,
                    )
                )
    return samples


def batch(samples, device):
    size = max(len(s[0]) for s in samples)
    x = torch.zeros(len(samples), size, samples[0][0].shape[-1])
    targets = torch.zeros(len(samples), size, 2)
    mask = torch.zeros_like(targets)
    for index, (features, speech, endpoint, weight, _) in enumerate(samples):
        length = len(features)
        x[index, :length] = features
        targets[index, :length, 0] = torch.from_numpy(speech)
        targets[index, :length, 1] = torch.from_numpy(endpoint)
        mask[index, :length, 0] = 1
        mask[index, :length, 1] = torch.from_numpy(weight)
    return x.to(device), targets.to(device), mask.to(device)


def loss_for(model, examples, device):
    x, y, mask = batch(examples, device)
    logits, _ = model.score(x)
    losses = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    speech, endpoint = (losses * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1))
    return 0.5 * speech + endpoint


def validation_loss(model, samples, device, batch_size):
    total = 0.0
    for start in range(0, len(samples), batch_size):
        subset = samples[start : start + batch_size]
        total += loss_for(model, subset, device).item() * len(subset)
    return total / len(samples)


def dev_scores(model, samples, rows):
    scores = []
    with torch.inference_mode():
        for sample, row in zip(samples, rows, strict=True):
            features, _, _, _, boundary = sample
            logits, _ = model.score(features.unsqueeze(0))
            # A bounded observation window, fixed before inspecting the test split.
            completion = logits[0, :, 1].sigmoid()
            probability = completion[boundary + 6 : boundary + 24].max().item()
            scores.append((probability, row["complete"]))
    results = []
    positives = sum(label for _, label in scores)
    negatives = len(scores) - positives
    if min(positives, negatives) == 0:
        raise ValueError("Dev requires both completion classes")
    for threshold in (0.35, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.90, 0.95):
        false_end = sum(p >= threshold and not label for p, label in scores) / negatives
        miss = sum(p < threshold and label for p, label in scores) / positives
        results.append(
            {"threshold": threshold, "false_end_rate": false_end, "miss_rate": miss}
        )
    return (
        min(
            results,
            key=lambda x: (2 * x["false_end_rate"] + x["miss_rate"], -x["threshold"]),
        ),
        results,
    )


def train(args):
    if min(args.epochs, args.batch_size, args.threads) < 1:
        raise ValueError("Epochs, batch size and threads must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if args.output.suffix != ".pt" or any(
        p.exists() for p in (args.output, args.output.with_suffix(".json"))
    ):
        raise ValueError("Training needs a fresh .pt output path")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = load_manifest(args.manifest)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "dev"]
    if any(len({r["complete"] for r in split}) < 2 for split in (training, validation)):
        raise ValueError(
            "Train and dev must each contain complete and incomplete clips"
        )
    if args.balance_languages and any(
        not isinstance(row.get("language"), str) or not row["language"].strip()
        for row in (*training, *validation)
    ):
        raise ValueError(
            "Language balancing requires a language on every train/dev clip"
        )
    if not 1 <= args.hidden_size <= 256:
        raise ValueError("Hidden size must be in 1..=256")
    model = JointModel(prosody=args.prosody, hidden=args.hidden_size)
    architecture = "joint_prosody" if args.prosody else "joint"
    if args.initialize_from is not None:
        checkpoint = torch.load(
            args.initialize_from, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("architecture", "joint") != architecture
            or checkpoint.get("hidden_size", HIDDEN) != args.hidden_size
        ):
            raise ValueError(
                "Initial checkpoint architecture differs from requested model"
            )
        model.load_state_dict(checkpoint["state_dict"])
    variants = 2
    train_set = features_for(
        training,
        model.frontend,
        args.seed,
        variants,
        supervise_pauses=args.supervise_pauses,
        early_endpoint=args.early_endpoint,
        frame_jitter=args.frame_jitter,
        annotated_cutoffs=args.annotated_cutoffs,
    )
    dev_set = features_for(
        validation,
        model.frontend,
        args.seed,
        1,
        supervise_pauses=args.supervise_pauses,
        early_endpoint=args.early_endpoint,
        annotated_cutoffs=args.annotated_cutoffs,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0001
    )
    order = list(range(len(train_set)))
    weights = None
    dev_groups = [dev_set]
    if args.balance_languages:
        counts = Counter(row["language"] for row in training)
        weights = [
            1 / counts[row["language"]] for row in training for _ in range(variants)
        ]
        by_language = {}
        for row, sample in zip(validation, dev_set, strict=True):
            by_language.setdefault(row["language"], []).append(sample)
        dev_groups = list(by_language.values())
    best, best_loss, history = None, float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if weights is None:
            random.shuffle(order)
            sampled = order
        else:
            sampled = random.choices(order, weights=weights, k=len(order))
        train_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            subset = [train_set[i] for i in sampled[start : start + args.batch_size]]
            optimizer.zero_grad()
            loss = loss_for(model, subset, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            train_loss += loss.item() * len(subset)
        model.eval()
        with torch.inference_mode():
            dev_loss = sum(
                validation_loss(model, group, device, args.batch_size)
                for group in dev_groups
            ) / len(dev_groups)
        if dev_loss < best_loss:
            best_loss = dev_loss
            best = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_set),
            "dev_loss": dev_loss,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
    if best is None:
        raise ValueError("No finite validation checkpoint")
    model.cpu().eval().load_state_dict(best)
    selected, sweep = dev_scores(model, dev_set, validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "hidden_size": args.hidden_size,
            "state_dict": best,
        },
        args.output,
    )
    completion_labels = "Corpus clip labels with matched 1.28 s observation tails, annotated endpoint proxy."
    if args.early_endpoint:
        completion_labels += (
            " Complete targets start on the first quiet frame, and both classes "
            "have extra weight on the first three quiet frames. Speech-bounded "
            "pauses of at least 64 ms are continued targets."
        )
    elif args.supervise_pauses:
        completion_labels += (
            " Quiet frames in speech-bounded pauses of at least 200 ms have "
            "continued labels weighted equally to the observation tail."
        )
    metadata = {
        "architecture": architecture,
        "hidden_size": args.hidden_size,
        "supervise_pauses": args.supervise_pauses,
        "early_endpoint": args.early_endpoint,
        "frame_jitter": args.frame_jitter,
        "annotated_cutoffs": args.annotated_cutoffs,
        "learning_rate": args.learning_rate,
        "initial_checkpoint_sha256": (
            hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
            if args.initialize_from is not None
            else None
        ),
        "balance_languages": args.balance_languages,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": args.seed,
        "torch": torch.__version__,
        "device": str(device),
        "train_clips": len(training),
        "dev_clips": len(validation),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "training_history": history,
        "best_dev_loss": best_loss,
        "dev_threshold": selected,
        "dev_sweep": sweep,
        "dev_threshold_scope": "Clip-tail scores, weighted false ends 2:1 over misses. Native policy separately evaluated.",
        "speech_labels": "Weak frame RMS labels, with known added silence and noise frames.",
        "completion_labels": completion_labels,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dev_threshold": selected,
                "parameters": metadata["parameters"],
            }
        ),
        flush=True,
    )


def export(args):
    if args.output.suffix != ".onnx" or args.output.exists():
        raise ValueError("Export needs a fresh .onnx output path")
    import onnx

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "joint")
    if architecture not in {"joint", "joint_prosody"}:
        raise ValueError(f"Unknown checkpoint architecture: {architecture}")
    hidden = checkpoint.get("hidden_size", HIDDEN)
    if type(hidden) is not int or not 1 <= hidden <= 256:
        raise ValueError("Invalid checkpoint hidden size")
    model = JointModel(prosody=architecture == "joint_prosody", hidden=hidden).eval()
    model.load_state_dict(checkpoint["state_dict"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    torch.onnx.export(
        model,
        (torch.zeros(1, FRAME), torch.zeros(1, 1, hidden)),
        str(temporary),
        input_names=["pcm", "state"],
        output_names=["probabilities", "next_state"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(temporary))
    temporary.rename(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


def prepare_eval(args):
    rows = [r for r in load_manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        raise ValueError("Requested split is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / f"{args.split}.jsonl").open("w", encoding="utf-8") as file:
        for index, row in enumerate(rows):
            audio, _, _, _, boundary = prepare_clip(
                row["path"], row["complete"],
                cutoff_samples=cutoff_for(row, args.annotated_cutoffs),
            )
            name = f"clip-{index:05}.wav"
            with wave.open(str(args.output / name), "wb") as wav:
                wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
                wav.writeframes(np.rint(audio * 32768).astype("<i2").tobytes())
            prepared = {
                "id": row["id"],
                "audio": name,
                "reference_end_ms": boundary * 32 if row["complete"] else None,
            }
            file.write(json.dumps(prepared) + "\n")
    print(
        json.dumps(
            {
                "split": args.split,
                "clips": len(rows),
                "manifest": str(args.output / f"{args.split}.jsonl"),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("manifest", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=18)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--seed", type=int, default=20261106)
    training.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    pause_objective = training.add_mutually_exclusive_group()
    pause_objective.add_argument(
        "--supervise-pauses",
        action="store_true",
        help="Emphasize continued labels in within-clip speech pauses of at least 200 ms",
    )
    pause_objective.add_argument(
        "--early-endpoint",
        action="store_true",
        help="Train on the first quiet frames and emphasize internal pauses of at least 64 ms",
    )
    training.add_argument(
        "--balance-languages",
        action="store_true",
        help="Sample training languages equally and average development loss by language",
    )
    training.add_argument(
        "--prosody",
        action="store_true",
        help="Add pitch-period features and nonlinear projection and output layers",
    )
    training.add_argument(
        "--frame-jitter",
        action="store_true",
        help="Randomize the position of speech within frames in augmented training clips",
    )
    training.add_argument(
        "--annotated-cutoffs",
        action="store_true",
        help="Keep the original end of clips with source silence annotations",
    )
    training.add_argument(
        "--initialize-from", type=Path, help="Fine-tune an existing Cue checkpoint"
    )
    training.add_argument("--learning-rate", type=float, default=0.003)
    training.add_argument("--hidden-size", type=int, default=HIDDEN)
    training.set_defaults(run=train)
    exporter = commands.add_parser("export")
    exporter.add_argument("checkpoint", type=Path)
    exporter.add_argument("--output", type=Path, required=True)
    exporter.set_defaults(run=export)
    evaluate = commands.add_parser("prepare-eval")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=["dev", "test"], default="test")
    evaluate.add_argument(
        "--annotated-cutoffs", action="store_true",
        help="Keep the original end of clips with source silence annotations, as in training",
    )
    evaluate.set_defaults(run=prepare_eval)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
