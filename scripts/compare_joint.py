"""Compare Cue's two acoustic backends on the same annotated WAV manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
EXE_NAME = "sao-turn.exe" if sys.platform == "win32" else "sao-turn"


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def outcome(row: dict) -> str:
    reference, endpoint = row["reference_end_ms"], row["baseline_end_ms"]
    if reference is None:
        return "held_incomplete" if endpoint is None else "ended_incomplete"
    if endpoint is None:
        return "missed_complete"
    return "early_complete" if endpoint < reference else "correct_complete"


def paired_outcomes(legacy: list[dict], joint: list[dict]) -> dict:
    counts = Counter()
    delay_deltas = []
    for old, new in zip(legacy, joint, strict=True):
        if (old["id"], old["reference_end_ms"]) != (new["id"], new["reference_end_ms"]):
            raise ValueError("Benchmarks returned different segment IDs or references")
        old_outcome, new_outcome = outcome(old), outcome(new)
        counts[f"{old_outcome} -> {new_outcome}"] += 1
        if old_outcome == new_outcome == "correct_complete":
            delay_deltas.append(new["baseline_end_ms"] - old["baseline_end_ms"])
    delay_deltas.sort()
    count = len(delay_deltas)

    def percentile(percent: int) -> int | None:
        return delay_deltas[(count * percent + 99) // 100 - 1] if count else None

    return {
        "outcomes_legacy_to_joint": dict(counts),
        "joint_minus_legacy_delay_ms_on_shared_correct_endpoints": {
            "count": count,
            "p50": percentile(50),
            "p95": percentile(95),
        },
    }


def run_backend(
    command: list[str], manifest: Path, output: Path, name: str
) -> tuple[dict, list[dict]]:
    metrics_path = output / f"{name}.json"
    rows_path = output / f"{name}.jsonl"
    log_path = output / f"{name}.stderr.log"
    started = perf_counter()
    with metrics_path.open("xb") as metrics, log_path.open("xb") as log:
        result = subprocess.run(
            [*command, "benchmark", str(manifest), "--results", str(rows_path)],
            cwd=ROOT,
            stdout=metrics,
            stderr=log,
            check=False,
        )
    wall_seconds = perf_counter() - started
    if result.returncode:
        raise ValueError(f"{name} benchmark exited {result.returncode}. See {log_path}")

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = read_rows(rows_path)
    comparison = report["comparison"]
    if comparison["sessions"] != len(rows):
        raise ValueError(f"{name} benchmark report and row counts differ")
    metrics = comparison["baseline"].copy()
    calls = metrics.pop("inference_ms")
    summary = {
        "endpoint_metrics": metrics,
        "config": report["config"],
        "timed_call_ms": calls,
        "timed_call_scope": (
            "Joint speech and completion inference on each audio frame"
            if name == "joint"
            else "Requested Smart Turn completion calls, including feature extraction"
        ),
        "benchmark_wall_seconds_including_startup_and_io": wall_seconds,
    }
    compute = report.get("acoustic_compute")
    if compute is not None:
        processed_seconds = compute["frames"] * 0.032
        summary["acoustic_compute"] = compute
        if processed_seconds:
            summary["acoustic_compute_ms_per_processed_audio_second"] = (
                compute["total_ms"] / processed_seconds
            )
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--exe", type=Path, default=ROOT / "target/release" / EXE_NAME)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--joint-model", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "assets/models/smart-turn-v3.2-cpu.onnx"
    )
    parser.add_argument(
        "--vad-model", type=Path, default=ROOT / "assets/models/silero-vad.onnx"
    )
    parser.add_argument(
        "--config", type=Path, help="Shared endpoint policy configuration"
    )
    parser.add_argument(
        "--joint-config", type=Path, help="Override the joint policy configuration"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New output directory"
    )
    args = parser.parse_args()

    try:
        manifest = args.manifest.resolve(strict=True)
        command = [
            str(args.exe.resolve(strict=True)),
            "--runtime",
            str(args.runtime.resolve(strict=True)),
            "--threads",
            "1",
        ]
        legacy_command = [
            *command,
            "--model",
            str(args.model.resolve(strict=True)),
            "--vad-model",
            str(args.vad_model.resolve(strict=True)),
        ]
        joint_command = [
            *command,
            "--joint-model",
            str(args.joint_model.resolve(strict=True)),
        ]
        for invocation, config in (
            (legacy_command, args.config),
            (joint_command, args.joint_config or args.config),
        ):
            if config is not None:
                invocation.extend(["--config", str(config.resolve(strict=True))])
        output = args.output.resolve()
        output.mkdir(parents=True)
        legacy, legacy_rows = run_backend(legacy_command, manifest, output, "legacy")
        joint, joint_rows = run_backend(joint_command, manifest, output, "joint")
        summary = {
            "manifest": str(manifest),
            "platform": platform.platform(),
            "ort_threads": 1,
            "segments": len(legacy_rows),
            "policy": (
                "First end decision from each independently run context-disabled policy"
            ),
            "scope": (
                "Audio-clock endpoints. Delay distributions use correct complete segments. "
                "Acoustic compute covers the calls made by each native benchmark. "
                "Transport, live queueing and downstream processing are outside this measurement."
            ),
            "legacy": legacy,
            "joint": joint,
            "paired": paired_outcomes(legacy_rows, joint_rows),
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Benchmark error: {error}\n")
    print(output / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
