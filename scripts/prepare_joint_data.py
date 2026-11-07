"""Prepare a bounded English subset of the public Smart Turn v3.2 audio data.

Requires requests, numpy, scipy and soundfile. No model weights are used. The
dataset's endpoint annotations are clip labels, not frame-level VAD labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
from math import gcd
from pathlib import Path
import re
import threading
from urllib.parse import urlencode, urlsplit

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from scipy.signal import resample_poly
import soundfile as sf
from urllib3.util.retry import Retry

from joint_manifest import validate_saved_wav

DATASETS = {
    name: f"pipecat-ai/smart-turn-data-v3.2-{name}" for name in ("train", "test")
}
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
API = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100
MAX_SOURCE_BYTES = 3_000_000
SAMPLE_RATE = 16_000
MAX_SOURCE_SECONDS = 33
MAX_RETAINED_SAMPLES = 81_920  # 160 streaming frames, 5.12 seconds
MIN_SECONDS = 0.35
CLIENT = threading.local()


def fetch(url: str, max_bytes: int) -> bytes:
    if not hasattr(CLIENT, "session"):
        CLIENT.session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            respect_retry_after_header=False,
        )
        CLIENT.session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        with CLIENT.session.get(
            url, stream=True, timeout=(8, 12), allow_redirects=False
        ) as response:
            if response.status_code != 200:
                raise ValueError(f"download HTTP status {response.status_code}")
            result = bytearray()
            for chunk in response.iter_content(chunk_size=65536):
                result.extend(chunk)
                if len(result) > max_bytes:
                    raise ValueError("download exceeds per-file limit")
            return bytes(result)
    except requests.RequestException as error:
        # Avoid persisting temporary signed URLs in network errors.
        raise ValueError(f"download failed ({type(error).__name__})") from None


def page(dataset: str, offset: int) -> list[dict]:
    query = urlencode(
        dict(
            dataset=dataset,
            config="default",
            split="train",
            offset=offset,
            length=PAGE_SIZE,
        )
    )
    rows = json.loads(fetch(f"{API}/rows?{query}", MAX_SOURCE_BYTES))
    return rows["rows"]


def clip(source: str, item: dict) -> tuple[dict, bytes]:
    row = item["row"]
    source_id = row["id"]
    if not re.fullmatch(r"[a-zA-Z0-9-]{1,100}", source_id):
        raise ValueError("unexpected source ID")
    audio_url = row["audio"][0]["src"]
    parsed = urlsplit(audio_url)
    prefix = f"/cached-assets/{source}/--/"
    if parsed.scheme != "https" or parsed.netloc != "datasets-server.huggingface.co":
        raise ValueError("unexpected audio host")
    if not parsed.path.startswith(prefix):
        raise ValueError("unexpected audio path")
    revision = parsed.path[len(prefix) :].split("/", 1)[0]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("missing dataset revision")

    raw = fetch(audio_url, MAX_SOURCE_BYTES)
    with sf.SoundFile(io.BytesIO(raw)) as audio_file:
        original_rate = audio_file.samplerate
        source_samples = len(audio_file)
        if not MIN_SECONDS <= source_samples / original_rate <= MAX_SOURCE_SECONDS:
            raise ValueError("clip duration outside bounds")
        samples = audio_file.read(dtype="float32", always_2d=True).mean(axis=1)
    if original_rate != SAMPLE_RATE:
        divisor = gcd(original_rate, SAMPLE_RATE)
        samples = resample_poly(
            samples, SAMPLE_RATE // divisor, original_rate // divisor
        )
    samples = samples[-MAX_RETAINED_SAMPLES:]
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-4:
        raise ValueError("invalid or silent audio")
    pcm = np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2")
    wav = io.BytesIO()
    sf.write(wav, pcm, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return {
        "id": source_id,
        "audio": f"audio/{source_id}.wav",
        "complete": row["endpoint_bool"],
        "language": row["language"],
        "synthetic": row["synthetic"],
        "source": source,
        "source_collection": row["dataset"],
        "source_revision": revision,
        "source_row": item["row_idx"],
        "source_samples": source_samples,
        "source_sample_rate": original_rate,
        "download_bytes": len(raw),
        "license": "CC-BY-4.0",
        "samples": len(pcm),
        "pcm_sha256": sha256(pcm.tobytes()).hexdigest(),
    }, wav.getvalue()


def choose_split(source_part: str, row: dict, train_count: int, dev_count: int) -> str:
    if source_part == "test":
        return "test"
    bucket = int(sha256(row["id"].encode()).hexdigest()[:8], 16) / 2**32
    return "dev" if bucket < dev_count / (train_count + dev_count) else "train"


def write_manifest(folder: Path, rows: list[dict]) -> None:
    pending = folder / "manifest.jsonl.partial"
    pending.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pending.replace(folder / "manifest.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/joint-data"))
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--dev", type=int, default=120)
    parser.add_argument("--test", type=int, default=180)
    parser.add_argument(
        "--max-bytes", type=int, default=250_000_000,
        help="Maximum retained WAV bytes; total network transfer may be larger",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-pages", type=int, default=180)
    parser.add_argument(
        "--resume", action="store_true", help="Continue with the same split settings"
    )
    args = parser.parse_args()
    if any(x <= 0 or x % 2 for x in (args.train, args.dev, args.test)):
        parser.error("split counts must be positive and even")
    if not 1 <= args.workers <= 24 or args.max_bytes <= 0 or args.max_pages <= 0:
        parser.error("invalid worker, page or byte limit")
    folder = args.output
    rows: list[dict] = []
    counts: Counter = Counter()
    rejected: Counter = Counter()
    ids, hashes = set(), set()
    revisions: dict[str, str] = {}
    total_bytes = 0
    quotas = {"train": args.train // 2, "dev": args.dev // 2, "test": args.test // 2}
    if args.resume:
        rows = [
            json.loads(line)
            for line in (folder / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for row in rows:
            if (
                not re.fullmatch(r"[a-zA-Z0-9-]{1,100}", row["id"])
                or row["audio"] != f"audio/{row['id']}.wav"
            ):
                raise ValueError("invalid existing audio path")
            if row["id"] in ids or row["pcm_sha256"] in hashes:
                raise ValueError("duplicate in existing manifest")
            source = row["source"]
            part = "test" if source == DATASETS["test"] else "train"
            if source != DATASETS[part] or row["split"] != choose_split(
                part, row, args.train, args.dev
            ):
                raise ValueError("existing split settings differ")
            total_bytes += validate_saved_wav(
                folder / row["audio"],
                samples=row["samples"],
                pcm_sha256=row["pcm_sha256"],
                sample_rate=SAMPLE_RATE,
                max_samples=MAX_RETAINED_SAMPLES,
            )
            ids.add(row["id"])
            hashes.add(row["pcm_sha256"])
            counts[row["split"], row["complete"]] += 1
            if (
                revisions.setdefault(source, row["source_revision"])
                != row["source_revision"]
            ):
                raise ValueError("multiple revisions in existing manifest")
        if total_bytes > args.max_bytes or any(
            count > quotas[split] for (split, _), count in counts.items()
        ):
            raise ValueError("existing manifest exceeds requested limits")
    else:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "audio").mkdir()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Preserve a separate official test partition. Exact duplicates never
        # enter training even if their source IDs differ.
        for part in ("test", "train"):
            source = DATASETS[part]
            splits = ("test",) if part == "test" else ("train", "dev")
            for batch in range(0, args.max_pages, 8):
                if all(
                    counts[s, label] >= quotas[s]
                    for s in splits
                    for label in (False, True)
                ):
                    break
                offsets = [
                    page_number * PAGE_SIZE
                    for page_number in range(batch, min(batch + 8, args.max_pages))
                ]
                pages = [pool.submit(page, source, offset) for offset in offsets]
                candidates = []
                planned: Counter = Counter()
                for pending in pages:
                    try:
                        items = pending.result()
                    except ValueError:
                        rejected["metadata page"] += 1
                        continue
                    for item in items:
                        row = item["row"]
                        if row["language"] != "eng" or row["synthetic"] is not False:
                            continue
                        if (
                            not isinstance(row["endpoint_bool"], bool)
                            or row["id"] in ids
                        ):
                            continue
                        split = choose_split(part, row, args.train, args.dev)
                        key = (split, row["endpoint_bool"])
                        if counts[key] + planned[key] >= quotas[split]:
                            continue
                        planned[key] += 1
                        candidates.append((split, pool.submit(clip, source, item)))
                for split, pending in candidates:
                    try:
                        row, wav = pending.result()
                    except (ValueError, RuntimeError):
                        rejected["audio decode or download"] += 1
                        continue
                    previous = revisions.setdefault(source, row["source_revision"])
                    if row["source_revision"] != previous:
                        raise RuntimeError(
                            "dataset revision changed during acquisition"
                        )
                    if row["id"] in ids or row["pcm_sha256"] in hashes:
                        rejected["duplicate ID or PCM"] += 1
                        continue
                    if total_bytes + len(wav) > args.max_bytes:
                        rejected["byte limit"] += 1
                        continue
                    row["split"] = split
                    (folder / row["audio"]).write_bytes(wav)
                    total_bytes += len(wav)
                    ids.add(row["id"])
                    hashes.add(row["pcm_sha256"])
                    counts[split, row["complete"]] += 1
                    rows.append(row)
                write_manifest(folder, rows)
                print(
                    f"{part} through row {offsets[-1] + PAGE_SIZE}: {len(rows)} clips, "
                    f"{total_bytes / 1e6:.1f} MB, {dict(counts)}, {dict(rejected)}",
                    flush=True,
                )
                if rejected["byte limit"]:
                    break

    summary = {
        "datasets": {
            name: f"https://huggingface.co/datasets/{name}"
            for name in DATASETS.values()
        },
        "attribution": "Pipecat Smart Turn v3.2 contributors, as listed in the upstream dataset cards",
        "license_url": LICENSE_URL,
        "revisions": revisions,
        "sample_rate": SAMPLE_RATE,
        "audio_changes": "averaged channels to mono, polyphase resampled if needed, retained last 5.12 seconds, quantized to PCM16",
        "selection": "balanced human English clips from the dataset viewer; deterministic ID-based validation partition; exact decoded PCM deduplication across partitions",
        "limitations": "Clip-level labels only. No speaker IDs supplied. Dataset-origin and near-duplicate separation are not established; the official test set is upstream related.",
        "counts": {
            split: {str(label).lower(): counts[split, label] for label in (False, True)}
            for split in quotas
        },
        "audio_bytes": total_bytes,
        "accepted_source_download_bytes": sum(row["download_bytes"] for row in rows),
        "manifest_sha256": sha256((folder / "manifest.jsonl").read_bytes()).hexdigest(),
        "excluded_this_run": dict(rejected),
    }
    (folder / "provenance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if any(
        counts[split, label] != quotas[split]
        for split in quotas
        for label in (False, True)
    ):
        raise SystemExit(
            "Acquisition stopped before every quota was filled. See the manifest and provenance."
        )


if __name__ == "__main__":
    main()
