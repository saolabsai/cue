"""Standard-library integrity checks for retained joint-model training audio."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
import wave


def validate_saved_wav(
    path: Path,
    *,
    samples: int,
    pcm_sha256: str,
    sample_rate: int,
    max_samples: int,
) -> int:
    """Verify manifest claims against PCM bytes; return retained file size.

    A resume must fail on changed or truncated audio rather than trusting an
    old manifest hash for duplicate detection and provenance.
    """
    if type(samples) is not int or not 0 < samples <= max_samples:
        raise ValueError("Invalid retained sample count")
    if not isinstance(pcm_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", pcm_sha256):
        raise ValueError("Invalid retained PCM checksum")
    try:
        with wave.open(str(path), "rb") as audio:
            if (
                audio.getframerate(),
                audio.getnchannels(),
                audio.getsampwidth(),
                audio.getcomptype(),
            ) != (sample_rate, 1, 2, "NONE"):
                raise ValueError("Existing audio must be mono PCM16 at the expected sample rate")
            if audio.getnframes() != samples:
                raise ValueError("Existing audio sample count differs from the manifest")
            pcm = audio.readframes(samples)
    except (wave.Error, EOFError) as error:
        raise ValueError("Existing audio is not a valid PCM WAV") from error
    if len(pcm) != samples * 2:
        raise ValueError("Existing audio has truncated PCM data")
    if sha256(pcm).hexdigest() != pcm_sha256:
        raise ValueError("Existing audio PCM checksum differs from the manifest")
    return path.stat().st_size
