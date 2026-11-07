"""Offline resume integrity checks; no training or scientific packages needed."""

from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
import wave

from joint_manifest import validate_saved_wav


class ResumeAudioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "clip.wav"
        self.pcm = b"\x01\x00\xff\x7f\x00\x80\x00\x00"
        self.write_wav()

    def write_wav(self, rate=16000, channels=1, width=2):
        with wave.open(str(self.path), "wb") as audio:
            audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            audio.writeframes(self.pcm)

    def validate(self, **changes):
        arguments = dict(
            samples=4, pcm_sha256=sha256(self.pcm).hexdigest(),
            sample_rate=16000, max_samples=81920,
        )
        arguments.update(changes)
        return validate_saved_wav(self.path, **arguments)

    def test_valid_pcm_returns_file_size(self):
        self.assertEqual(self.validate(), self.path.stat().st_size)

    def test_changed_audio_with_same_length_is_rejected(self):
        original_hash = sha256(self.pcm).hexdigest()
        self.pcm = b"\x00" * len(self.pcm)
        self.write_wav()
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            self.validate(pcm_sha256=original_hash)

    def test_sample_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample count differs"):
            self.validate(samples=3)

    def test_non_pcm16_mono_16khz_is_rejected(self):
        for arguments in ({"rate": 8000}, {"channels": 2}, {"width": 1}):
            with self.subTest(arguments=arguments):
                self.write_wav(**arguments)
                with self.assertRaisesRegex(ValueError, "mono PCM16"):
                    self.validate()

    def test_truncated_pcm_is_rejected_even_if_remaining_bytes_match_hash(self):
        self.path.write_bytes(self.path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "truncated PCM"):
            self.validate(pcm_sha256=sha256(self.pcm[:-2]).hexdigest())

    def test_invalid_wav_header_is_rejected(self):
        self.path.write_bytes(b"not a WAV")
        with self.assertRaisesRegex(ValueError, "not a valid PCM WAV"):
            self.validate()

    def test_invalid_manifest_count_and_hash_are_rejected(self):
        for count in (0, -1, True, 81921, 4.0):
            with self.subTest(samples=count):
                with self.assertRaisesRegex(ValueError, "Invalid retained sample count"):
                    self.validate(samples=count)
        for checksum in (None, "", "g" * 64):
            with self.subTest(checksum=checksum):
                with self.assertRaisesRegex(ValueError, "Invalid retained PCM checksum"):
                    self.validate(pcm_sha256=checksum)

    def test_missing_audio_is_rejected(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
