"""Tiny preparation regressions; requires NumPy/Torch but never trains a model."""

from argparse import Namespace
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cue_joint_training_test", ROOT / "scripts/train_joint.py")
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


class PreparationCutoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        # Two audible frames, then three real quiet frames before the annotated cutoff.
        self.pcm = np.concatenate((np.full(1024, 4096, dtype="<i2"), np.zeros(1536, dtype="<i2")))
        with wave.open(str(self.folder / "source.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(self.pcm.tobytes())
        self.row = dict(
            id="synthetic-cutoff", audio="source.wav", split="test", complete=True,
            source_silence_start_s=1.16, source_prefix_offset_samples=16000,
        )
        self.manifest = self.folder / "manifest.jsonl"
        self.manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")

    def prepare(self, annotated):
        output = self.folder / ("annotated" if annotated else "proxy")
        with redirect_stdout(io.StringIO()):
            training.prepare_eval(Namespace(
                manifest=self.manifest, split="test", output=output,
                annotated_cutoffs=annotated,
            ))
        result = json.loads((output / "test.jsonl").read_text(encoding="utf-8"))
        with wave.open(str(output / result["audio"]), "rb") as audio:
            pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").copy()
        return result, pcm

    def test_proxy_mode_trims_real_quiet_before_observation_tail(self):
        result, pcm = self.prepare(False)
        expected = np.concatenate((np.zeros(3 * 512, dtype="<i2"), self.pcm[:1024], np.zeros(40 * 512, dtype="<i2")))
        self.assertEqual(result["reference_end_ms"], 5 * 32)
        np.testing.assert_array_equal(pcm, expected)

    def test_annotated_mode_preserves_clip_and_reference_cutoff(self):
        result, pcm = self.prepare(True)
        expected = np.concatenate((np.zeros(3 * 512, dtype="<i2"), self.pcm, np.zeros(40 * 512, dtype="<i2")))
        self.assertEqual(result["reference_end_ms"], 8 * 32)
        np.testing.assert_array_equal(pcm, expected)
        self.assertEqual(training.cutoff_for(self.row, True), len(self.pcm))

    def test_inconsistent_annotation_is_rejected(self):
        self.row["source_silence_start_s"] = 2.0
        self.manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Annotated cutoff does not match"):
            self.prepare(True)

    def test_missing_annotation_falls_back_to_proxy(self):
        self.row.pop("source_silence_start_s")
        self.row.pop("source_prefix_offset_samples")
        self.manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        result, pcm = self.prepare(True)
        self.assertEqual(result["reference_end_ms"], 5 * 32)
        self.assertEqual(len(pcm), 45 * 512)


if __name__ == "__main__":
    unittest.main()
