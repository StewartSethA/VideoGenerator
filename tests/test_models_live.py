"""Explicit opt-in real inference checks; never substitute mocks for these."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from videogenerator import backends


ENABLED = set(filter(None, os.environ.get("VIDEOGENERATOR_LIVE_MODELS", "").split(",")))


class LiveModelTests(unittest.TestCase):
    def check_model(self, name, suffix, width=512, height=512, duration=2):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / ("result" + suffix)
            backends.generate(name, "A calm ocean at sunrise", output,
                              width=width, height=height, duration=duration, seed=42)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 100)
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-t", "1",
                            "-f", "null", "-"], check=True, capture_output=True)

    @unittest.skipUnless("sdxl" in ENABLED, "Opt in to SDXL weights/inference")
    def test_sdxl(self):
        self.check_model("sdxl", ".png")

    @unittest.skipUnless("cogvideox" in ENABLED, "Opt in to CogVideoX weights/CUDA")
    def test_cogvideox(self):
        self.check_model("cogvideox", ".mp4", 720, 480, 6)

    @unittest.skipUnless("wan" in ENABLED, "Opt in to Wan weights/CUDA")
    def test_wan(self):
        self.check_model("wan", ".mp4", 832, 480)

    @unittest.skipUnless("ltx" in ENABLED, "Opt in to LTX weights/CUDA")
    def test_ltx(self):
        self.check_model("ltx", ".mp4")

    @unittest.skipUnless("musicgen" in ENABLED, "Opt in to MusicGen weights/inference")
    def test_musicgen(self):
        self.check_model("musicgen", ".wav")

    @unittest.skipUnless("bark" in ENABLED, "Opt in to Bark weights/inference")
    def test_bark(self):
        self.check_model("bark", ".wav")
