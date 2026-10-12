import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videogenerator.project import create_project
from videogenerator.workflow import add_generation, estimate_plan, generate_storyboard


class WorkflowTests(unittest.TestCase):
    def test_alternative_does_not_mutate_original(self):
        project = create_project("A scene", duration=1, width=64, height=64)
        with tempfile.TemporaryDirectory() as tmp:
            def generate(backend, prompt, output, **kwargs):
                output.write_bytes(b"generated")
                return output
            with patch("videogenerator.workflow.ensure_models"), patch("videogenerator.backends.generate", side_effect=generate):
                alternative = add_generation(project, project["tracks"][0]["segments"][0]["id"],
                                             "sdxl", Path(tmp), seed=42)
            self.assertEqual(len(project["assets"]), 0)
            self.assertFalse(alternative["tracks"][-1]["enabled"])
            self.assertEqual(alternative["tracks"][-1]["segments"][0]["seed"], 42)
            self.assertEqual(alternative["assets"][-1]["kind"], "image")

    def test_generation_uses_model_size_not_export_canvas(self):
        project = create_project("Scene", width=320, height=180)
        def generate(backend, prompt, output, **kwargs):
            self.assertEqual((kwargs["width"], kwargs["height"]), (768, 512))
            output.write_bytes(b"video")
        with tempfile.TemporaryDirectory() as tmp, patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate", side_effect=generate):
            add_generation(project, "scene-1", "ltx", Path(tmp))

    def test_incompatible_modality_rejected(self):
        project = create_project("Scene")
        for options in ({"backend": "bark"}, {"backend": "title", "voice": "sdxl"},
                        {"backend": "title", "music": "espeak"}, {"backend": "missing"}):
            with self.assertRaises(ValueError):
                generate_storyboard(project, directory=Path("/tmp"), **options)

    def test_partial_output_removed_on_failure(self):
        def generate(backend, prompt, output, **kwargs):
            output.write_bytes(b"partial")
            raise RuntimeError("failed")
        with tempfile.TemporaryDirectory() as tmp, patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate", side_effect=generate):
            with self.assertRaises(RuntimeError):
                add_generation(create_project("Scene"), "scene-1", "sdxl", Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failed_generation_leaves_project_unchanged(self):
        project = create_project("A scene")
        with tempfile.TemporaryDirectory() as tmp, patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                add_generation(project, "scene-1", "sdxl", Path(tmp))
        self.assertEqual(len(project["tracks"]), 1)

    def test_music_covers_whole_storyboard(self):
        project = create_project("First. Second.", duration=2)
        calls = []
        def generate(backend, prompt, output, **kwargs):
            calls.append(kwargs)
            output.write_bytes(b"audio")
            return output
        with tempfile.TemporaryDirectory() as tmp, patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate", side_effect=generate):
            result = generate_storyboard(project, "title", Path(tmp), music="musicgen")
        self.assertEqual(calls[0]["duration"], 4)
        self.assertEqual(result["tracks"][-1]["segments"][0]["duck"], "soft")

    def test_unknown_segment_does_not_install_models(self):
        with patch("videogenerator.workflow.ensure_models") as install:
            with self.assertRaisesRegex(ValueError, "Unknown segment"):
                add_generation(create_project("Scene"), "missing", "sdxl", Path("/tmp"))
            install.assert_not_called()

    def test_estimate_planned_video_uses_generation_resolution(self):
        project = create_project("Scene")
        with patch("videogenerator.backends.estimate", return_value={}) as estimate:
            estimate_plan(project, "wan", "espeak", "musicgen")
        plan = estimate.call_args.args[0]
        self.assertEqual(len(plan["segments"]), 3)
        self.assertEqual(plan["segments"][0]["backend"], "wan")
        self.assertEqual((plan["segments"][0]["width"], plan["segments"][0]["height"]), (832, 480))
        self.assertEqual(project["tracks"][0]["segments"][0]["backend"], "title")
