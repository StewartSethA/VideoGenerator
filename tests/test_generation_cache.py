import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videogenerator.project import create_project
from videogenerator.workflow import add_generation


class GenerationCacheTests(unittest.TestCase):
    def test_second_identical_generation_hits_cache(self):
        project = create_project("Scene", duration=1, width=64, height=64)
        calls = []

        def fake_generate(backend, prompt, output, **kwargs):
            calls.append((backend, prompt, kwargs["seed"]))
            output.write_bytes(b"artifact")
            return output

        with tempfile.TemporaryDirectory() as tmp, \
             patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate", side_effect=fake_generate):
            first = add_generation(project, "scene-1", "sdxl", Path(tmp), seed=5)
            second = add_generation(project, "scene-1", "sdxl", Path(tmp), seed=5)

        self.assertEqual(len(calls), 1)
        self.assertFalse(first["tracks"][-1]["segments"][0]["generation"]["cache_hit"])
        self.assertTrue(second["tracks"][-1]["segments"][0]["generation"]["cache_hit"])
        self.assertEqual(
            first["tracks"][-1]["segments"][0]["generation"]["cache_key"],
            second["tracks"][-1]["segments"][0]["generation"]["cache_key"],
        )

    def test_prompt_or_seed_changes_cache_key(self):
        project = create_project("Scene", duration=1, width=64, height=64)

        def fake_generate(backend, prompt, output, **kwargs):
            output.write_bytes(f"{prompt}-{kwargs['seed']}".encode())
            return output

        with tempfile.TemporaryDirectory() as tmp, \
             patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate", side_effect=fake_generate):
            first = add_generation(project, "scene-1", "sdxl", Path(tmp), seed=1)
            project2 = create_project("Different", duration=1, width=64, height=64)
            second = add_generation(project2, "scene-1", "sdxl", Path(tmp), seed=1)
            third = add_generation(project, "scene-1", "sdxl", Path(tmp), seed=2)

        keys = {
            first["tracks"][-1]["segments"][0]["generation"]["cache_key"],
            second["tracks"][-1]["segments"][0]["generation"]["cache_key"],
            third["tracks"][-1]["segments"][0]["generation"]["cache_key"],
        }
        self.assertEqual(len(keys), 3)


if __name__ == "__main__":
    unittest.main()
