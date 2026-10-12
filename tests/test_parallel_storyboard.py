import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videogenerator.project import create_project
from videogenerator.workflow import generate_storyboard_parallel


class ParallelStoryboardTests(unittest.TestCase):
    def test_parallel_generation_builds_enabled_alternatives(self):
        project = create_project("First scene. Second scene.", duration=1, width=64, height=64)
        hardware = {
            "gpu_count": 2,
            "gpus": [{"name":"GPU0","memory_gb":24}, {"name":"GPU1","memory_gb":24}],
        }

        def fake_execute(assignments, commands, **kwargs):
            for command in commands.values():
                output = Path(command.argv[command.argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"video")
            return {key: object() for key in commands}

        with tempfile.TemporaryDirectory() as tmp, \
             patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.workflow.execute_commands", side_effect=fake_execute):
            result = generate_storyboard_parallel(
                project, "wan", Path(tmp), seed=10, hardware_info=hardware
            )

        alternatives = result["tracks"][1:]
        self.assertEqual(len(alternatives), 2)
        self.assertTrue(all(track["enabled"] for track in alternatives))
        generated = [track["segments"][0] for track in alternatives]
        self.assertEqual({segment["backend"] for segment in generated}, {"wan"})
        self.assertEqual({segment["seed"] for segment in generated}, {10, 11})
        self.assertTrue(all("generation" in segment for segment in generated))

    def test_parallel_auto_uses_router(self):
        project = create_project("Scene", duration=1, width=64, height=64)
        hardware = {"gpu_count":1, "gpus":[{"name":"GPU0","memory_gb":24}]}

        def fake_execute(assignments, commands, **kwargs):
            for command in commands.values():
                output = Path(command.argv[command.argv.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"video")
            return {}

        with tempfile.TemporaryDirectory() as tmp, \
             patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.workflow.choose_backend", return_value="wan") as choose, \
             patch("videogenerator.workflow.execute_commands", side_effect=fake_execute):
            result = generate_storyboard_parallel(project, "auto", Path(tmp), hardware_info=hardware)
        choose.assert_called()
        self.assertEqual(result["tracks"][-1]["segments"][0]["backend"], "wan")

    def test_parallel_rejects_insufficient_vram_before_launch(self):
        project = create_project("Scene")
        hardware = {"gpu_count":1, "gpus":[{"name":"GPU0","memory_gb":4}]}
        with tempfile.TemporaryDirectory() as tmp, patch("videogenerator.workflow.ensure_models"):
            with self.assertRaisesRegex(ValueError, "No worker"):
                generate_storyboard_parallel(project, "wan22_t2v", Path(tmp), hardware_info=hardware)


if __name__ == "__main__":
    unittest.main()
