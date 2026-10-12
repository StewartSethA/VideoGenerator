import os
import sys
import unittest
from unittest.mock import patch

from videogenerator.scheduler import Worker, Job, schedule, CommandJob, execute_commands


class SchedulerExecutionTests(unittest.TestCase):
    def test_execute_commands_sets_worker_gpu_and_preserves_order(self):
        workers = [Worker(id="0", vram_gb=24), Worker(id="1", vram_gb=80)]
        jobs = [Job(id="a", vram_gb=8), Job(id="b", vram_gb=40)]
        assignments = schedule(jobs, workers)
        commands = {
            "a": CommandJob(id="a", argv=[sys.executable, "-c", "print('a')"]),
            "b": CommandJob(id="b", argv=[sys.executable, "-c", "print('b')"]),
        }
        seen = []

        def runner(argv, **kwargs):
            seen.append((argv, kwargs["env"].get("CUDA_VISIBLE_DEVICES")))
            class Result:
                returncode = 0
                stdout = "ok"
                stderr = ""
            return Result()

        result = execute_commands(assignments, commands, runner=runner)
        self.assertEqual(set(result), {"a", "b"})
        gpu_for = {argv[-1]: gpu for argv, gpu in seen}
        self.assertEqual(gpu_for["print('b')"], "1")
        self.assertIn(gpu_for["print('a')"], {"0", "1"})

    def test_failed_command_is_reported(self):
        assignment = schedule([Job(id="a", vram_gb=0)], [Worker(id="cpu", vram_gb=0)])

        def runner(*args, **kwargs):
            class Result:
                returncode = 7
                stdout = ""
                stderr = "boom"
            return Result()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            execute_commands(
                assignment,
                {"a": CommandJob(id="a", argv=["false"])},
                runner=runner,
            )

    def test_cpu_worker_does_not_force_cuda_visible_devices(self):
        assignment = schedule([Job(id="a", vram_gb=0)], [Worker(id="cpu", vram_gb=0)])
        envs = []

        def runner(argv, **kwargs):
            envs.append(kwargs["env"])
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""
            return Result()

        execute_commands(assignment, {"a": CommandJob(id="a", argv=["true"])}, runner=runner)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", envs[0])


if __name__ == "__main__":
    unittest.main()
