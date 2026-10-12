import unittest
from videogenerator.scheduler import Worker, Job, schedule


class SchedulerTests(unittest.TestCase):
    def test_vram_aware_assignment(self):
        workers = [
            Worker(id="gpu0", vram_gb=16),
            Worker(id="gpu1", vram_gb=80),
        ]
        jobs = [
            Job(id="small", vram_gb=8),
            Job(id="large", vram_gb=40),
            Job(id="tiny", vram_gb=4),
        ]
        result = schedule(jobs, workers)
        assigned = {item.job.id:item.worker.id for item in result}
        self.assertEqual(assigned["large"], "gpu1")
        self.assertIn(assigned["small"], {"gpu0","gpu1"})

    def test_unschedulable_job_fails(self):
        with self.assertRaisesRegex(ValueError, "No worker"):
            schedule([Job(id="huge", vram_gb=90)], [Worker(id="gpu0", vram_gb=24)])

    def test_cpu_job_can_use_cpu_worker(self):
        result = schedule([Job(id="cpu", vram_gb=0)], [Worker(id="cpu0", vram_gb=0)])
        self.assertEqual(result[0].worker.id, "cpu0")


if __name__ == "__main__":
    unittest.main()
