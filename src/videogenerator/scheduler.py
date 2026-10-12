"""Small deterministic resource-aware scheduler used by local render workers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Worker:
    id: str
    vram_gb: float = 0
    slots: int = 1


@dataclass(frozen=True)
class Job:
    id: str
    vram_gb: float = 0


@dataclass(frozen=True)
class Assignment:
    job: Job
    worker: Worker
    slot: int


def schedule(jobs: list[Job], workers: list[Worker]) -> list[Assignment]:
    if not workers and jobs:
        raise ValueError("No workers available")
    load = {worker.id: 0 for worker in workers}
    by_id = {worker.id: worker for worker in workers}
    assignments = []
    # Place the hardest jobs first; return assignments independent of job order.
    for job in sorted(jobs, key=lambda item: (-item.vram_gb, item.id)):
        eligible = [
            worker for worker in workers
            if worker.vram_gb >= job.vram_gb and load[worker.id] < max(1, worker.slots)
        ]
        if not eligible:
            # Slots represent simultaneous capacity; if all are occupied, reuse the least-loaded
            # eligible worker for the next wave rather than declaring the workload impossible.
            eligible = [worker for worker in workers if worker.vram_gb >= job.vram_gb]
        if not eligible:
            raise ValueError(f"No worker can satisfy job {job.id!r} requiring {job.vram_gb} GiB VRAM")
        worker = min(eligible, key=lambda item: (load[item.id] / max(1, item.slots),
                                                 item.vram_gb, item.id))
        slot = load[worker.id] % max(1, worker.slots)
        assignments.append(Assignment(job=job, worker=worker, slot=slot))
        load[worker.id] += 1
    return assignments
