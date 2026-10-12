"""Small deterministic resource-aware scheduler used by local render workers."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess


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


@dataclass(frozen=True)
class CommandJob:
    id: str
    argv: list[str]
    cwd: str | None = None


def execute_commands(assignments: list[Assignment], commands: dict[str, CommandJob],
                     runner=subprocess.run) -> dict[str, object]:
    """Run assigned subprocess jobs concurrently, isolating CUDA visibility per worker."""
    if {assignment.job.id for assignment in assignments} != set(commands):
        raise ValueError("Assignments and command jobs must have identical ids")

    def run_one(assignment: Assignment):
        command = commands[assignment.job.id]
        env = os.environ.copy()
        if assignment.worker.id.isdigit() and assignment.worker.vram_gb > 0:
            env["CUDA_VISIBLE_DEVICES"] = assignment.worker.id
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        result = runner(
            command.argv, cwd=command.cwd, env=env,
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Job {command.id!r} failed with exit {result.returncode}: "
                f"{(result.stderr or result.stdout or '').strip()[-4000:]}"
            )
        return command.id, result

    completed = {}
    with ThreadPoolExecutor(max_workers=max(1, len({a.worker.id for a in assignments}))) as pool:
        futures = [pool.submit(run_one, assignment) for assignment in assignments]
        for future in as_completed(futures):
            identifier, result = future.result()
            completed[identifier] = result
    return completed
