"""Shared generation operations for the command line and web editor."""

import copy
import importlib.util
import subprocess
import uuid
from pathlib import Path

from . import backends
from .project import validate_project


def ensure_models(backend: str) -> None:
    spec = backends.get_backend(backend)
    if not spec.get("model_id"):
        return
    if all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers")):
        return
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Install optional models with: uv pip install 'videogenerator[models]'")
    subprocess.run(["uv", "sync", "--project", str(root), "--extra", "models"], check=True)


def add_generation(project: dict, segment_id: str, backend: str, directory: Path,
                   seed: int = 0, input_asset_id: str | None = None,
                   enable: bool = False, span_duration: float | None = None) -> dict:
    """Generate first, then return a new project with a switchable alternative track."""
    validate_project(project)
    segment = next((segment for track in project["tracks"]
                    for segment in track["segments"] if segment["id"] == segment_id), None)
    if segment is None:
        raise ValueError(f"Unknown segment: {segment_id}")
    segment = copy.deepcopy(segment)
    if span_duration is not None:
        segment["duration"] = span_duration
    spec = backends.get_backend(backend)
    kind = spec["kind"]
    if kind not in {"image", "video", "audio"}:
        raise ValueError("Title cards are rendered directly; choose an image, video or audio backend")
    input_asset = None
    if input_asset_id:
        input_asset = next((a for a in project["assets"] if a["id"] == input_asset_id), None)
        if input_asset is None:
            raise ValueError("Unknown conditioning asset")
    ensure_models(backend)
    directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    output = directory / f"{backend}-{identifier}{ {'image': '.png', 'video': '.mp4', 'audio': '.wav'}[kind]}"
    backends.generate(
        backend, segment.get("prompt") or segment.get("text") or project["prompt"],
        output, seed=seed, duration=segment["duration"],
        width=project["settings"]["width"], height=project["settings"]["height"],
        input_path=Path(input_asset["path"]) if input_asset else None,
        keyframes=segment.get("keyframes") or None,
    )
    result = copy.deepcopy(project)
    result["assets"].append({"id": identifier, "name": output.name,
                             "path": str(output.resolve()), "kind": kind})
    alternative = copy.deepcopy(segment)
    alternative.update(id=uuid.uuid4().hex, asset_id=identifier, backend=backend,
                       seed=seed, source_start=0, speed=1)
    if kind == "audio":
        alternative.update(role="voice" if backend in {"espeak", "bark"} else "music", volume=1)
    result["tracks"].append({
        "id": uuid.uuid4().hex, "name": f"{backend} / seed {seed} / {segment_id}",
        "kind": "audio" if kind == "audio" else "video",
        "enabled": enable, "segments": [alternative],
    })
    validate_project(result)
    return result


def generate_storyboard(project: dict, backend: str, directory: Path,
                        voice: str | None = None, music: str | None = None,
                        seed: int = 0) -> dict:
    result = project
    segments = list(project["tracks"][0]["segments"])
    for index, segment in enumerate(segments):
        if backend != "title" and not segment.get("asset_id"):
            result = add_generation(result, segment["id"], backend, directory, seed + index, enable=True)
        if voice:
            result = add_generation(result, segment["id"], voice, directory, seed + index, enable=True)
    if music and segments:
        total = max(s["start"] + s["duration"] for s in segments)
        result = add_generation(result, segments[0]["id"], music, directory, seed,
                                enable=True, span_duration=total)
        result["tracks"][-1]["segments"][0]["duck"] = "soft"
    return result
