"""Shared generation operations for the command line and web editor."""

import copy
import importlib.util
import subprocess
import uuid
import shutil
import sys
from pathlib import Path

from . import backends
from .project import validate_project
from .provenance import asset_fingerprints, generation_key
from .routing import choose_backend
from .scheduler import Worker, Job, CommandJob, schedule, execute_commands


GENERATION_SIZES = {
    "sdxl": (1024, 1024),
    "cogvideox": (720, 480),
    "wan": (832, 480),
    "ltx": (768, 512),
    "wan22_t2v": (832, 480),
    "wan22_i2v": (832, 480),
    "wan22_ti2v": (832, 480),
    "hunyuan15_t2v": (832, 480),
    "hunyuan15_i2v": (832, 480),
    "ltx2": (768, 512),
    "qwen_image": (1024, 1024),
    "flux2_klein": (1024, 1024),
}


def ensure_model_dependencies() -> None:
    if all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "transformers")):
        return
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Install optional models with: uv pip install 'videogenerator[models]'")
    subprocess.run(["uv", "sync", "--project", str(root), "--extra", "models"], check=True)


def ensure_models(backend: str) -> None:
    spec = backends.get_backend(backend)
    if not spec.get("model_id"):
        return
    ensure_model_dependencies()


def add_generation(project: dict, segment_id: str, backend: str, directory: Path,
                   seed: int = 0, input_asset_id: str | None = None,
                   enable: bool = False, span_duration: float | None = None) -> dict:
    """Generate first, then return a new project with a switchable alternative track."""
    validate_project(project)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be an integer between 0 and 2^32 - 1")
    segment = next((segment for track in project["tracks"]
                    for segment in track["segments"] if segment["id"] == segment_id), None)
    if segment is None:
        raise ValueError(f"Unknown segment: {segment_id}")
    segment = copy.deepcopy(segment)
    if span_duration is not None:
        segment["duration"] = span_duration

    assets_by_id = {asset["id"]: asset for asset in project["assets"]}
    keyframes = segment.get("keyframes") or []
    if backend == "auto":
        requested_inputs = ["image"] if input_asset_id or keyframes else []
        backend = choose_backend("video", input_kinds=requested_inputs,
                                 keyframes=len(keyframes), require_executable=True)
    spec = backends.get_backend(backend)
    kind = spec["kind"]
    if kind not in {"image", "video", "audio"}:
        raise ValueError("Title cards are rendered directly; choose an image, video or audio backend")

    input_asset = None
    if input_asset_id:
        input_asset = assets_by_id.get(input_asset_id)
        if input_asset is None:
            raise ValueError("Unknown conditioning asset")
        if input_asset["kind"] != "image" or "image" not in spec.get("inputs", []):
            raise ValueError(f"{backend} does not accept this conditioning asset type")

    effective_keyframes = [
        {**keyframe, "path": assets_by_id[keyframe["asset_id"]]["path"]}
        for keyframe in keyframes
    ]
    # A single first-frame reference can be mapped onto ordinary image-to-video backends.
    if keyframes and not spec.get("keyframes"):
        if (len(keyframes) == 1 and keyframes[0]["frame"] == 0
                and "image" in spec.get("inputs", [])):
            input_asset = assets_by_id[keyframes[0]["asset_id"]]
            if input_asset["kind"] != "image":
                raise ValueError("Frame-zero conditioning requires an image")
            effective_keyframes = []
        else:
            raise ValueError(f"{backend} does not support these keyframes")

    ensure_models(backend)
    directory.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    output = directory / f"{backend}-{identifier}{ {'image': '.png', 'video': '.mp4', 'audio': '.wav'}[kind]}"
    width, height = GENERATION_SIZES.get(
        backend, (project["settings"]["width"], project["settings"]["height"]))

    referenced = []
    if input_asset:
        referenced.append(input_asset)
    for keyframe in keyframes:
        asset = assets_by_id[keyframe["asset_id"]]
        if asset not in referenced:
            referenced.append(asset)
    prompt = segment.get("prompt") or segment.get("text") or project["prompt"]
    cache_key = generation_key(
        backend=backend, model_revision=spec.get("model_id"), prompt=prompt, seed=seed,
        width=width, height=height, duration=segment["duration"], keyframes=keyframes,
        assets=asset_fingerprints(referenced),
    )
    cache_dir = directory / ".cache"
    cached = cache_dir / f"{cache_key}{output.suffix}"
    cache_hit = cached.is_file()
    try:
        if cache_hit:
            shutil.copy2(cached, output)
        else:
            backends.generate(
                backend, prompt, output, seed=seed, duration=segment["duration"],
                width=width, height=height,
                input_path=Path(input_asset["path"]) if input_asset else None,
                keyframes=effective_keyframes or None,
            )
            cache_dir.mkdir(exist_ok=True)
            shutil.copy2(output, cached)
    except Exception:
        output.unlink(missing_ok=True)
        raise

    result = copy.deepcopy(project)
    result["assets"].append({"id": identifier, "name": output.name,
                             "path": str(output.resolve()), "kind": kind})
    alternative = copy.deepcopy(segment)
    alternative.update(id=uuid.uuid4().hex, asset_id=identifier, backend=backend,
                       seed=seed, source_start=0, speed=1)
    alternative["generation"] = {
        "backend": backend,
        "model_id": spec.get("model_id"),
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "seed": seed,
        "prompt": prompt,
        "width": width,
        "height": height,
        "duration": segment["duration"],
        "inputs": asset_fingerprints(referenced),
        "keyframes": copy.deepcopy(keyframes),
    }
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
    _check_modalities(backend, voice, music)
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


def generate_storyboard_parallel(project: dict, backend: str, directory: Path,
                                 voice: str | None = None, music: str | None = None,
                                 seed: int = 0, hardware_info: dict | None = None) -> dict:
    """Generate independent storyboard visuals across available GPUs in subprocesses."""
    validate_project(project)
    _check_modalities(backend, voice, music)
    directory.mkdir(parents=True, exist_ok=True)
    cache_dir = directory / ".cache"
    cache_dir.mkdir(exist_ok=True)
    hardware_info = hardware_info or backends.hardware()
    gpus = hardware_info.get("gpus", [])
    workers = [
        Worker(id=str(index), vram_gb=float(gpu.get("memory_gb", 0) or 0))
        for index, gpu in enumerate(gpus)
    ] or [Worker(id="cpu", vram_gb=0)]

    assets_by_id = {asset["id"]: asset for asset in project["assets"]}
    storyboard = list(project["tracks"][0]["segments"])
    plans = []
    jobs = []
    commands = {}
    installed = set()

    for index, segment in enumerate(storyboard):
        if segment.get("asset_id"):
            continue
        keyframes = segment.get("keyframes") or []
        selected = backend
        if selected == "auto":
            selected = choose_backend(
                "video",
                input_kinds=["image"] if keyframes else [],
                keyframes=len(keyframes),
                require_executable=True,
            )
        spec = backends.get_backend(selected)
        if spec["kind"] not in {"image", "video"}:
            raise ValueError("Parallel storyboard backend must generate images or video")
        if selected not in installed:
            ensure_models(selected)
            installed.add(selected)

        width, height = GENERATION_SIZES.get(
            selected, (project["settings"]["width"], project["settings"]["height"])
        )
        extension = ".png" if spec["kind"] == "image" else ".mp4"
        identifier = uuid.uuid4().hex
        output = directory / f"{selected}-{identifier}{extension}"
        prompt = segment.get("prompt") or segment.get("text") or project["prompt"]
        referenced = []
        argv = [
            sys.executable, "-m", "videogenerator", "generate", selected, prompt,
            "--output", str(output), "--seed", str(seed + index),
            "--duration", str(segment["duration"]), "--width", str(width),
            "--height", str(height),
        ]

        effective_keyframes = []
        for keyframe in keyframes:
            asset = assets_by_id[keyframe["asset_id"]]
            if asset not in referenced:
                referenced.append(asset)
            effective_keyframes.append({**keyframe, "path": asset["path"]})

        if effective_keyframes and not spec.get("keyframes"):
            if (len(effective_keyframes) == 1 and effective_keyframes[0]["frame"] == 0
                    and "image" in spec.get("inputs", [])):
                input_asset = assets_by_id[keyframes[0]["asset_id"]]
                if input_asset["kind"] != "image":
                    raise ValueError("Frame-zero conditioning requires an image")
                argv += ["--input", input_asset["path"]]
                effective_keyframes = []
            else:
                raise ValueError(f"{selected} does not support these keyframes")
        else:
            for keyframe in effective_keyframes:
                argv += ["--keyframe", f"{keyframe['frame']}={keyframe['path']}"]

        key = generation_key(
            backend=selected, model_revision=spec.get("model_id"), prompt=prompt,
            seed=seed + index, width=width, height=height,
            duration=segment["duration"], keyframes=keyframes,
            assets=asset_fingerprints(referenced),
        )
        cached = cache_dir / f"{key}{extension}"
        cache_hit = cached.is_file()
        if cache_hit:
            shutil.copy2(cached, output)
        else:
            required_vram = float(spec.get("vram_gb", 0) or 0)
            if not gpus and spec.get("device") != "cuda":
                required_vram = 0
            job = Job(id=identifier, vram_gb=required_vram)
            jobs.append(job)
            commands[identifier] = CommandJob(id=identifier, argv=argv)

        plans.append({
            "identifier": identifier, "segment": segment, "backend": selected,
            "spec": spec, "output": output, "cache": cached, "cache_key": key,
            "cache_hit": cache_hit, "prompt": prompt, "seed": seed + index,
            "width": width, "height": height, "keyframes": keyframes,
            "referenced": referenced,
        })

    if jobs:
        assignments = schedule(jobs, workers)
        execute_commands(assignments, commands)
        for plan in plans:
            if not plan["cache_hit"]:
                shutil.copy2(plan["output"], plan["cache"])

    result = copy.deepcopy(project)
    for plan in plans:
        spec = plan["spec"]
        kind = spec["kind"]
        result["assets"].append({
            "id": plan["identifier"], "name": plan["output"].name,
            "path": str(plan["output"].resolve()), "kind": kind,
        })
        alternative = copy.deepcopy(plan["segment"])
        alternative.update(
            id=uuid.uuid4().hex, asset_id=plan["identifier"],
            backend=plan["backend"], seed=plan["seed"], source_start=0, speed=1,
        )
        alternative["generation"] = {
            "backend": plan["backend"], "model_id": spec.get("model_id"),
            "cache_key": plan["cache_key"], "cache_hit": plan["cache_hit"],
            "seed": plan["seed"], "prompt": plan["prompt"],
            "width": plan["width"], "height": plan["height"],
            "duration": plan["segment"]["duration"],
            "inputs": asset_fingerprints(plan["referenced"]),
            "keyframes": copy.deepcopy(plan["keyframes"]),
        }
        result["tracks"].append({
            "id": uuid.uuid4().hex,
            "name": f"{plan['backend']} / seed {plan['seed']} / {plan['segment']['id']}",
            "kind": "video", "enabled": True, "segments": [alternative],
        })

    if voice:
        for index, segment in enumerate(storyboard):
            result = add_generation(
                result, segment["id"], voice, directory, seed + index, enable=True
            )
    if music and storyboard:
        total = max(s["start"] + s["duration"] for s in storyboard)
        result = add_generation(
            result, storyboard[0]["id"], music, directory, seed,
            enable=True, span_duration=total,
        )
        result["tracks"][-1]["segments"][0]["duck"] = "soft"
    validate_project(result)
    return result


def _check_modalities(backend, voice, music):
    if backend not in {"title", "auto"} and backends.get_backend(backend)["kind"] not in {"image", "video"}:
        raise ValueError("Storyboard backend must generate image or video assets")
    if voice and backends.get_backend(voice).get("modality") != "voice":
        raise ValueError("Voice backend must generate speech")
    if music:
        spec = backends.get_backend(music)
        if spec["kind"] != "audio" or spec.get("modality") == "voice":
            raise ValueError("Music backend must generate a musical score")


def estimate_plan(project: dict, backend: str | None = None, voice: str | None = None,
                  music: str | None = None, hardware_info: dict | None = None) -> dict:
    validate_project(project)
    _check_modalities(backend or "title", voice, music)
    segments = []
    storyboard = project["tracks"][0]["segments"] if project["tracks"] else []
    for track in project["tracks"]:
        if not track["enabled"]:
            continue
        for original in track["segments"]:
            segment = copy.deepcopy(original)
            if backend and track is project["tracks"][0] and not segment.get("asset_id"):
                if backend == "auto":
                    input_kinds = ["image"] if segment.get("keyframes") else []
                    segment["backend"] = choose_backend(
                        "video", input_kinds=input_kinds,
                        keyframes=len(segment.get("keyframes") or []),
                    )
                else:
                    segment["backend"] = backend
            selected = segment["backend"]
            if selected in GENERATION_SIZES:
                segment["width"], segment["height"] = GENERATION_SIZES[selected]
            segments.append(segment)
    if voice:
        segments.extend({"id": f"planned-voice-{s['id']}", "backend": voice,
                         "duration": s["duration"]} for s in storyboard)
    if music and storyboard:
        segments.append({"id": "planned-score", "backend": music,
                         "duration": max(s["start"] + s["duration"] for s in storyboard)})
    return backends.estimate({"settings": project["settings"], "segments": segments}, hardware_info)
