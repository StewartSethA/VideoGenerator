"""Portable, local-media timeline documents."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path


_KINDS = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image",
    ".bmp": "image", ".gif": "image", ".mp4": "video", ".mov": "video",
    ".mkv": "video", ".webm": "video", ".avi": "video", ".wav": "audio",
    ".mp3": "audio", ".m4a": "audio", ".aac": "audio", ".ogg": "audio",
    ".flac": "audio",
}


def import_assets(folder: Path) -> list[dict]:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("Asset folder must be a directory")
    result = []
    def chronological(path):
        return [(1, int(part)) if part.isdigit() else (0, part.lower())
                for part in re.split(r"(\d+)", str(path.relative_to(folder)))]

    for path in sorted(folder.rglob("*"), key=chronological):
        kind = _KINDS.get(path.suffix.lower())
        if not kind or not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(folder):
            raise ValueError("Asset symlink escapes asset folder")
        sidecar = path.with_suffix(".txt")
        description = ""
        tags = []
        if sidecar.is_file() and sidecar.resolve().is_relative_to(folder):
            lines = sidecar.read_text(encoding="utf-8", errors="replace").splitlines()
            description_lines = []
            for line in lines:
                if line.lower().startswith("tags:"):
                    tags.extend(tag.strip() for tag in line[5:].split(",") if tag.strip())
                elif line.strip():
                    description_lines.append(line.strip())
            description = " ".join(description_lines)[:20000]
            tags = tags[:100]
        result.append({
            "id": f"asset-{len(result) + 1}", "name": path.name,
            "path": str(resolved), "kind": kind,
            "description": description, "tags": tags,
        })
        if len(result) > 1000:
            raise ValueError("Too many assets")
    return result


def _segment(identifier, start, duration, text="", asset_id=None, role="source"):
    return {
        "id": identifier, "start": start, "duration": duration,
        "source_start": 0, "speed": 1, "asset_id": asset_id,
        "text": text, "prompt": text, "seed": 0, "backend": "title",
        "volume": 1, "fade_in": 0, "fade_out": 0, "duck": "none",
        "role": role, "keyframes": [],
        "transform": {"x": 0, "y": 0, "scale": 1, "rotation": 0, "opacity": 1},
        "effects": [],
    }


def create_project(prompt: str, assets: Path | None = None,
                   script: str | None = None, duration: float = 4,
                   width: int = 1280, height: int = 720, fps: int = 24) -> dict:
    _text(prompt, "prompt")
    if script is not None:
        _text(script, "script")
    _number(duration, "duration", 0.01, 86400)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", script or prompt)
                 if s.strip()] or ["Untitled"]
    if len(sentences) > 1000:
        raise ValueError("Too many storyboard segments")
    total_duration = duration * len(sentences)
    _number(total_duration, "total duration", 0.01, 86400)
    imported = import_assets(assets) if assets is not None else []
    visuals = [a for a in imported if a["kind"] != "audio"]
    storyboard, sources = [], []
    used = set()
    for index, sentence in enumerate(sentences):
        words = set(re.findall(r"[^\W_]+", sentence.lower()))
        candidates = [a for a in visuals if a["id"] not in used] or visuals
        # Descriptive filenames win; chronological filename order is the fallback.
        def asset_words(asset):
            text = " ".join([
                Path(asset["name"]).stem,
                asset.get("description", ""),
                " ".join(asset.get("tags", [])),
            ]).lower()
            return set(re.findall(r"[^\W_]+", text))
        selected = max(candidates, key=lambda a: len(words & asset_words(a))) if candidates else None
        if selected:
            used.add(selected["id"])
        start = duration * index
        length = duration
        storyboard.append(_segment(f"scene-{index + 1}", start, length,
                                   sentence, selected["id"] if selected else None))
        if selected and selected["kind"] == "video":
            sources.append(_segment(f"source-{index + 1}", start, length,
                                    asset_id=selected["id"]))
    tracks = [{"id": "storyboard", "name": "Storyboard", "kind": "video",
               "enabled": True, "segments": storyboard}]
    if sources:
        tracks.append({"id": "source-audio", "name": "Source audio", "kind": "audio",
                       "enabled": True, "segments": sources})
    music = [a for a in imported if a["kind"] == "audio"]
    for index, asset in enumerate(music):
        tracks.append({
            "id": f"music-{index + 1}", "name": asset["name"], "kind": "audio",
            "enabled": True, "segments": [
                _segment(f"music-segment-{index + 1}", 0, total_duration,
                         asset_id=asset["id"], role="music")
            ],
        })
    project = {
        "version": 1, "prompt": prompt,
        "settings": {"width": width, "height": height, "fps": fps},
        "assets": imported, "global_asset_ids": [a["id"] for a in music],
        "tracks": tracks, "captions": [
            {"start": scene["start"], "duration": scene["duration"], "text": scene["text"]}
            for scene in storyboard if scene["asset_id"] is not None
        ],
    }
    validate_project(project)
    return project


def _text(value, name, limit=20000):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise ValueError(f"Invalid {name}")


def _number(value, name, low=0, high=86400):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not low <= value <= high or not math.isfinite(value)):
        raise ValueError(f"Invalid {name}")


def _list(value, name, limit=1000):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"Invalid {name}")
    return value


def validate_project(project: dict, asset_root: Path | None = None) -> None:
    """Validate documents, limiting each to 1,000 tracks and 1,000 total segments."""
    if not isinstance(project, dict):
        raise ValueError("Project must be an object")
    if type(project.get("version")) is not int or project["version"] != 1:
        raise ValueError("Unsupported project version")
    _text(project.get("prompt"), "prompt")
    settings = project.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Invalid settings")
    for name, low, high in (("width", 16, 4096), ("height", 16, 4096), ("fps", 1, 120)):
        value = settings.get(name)
        if type(value) is not int:
            raise ValueError(f"Invalid {name}")
        _number(value, name, low, high)
        if name != "fps" and value % 2:
            raise ValueError("Dimensions must be even")
    audio_settings = settings.get("audio")
    if audio_settings is not None:
        if not isinstance(audio_settings, dict) or type(audio_settings.get("normalize", False)) is not bool:
            raise ValueError("Invalid audio settings")
        for name, low, high, default in (
            ("target_lufs", -70, -5, -14), ("true_peak", -9, 0, -1), ("lra", 1, 50, 11)
        ):
            _number(audio_settings.get(name, default), name, low, high)
    root = Path(asset_root).resolve() if asset_root is not None else None
    if root is not None and not root.is_dir():
        raise ValueError("Asset root must be a directory")
    ids = set()

    def identifier(item):
        if not isinstance(item, dict):
            raise ValueError("Expected object")
        value = item.get("id")
        _text(value, "id", 200)
        if not value or value in ids:
            raise ValueError("Missing or duplicate id")
        ids.add(value)
        return value

    assets = {}
    for asset in _list(project.get("assets"), "assets"):
        aid = identifier(asset)
        _text(asset.get("name"), "asset name", 1000)
        _text(asset.get("path"), "asset path", 4096)
        if asset.get("kind") not in ("video", "image", "audio"):
            raise ValueError("Invalid asset kind")
        if "description" in asset:
            _text(asset.get("description"), "asset description")
        if "tags" in asset:
            tags = _list(asset.get("tags"), "asset tags", 100)
            for tag in tags:
                _text(tag, "asset tag", 200)
        path = Path(asset["path"])
        if not path.is_absolute():
            raise ValueError("Asset must be an absolute local file")
        if root is not None:
            # Reject lexical escapes before touching the filesystem, then reject
            # symlink escapes before opening/probing the referenced media.
            if os.path.commonpath((str(root), os.path.abspath(path))) != str(root):
                raise ValueError("Asset path escapes asset root")
            if os.path.commonpath((str(root), str(path.resolve()))) != str(root):
                raise ValueError("Asset symlink escapes asset root")
        if not path.is_file():
            raise ValueError("Asset must be an existing local file")
        assets[aid] = asset
    global_ids = _list(project.get("global_asset_ids"), "global assets")
    if any(not isinstance(aid, str) or aid not in assets for aid in global_ids):
        raise ValueError("Unknown global asset")
    if len(set(global_ids)) != len(global_ids):
        raise ValueError("Duplicate global asset")
    count = 0
    for track in _list(project.get("tracks"), "tracks", 1000):
        identifier(track)
        _text(track.get("name"), "track name", 1000)
        if track.get("kind") not in ("video", "audio") or type(track.get("enabled")) is not bool:
            raise ValueError("Invalid track kind or enabled flag")
        for segment in _list(track.get("segments"), "segments"):
            count += 1
            if count > 1000:
                raise ValueError("Too many segments")
            identifier(segment)
            for key in ("start", "source_start", "fade_in", "fade_out"):
                _number(segment.get(key), key)
            _number(segment.get("duration"), "duration", 0.001)
            _number(segment.get("speed"), "speed", 0.05, 20)
            _number(segment.get("volume"), "volume", 0, 10)
            if segment["start"] + segment["duration"] > 86400:
                raise ValueError("Timeline is too long")
            if segment["source_start"] + segment["duration"] * segment["speed"] > 86400:
                raise ValueError("Source interval is too long")
            if any(segment[key] > segment["duration"] for key in ("fade_in", "fade_out")):
                raise ValueError("Fade exceeds segment duration")
            for key in ("text", "prompt"):
                _text(segment.get(key), key)
            _text(segment.get("backend"), "backend", 100)
            seed = segment.get("seed")
            if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                raise ValueError("Invalid seed")
            if segment.get("duck") not in ("none", "soft", "hard"):
                raise ValueError("Invalid duck")
            if segment.get("role") not in ("voice", "music", "source"):
                raise ValueError("Invalid role")
            keyframes = _list(segment.get("keyframes"), "keyframes", 64)
            seen_frames = set()
            for keyframe in keyframes:
                if not isinstance(keyframe, dict):
                    raise ValueError("Invalid keyframe")
                frame = keyframe.get("frame")
                if type(frame) is not int or frame < 0 or frame > int(segment["duration"] * settings["fps"]):
                    raise ValueError("Invalid keyframe frame")
                if frame in seen_frames:
                    raise ValueError("Duplicate keyframe frame")
                seen_frames.add(frame)
                key_asset = keyframe.get("asset_id")
                if not isinstance(key_asset, str) or key_asset not in assets:
                    raise ValueError("Unknown keyframe asset")
                if assets[key_asset]["kind"] not in ("image", "video"):
                    raise ValueError("Keyframe asset must be visual")
                if "prompt" in keyframe:
                    _text(keyframe.get("prompt"), "keyframe prompt")
            transform = segment.get("transform", {"x": 0, "y": 0, "scale": 1, "rotation": 0, "opacity": 1})
            if not isinstance(transform, dict):
                raise ValueError("Invalid transform")
            for key, low, high, default in (
                ("x", -16384, 16384, 0), ("y", -16384, 16384, 0),
                ("scale", 0.01, 20, 1), ("rotation", -3600, 3600, 0),
                ("opacity", 0, 1, 1),
            ):
                _number(transform.get(key, default), f"transform {key}", low, high)
            effects = _list(segment.get("effects", []), "effects", 32)
            for effect in effects:
                if not isinstance(effect, dict) or effect.get("type") not in {"grayscale", "blur", "sharpen", "eq"}:
                    raise ValueError("Invalid effect")
                kind_effect = effect["type"]
                if kind_effect == "blur":
                    _number(effect.get("radius", 1), "blur radius", 0, 100)
                elif kind_effect == "sharpen":
                    _number(effect.get("amount", 1), "sharpen amount", 0, 10)
                elif kind_effect == "eq":
                    _number(effect.get("brightness", 0), "brightness", -1, 1)
                    _number(effect.get("contrast", 1), "contrast", 0, 4)
                    _number(effect.get("saturation", 1), "saturation", 0, 4)
            generation = segment.get("generation")
            if generation is not None:
                if not isinstance(generation, dict):
                    raise ValueError("Invalid generation provenance")
                for key in ("backend", "cache_key"):
                    if key in generation:
                        _text(generation[key], f"generation {key}", 1000)
            aid = segment.get("asset_id")
            if aid is not None:
                if not isinstance(aid, str) or aid not in assets:
                    raise ValueError("Unknown segment asset")
                kind = assets[aid]["kind"]
                if (track["kind"] == "video" and kind == "audio"
                        or track["kind"] == "audio" and kind == "image"):
                    raise ValueError("Incompatible segment asset")
            elif track["kind"] == "audio":
                raise ValueError("Audio segments require an asset")
    for caption in _list(project.get("captions"), "captions"):
        if not isinstance(caption, dict):
            raise ValueError("Invalid caption")
        _number(caption.get("start"), "caption start")
        _number(caption.get("duration"), "caption duration", 0.001)
        if caption["start"] + caption["duration"] > 86400:
            raise ValueError("Caption exceeds timeline limit")
        _text(caption.get("text"), "caption text")


def save_project(project: dict, path: Path) -> None:
    validate_project(project)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".project-", suffix=".json",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(project, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_project(path: Path) -> dict:
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise ValueError("Project file is too large")
    try:
        project = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid project JSON") from exc
    validate_project(project)
    return project
