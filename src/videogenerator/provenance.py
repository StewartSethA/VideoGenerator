"""Deterministic generation provenance and content-addressed cache helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_key(*, backend: str, model_revision: str | None, prompt: str, seed: int,
                   width: int, height: int, duration: float, keyframes: list,
                   assets: list) -> str:
    payload = {
        "backend": backend,
        "model_revision": model_revision,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "duration": duration,
        "keyframes": keyframes,
        "assets": assets,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def attach_provenance(segment: dict, provenance: dict) -> dict:
    result = copy.deepcopy(segment)
    result["generation"] = copy.deepcopy(provenance)
    return result


def asset_fingerprints(assets: list[dict]) -> list[dict]:
    result = []
    for asset in assets:
        item = {"id": asset["id"], "kind": asset["kind"], "name": asset["name"]}
        path = Path(asset["path"])
        if path.is_file():
            item["sha256"] = file_digest(path)
        result.append(item)
    return result
