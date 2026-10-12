"""Install/cache/launch optional external generation ecosystems.

This module intentionally uses only the Python standard library at import time.
Heavy dependencies and model weights are opt-in.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path


def _registry() -> dict:
    path = files("videogenerator").joinpath("model_registry.toml")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def catalog() -> dict:
    data = _registry()
    return {
        "models": {name: dict(spec) for name, spec in data.get("models", {}).items()},
        "frameworks": {name: dict(spec) for name, spec in data.get("frameworks", {}).items()},
    }


def home() -> Path:
    return Path(os.environ.get("VIDEOGENERATOR_HOME", Path.home() / ".cache" / "videogenerator")).expanduser()


def framework_root(name: str) -> Path:
    return home() / "frameworks" / name


def framework_env(name: str) -> Path:
    return home() / "envs" / name


def model_root(name: str) -> Path:
    return home() / "models" / name


def _run(command: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def _framework(name: str) -> dict:
    try:
        return catalog()["frameworks"][name]
    except KeyError as exc:
        raise ValueError(f"Unknown framework {name!r}") from exc


def _model(name: str) -> dict:
    try:
        return catalog()["models"][name]
    except KeyError as exc:
        raise ValueError(f"Unknown ecosystem model {name!r}") from exc


def install_framework(name: str, update: bool = False) -> Path:
    spec = _framework(name)
    root = framework_root(name)
    root.parent.mkdir(parents=True, exist_ok=True)
    framework_env(name).parent.mkdir(parents=True, exist_ok=True)

    if root.exists():
        if update:
            _run(["git", "pull", "--ff-only"], root)
    else:
        if not shutil.which("git"):
            raise RuntimeError("git is required to install external frameworks")
        _run(["git", "clone", "--depth", "1", spec["repo"], str(root)])

    mode = spec.get("install", "requirements")
    if mode == "uv":
        if not shutil.which("uv"):
            raise RuntimeError("uv is required for this framework")
        _run(["uv", "sync"], root)
        return root

    env = framework_env(name)
    python_version = str(spec.get("python", "3.11"))
    if not env.exists():
        if not shutil.which("uv"):
            raise RuntimeError("uv is required to create framework environments")
        _run(["uv", "venv", "--python", python_version, str(env)])

    py = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if mode == "pyproject" and (root / "pyproject.toml").exists():
        _run(["uv", "pip", "install", "--python", str(py), "-e", str(root)])
    elif (root / "requirements.txt").exists():
        _run(["uv", "pip", "install", "--python", str(py), "-r", str(root / "requirements.txt")])
    elif (root / "pyproject.toml").exists():
        _run(["uv", "pip", "install", "--python", str(py), "-e", str(root)])
    return root


def launch_command(name: str, port: int | None = None) -> tuple[list[str], Path]:
    spec = _framework(name)
    root = framework_root(name)
    if not root.exists():
        raise RuntimeError(f"{name} is not installed; run: videogenerator framework install {name}")
    template = spec.get("launch")
    if not template:
        raise RuntimeError(f"{name} has no declared native UI launcher")
    port = port or 7860

    if template.startswith("uv "):
        command = shlex.split(template.format(port=port))
    else:
        env = framework_env(name)
        py = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        expanded = template.format(port=port).replace("python ", shlex.quote(str(py)) + " ", 1)
        command = shlex.split(expanded)
    return command, root


def launch_framework(name: str, port: int | None = None, install: bool = True) -> subprocess.Popen:
    if install and not framework_root(name).exists():
        install_framework(name)
    command, cwd = launch_command(name, port)
    return subprocess.Popen(command, cwd=cwd)


def install_model(name: str, token: str | None = None, revision: str | None = None) -> Path:
    spec = _model(name)
    provider = spec.get("provider")
    destination = model_root(name)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if provider != "huggingface":
        raise RuntimeError(f"Unsupported model provider {provider!r}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is part of the optional model dependencies; run: uv sync --extra models"
        ) from exc

    snapshot_download(
        repo_id=spec["repo_id"],
        local_dir=destination,
        token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        revision=revision,
    )
    return destination


def status() -> dict:
    data = catalog()
    return {
        "home": str(home()),
        "tools": {
            "git": bool(shutil.which("git")),
            "uv": bool(shutil.which("uv")),
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "ffprobe": bool(shutil.which("ffprobe")),
        },
        "frameworks": {
            name: {
                "installed": framework_root(name).exists(),
                "path": str(framework_root(name)),
                "native_ui": bool(spec.get("native_ui")),
            }
            for name, spec in data["frameworks"].items()
        },
        "models": {
            name: {
                "cached": model_root(name).exists(),
                "path": str(model_root(name)),
                "repo_id": spec.get("repo_id"),
            }
            for name, spec in data["models"].items()
        },
    }


def format_catalog() -> str:
    return json.dumps(catalog(), indent=2, sort_keys=True)
