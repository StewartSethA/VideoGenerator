"""Capability-based routing across built-in and optional ecosystem generators."""

from __future__ import annotations

from . import backends, ecosystem


def capabilities() -> list[dict]:
    rows = []
    for item in backends.catalog():
        kind = item["kind"]
        output = "image" if kind == "image" else "video" if kind in {"video", "title"} else "audio"
        rows.append({
            "id": item["id"],
            "output": output,
            "inputs": list(item.get("inputs", [])),
            "keyframes": bool(item.get("keyframes", False)),
            "executable": True,
            "source": "builtin",
            "vram_gb": item.get("vram_gb", 0),
        })
    for identifier, item in ecosystem.catalog()["models"].items():
        rows.append({
            "id": identifier,
            "output": item.get("output", item.get("kind", "unknown")),
            "inputs": list(item.get("inputs", [])),
            "keyframes": bool(item.get("keyframes", False)),
            "executable": bool(item.get("executable", False)),
            "source": "ecosystem",
            "vram_gb": item.get("vram_gb", 0),
        })
    return rows


def candidates(output: str, *, input_kinds: list[str] | None = None,
               keyframes: int = 0) -> list[dict]:
    input_kinds = input_kinds or []
    if output not in {"image", "video", "audio", "music", "voice", "audiovisual"}:
        raise ValueError(f"Unknown output kind {output!r}")
    result = []
    for row in capabilities():
        if row["output"] not in {output, "audiovisual"}:
            continue
        supported = set(row["inputs"])
        if any(kind not in supported for kind in input_kinds):
            continue
        if keyframes and not row["keyframes"]:
            continue
        score = 0
        score += 100 if row["executable"] else 0
        score += 10 * len(input_kinds)
        score += 20 if keyframes and row["keyframes"] else 0
        # Prefer purpose-built modern ecosystem candidates when executable is not required.
        if row["id"] == "ltx2" and keyframes:
            score += 10
        if row["id"] == "ltx" and input_kinds == ["image"]:
            score += 5
        if row["id"] == "sdxl" and output == "image" and not input_kinds:
            score += 5
        result.append({**row, "score": score})
    return sorted(result, key=lambda row: (-row["score"], row["id"]))


def choose_backend(output: str, *, input_kinds: list[str] | None = None,
                   keyframes: int = 0, require_executable: bool = True) -> str:
    options = candidates(output, input_kinds=input_kinds, keyframes=keyframes)
    if require_executable:
        options = [item for item in options if item["executable"]]
    if not options:
        qualifier = " executable" if require_executable else ""
        raise ValueError(f"No{qualifier} backend supports output={output!r}, "
                         f"inputs={input_kinds or []!r}, keyframes={keyframes}")
    return options[0]["id"]
