"""Deterministic transcript cuts and optional, locally executed Whisper transcription."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import wave
from collections import Counter
from pathlib import Path

from .project import _segment, create_project, validate_project
from .render import _INPUT_SAFETY


_VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
_MODEL = "openai/whisper-small"


def _source(source: Path) -> Path:
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError("Source must be an existing local media file")
    if source.suffix.lower() not in _VIDEO | _AUDIO:
        raise ValueError("Source must be a supported audio or video file")
    return source


def _run(arguments: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(arguments, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Install FFmpeg and ffprobe to process source media") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Media processing failed: {(exc.stderr or '')[-2000:]}") from exc


def _number(value, name: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 86400 or not math.isfinite(value)
            or positive and value < 0.001):
        raise ValueError(f"Invalid {name}")
    return float(value)


def _words(transcript: list[dict], duration: float) -> list[dict]:
    if not isinstance(transcript, list) or not transcript or len(transcript) > 1000:
        raise ValueError("Transcript must contain 1–1000 timestamped entries")
    words = []
    previous_end = 0.0
    for entry in transcript:
        if not isinstance(entry, dict):
            raise ValueError("Transcript entries must be objects")
        start = _number(entry.get("start"), "transcript start")
        length = _number(entry.get("duration"), "transcript duration", positive=True)
        text = entry.get("text")
        if (not isinstance(text, str) or not text.strip()
                or len(text) > 20000 or "\x00" in text):
            raise ValueError("Transcript entries require nonempty text")
        end = start + length
        if start < previous_end - 1e-9 or end > duration + 1e-9:
            raise ValueError("Transcript must be ordered, non-overlapping and within source bounds")
        words.append({"start": start, "duration": length, "text": text.strip()})
        previous_end = end
    return words


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold()))


def _removed(words: list[dict], phrases: list[str]) -> set[int]:
    patterns = {_tokens(phrase) for phrase in phrases}
    removed = set()
    normalized = [_tokens(word["text"]) for word in words]
    for start in range(len(words)):
        for pattern in patterns:
            candidate = ()
            for end in range(start, len(words)):
                candidate += normalized[end]
                if candidate == pattern:
                    removed.update(range(start, end + 1))
                    break
                if not candidate or candidate != pattern[:len(candidate)]:
                    break
    return removed


def select_highlights(words: list[dict], target_duration: float) -> list[int]:
    """Select informative transcript entries under a duration budget.

    The default scorer is deterministic and local. It removes common filler phrases,
    groups speech into utterances, rewards lexical information, penalizes repetition,
    and returns retained entry indices in original order.
    """
    if (isinstance(target_duration, bool) or not isinstance(target_duration, (int, float))
            or not math.isfinite(target_duration) or target_duration <= 0):
        raise ValueError("target_duration must be finite and positive")
    if not isinstance(words, list) or not words:
        raise ValueError("words must be a nonempty transcript list")
    filler = _removed(words, ["um", "uh", "erm", "hmm", "you know"])
    available = [i for i in range(len(words)) if i not in filler]
    if not available:
        return []

    tokens_by_index = [_tokens(entry.get("text", "")) for entry in words]
    frequencies = Counter(
        token for i in available for token in tokens_by_index[i]
        if len(token) > 1
    )

    blocks = []
    current = []
    previous_end = None
    for index in available:
        entry = words[index]
        start = float(entry["start"])
        end = start + float(entry["duration"])
        boundary = (
            current and previous_end is not None and start - previous_end > 0.75
        )
        if boundary:
            blocks.append(current)
            current = []
        current.append(index)
        previous_end = end
        if re.search(r"[.!?][\"')\]]*$", entry.get("text", "").strip()):
            blocks.append(current)
            current = []
            previous_end = None
    if current:
        blocks.append(current)

    def block_score(indices):
        duration = sum(float(words[i]["duration"]) for i in indices)
        tokens = [token for i in indices for token in tokens_by_index[i] if len(token) > 1]
        if not tokens or duration <= 0:
            return -1e9
        rarity = sum(1 / math.sqrt(max(1, frequencies[token])) for token in tokens)
        diversity = len(set(tokens)) / len(tokens)
        repetition = 1 - diversity
        density = len(tokens) / duration
        return (rarity + diversity * 2 + min(density, 4) * 0.3 - repetition * 2) / duration

    ranked = sorted(blocks, key=lambda block: (-block_score(block), block[0]))
    selected = []
    used = 0.0
    for block in ranked:
        duration = sum(float(words[i]["duration"]) for i in block)
        if used + duration <= target_duration + 1e-9:
            selected.extend(block)
            used += duration

    if not selected:
        entries = sorted(
            available,
            key=lambda i: (
                -(sum(1 / math.sqrt(max(1, frequencies[t])) for t in tokens_by_index[i])
                  / max(float(words[i]["duration"]), 1e-6)),
                i,
            ),
        )
        for index in entries:
            duration = float(words[index]["duration"])
            if used + duration <= target_duration + 1e-9:
                selected.append(index)
                used += duration
    return sorted(selected)


def edit_media(source: Path, transcript: list[dict], *,
               remove_fillers: bool = True, max_pause: float = 0.35,
               drop_phrases: list[str] | None = None,
               keep_indices: list[int] | set[int] | None = None,
               width: int = 1280, height: int = 720, fps: int = 24) -> dict:
    """Cut exact transcript phrases and long pauses, not semantic summarization.

    Word-level entries give precise cuts. A phrase must occupy complete entries;
    speech inside a larger sentence entry cannot safely be cut without timestamps.
    """
    source = _source(source)
    max_pause = _number(max_pause, "max_pause")
    if type(remove_fillers) is not bool:
        raise ValueError("remove_fillers must be a boolean")
    if drop_phrases is None:
        drop_phrases = []
    if (not isinstance(drop_phrases, list) or len(drop_phrases) > 1000
            or any(not isinstance(p, str) or not p.strip() or len(p) > 20000
                   or "\x00" in p or not _tokens(p) for p in drop_phrases)):
        raise ValueError("drop_phrases must contain nonempty exact phrases")
    probe = _run(["ffprobe", "-v", "error", *_INPUT_SAFETY,
                  "-show_entries", "format=duration",
                  "-of", "default=noprint_wrappers=1:nokey=1", str(source)])
    try:
        duration = _number(float(probe.stdout.strip()), "source duration", positive=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Source must have a finite, positive duration") from exc
    words = _words(transcript, duration)
    phrases = list(drop_phrases)
    if remove_fillers:
        phrases += ["um", "uh", "erm", "hmm", "you know"]
    removed = _removed(words, phrases)
    if keep_indices is not None:
        if (not isinstance(keep_indices, (list, set, tuple))
                or any(type(i) is not int or not 0 <= i < len(words) for i in keep_indices)):
            raise ValueError("keep_indices must contain valid transcript indices")
        keep = set(keep_indices)
        removed.update(i for i in range(len(words)) if i not in keep)
    retained = [i for i in range(len(words)) if i not in removed]
    if not retained:
        raise ValueError("No speech remains after transcript editing")

    ranges = []
    captions = []
    timeline = 0.0
    previous_index = None
    for index in retained:
        word = words[index]
        start, end = word["start"], word["start"] + word["duration"]
        if previous_index is not None and index == previous_index + 1:
            gap = max(0.0, start - ranges[-1][1])
            pause = min(gap, max_pause)
            ranges[-1][1] += pause
            timeline += pause
            if gap <= max_pause + 1e-9:
                ranges[-1][1] = end
            else:
                ranges.append([start, end])
        else:
            ranges.append([start, end])
        captions.append({"start": timeline, "duration": word["duration"], "text": word["text"]})
        timeline += word["duration"]
        previous_index = index

    selected_text = " ".join(word["text"] for i, word in enumerate(words) if i not in removed)
    project = create_project(selected_text, duration=max(0.01, timeline),
                             width=width, height=height, fps=fps)
    project["name"] = selected_text[:200]
    project["description"] = "Transcript-based excerpt; exact phrase and pause cuts, not a semantic summary."
    kind = "video" if source.suffix.lower() in _VIDEO else "audio"
    project["assets"] = [{"id": "edited-source", "name": source.name,
                          "path": str(source), "kind": kind}]
    project["global_asset_ids"] = []
    project["tracks"] = []
    for track_kind in (("video", "audio") if kind == "video" else ("audio",)):
        segments = []
        position = 0.0
        for index, (start, end) in enumerate(ranges):
            segment = _segment(f"edited-{track_kind}-{index + 1}", position,
                               end - start, asset_id="edited-source")
            segment["source_start"] = start
            segments.append(segment)
            position += end - start
        project["tracks"].append({
            "id": f"edited-{track_kind}", "name": f"Original {track_kind}",
            "kind": track_kind, "enabled": True, "segments": segments,
        })
    project["captions"] = captions
    validate_project(project)
    return project


def auto_edit_media(source: Path, transcript: list[dict], target_duration: float, *,
                    remove_fillers: bool = True, max_pause: float = 0.35,
                    width: int = 1280, height: int = 720, fps: int = 24) -> dict:
    """Create a concise chronological edit from transcript information density."""
    keep = select_highlights(transcript, target_duration)
    if not keep:
        raise ValueError("Automatic edit found no speech to retain")
    return edit_media(
        source, transcript, remove_fillers=remove_fillers, max_pause=max_pause,
        keep_indices=keep, width=width, height=height, fps=fps,
    )


def transcribe(source: Path, model_id: str = _MODEL) -> list[dict]:
    """Download the allowlisted Whisper model on request and transcribe locally."""
    if model_id != _MODEL:
        raise ValueError(f"Only the allowlisted model {_MODEL!r} is supported")
    source = _source(source)
    try:
        import numpy as np
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Transcription needs optional models: run uv sync --extra models") from exc

    with tempfile.TemporaryDirectory(prefix=".transcription-") as directory:
        audio_path = Path(directory) / "source.wav"
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              *_INPUT_SAFETY, "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(audio_path)])
        with wave.open(str(audio_path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("Transcription requires mono 16kHz PCM audio")
            duration = audio.getnframes() / 16000
            samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            samples = samples.astype(np.float32) / 32768.0
        recognizer = pipeline(
            "automatic-speech-recognition", model=model_id, device=-1,
            trust_remote_code=False, model_kwargs={"use_safetensors": True},
        )
        result = recognizer({"array": samples, "sampling_rate": 16000},
                            return_timestamps="word", chunk_length_s=30)
    if not isinstance(result, dict) or not isinstance(result.get("chunks"), list):
        raise ValueError("Transcription did not return word timestamps")
    words = []
    for chunk in result["chunks"]:
        if not isinstance(chunk, dict):
            raise ValueError("Invalid transcription word")
        timestamp = chunk.get("timestamp")
        if (not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2
                or timestamp[0] is None or timestamp[1] is None):
            raise ValueError("Transcription word is missing its start or end timestamp")
        start = _number(timestamp[0], "transcription start")
        end = _number(timestamp[1], "transcription end")
        words.append({"start": start, "duration": end - start, "text": chunk.get("text")})
    return _words(words, duration)
