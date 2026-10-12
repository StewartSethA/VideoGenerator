"""Deterministic timeline edit operations."""

from __future__ import annotations

import copy
import uuid

from .project import validate_project


def _track(project: dict, track_id: str) -> dict:
    track = next((item for item in project["tracks"] if item["id"] == track_id), None)
    if track is None:
        raise ValueError(f"Unknown track: {track_id}")
    return track


def split_segment(project: dict, track_id: str, segment_id: str, offset: float) -> dict:
    """Split a segment at an output-time offset measured from its own start."""
    validate_project(project)
    result = copy.deepcopy(project)
    track = _track(result, track_id)
    index = next((i for i, item in enumerate(track["segments"]) if item["id"] == segment_id), None)
    if index is None:
        raise ValueError(f"Unknown segment: {segment_id}")
    segment = track["segments"][index]
    if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not 0 < offset < segment["duration"]:
        raise ValueError("Split offset must be inside the segment")
    left = copy.deepcopy(segment)
    right = copy.deepcopy(segment)
    left["duration"] = float(offset)
    left["fade_out"] = min(left["fade_out"], left["duration"])
    right["id"] = uuid.uuid4().hex
    right["start"] = segment["start"] + float(offset)
    right["duration"] = segment["duration"] - float(offset)
    right["source_start"] = segment["source_start"] + float(offset) * segment["speed"]
    right["fade_in"] = min(right["fade_in"], right["duration"])
    track["segments"][index:index + 1] = [left, right]
    validate_project(result)
    return result


def crossfade(project: dict, track_id: str, left_id: str, right_id: str,
              duration: float) -> dict:
    """Overlap two visual/audio segments and apply matching fade-out/fade-in."""
    validate_project(project)
    result = copy.deepcopy(project)
    track = _track(result, track_id)
    left = next((item for item in track["segments"] if item["id"] == left_id), None)
    right = next((item for item in track["segments"] if item["id"] == right_id), None)
    if left is None or right is None:
        raise ValueError("Unknown crossfade segment")
    if left is right:
        raise ValueError("Crossfade requires two different segments")
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or duration <= 0 or duration > min(left["duration"], right["duration"])):
        raise ValueError("Crossfade duration must fit both segments")
    duration = float(duration)
    right["start"] = left["start"] + left["duration"] - duration
    left["fade_out"] = duration
    right["fade_in"] = duration
    validate_project(result)
    return result
