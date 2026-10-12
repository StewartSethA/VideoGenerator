"""Deterministic local FFmpeg rendering; no model or shell is involved."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

from .project import validate_project


_INPUT_SAFETY = [
    "-protocol_whitelist", "file,pipe",
    "-format_whitelist",
    "mov,matroska,webm,avi,image2,png_pipe,jpeg_pipe,webp_pipe,bmp_pipe,"
    "ppm_pipe,gif,wav,mp3,flac,ogg,aac",
]


def _run(args):
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg and ffprobe must be installed") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Media processing failed: {exc.stderr[-6000:]}") from exc


def _tempo(speed):
    factors = []
    while speed > 2:
        factors.append(2)
        speed /= 2
    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5
    factors.append(speed)
    return ",".join(f"atempo={factor:.12g}" for factor in factors)


def render(project: dict, output: Path, asset_root: Path | None = None) -> Path:
    validate_project(project, asset_root)
    output = Path(output)
    if output.suffix.lower() not in (".mp4", ".mov", ".mkv"):
        raise ValueError("Output must be MP4, MOV, or MKV")
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = (project["settings"][key] for key in ("width", "height", "fps"))
    tracks = [track for track in project["tracks"] if track["enabled"]]
    spans = [s for track in tracks for s in track["segments"]] + project["captions"]
    duration = max((s["start"] + s["duration"] for s in spans), default=4)
    duration = math.ceil(duration * fps - 1e-9) / fps
    assets = {a["id"]: a for a in project["assets"]}
    # Keep scratch files on the same filesystem and remove them even on failure.
    with tempfile.TemporaryDirectory(
        prefix=".videogenerator-", dir=output.parent.resolve(),
    ) as directory:
        scratch = Path(directory)
        args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-filter_complex_threads", "1", "-threads", "1"]
        filters = [f"color=c=0x101827:s={width}x{height}:r={fps}:d={duration}[base]"]
        input_index = 0
        text_index = 0
        visual_index = 0
        audio_labels = []
        voices = [s for t in tracks if t["kind"] == "audio" for s in t["segments"]
                  if s["role"] == "voice" and s["volume"] > 0]
        stream_cache = {}

        def has_audio(asset):
            path = asset["path"]
            if path not in stream_cache:
                data = json.loads(_run([
                    "ffprobe", "-v", "error", *_INPUT_SAFETY,
                    "-show_entries", "stream=codec_type",
                    "-of", "json", path,
                ]).stdout)
                stream_cache[path] = any(s["codec_type"] == "audio" for s in data["streams"])
            return stream_cache[path]

        def text_filter(text, title=False):
            nonlocal text_index
            filename = scratch / f"text-{text_index}.txt"
            text_index += 1
            font_size = max(12, round(height * (0.065 if title else 0.045)))
            wrap = max(10, int(width * 0.82 / (font_size * 0.6)))
            lines = []
            for line in text.splitlines() or [""]:
                lines.extend(textwrap.wrap(line, width=wrap) or [""])
            filename.write_text("\n".join(lines), encoding="utf-8")
            # Generated paths contain no user text; expansion=none disables %{...}.
            safe_path = filename.as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
            y = "(h-text_h)/2" if title else "h-text_h-h*0.08"
            return (f"drawtext=font=Sans:textfile='{safe_path}':expansion=none:"
                    f"fontsize={font_size}:fontcolor=white:x=(w-text_w)/2:y={y}:"
                    f"line_spacing={round(font_size * 0.25)}:"
                    "box=1:boxcolor=black@0.55:boxborderw=12")

        for track in tracks:
            for segment in track["segments"]:
                start, length = segment["start"], segment["duration"]
                end = start + length
                asset = assets.get(segment["asset_id"])
                if track["kind"] == "audio" and not has_audio(asset):
                    continue
                index = None
                if asset:
                    index = input_index
                    input_index += 1
                    args += _INPUT_SAFETY
                    if asset["kind"] == "image":
                        args += ["-loop", "1", "-framerate", str(fps)]
                    args += ["-threads", "1", "-i", asset["path"]]
                if track["kind"] == "video":
                    label = f"visual{visual_index}"
                    if asset:
                        chain = (f"[{index}:v]trim=start={segment['source_start']}:"
                                 f"duration={length * segment['speed']},"
                                 f"setpts=(PTS-STARTPTS)/{segment['speed']},fps={fps},"
                                 f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                                 f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x101827,"
                                 "setsar=1,format=rgba")
                    else:
                        chain = (f"color=c=0x17233d:s={width}x{height}:r={fps}:d={length},"
                                 f"format=rgba,{text_filter(segment['text'], True)}")
                    for effect in segment.get("effects", []):
                        effect_type = effect["type"]
                        if effect_type == "grayscale":
                            chain += ",hue=s=0"
                        elif effect_type == "blur":
                            chain += f",boxblur={effect.get('radius', 1)}:1"
                        elif effect_type == "sharpen":
                            chain += f",unsharp=5:5:{effect.get('amount', 1)}:5:5:0"
                        elif effect_type == "eq":
                            chain += (
                                f",eq=brightness={effect.get('brightness', 0)}:"
                                f"contrast={effect.get('contrast', 1)}:"
                                f"saturation={effect.get('saturation', 1)}"
                            )
                    transform = segment.get(
                        "transform",
                        {"x": 0, "y": 0, "scale": 1, "rotation": 0, "opacity": 1},
                    )
                    if transform.get("scale", 1) != 1:
                        chain += f",scale=iw*{transform['scale']}:ih*{transform['scale']}"
                    if transform.get("rotation", 0):
                        radians = transform["rotation"] * math.pi / 180
                        chain += (
                            f",rotate={radians}:c=none:ow=rotw(iw):oh=roth(ih)"
                        )
                    if transform.get("opacity", 1) != 1:
                        chain += f",colorchannelmixer=aa={transform['opacity']}"
                    if segment["fade_in"]:
                        chain += f",fade=t=in:st=0:d={segment['fade_in']}:alpha=1"
                    if segment["fade_out"]:
                        chain += (f",fade=t=out:st={length - segment['fade_out']}:"
                                  f"d={segment['fade_out']}:alpha=1")
                    filters.append(chain + f",setpts=PTS+{start}/TB[{label}]")
                    previous = "base" if visual_index == 0 else f"composite{visual_index - 1}"
                    x = transform.get("x", 0)
                    y = transform.get("y", 0)
                    filters.append(
                        f"[{previous}][{label}]overlay="
                        f"x=(main_w-overlay_w)/2+{x}:y=(main_h-overlay_h)/2+{y}:"
                        f"eof_action=pass:repeatlast=0:"
                        f"enable='gte(t,{start})*lt(t,{end})'[composite{visual_index}]"
                    )
                    visual_index += 1
                else:
                    label = f"audio{len(audio_labels)}"
                    chain = (
                        f"[{index}:a]atrim=start={segment['source_start']}:"
                        f"duration={length * segment['speed']},asetpts=PTS-STARTPTS,"
                        f"{_tempo(segment['speed'])},aresample=48000,"
                        "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                        f"apad,atrim=duration={length},volume={segment['volume']}"
                    )
                    if segment["fade_in"]:
                        chain += f",afade=t=in:st=0:d={segment['fade_in']}"
                    if segment["fade_out"]:
                        chain += (f",afade=t=out:st={length - segment['fade_out']}:"
                                  f"d={segment['fade_out']}")
                    if segment["role"] != "voice" and segment["duck"] != "none" and voices:
                        intervals = "+".join(
                            f"gte(t,{v['start'] - start})*lt(t,{v['start'] + v['duration'] - start})"
                            for v in voices
                        )
                        amount = 0.25 if segment["duck"] == "soft" else 0.08
                        chain += f",volume='if(gt({intervals},0),{amount},1)':eval=frame"
                    chain += f",adelay={round(start * 48000)}S:all=1[{label}]"
                    filters.append(chain)
                    audio_labels.append(label)
        video = f"composite{visual_index - 1}" if visual_index else "base"
        for index, caption in enumerate(project["captions"]):
            label = f"caption{index}"
            filters.append(
                f"[{video}]{text_filter(caption['text'])}:"
                f"enable='gte(t,{caption['start']})*lt(t,{caption['start'] + caption['duration']})'[{label}]"
            )
            video = label
        filters.append(f"[{video}]format=yuv420p[video]")
        filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={duration}[silence]")
        inputs = "".join(f"[{label}]" for label in ["silence"] + audio_labels)
        filters.append(
            f"{inputs}amix=inputs={len(audio_labels) + 1}:normalize=0:"
            f"duration=longest,atrim=duration={duration}[mixed]"
        )
        audio_settings = project["settings"].get("audio", {})
        if audio_settings.get("normalize", False):
            target = audio_settings.get("target_lufs", -14)
            peak = audio_settings.get("true_peak", -1)
            lra = audio_settings.get("lra", 11)
            filters.append(
                f"[mixed]loudnorm=I={target}:TP={peak}:LRA={lra}[audio]"
            )
        else:
            filters.append("[mixed]anull[audio]")
        filter_file = scratch / "filters.txt"
        filter_file.write_text(";\n".join(filters), encoding="utf-8")
        rendered = scratch / ("rendered" + output.suffix.lower())
        args += [
            "-filter_complex_script", str(filter_file), "-map", "[video]", "-map", "[audio]",
            "-t", str(duration), "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-threads", "1", "-c:a", "aac", "-b:a", "192k",
        ]
        if output.suffix.lower() in (".mp4", ".mov"):
            args += ["-movflags", "+faststart"]
        _run(args + [str(rendered)])
        os.replace(rendered, output)
    return output
