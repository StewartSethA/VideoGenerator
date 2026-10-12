import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import backends
from .project import create_project, load_project, save_project
from .render import render
from .workflow import (
    add_generation, ensure_models, estimate_plan, generate_storyboard,
    generate_storyboard_parallel,
)


def _keyframe(value: str) -> dict:
    try:
        frame_text, path = value.split("=", 1)
        frame = int(frame_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("keyframe must be FRAME=PATH") from exc
    if frame < 0 or not path:
        raise argparse.ArgumentTypeError("keyframe must use a nonnegative frame and nonempty path")
    return {"frame": frame, "path": path}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local-first prompt/asset video generator and editor")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("new", "run"):
        item = commands.add_parser(name, help="Create a storyboard" if name == "new" else "Create and export a video")
        item.add_argument("prompt")
        item.add_argument("--assets", type=Path)
        item.add_argument("--script", type=Path, help="UTF-8 script; one segment per sentence or line")
        item.add_argument("--project", type=Path, default=Path("projects/local/project.json"))
        item.add_argument("--duration", type=float, default=4, help="Seconds per storyboard segment")
        item.add_argument("--width", type=int, default=1280)
        item.add_argument("--height", type=int, default=720)
        item.add_argument("--fps", type=int, default=24)
        if name == "run":
            item.add_argument("--backend", default="title")
            item.add_argument("--voice", help="espeak or bark")
            item.add_argument("--music", help="musicgen")
            item.add_argument("--seed", type=int, default=0)
            item.add_argument("--parallel", action="store_true",
                              help="Generate independent visual scenes across detected GPUs")
            item.add_argument("--output", type=Path, default=Path("outputs/video.mp4"))
    item = commands.add_parser("render", help="Export enabled tracks in a project")
    item.add_argument("project", type=Path)
    item.add_argument("--output", type=Path, default=Path("outputs/video.mp4"))
    item = commands.add_parser("split", help="Split a timeline segment at an offset in seconds")
    item.add_argument("project", type=Path)
    item.add_argument("track")
    item.add_argument("segment")
    item.add_argument("offset", type=float)
    item = commands.add_parser("crossfade", help="Overlap two segments with matched fades")
    item.add_argument("project", type=Path)
    item.add_argument("track")
    item.add_argument("left")
    item.add_argument("right")
    item.add_argument("duration", type=float)
    item = commands.add_parser("regenerate", help="Add a disabled alternative track for a segment")
    item.add_argument("project", type=Path)
    item.add_argument("segment")
    item.add_argument("--backend", required=True)
    item.add_argument("--input-asset", help="Conditioning asset ID (only for supported backends)")
    item.add_argument("--seed", type=int, default=0)
    item.add_argument("--enable", action="store_true")
    item = commands.add_parser("generate", help="Run one generator and download its weights on demand")
    item.add_argument("backend")
    item.add_argument("prompt")
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--input", type=Path)
    item.add_argument("--seed", type=int, default=0)
    item.add_argument("--duration", type=float, default=4)
    item.add_argument("--width", type=int, default=512)
    item.add_argument("--height", type=int, default=512)
    item.add_argument("--keyframe", action="append", type=_keyframe, default=[],
                      help="Frame-conditioned image as FRAME=PATH; repeat when supported")
    item = commands.add_parser("download", help="Install model dependencies and cache requested weights")
    item.add_argument("backends", nargs="+")
    item = commands.add_parser("setup", help="Check system tools and optionally install/cache models")
    item.add_argument("--models", nargs="*", default=[])
    commands.add_parser("backends", help="List built-in capabilities, licenses and model IDs")
    item = commands.add_parser("routes", help="Show capability-ranked generation backends")
    item.add_argument("--output", required=True, choices=["image","video","audio","music","voice","audiovisual"])
    item.add_argument("--input", action="append", default=[], choices=["image","video","audio"])
    item.add_argument("--keyframes", type=int, default=0)
    item.add_argument("--include-unavailable", action="store_true")
    commands.add_parser("ecosystem", help="List optional external frameworks and modern model checkpoints")
    commands.add_parser("ecosystem-status", help="Show installed external frameworks and cached models")
    item = commands.add_parser("framework", help="Install, update or launch an external native framework")
    framework_commands = item.add_subparsers(dest="framework_command", required=True)
    install = framework_commands.add_parser("install")
    install.add_argument("name")
    install.add_argument("--update", action="store_true")
    install.add_argument("--dry-run", action="store_true")
    launch = framework_commands.add_parser("launch")
    launch.add_argument("name")
    launch.add_argument("--port", type=int)
    launch.add_argument("--no-install", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    model = commands.add_parser("model", help="Cache an optional external model checkpoint")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_install = model_commands.add_parser("install")
    model_install.add_argument("name")
    model_install.add_argument("--revision")
    model_install.add_argument("--dry-run", action="store_true")
    commands.add_parser("hardware", help="Detect CPU/RAM/NVIDIA GPU hardware")
    commands.add_parser("doctor", help="Check tools, hardware, backends and optional ecosystem status")
    item = commands.add_parser("estimate", help="Heuristic hardware-aware generation estimate")
    item.add_argument("project", type=Path)
    item.add_argument("--hardware", type=Path, help="JSON hardware profile for another machine")
    item.add_argument("--backend", help="Plan generation for storyboard spans without assets")
    item.add_argument("--voice", help="Plan narration for every storyboard span")
    item.add_argument("--music", help="Plan a whole-sequence score")
    item = commands.add_parser("edit", help="Trim a source by human/local Whisper transcript and add captions")
    item.add_argument("source", type=Path)
    item.add_argument("--transcript", type=Path, help="JSON list of {start, duration, text}; word timestamps preferred")
    item.add_argument("--keep-fillers", action="store_true")
    item.add_argument("--max-pause", type=float, default=0.35)
    item.add_argument("--drop-phrase", action="append", default=[])
    item.add_argument("--project", type=Path, default=Path("projects/edited/project.json"))
    item.add_argument("--output", type=Path, help="Also export the edited project")
    item = commands.add_parser("transcribe", help="Download/use local Whisper-small for word timestamps")
    item.add_argument("source", type=Path)
    item.add_argument("--output", type=Path, required=True)
    item = commands.add_parser("serve", help="Start the localhost web editor")
    item.add_argument("--workspace", type=Path, default=Path("projects/local"))
    item.add_argument("--port", type=int, default=7860)
    item.add_argument("--no-browser", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in {"new", "run"}:
            project = create_project(args.prompt, args.assets,
                                     args.script.read_text(encoding="utf-8") if args.script else None,
                                     args.duration, args.width, args.height, args.fps)
            if args.command == "run":
                generator = generate_storyboard_parallel if args.parallel else generate_storyboard
                project = generator(
                    project, args.backend, args.project.parent / "generated",
                    args.voice, args.music, args.seed
                )
            save_project(project, args.project)
            if args.command == "run":
                print(render(project, args.output))
            else:
                print(args.project)
        elif args.command == "render":
            print(render(load_project(args.project), args.output))
        elif args.command in {"split", "crossfade"}:
            from .operations import split_segment, crossfade
            project = load_project(args.project)
            if args.command == "split":
                project = split_segment(project, args.track, args.segment, args.offset)
            else:
                project = crossfade(project, args.track, args.left, args.right, args.duration)
            save_project(project, args.project)
            print(args.project)
        elif args.command == "regenerate":
            project = add_generation(load_project(args.project), args.segment, args.backend,
                                     args.project.parent / "generated", args.seed, args.input_asset, args.enable)
            save_project(project, args.project)
            print(project["tracks"][-1]["id"])
        elif args.command == "generate":
            ensure_models(args.backend)
            print(backends.generate(
                args.backend, args.prompt, args.output, seed=args.seed,
                duration=args.duration, width=args.width, height=args.height,
                input_path=args.input, keyframes=args.keyframe or None,
            ))
        elif args.command in {"setup", "download"}:
            if args.command == "setup":
                missing = [name for name in ("ffmpeg", "ffprobe") if not shutil.which(name)]
                if not (shutil.which("espeak") or shutil.which("espeak-ng")):
                    missing.append("espeak")
                if missing:
                    raise RuntimeError("Missing system tools: " + ", ".join(missing) +
                                       ". Ubuntu/Debian: sudo apt-get install ffmpeg espeak")
            for backend in args.models if args.command == "setup" else args.backends:
                ensure_models(backend)
                print(backends.download(backend))
            print("Ready")
        elif args.command == "backends":
            print(json.dumps(backends.catalog(), indent=2))
        elif args.command == "routes":
            from .routing import candidates
            rows = candidates(args.output, input_kinds=args.input, keyframes=args.keyframes)
            if not args.include_unavailable:
                rows = [row for row in rows if row["executable"]]
            print(json.dumps(rows, indent=2))
        elif args.command in {"ecosystem", "ecosystem-status", "framework", "model"}:
            from . import ecosystem
            if args.command == "ecosystem":
                print(ecosystem.format_catalog())
            elif args.command == "ecosystem-status":
                print(json.dumps(ecosystem.status(), indent=2))
            elif args.command == "framework":
                if args.dry_run:
                    print(json.dumps(ecosystem.framework_plan(args.name, getattr(args, "port", None)), indent=2))
                elif args.framework_command == "install":
                    print(ecosystem.install_framework(args.name, update=args.update))
                else:
                    process = ecosystem.launch_framework(
                        args.name, args.port, install=not args.no_install
                    )
                    print(f"Started {args.name} as PID {process.pid}")
            elif args.model_command == "install":
                if args.dry_run:
                    print(json.dumps(ecosystem.model_plan(args.name, args.revision), indent=2))
                else:
                    print(ecosystem.install_model(args.name, revision=args.revision))
        elif args.command == "hardware":
            print(json.dumps(backends.hardware(), indent=2))
        elif args.command == "doctor":
            from . import ecosystem
            from .routing import capabilities
            report = {
                "tools": ecosystem.status()["tools"],
                "hardware": backends.hardware(),
                "builtin_backends": [row["id"] for row in capabilities() if row["source"] == "builtin"],
                "ecosystem": ecosystem.status(),
            }
            print(json.dumps(report, indent=2))
        elif args.command == "estimate":
            info = json.loads(args.hardware.read_text()) if args.hardware else None
            print(json.dumps(estimate_plan(load_project(args.project), args.backend,
                                           args.voice, args.music, info), indent=2))
        elif args.command in {"edit", "transcribe"}:
            from .editing import edit_media, transcribe
            if args.command == "edit" and args.transcript:
                transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
            else:
                from .workflow import ensure_model_dependencies
                ensure_model_dependencies()
                transcript = transcribe(args.source)
            if args.command == "transcribe":
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
                print(args.output)
            else:
                project = edit_media(args.source, transcript, remove_fillers=not args.keep_fillers,
                                     max_pause=args.max_pause, drop_phrases=args.drop_phrase)
                save_project(project, args.project)
                print(render(project, args.output) if args.output else args.project)
        elif args.command == "serve":
            from .web import serve
            serve(args.workspace, args.port, open_browser=not args.no_browser)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
