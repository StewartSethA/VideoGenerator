"""Local, opt-in model adapters. Importing this module never imports ML packages."""

from __future__ import annotations

import copy
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import wave
from functools import lru_cache


_CATALOG = [
    dict(id="title", name="FFmpeg title card", kind="title", model_id=None,
         license="System FFmpeg license", generative=False, inputs=[], device="cpu",
         vram_gb=0, ram_gb=1, seconds_per_second=0.2, steps=0),
    dict(id="espeak", name="eSpeak / eSpeak NG system voice", kind="audio", modality="voice", model_id=None,
         license="eSpeak: GPL-2.0-or-later; eSpeak NG: GPL-3.0-or-later",
         generative=False, inputs=[], device="cpu",
         vram_gb=0, ram_gb=0.1, seconds_per_second=0.05, steps=0),
    dict(id="sdxl", name="Stable Diffusion XL", kind="image",
         model_id="stabilityai/stable-diffusion-xl-base-1.0",
         pipeline="StableDiffusionXLPipeline", license="OpenRAIL++",
         generative=True, inputs=[], device="cpu or cuda", vram_gb=8, ram_gb=16,
         seconds_per_second=12, steps=30, size_multiple=8),
    dict(id="cogvideox", name="CogVideoX 2B", kind="video",
         model_id="THUDM/CogVideoX-2b", pipeline="CogVideoXPipeline",
         license="Apache-2.0", generative=True, inputs=[], device="cuda",
         vram_gb=12, ram_gb=24, seconds_per_second=60, steps=50,
         size_multiple=16, frames=49, fps=8,
         duration_note="Generates 49 frames; export frame rate fits requested duration."),
    dict(id="wan", name="Wan 2.1 T2V 1.3B", kind="video",
         model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", pipeline="WanPipeline",
         license="Apache-2.0", generative=True, inputs=[], device="cuda",
         vram_gb=10, ram_gb=24, seconds_per_second=45, steps=50,
         size_multiple=16, frame_multiple=4, fps=16),
    dict(id="ltx", name="LTX Video", kind="video",
         model_id="Lightricks/LTX-Video", pipeline="LTXPipeline",
         license="Checkpoint-specific; inspect the downloaded model card/LICENSE before use", generative=True,
         inputs=["image"], device="cuda", vram_gb=10, ram_gb=16,
         seconds_per_second=15, steps=50, size_multiple=32,
         frame_multiple=8, fps=24),
    dict(id="musicgen", name="MusicGen small", kind="audio",
         model_id="facebook/musicgen-small", license="CC-BY-NC-4.0",
         generative=True, inputs=[], device="cpu or cuda", vram_gb=2,
         ram_gb=4, seconds_per_second=3, steps=0, max_duration=30),
    dict(id="bark", name="Bark small", kind="audio", modality="voice",
         model_id="suno/bark-small", license="MIT", generative=True,
         inputs=[], device="cpu or cuda", vram_gb=4, ram_gb=8,
         seconds_per_second=5, steps=0,
         duration_note="Speech duration follows text; duration is not a speech timing control."),
]


def catalog() -> list[dict]:
    """Return independent copies of the supported, locally runnable backends."""
    result = copy.deepcopy(_CATALOG)
    for backend in result:
        backend["estimate_basis"] = "Illustrative single-GPU/CPU baseline; not benchmarked or calibrated."
    return result


def get_backend(id: str) -> dict:
    for backend in catalog():
        if backend["id"] == id:
            return backend
    raise ValueError(f"Unknown backend {id!r}; choose from {', '.join(b['id'] for b in _CATALOG)}")


def _dependency(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing optional model dependency {name!r}. Run uv sync --extra models."
        ) from exc


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _system_program(backend_id: str) -> str:
    names = ("espeak", "espeak-ng") if backend_id == "espeak" else ("ffmpeg",)
    for name in names:
        executable = shutil.which(name)
        if executable:
            return executable
    raise RuntimeError(f"Install the system {'eSpeak (espeak or espeak-ng)' if backend_id == 'espeak' else 'FFmpeg'} package.")


def _audio_uses_safetensors(backend: dict) -> bool:
    """Check upstream inventory rather than assume old audio checkpoints were converted."""
    files = _dependency("huggingface_hub").list_repo_files(
        repo_id=backend["model_id"], token=_token())
    if any("/" not in name and name.startswith("model") and name.endswith(".safetensors")
           for name in files):
        return True
    if any("/" not in name and name.startswith("pytorch_model") and name.endswith(".bin")
           for name in files):
        return False
    raise RuntimeError(f"No supported weight format found for {backend['model_id']}.")


def download(backend_id: str) -> str:
    """Download inert model weights/configuration to the normal Hugging Face cache."""
    backend = get_backend(backend_id)
    if backend["model_id"] is None:
        return _system_program(backend_id)
    hub = _dependency("huggingface_hub")
    allow_patterns = ["*.json", "*.safetensors", "*.txt", "*.model",
                      "*.tiktoken", "*.vocab", "*.merges", "*.md", "LICENSE*"]
    ignore_patterns = ["*.py", "*.bin", "*.pt", "*.pth", "*.pkl", "*.pickle"]
    if backend_id in {"musicgen", "bark"} and not _audio_uses_safetensors(backend):
        # Downloading bytes is inert; deserialization has a separate restricted-loader guard.
        allow_patterns.append("pytorch_model*.bin")
        ignore_patterns.remove("*.bin")
    return str(hub.snapshot_download(
        repo_id=backend["model_id"], token=_token(),
        allow_patterns=allow_patterns, ignore_patterns=ignore_patterns,
    ))


@lru_cache(maxsize=1)
def _load_model(backend_id: str, device: str, image_input: bool):
    backend = get_backend(backend_id)
    torch = _dependency("torch")
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if backend_id in {"wan", "ltx"} else torch.float16
    kwargs = dict(torch_dtype=dtype, use_safetensors=True, token=_token())
    if backend_id in {"musicgen", "bark"}:
        transformers = _dependency("transformers")
        kwargs["use_safetensors"] = _audio_uses_safetensors(backend)
        if not kwargs["use_safetensors"]:
            try:
                version = tuple(int(part) for part in str(torch.__version__).split("+")[0].split(".")[:2])
            except ValueError as exc:
                raise RuntimeError("Cannot verify PyTorch's safe binary loader; run uv sync --extra models.") from exc
            if version < (2, 6):
                raise RuntimeError("Binary audio checkpoints require PyTorch >=2.6 with restricted "
                                   "weights_only loading. Run uv sync --extra models.")
        kwargs["weights_only"] = True
        processor = transformers.AutoProcessor.from_pretrained(
            backend["model_id"], token=_token(), trust_remote_code=False)
        cls = getattr(transformers, "MusicgenForConditionalGeneration"
                      if backend_id == "musicgen" else "BarkModel")
        model = cls.from_pretrained(backend["model_id"], trust_remote_code=False, **kwargs)
        return model.to(device), processor
    diffusers = _dependency("diffusers")
    cls = getattr(diffusers, "LTXImageToVideoPipeline" if image_input else backend["pipeline"])
    if backend_id == "wan":
        # Wan's VAE needs float32 even when the transformer uses bfloat16.
        kwargs["vae"] = diffusers.AutoencoderKLWan.from_pretrained(
            backend["model_id"], subfolder="vae", torch_dtype=torch.float32,
            use_safetensors=True, token=_token())
    model = cls.from_pretrained(backend["model_id"], **kwargs)
    if device == "cuda":
        # Offload also makes image/video models usable on smaller CUDA cards.
        model.enable_model_cpu_offload()
        if hasattr(model, "vae") and hasattr(model.vae, "enable_tiling"):
            model.vae.enable_tiling()
    else:
        model.to(device)
    return model, None


def _write_audio(output: Path, audio, sample_rate: int) -> None:
    """Write model PCM without scipy/soundfile; clamp rather than wrapping samples."""
    audio = audio.detach().float().cpu()
    # MusicGen: [batch, channels, time]; Bark: [batch, time].
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    pcm = (audio.clamp(-1, 1) * 32767).round().to(dtype=_dependency("torch").int16)
    raw = pcm.transpose(0, 1).contiguous().numpy().astype("<i2", copy=False).tobytes()
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(pcm.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)


def _video_frames(backend: dict, duration: float) -> int:
    if "frames" in backend:
        return backend["frames"]
    multiple = backend["frame_multiple"]
    return max(multiple + 1, math.ceil((duration * backend["fps"] - 1) / multiple) * multiple + 1)


def generate(backend_id: str, prompt: str, output: Path, *, seed: int = 0,
             duration: float = 4, width: int = 512, height: int = 512,
             input_path: Path | None = None, keyframes: list | None = None) -> Path:
    """Generate one artifact, fetching weights on first requested use only."""
    backend = get_backend(backend_id)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if keyframes:
        raise ValueError(f"{backend_id} does not support keyframes")
    if input_path is not None and "image" not in backend["inputs"]:
        raise ValueError(f"{backend_id} does not support an input image")
    if input_path is not None and not Path(input_path).is_file():
        raise ValueError("input_path must be an existing image file")
    if backend["kind"] in {"image", "video", "title"}:
        multiple = backend.get("size_multiple", 1)
        if any(not isinstance(v, int) or v <= 0 or v % multiple for v in (width, height)):
            raise ValueError(f"{backend_id} width and height must be positive multiples of {multiple}")
    if duration > backend.get("max_duration", math.inf):
        raise ValueError(f"{backend_id} supports at most {backend['max_duration']} seconds")
    output = Path(output)
    if backend_id == "title":
        raise ValueError("Title cards are a non-generative FFmpeg baseline; use the timeline renderer.")
    if backend["kind"] in {"audio", "voice"} and output.suffix.lower() != ".wav":
        raise ValueError("Audio and voice output must use the .wav extension")
    if backend_id == "espeak":
        executable = _system_program("espeak")
        output.parent.mkdir(parents=True, exist_ok=True)
        # stdin prevents text beginning with '-' from becoming command-line options.
        subprocess.run([executable, "-w", str(output), "--stdin"], input=prompt,
                       text=True, check=True, capture_output=True)
        return output
    torch = _dependency("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if backend["device"] == "cuda" and device != "cuda":
        raise RuntimeError(f"{backend_id} video generation requires an NVIDIA CUDA GPU; "
                           "use SDXL images or title cards on CPU, or select a CUDA machine.")
    if device == "cuda" and backend_id in {"wan", "ltx"} and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{backend_id} requires a CUDA GPU supporting bfloat16 (Ampere or newer).")
    try:
        model, processor = _load_model(backend_id, device, input_path is not None)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Missing or outdated optional model dependencies. "
                           "Run uv sync --extra models.") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    if processor is not None:
        transformers = _dependency("transformers")
        transformers.set_seed(seed)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        options = {"do_sample": True}
        if backend_id == "musicgen":
            options["max_new_tokens"] = max(1, math.ceil(duration * 50))
        with torch.inference_mode():
            audio = model.generate(**inputs, **options)
        sample_rate = (model.config.audio_encoder.sampling_rate if backend_id == "musicgen"
                       else model.generation_config.sample_rate)
        _write_audio(output, audio, sample_rate)
        return output
    options = dict(prompt=prompt, width=width, height=height,
                   num_inference_steps=backend["steps"],
                   generator=torch.Generator(device="cpu").manual_seed(seed))
    if backend["kind"] == "image":
        model(**options).images[0].save(output)
    else:
        frames = _video_frames(backend, duration)
        options["num_frames"] = frames
        if backend_id == "ltx":
            options["frame_rate"] = backend["fps"]
            if input_path is not None:
                options["image"] = _dependency("diffusers.utils").load_image(str(input_path))
        result = model(**options)
        _dependency("diffusers.utils").export_to_video(
            result.frames[0], str(output), fps=frames / duration)
    return output


def hardware() -> dict:
    """Inspect system capacity using only the standard library and nvidia-smi."""
    memory_gb = None
    try:
        memory_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        pass
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=5)
        for line in result.stdout.splitlines():
            name, memory = line.rsplit(",", 1)
            gpus.append(dict(name=name.strip(), memory_gb=float(memory.strip()) / 1024))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return dict(cpu_count=os.cpu_count() or 1, memory_gb=memory_gb,
                gpu_count=len(gpus), gpus=gpus)


def estimate(project: dict, hardware_info: dict | None = None) -> dict:
    """Illustrative ranges, not measured benchmarks; generation and rendering differ."""
    hw = hardware() if hardware_info is None else hardware_info
    gpus = hw.get("gpus", [])
    gpu_count = hw.get("gpu_count", len(gpus))
    gpu_memory = max((gpu.get("memory_gb", gpu.get("vram_gb", 0)) for gpu in gpus), default=0)
    gpu_memory = max(gpu_memory, hw.get("vram_gb", hw.get("gpu_memory_gb", 0)) or 0)
    cpu_count = max(1, hw.get("cpu_count", hw.get("cpu", 1)) or 1)
    memory = hw.get("memory_gb", hw.get("ram_gb"))
    warnings = ["Heuristic illustrative ranges, not benchmarked or calibrated. "
                "First-use downloads, model loading, offload and thermal effects are excluded."]
    segments = []
    generation_total = 0.0
    render_total = 0.0
    settings = project.get("settings", {})
    source_segments = project.get("segments")
    if source_segments is None:
        source_segments = [segment for track in project.get("tracks", [])
                           if track.get("enabled", True) for segment in track.get("segments", [])]
    for index, segment in enumerate(source_segments):
        backend = get_backend(segment.get("backend", segment.get("backend_id", "title")))
        duration = float(segment.get("duration", 4))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Segment duration must be finite and positive")
        width = segment.get("width", settings.get("width", 512))
        height = segment.get("height", settings.get("height", 512))
        if width <= 0 or height <= 0:
            raise ValueError("Segment sizes must be positive")
        pixels = width * height / 512**2 if backend["kind"] in {"image", "video", "title"} else 1
        steps = max(1, segment.get("steps", backend["steps"]))
        frames = _video_frames(backend, duration) if backend["generative"] and backend["kind"] == "video" else None
        generation_duration = frames / backend["fps"] if frames else duration
        work = (1 if backend["kind"] == "image" else generation_duration) * backend["seconds_per_second"] * pixels
        if backend["steps"]:
            work *= steps / backend["steps"]
        feasible = True
        notes = []
        if backend["device"] == "cuda" and not gpu_count:
            feasible = False
            notes.append("Requires a CUDA GPU; CPU video generation is unsupported.")
        if memory is not None and memory < backend["ram_gb"]:
            feasible = False
            notes.append(f"Suggested RAM is at least {backend['ram_gb']} GiB.")
        if gpu_count and gpu_memory and gpu_memory < backend["vram_gb"]:
            notes.append("Below suggested VRAM; CPU offload may help but can be much slower or still run out of memory.")
            work *= 3
        if not gpu_count and backend["generative"]:
            work *= 15 / min(cpu_count / 4, 4)
        elif backend["device"] == "cpu":
            work /= min(cpu_count / 2, 4)
        # More GPUs do not accelerate these single-device adapters.
        needs_generation = not segment.get("asset_id")
        generation = work if needs_generation and (backend["generative"] or backend["id"] == "espeak") else 0
        if not needs_generation:
            feasible, notes = True, []
        render = duration * 0.2 * pixels / min(cpu_count / 2, 4)
        generation_total += generation
        render_total += render
        segments.append(dict(index=index, id=segment.get("id"), backend=backend["id"], feasible=feasible,
                             generation_frames=frames if needs_generation else None,
                             generation_seconds_range=[round(generation * .5, 2), round(generation * 3, 2)],
                             render_seconds_range=[round(render * .5, 2), round(render * 3, 2)],
                             suggested_vram_gb=backend["vram_gb"], suggested_ram_gb=backend["ram_gb"],
                             warnings=notes))
    total = generation_total + render_total
    return dict(heuristic=True, calibrated=False, hardware=hw, segments=segments,
                feasible=all(segment["feasible"] for segment in segments),
                generation_seconds_range=[round(generation_total * .5, 2), round(generation_total * 3, 2)],
                render_seconds_range=[round(render_total * .5, 2), round(render_total * 3, 2)],
                total_seconds_range=[round(total * .5, 2), round(total * 3, 2)], warnings=warnings)
