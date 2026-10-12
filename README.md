# VideoGenerator

Local-first prompt/asset-to-video CLI and a lightweight multitrack web editor.
The default path works on a CPU without weights, accounts, subscriptions, or
Python runtime dependencies: **FFmpeg title cards/assets → optional eSpeak
narration → captioned MP4**. It is an editor/assembler, not an AI video model.

## One-command start

Requires Python 3 with `venv`, an internet connection on first setup, and FFmpeg.
On Ubuntu/Debian the launcher installs missing FFmpeg/eSpeak packages using
`apt-get` (may ask for sudo). Other platforms need these system tools installed
first; on macOS: `brew install ffmpeg espeak`.

```sh
./start.sh                              # install uv/environment, open localhost editor
./start.sh run "A short introduction." --voice espeak --output outputs/intro.mp4
./start.sh run "Show the garden. Then the sunset." --assets ./my-assets \
  --script ./script.txt --duration 4 --output outputs/story.mp4
```

Alternatively, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked
uv run videogenerator setup
uv run videogenerator serve
```

The browser is at `http://127.0.0.1:7860`. Uploaded assets and project edits stay
in `projects/local`; export is `projects/local/export.mp4`. The server accepts
only localhost hosts, uses same-origin mutation tokens, and restricts media paths
to that workspace. **Do not expose it publicly or put it behind a public proxy.**
For a remote machine use SSH port forwarding: `ssh -L 7860:127.0.0.1:7860 host`.
Local JSON projects from untrusted sources still need review before use.

Model selection in the CLI/editor automatically installs the optional Python
`models` extra using uv, then downloads the selected model on first use. Allow
substantial disk space and download time. Ordinary startup does not install Torch
or download weights. In a installed wheel (rather than a source checkout), install
the extra yourself: `uv pip install 'videogenerator[models]'`.

## Generators, quality and licenses

These are specific supported adapters, **not “every free model”** or a claim of
current state-of-the-art quality. Model quality varies with prompt, checkpoint,
resolution and hardware. More recent/larger video models often exceed this
application's practical single-GPU presets. Review each linked model card before
using or distributing output; code and weight licenses are different.

| Backend | What it does / practical quality | Weight license / account considerations |
|---|---|---|
| `title` | Fast, deterministic FFmpeg slides and asset assembly; not neural video | System FFmpeg's build/license applies; no account |
| `espeak` | Reliable script-to-speech on CPU; robotic rather than natural | eSpeak GPL-2.0 / eSpeak NG GPL-3.0; no account |
| `sdxl` | [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): strong still images; not animated video | CreativeML Open RAIL++-M, with use restrictions |
| `cogvideox` | [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b): short text-to-video; older/smaller accessible model | Apache-2.0 for this 2B checkpoint, not all CogVideo checkpoints |
| `wan` | [Wan2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers): capable short 480p text-to-video | Apache-2.0 |
| `ltx` | [LTX-Video](https://huggingface.co/Lightricks/LTX-Video): short text-to-video or initial-image guidance | Check the exact checkpoint's weight license; do not infer it from Apache-licensed repository code |
| `musicgen` | [MusicGen small](https://huggingface.co/facebook/musicgen-small): short prompt-driven music, up to 30 seconds per request | **CC-BY-NC-4.0 weights: not a commercial default** |
| `bark` | [Bark small](https://huggingface.co/suno/bark-small): expressive speech/sounds; may deviate from script and timing | MIT; use eSpeak for deterministic narration |
| Whisper editing | [Whisper-small](https://huggingface.co/openai/whisper-small): local transcription and word timestamps | Apache-2.0 |

No adapter charges a subscription or API fee. Public downloads are normally
available without login, but **current gating/account requirements were not
verified against the live model host in this development environment** and can
change. For gated/restricted weights accept the upstream terms and set `HF_TOKEN`
in your environment; never put a token in project JSON. Electricity, hardware,
cloud GPU rental and bandwidth are still costs. “Free download” does not mean
unrestricted commercial licensing.

Text-to-video adapters require NVIDIA CUDA and use CPU offloading/VAE tiling.
Plan for approximately 8–16+ GB VRAM and 16–32+ GB system RAM, with enough disk
for text encoders as well as denoisers. Upstream Wan reports 8.19 GB for one
optimized 480p setup, **not a guarantee that an 8 GB card is sufficient**.
CPU image/audio inference is supported but can be very slow. Other accelerators,
model sharding and multi-GPU parallel inference are not implemented.

```sh
uv run videogenerator backends                         # capabilities/licenses/IDs
uv run videogenerator setup --models sdxl wan           # dependencies + explicit weight caching
uv run videogenerator generate sdxl "Mountain sunrise" --output outputs/sunrise.png
uv run videogenerator generate cogvideox "Waves at sunset" \
  --width 720 --height 480 --duration 6 --output outputs/waves.mp4
uv run videogenerator generate ltx "Camera moves through a garden" \
  --input ./garden.png --width 768 --height 512 --output outputs/garden.mp4
uv run videogenerator run "A mountain sunrise." --backend sdxl --voice espeak
```

Unsupported conditioning types/keyframes fail explicitly, rather than pretending
to guide a model. LTX supports a single initial image; arbitrary intermediate
keyframe conditioning is not implemented. Dimensions/frame counts are validated
per backend; some models have fixed frame presets. Assembly trims/pads generated
media to the chosen span. Speech duration follows the text and can require a
longer span or speed adjustment.

Weights live in the normal Hugging Face cache (`HF_HOME` may relocate it).
The environment is locked in `uv.lock`; model IDs currently follow upstream
default revisions, **not immutable weight revisions**. A previously downloaded
cache can be used offline with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
once all required files/dependencies are available.

## Timeline and web editor

- Storyboard is the base video track. Filename words choose matching assets;
  numeric filename order provides a fallback. Audio files become global music
  tracks; video sound is separated into source-audio tracks.
- The timeline and storyboard bins share the same time scale. Click a span,
  choose an asset and edit start/duration/source offset, prompt, speed, gain,
  fades and ducking. Frame spans use `start = frame / fps`.
- Add any number of audio/video tracks. Later enabled visual tracks overlay
  earlier tracks over their anchored spans. Duplicate or regenerate a segment
  as a **disabled alternative track**, then enable/disable to compare seeds or
  backends without destroying the original.
- Add narration or score later by regenerating a span with an audio backend.
  Music tracks support soft/hard voice ducking. Source video audio is independently
  selectable; audio speed uses FFmpeg `atempo` to preserve pitch.
- Save edits before exporting. The editor runs generation/export jobs in the
  background, previews completed MP4s and offers a download. One job runs at a
  time. First model installation/download can take much longer than generation.
- Default scripted asset scenes receive aligned caption overlays. Human
  transcripts can be edited in caption JSON. Advanced project JSON exposes all
  tracks, global assets and frame-span settings.

Project documents are versioned JSON. Paths are absolute local files; server
projects may reference only files inside their workspace. CLI projects may
reference local assets anywhere. Import copies are not implicit in the CLI.

```sh
uv run videogenerator new "First scene. Second scene." --assets ./assets --project projects/demo.json
uv run videogenerator regenerate projects/demo.json scene-1 --backend sdxl --seed 123
# Enable the alternative in JSON or in a workspace editor:
uv run videogenerator render projects/demo.json --output outputs/demo.mp4
uv run videogenerator serve --workspace projects --no-browser
```

Use `--project projects/project.json` when serving `--workspace projects`;
the server reads that exact filename. Files referenced from outside this folder
must be uploaded/copied into the workspace first.

## Transcript-guided automatic edits

Word-level human transcripts use `[{"start":0.2,"duration":0.4,"text":"Hello"}]`.
Phrase-level transcripts work, but individual filler words cannot be removed
inside a phrase without word timestamps.

```sh
uv run videogenerator transcribe ./interview.mp4 --output outputs/transcript.json
uv run videogenerator edit ./interview.mp4 --transcript outputs/transcript.json \
  --max-pause 0.35 --drop-phrase "off topic" --output outputs/edited.mp4
# Without --transcript, edit requests local Whisper-small automatically.
```

Editing keeps selected speech ranges, trims long gaps and fillers, separates
original audio/video, and retimes captions. It is deterministic transcript-based
excerpting, **not a semantic “boring segment” detector or LLM summarizer**.
Manual captions/transcripts avoid any transcription model download.

## Estimates

```sh
uv run videogenerator hardware
uv run videogenerator estimate projects/local/project.json
uv run videogenerator estimate projects/local/project.json --hardware ./target-hardware.json
```

Estimates report per-piece and total heuristic ranges, RAM/VRAM warnings and the
detected/overridden hardware. They are planning aids, **not measured benchmarks
or guarantees**. Model loading/download time is separate; offload, GPU
architecture, resolution and backend settings matter. Existing assets need
assembly, not generation. GPU count is detected but does not imply speedup:
inference is sequential on one device. Use `hardware` output as the template
for a target hardware profile.

## Existing ecosystems and deployment

There is no verified universal repository bundling *all* local generators,
professional editing, arbitrary-keyframe regeneration and unrestricted weights.
For broader model coverage, [ComfyUI](https://github.com/Comfy-Org/ComfyUI) is a
strong node/workflow ecosystem with desktop installs and native local UI.
It still requires appropriate workflows, dependencies and weights; it is not
automatically a timeline NLE. The [Wan examples](https://github.com/Comfy-Org/ComfyUI_examples/tree/master/wan)
and [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) are starting
points. [AudioCraft](https://github.com/facebookresearch/audiocraft) provides
MusicGen's native Gradio demos. These native applications are alternatives,
**not installed, launched or integrated by this repository**.

For this application, `./start.sh` is the native one-liner. A CPU Linux container:

```sh
docker build -t videogenerator .
mkdir -p projects/local
docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v "$(pwd)/projects/local:/data" videogenerator
```

Host networking preserves the localhost security boundary; this example targets
Linux. For other platforms use the direct launcher. GPU containers require a
CUDA-compatible Torch build and NVIDIA Container Toolkit, and are not verified
here. A rented GPU VM plus SSH forwarding is another deployment option; rented
compute is paid and provider accounts are required. No unauthenticated public
“one-click cloud” endpoint is provided.

## Validation and boundaries

```sh
uv run python -m unittest discover -s tests -v
# Opt-in REAL inference/download tests; substantial time/disk/GPU required:
VIDEOGENERATOR_LIVE_MODELS=sdxl,cogvideox,wan,ltx,musicgen,bark \
  uv run --extra models python -m unittest discover -s tests -p test_models_live.py -v
```

Offline tests cover project validation, real small FFmpeg/eSpeak workflows,
HTTP security/asset isolation, transcript edits, mocked adapter dispatch and
estimation. **Mock tests do not establish model quality, actual VRAM use, or
real GPU compatibility. No large-weight GPU inference was verified in this
environment.** Live tests are deliberately opt-in, not silently replaced with
fake output.

This is a V1/V1.5 foundation, not fulfillment of every aspirational feature:
advanced crossfades/effects, interactive drag editing, automatic semantic
summarization, arbitrary intermediate keyframes, relighting/background swaps,
external framework/native-UI orchestration, configurable hosted paid providers,
multi-GPU scheduling and calibrated time/cost benchmarks remain future work.
Useful production additions include undo/version history, autosave recovery,
proxy media, color management, loudness normalization, subtitle export,
accessibility review and reproducible model revisions.
