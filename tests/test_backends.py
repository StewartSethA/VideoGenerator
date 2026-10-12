import importlib
import shutil
import sys
import unittest
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from videogenerator import backends


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path("tests") / (".backend-test-" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        backends._load_model.cache_clear()
        self.addCleanup(backends._load_model.cache_clear)

    def mock_dependencies(self, cuda=False):
        torch = MagicMock()
        torch.cuda.is_available.return_value = cuda
        torch.cuda.is_bf16_supported.return_value = True
        modules = {"torch": torch, "diffusers": MagicMock(),
                   "diffusers.utils": MagicMock(), "transformers": MagicMock(),
                   "huggingface_hub": MagicMock()}
        torch.__version__ = "2.14.1"
        modules["huggingface_hub"].list_repo_files.return_value = ["model.safetensors"]
        patcher = patch.object(backends, "_dependency", side_effect=modules.__getitem__)
        patcher.start()
        self.addCleanup(patcher.stop)
        return modules

    def test_import_does_not_load_ml_modules(self):
        original = importlib.import_module
        imported = []

        def tracked(name, *args, **kwargs):
            imported.append(name)
            return original(name, *args, **kwargs)

        with patch("importlib.import_module", side_effect=tracked):
            importlib.reload(backends)
            backends.catalog()
        self.assertFalse(set(imported) & {"torch", "diffusers", "transformers"})

    def test_catalog_is_independent_and_unknown_rejected(self):
        catalog = backends.catalog()
        self.assertEqual({b["id"] for b in catalog},
                         {"title", "espeak", "sdxl", "cogvideox", "wan", "ltx", "musicgen", "bark"})
        catalog[0]["inputs"].append("unsafe")
        self.assertEqual(backends.get_backend("title")["inputs"], [])
        self.assertFalse(backends.get_backend("title")["generative"])
        self.assertEqual(backends.get_backend("title")["kind"], "title")
        self.assertTrue(all(b["kind"] in {"title", "image", "video", "audio"}
                            for b in backends.catalog()))
        self.assertIn("eSpeak: GPL-2.0-or-later", backends.get_backend("espeak")["license"])
        self.assertIn("eSpeak NG: GPL-3.0-or-later", backends.get_backend("espeak")["license"])
        self.assertEqual(backends.get_backend("cogvideox")["license"], "Apache-2.0")
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            backends.get_backend("unknown")

    def test_download_is_safe_and_uses_standard_hf_cache(self):
        modules = self.mock_dependencies()
        hub = modules["huggingface_hub"]
        hub.snapshot_download.return_value = "hf-cache/model"
        with patch.dict("os.environ", {"HF_TOKEN": "environment-token"}):
            self.assertEqual(backends.download("wan"), "hf-cache/model")
        kwargs = hub.snapshot_download.call_args.kwargs
        self.assertEqual(kwargs["repo_id"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(kwargs["token"], "environment-token")
        self.assertNotIn("trust_remote_code", kwargs)
        self.assertNotIn("cache_dir", kwargs)
        self.assertNotIn("*.py", kwargs["allow_patterns"])
        self.assertIn("*.bin", kwargs["ignore_patterns"])

    def test_missing_dependency_has_install_guidance(self):
        with patch("importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "uv sync --extra models"):
                backends.download("sdxl")

    def test_input_validation_precedes_model_loading(self):
        with patch.object(backends, "_dependency") as dependency:
            cases = [
                ("unknown", {}),
                ("wan", {"width": 513}),
                ("ltx", {"height": 520}),
                ("sdxl", {"input_path": self.directory / "missing.png"}),
                ("ltx", {"input_path": self.directory / "missing.png"}),
                ("ltx", {"keyframes": [{"frame": 0}]}),
                ("musicgen", {"duration": 31}),
                ("sdxl", {"duration": float("nan")}),
                ("sdxl", {"duration": 0}),
                ("sdxl", {"seed": -1}),
            ]
            for backend, options in cases:
                with self.subTest(backend=backend, options=options):
                    with self.assertRaises(ValueError):
                        backends.generate(backend, "prompt", self.directory / "out.wav", **options)
            dependency.assert_not_called()

    def test_cpu_video_rejected_upfront(self):
        modules = self.mock_dependencies(cuda=False)
        for backend in ("wan", "ltx", "cogvideox"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                    backends.generate(backend, "scene", self.directory / "out.mp4")
        modules["diffusers"].WanPipeline.from_pretrained.assert_not_called()

    def test_missing_offload_dependency_has_install_guidance(self):
        modules = self.mock_dependencies(cuda=True)
        modules["diffusers"].WanPipeline.from_pretrained.return_value.enable_model_cpu_offload.side_effect = ImportError("accelerate")
        with self.assertRaisesRegex(RuntimeError, "uv sync --extra models"):
            backends.generate("wan", "scene", self.directory / "out.mp4")

    def test_sdxl_cpu_float32_and_model_cache(self):
        modules = self.mock_dependencies()
        cls = modules["diffusers"].StableDiffusionXLPipeline
        pipeline = cls.from_pretrained.return_value
        for name in ("one.png", "two.png"):
            backends.generate("sdxl", "a tree", self.directory / name, seed=42)
        cls.from_pretrained.assert_called_once()
        self.assertEqual(cls.from_pretrained.call_args.kwargs["torch_dtype"],
                         modules["torch"].float32)
        pipeline.to.assert_called_once_with("cpu")
        pipeline.enable_model_cpu_offload.assert_not_called()
        kwargs = pipeline.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "a tree")
        self.assertEqual(kwargs["width"], 512)
        self.assertEqual(kwargs["num_inference_steps"], 30)
        modules["torch"].Generator.return_value.manual_seed.assert_called_with(42)
        pipeline.return_value.images[0].save.assert_called_with(self.directory / "two.png")

    def test_video_pipeline_frames_dtypes_and_offload(self):
        modules = self.mock_dependencies(cuda=True)
        for backend, cls_name, dtype, frame_multiple in (
            ("wan", "WanPipeline", "bfloat16", 4),
            ("ltx", "LTXPipeline", "bfloat16", 8),
            ("cogvideox", "CogVideoXPipeline", "float16", None),
        ):
            with self.subTest(backend=backend):
                pipeline = getattr(modules["diffusers"], cls_name).from_pretrained.return_value
                result = backends.generate(backend, "a scene", self.directory / "out.mp4",
                                           duration=4, width=512, height=512)
                self.assertEqual(result, self.directory / "out.mp4")
                kwargs = pipeline.call_args.kwargs
                if frame_multiple:
                    self.assertEqual((kwargs["num_frames"] - 1) % frame_multiple, 0)
                else:
                    self.assertEqual(kwargs["num_frames"], 49)
                load_kwargs = getattr(modules["diffusers"], cls_name).from_pretrained.call_args.kwargs
                self.assertEqual(load_kwargs["torch_dtype"], getattr(modules["torch"], dtype))
                self.assertTrue(load_kwargs["use_safetensors"])
                pipeline.enable_model_cpu_offload.assert_called_once()
                export = modules["diffusers.utils"].export_to_video.call_args
                self.assertAlmostEqual(export.kwargs["fps"], kwargs["num_frames"] / 4)
        wan_vae = modules["diffusers"].AutoencoderKLWan.from_pretrained
        self.assertEqual(wan_vae.call_args.kwargs["torch_dtype"], modules["torch"].float32)

    def test_ltx_single_image_dispatch(self):
        modules = self.mock_dependencies(cuda=True)
        image = self.directory / "initial.png"
        image.write_bytes(b"mock image")
        backends.generate("ltx", "movement", self.directory / "out.mp4", input_path=image)
        cls = modules["diffusers"].LTXImageToVideoPipeline
        self.assertTrue(cls.from_pretrained.called)
        modules["diffusers.utils"].load_image.assert_called_once_with(str(image))
        self.assertEqual(cls.from_pretrained.return_value.call_args.kwargs["image"],
                         modules["diffusers.utils"].load_image.return_value)

    def test_musicgen_and_bark_cpu_dispatch(self):
        modules = self.mock_dependencies()
        transformers = modules["transformers"]
        processor = transformers.AutoProcessor.from_pretrained.return_value
        processor.return_value = {"input_ids": MagicMock()}
        for backend, cls_name in (("musicgen", "MusicgenForConditionalGeneration"),
                                  ("bark", "BarkModel")):
            with self.subTest(backend=backend), patch.object(backends, "_write_audio") as write:
                backends.generate(backend, "sound", self.directory / "out.wav", duration=2, seed=7)
                cls = getattr(transformers, cls_name)
                self.assertEqual(cls.from_pretrained.call_args.kwargs["torch_dtype"],
                                 modules["torch"].float32)
                self.assertFalse(cls.from_pretrained.call_args.kwargs["trust_remote_code"])
                self.assertTrue(cls.from_pretrained.call_args.kwargs["weights_only"])
                model = cls.from_pretrained.return_value.to.return_value
                kwargs = model.generate.call_args.kwargs
                if backend == "musicgen":
                    self.assertEqual(kwargs["max_new_tokens"], 100)
                else:
                    self.assertNotIn("max_new_tokens", kwargs)
                self.assertTrue(write.called)
        transformers.set_seed.assert_called_with(7)

    def test_audio_binary_checkpoint_uses_restricted_loader(self):
        modules = self.mock_dependencies()
        modules["huggingface_hub"].list_repo_files.return_value = ["pytorch_model.bin"]
        modules["transformers"].AutoProcessor.from_pretrained.return_value.return_value = {}
        with patch.object(backends, "_write_audio"):
            backends.generate("musicgen", "sound", self.directory / "out.wav")
        kwargs = modules["transformers"].MusicgenForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertFalse(kwargs["use_safetensors"])
        self.assertTrue(kwargs["weights_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        backends.download("musicgen")
        download = modules["huggingface_hub"].snapshot_download.call_args.kwargs
        self.assertIn("pytorch_model*.bin", download["allow_patterns"])
        self.assertIn("*.py", download["ignore_patterns"])
        self.assertNotIn("*.bin", download["ignore_patterns"])

    def test_audio_binary_checkpoint_rejects_unsafe_torch(self):
        modules = self.mock_dependencies()
        modules["huggingface_hub"].list_repo_files.return_value = ["pytorch_model.bin"]
        modules["torch"].__version__ = "2.5.1"
        with self.assertRaisesRegex(RuntimeError, "PyTorch >=2.6"):
            backends.generate("bark", "speech", self.directory / "out.wav")
        modules["transformers"].BarkModel.from_pretrained.assert_not_called()

    def test_audio_no_supported_weights_fails_explicitly(self):
        modules = self.mock_dependencies()
        modules["huggingface_hub"].list_repo_files.return_value = ["config.json", "unsafe.py"]
        with self.assertRaisesRegex(RuntimeError, "No supported weight format"):
            backends.download("bark")
        modules["huggingface_hub"].snapshot_download.assert_not_called()

    @unittest.skipUnless(shutil.which("espeak") or shutil.which("espeak-ng"), "eSpeak not installed")
    def test_real_espeak_wav_generation(self):
        output = self.directory / "voice.wav"
        self.assertEqual(backends.generate("espeak", "- hello world", output), output)
        with wave.open(str(output), "rb") as wav:
            self.assertGreater(wav.getnframes(), 0)
            self.assertGreater(wav.getframerate(), 0)
            self.assertEqual(wav.getnchannels(), 1)

    def test_title_guidance_and_system_download(self):
        with self.assertRaisesRegex(ValueError, "timeline renderer"):
            backends.generate("title", "Title", self.directory / "out.mp4")
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            self.assertEqual(backends.download("title"), "/usr/bin/ffmpeg")
        with patch("shutil.which", return_value="/usr/bin/espeak") as which:
            self.assertEqual(backends.download("espeak"), "/usr/bin/espeak")
            which.assert_called_once_with("espeak")
        with patch("shutil.which", side_effect=[None, "/usr/bin/espeak-ng"]):
            self.assertEqual(backends.download("espeak"), "/usr/bin/espeak-ng")
        with patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Install"):
                backends.download("espeak")

    def test_hardware_does_not_import_torch(self):
        completed = SimpleNamespace(stdout="GPU One, 8192\nGPU Two, 24576\n")
        with patch("subprocess.run", return_value=completed), patch.object(backends, "_dependency") as dependency:
            hardware = backends.hardware()
        self.assertEqual(hardware["gpu_count"], 2)
        self.assertEqual(hardware["gpus"][1]["memory_gb"], 24)
        dependency.assert_not_called()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(backends.hardware()["gpu_count"], 0)

    def test_estimate_duration_resolution_steps_and_hardware(self):
        gpu = dict(cpu_count=8, memory_gb=64, gpu_count=1, gpus=[dict(memory_gb=24)])
        base = dict(backend="wan", duration=2, width=512, height=512)

        def upper(segment, hw=gpu):
            return backends.estimate({"segments": [segment]}, hw)["total_seconds_range"][1]

        self.assertGreater(upper({**base, "duration": 4}), upper(base))
        self.assertGreater(upper({**base, "width": 1024}), upper(base))
        self.assertGreater(upper({**base, "steps": 100}), upper(base))
        self.assertGreater(upper(base, {**gpu, "gpus": [dict(memory_gb=4)]}), upper(base))
        cpu = dict(cpu_count=4, memory_gb=64, gpu_count=0)
        self.assertFalse(backends.estimate({"segments": [base]}, cpu)["feasible"])
        image = {**base, "backend": "sdxl"}
        self.assertGreater(upper(image, cpu), upper(image, {**cpu, "cpu_count": 16}))
        result = backends.estimate({"segments": [base]}, gpu)
        self.assertTrue(result["heuristic"])
        self.assertFalse(result["calibrated"])
        self.assertIn("not benchmarked", result["warnings"][0])
        self.assertGreater(result["generation_seconds_range"][0], result["render_seconds_range"][1])
        low_ram = {**gpu, "memory_gb": 2}
        self.assertFalse(backends.estimate({"segments": [base]}, low_ram)["feasible"])
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            upper({**base, "backend": "made-up"})

    def test_estimate_timeline_tracks_and_existing_assets(self):
        project = {
            "settings": {"width": 1024, "height": 512},
            "tracks": [
                {"enabled": True, "segments": [
                    {"id": "generated", "backend": "sdxl", "duration": 4},
                    {"id": "cached", "backend": "wan", "duration": 4, "asset_id": "existing"},
                ]},
                {"enabled": False, "segments": [{"backend": "musicgen", "duration": 4}]},
            ],
        }
        result = backends.estimate(project, dict(cpu_count=8, memory_gb=32, gpu_count=0))
        self.assertEqual(len(result["segments"]), 2)
        self.assertTrue(result["segments"][1]["feasible"])
        self.assertEqual(result["segments"][1]["generation_seconds_range"], [0, 0])
        self.assertGreater(result["segments"][1]["render_seconds_range"][1], 0)
        self.assertGreater(result["segments"][0]["generation_seconds_range"][1], 0)
        project["tracks"][0]["segments"][0]["duration"] = 8
        longer = backends.estimate(project, dict(cpu_count=8, memory_gb=32, gpu_count=0))
        self.assertEqual(result["generation_seconds_range"], longer["generation_seconds_range"])


if __name__ == "__main__":
    unittest.main()
