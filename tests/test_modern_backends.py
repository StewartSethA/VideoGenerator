import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from videogenerator import backends


class ModernBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        backends._load_model.cache_clear()
        self.addCleanup(backends._load_model.cache_clear)

    def deps(self):
        torch = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        torch.bfloat16 = "bf16"
        torch.float16 = "fp16"
        torch.float32 = "fp32"
        modules = {
            "torch": torch,
            "diffusers": MagicMock(),
            "diffusers.utils": MagicMock(),
            "diffusers.pipelines.ltx2.pipeline_ltx2_condition": MagicMock(),
            "transformers": MagicMock(),
            "huggingface_hub": MagicMock(),
        }
        patcher = patch.object(backends, "_dependency", side_effect=modules.__getitem__)
        patcher.start()
        self.addCleanup(patcher.stop)
        return modules

    def test_modern_backends_are_first_class(self):
        catalog = {b["id"]: b for b in backends.catalog()}
        for identifier in ("wan22_t2v", "wan22_i2v", "wan22_ti2v",
                           "hunyuan15_t2v", "hunyuan15_i2v", "ltx2",
                           "qwen_image", "flux2_klein"):
            self.assertIn(identifier, catalog)
            self.assertTrue(catalog[identifier]["generative"])
        self.assertTrue(catalog["ltx2"]["keyframes"])
        self.assertEqual(catalog["wan22_i2v"]["inputs"], ["image"])

    def test_generic_diffusion_image_backend(self):
        modules = self.deps()
        pipe = modules["diffusers"].DiffusionPipeline.from_pretrained.return_value
        output = self.root / "image.png"
        backends.generate("qwen_image", "lettering", output)
        modules["diffusers"].DiffusionPipeline.from_pretrained.assert_called_once()
        pipe.return_value.images[0].save.assert_called_once_with(output)

    def test_generic_i2v_dispatch_loads_image(self):
        modules = self.deps()
        image = self.root / "ref.png"
        image.write_bytes(b"x")
        pipe = modules["diffusers"].DiffusionPipeline.from_pretrained.return_value
        pipe.return_value.frames = [[MagicMock()]]
        backends.generate("wan22_i2v", "motion", self.root / "video.mp4",
                          input_path=image, duration=2)
        modules["diffusers.utils"].load_image.assert_called_once_with(str(image))
        self.assertIn("image", pipe.call_args.kwargs)
        self.assertTrue(modules["diffusers.utils"].export_to_video.called)

    def test_ltx2_keyframes_become_conditions(self):
        modules = self.deps()
        first = self.root / "first.png"; first.write_bytes(b"1")
        last = self.root / "last.png"; last.write_bytes(b"2")
        pipe = modules["diffusers"].LTX2ConditionPipeline.from_pretrained.return_value
        pipe.vae.config.temporal_compression_ratio = 8
        pipe.return_value.frames = [[MagicMock()]]
        keyframes = [
            {"frame": 0, "path": str(first)},
            {"frame": 120, "path": str(last)},
        ]
        backends.generate("ltx2", "transition", self.root / "ltx.mp4",
                          duration=5, width=768, height=512, keyframes=keyframes)
        cond = modules["diffusers.pipelines.ltx2.pipeline_ltx2_condition"].LTX2VideoCondition
        self.assertEqual(cond.call_count, 2)
        self.assertEqual(cond.call_args_list[0].kwargs["index"], 0)
        self.assertEqual(cond.call_args_list[1].kwargs["index"], -1)
        self.assertEqual(len(pipe.call_args.kwargs["conditions"]), 2)

    def test_non_ltx_keyframes_fail_before_loading(self):
        with patch.object(backends, "_dependency") as dependency:
            with self.assertRaisesRegex(ValueError, "keyframes"):
                backends.generate("wan22_t2v", "scene", self.root / "x.mp4",
                                  keyframes=[{"frame": 0, "path": "x.png"}])
            dependency.assert_not_called()


if __name__ == "__main__":
    unittest.main()
