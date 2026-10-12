import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from videogenerator.cli import main
from videogenerator import ecosystem


class DoctorDryRunTests(unittest.TestCase):
    def test_framework_plan_requires_no_checkout(self):
        plan = ecosystem.framework_plan("comfyui", port=8188)
        self.assertEqual(plan["name"], "comfyui")
        self.assertIn("ComfyUI", plan["repo"])
        self.assertIn("8188", " ".join(plan["launch"]))

    def test_model_plan_requires_no_download(self):
        plan = ecosystem.model_plan("qwen_image")
        self.assertEqual(plan["repo_id"], "Qwen/Qwen-Image")
        self.assertTrue(plan["destination"])

    def test_doctor_command_is_side_effect_free(self):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            self.assertEqual(main(["doctor"]), 0)
        self.assertIn("tools", out.getvalue())

    def test_heavy_cli_dry_runs_do_not_install_or_launch(self):
        with patch("videogenerator.ecosystem.install_framework") as install, \
             patch("videogenerator.ecosystem.launch_framework") as launch, \
             patch("videogenerator.ecosystem.install_model") as model:
            for argv in (
                ["framework","install","comfyui","--dry-run"],
                ["framework","launch","comfyui","--port","8188","--dry-run"],
                ["model","install","qwen_image","--dry-run"],
            ):
                with self.subTest(argv=argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(argv), 0)
            install.assert_not_called()
            launch.assert_not_called()
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
