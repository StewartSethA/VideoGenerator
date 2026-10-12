import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videogenerator import ecosystem


class EcosystemTests(unittest.TestCase):
    def test_registry_is_generic_and_has_expected_categories(self):
        data = ecosystem.catalog()
        self.assertIn("comfyui", data["frameworks"])
        self.assertIn("wan22_t2v_a14b", data["models"])
        serialized = ecosystem.format_catalog()
        self.assertIn("Qwen/Qwen-Image", serialized)
        self.assertIn("ComfyUI", serialized)

    def test_unknown_entries_fail_explicitly(self):
        with self.assertRaises(ValueError):
            ecosystem.install_framework("missing")
        with self.assertRaises(ValueError):
            ecosystem.install_model("missing")

    def test_launch_command_uses_framework_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"VIDEOGENERATOR_HOME": tmp}
        ):
            root = ecosystem.framework_root("comfyui")
            root.mkdir(parents=True)
            command, cwd = ecosystem.launch_command("comfyui", 9001)
            self.assertEqual(cwd, root)
            self.assertIn("main.py", command)
            self.assertIn("9001", command)

    def test_status_does_not_install_anything(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"VIDEOGENERATOR_HOME": tmp}
        ):
            value = ecosystem.status()
            self.assertIn("tools", value)
            self.assertFalse(value["frameworks"]["comfyui"]["installed"])


if __name__ == "__main__":
    unittest.main()
