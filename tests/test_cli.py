import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videogenerator.cli import main


class CLITests(unittest.TestCase):
    def test_new_creates_saved_storyboard(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            project = Path(tmp) / "project.json"
            self.assertEqual(main(["new", "A scene", "--project", str(project)]), 0)
            self.assertTrue(project.is_file())

    def test_error_returns_nonzero_without_traceback(self):
        with contextlib.redirect_stderr(io.StringIO()) as stream:
            self.assertEqual(main(["generate", "missing", "Prompt", "--output", "/tmp/no.mp4"]), 1)
        self.assertIn("Unknown backend", stream.getvalue())
        self.assertNotIn("Traceback", stream.getvalue())

    def test_run_connects_generation_and_export(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch("videogenerator.cli.generate_storyboard", side_effect=lambda project, *args: project) as generate, \
             patch("videogenerator.cli.render", return_value=Path(tmp) / "out.mp4") as render:
            result = main(["run", "Prompt", "--voice", "espeak", "--project", str(Path(tmp) / "p.json")])
            self.assertEqual(result, 0)
            generate.assert_called_once()
            render.assert_called_once()
