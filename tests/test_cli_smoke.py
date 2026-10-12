import io
import unittest
from contextlib import redirect_stdout, redirect_stderr

from videogenerator.cli import main, parser


class CliSmokeTests(unittest.TestCase):
    def test_all_documented_top_level_commands_parse(self):
        p = parser()
        commands = {
            "new": ["new","hello"],
            "run": ["run","hello"],
            "render": ["render","project.json"],
            "split": ["split","project.json","storyboard","scene-1","1.0"],
            "crossfade": ["crossfade","project.json","storyboard","scene-1","scene-2","0.5"],
            "regenerate": ["regenerate","project.json","scene-1","--backend","sdxl"],
            "generate": ["generate","sdxl","hello","--output","x.png"],
            "download": ["download","sdxl"],
            "setup": ["setup"],
            "backends": ["backends"],
            "routes": ["routes","--output","video"],
            "ecosystem": ["ecosystem"],
            "ecosystem-status": ["ecosystem-status"],
            "framework": ["framework","install","comfyui"],
            "model": ["model","install","qwen_image"],
            "hardware": ["hardware"],
            "estimate": ["estimate","project.json"],
            "edit": ["edit","source.mp4"],
            "transcribe": ["transcribe","source.mp4","--output","t.json"],
            "serve": ["serve","--no-browser"],
        }
        for name, argv in commands.items():
            with self.subTest(name=name):
                self.assertEqual(p.parse_args(argv).command, name)

    def test_informational_commands_do_not_fail(self):
        for argv in (["backends"], ["ecosystem"], ["ecosystem-status"],
                     ["routes","--output","image"]):
            with self.subTest(argv=argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)


if __name__ == "__main__":
    unittest.main()
