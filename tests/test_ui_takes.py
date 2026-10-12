import unittest
from pathlib import Path


class UiTakeTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/"src"/"videogenerator"/"static"
        self.html=(root/"index.html").read_text(encoding="utf-8")
        self.js=(root/"ui.js").read_text(encoding="utf-8")

    def test_take_browser_and_playhead_are_present(self):
        self.assertIn('id="takes"',self.html)
        self.assertIn('id="playhead"',self.html)
        self.assertIn("source_segment_id",self.js)
        self.assertIn("activateTake",self.js)

    def test_span_drag_and_resize_hooks_are_present(self):
        self.assertIn("pointerdown",self.js)
        self.assertIn("resize-handle",self.js)
        self.assertIn('id="split-playhead"',self.html)
        self.assertIn('id="crossfade-next"',self.html)


if __name__=="__main__":
    unittest.main()
