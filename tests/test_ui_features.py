import unittest
from pathlib import Path


class UiFeatureTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/"src"/"videogenerator"/"static"
        self.html=(root/"index.html").read_text(encoding="utf-8")
        self.js=(root/"ui.js").read_text(encoding="utf-8")

    def test_effects_and_mastering_controls_exist(self):
        self.assertIn('id="effects"',self.html)
        self.assertIn('id="normalize-audio"',self.html)
        self.assertIn('id="target-lufs"',self.html)
        for field in ("x","y","scale","rotation","opacity"):
            self.assertIn(f'transform-{field}',self.js)

    def test_keyframes_are_presented_as_supported_feature_not_stub_warning(self):
        self.assertIn('id="keyframes"',self.html)
        self.assertNotIn("unsupported requests fail explicitly",self.html.lower())


if __name__=="__main__":
    unittest.main()
