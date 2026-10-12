import copy
import tempfile
import unittest
from pathlib import Path

from videogenerator.project import create_project, validate_project
from videogenerator.provenance import generation_key, attach_provenance


class KeyframeProvenanceTests(unittest.TestCase):
    def test_keyframes_validate_and_are_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            image.write_bytes(b"x")
            project = create_project("Scene")
            project["assets"] = [{"id":"ref","name":"reference.png","path":str(image),"kind":"image"}]
            scene = project["tracks"][0]["segments"][0]
            scene["keyframes"] = [
                {"frame": 48, "asset_id": "ref"},
                {"frame": 0, "asset_id": "ref", "prompt": "opening"},
            ]
            validate_project(project)
            self.assertEqual([k["frame"] for k in scene["keyframes"]], [48, 0])

    def test_invalid_keyframes_are_rejected(self):
        project = create_project("Scene")
        scene = project["tracks"][0]["segments"][0]
        for value in (
            [{"frame": -1, "asset_id": "x"}],
            [{"frame": 0}],
            [{"frame": 0, "asset_id": "x"}],
            [{"frame": 1.5, "asset_id": "x"}],
        ):
            with self.subTest(value=value):
                candidate = copy.deepcopy(project)
                candidate["tracks"][0]["segments"][0]["keyframes"] = value
                with self.assertRaises(ValueError):
                    validate_project(candidate)

    def test_generation_key_is_stable_and_sensitive(self):
        base = dict(backend="sdxl", model_revision="abc", prompt="scene", seed=7,
                    width=512, height=512, duration=4, keyframes=[], assets=[])
        first = generation_key(**base)
        self.assertEqual(first, generation_key(**base))
        changed = dict(base, seed=8)
        self.assertNotEqual(first, generation_key(**changed))

    def test_attach_provenance_is_nondestructive(self):
        segment = {"id":"s","prompt":"p"}
        result = attach_provenance(segment, {"backend":"sdxl","cache_key":"abc"})
        self.assertNotIn("generation", segment)
        self.assertEqual(result["generation"]["cache_key"], "abc")


if __name__ == "__main__":
    unittest.main()
