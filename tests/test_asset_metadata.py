import tempfile
import unittest
from pathlib import Path

from videogenerator.project import create_project, import_assets


class AssetMetadataTests(unittest.TestCase):
    def test_sidecar_description_and_tags_improve_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"a.png").write_bytes(b"a")
            (root/"a.txt").write_text("sunset ocean beach\ntags: coast, evening",encoding="utf-8")
            (root/"b.png").write_bytes(b"b")
            assets=import_assets(root)
            a=next(x for x in assets if x["name"]=="a.png")
            self.assertIn("sunset",a["description"])
            self.assertIn("coast",a["tags"])
            project=create_project("An evening coast at sunset.",root)
            scene=project["tracks"][0]["segments"][0]
            selected=next(x for x in project["assets"] if x["id"]==scene["asset_id"])
            self.assertEqual(selected["name"],"a.png")


if __name__=="__main__":
    unittest.main()
