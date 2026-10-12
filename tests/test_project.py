import copy
import json
import tempfile
import unittest
from pathlib import Path

from videogenerator.project import (
    create_project, import_assets, load_project, save_project, validate_project,
)


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".project-test-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()

    def test_prompt_storyboard_and_roundtrip(self):
        project = create_project("First scene. Second scene!", duration=6, width=320, height=180)
        segments = project["tracks"][0]["segments"]
        self.assertEqual([s["text"] for s in segments], ["First scene.", "Second scene!"])
        self.assertEqual([s["start"] for s in segments], [0, 6])
        self.assertEqual([s["duration"] for s in segments], [6, 6])
        path = self.folder / "project.json"
        save_project(project, path)
        self.assertEqual(load_project(path), project)

    def test_descriptive_assets_and_source_audio(self):
        for name in ("01_ocean.mp4", "02_forest.png", "music.wav", "ignore.txt"):
            (self.folder / name).write_bytes(b"fixture")
        assets = import_assets(self.folder)
        self.assertEqual(len(assets), 3)
        project = create_project("Forest. Ocean.", self.folder, duration=4)
        by_id = {a["id"]: a for a in project["assets"]}
        scenes = project["tracks"][0]["segments"]
        self.assertEqual(by_id[scenes[0]["asset_id"]]["name"], "02_forest.png")
        self.assertEqual(by_id[scenes[1]["asset_id"]]["name"], "01_ocean.mp4")
        source = next(t for t in project["tracks"] if t["id"] == "source-audio")
        self.assertEqual(source["segments"][0]["start"], 4)
        self.assertEqual(len(project["global_asset_ids"]), 1)
        music = next(t for t in project["tracks"] if t["id"] == "music-1")
        self.assertEqual(music["segments"][0]["duration"], 8)
        validate_project(project, self.folder)

    def test_invalid_numbers_and_types(self):
        base = create_project("Hello")
        for field, value in (
            ("start", -1), ("duration", 0), ("duration", float("nan")),
            ("start", 10**1000),
            ("source_start", float("inf")), ("speed", 0), ("speed", True),
            ("volume", 11), ("fade_in", 5), ("seed", "0"), ("text", 17),
            ("keyframes", [{}]), ("duck", "unknown"), ("role", "unknown"),
            ("asset_id", []),
        ):
            with self.subTest(field=field, value=value):
                project = copy.deepcopy(base)
                project["tracks"][0]["segments"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_project(project)
        for field, value in (("width", 15), ("height", 721), ("fps", 0), ("fps", 24.0)):
            with self.subTest(field=field):
                project = copy.deepcopy(base)
                project["settings"][field] = value
                with self.assertRaises(ValueError):
                    validate_project(project)

    def test_timestamp_order_and_script(self):
        for name in ("scene_10.mp4", "scene_2.mp4", "scene_1.mp4"):
            (self.folder / name).write_bytes(b"fixture")
        project = create_project("unused", self.folder,
                                 script="One.\nTwo.\nThree.", duration=3)
        assets = {asset["id"]: asset for asset in project["assets"]}
        scenes = project["tracks"][0]["segments"]
        self.assertEqual([assets[s["asset_id"]]["name"] for s in scenes],
                         ["scene_1.mp4", "scene_2.mp4", "scene_10.mp4"])
        self.assertEqual([s["text"] for s in scenes], ["One.", "Two.", "Three."])
    def test_schema_duplicates_and_unknown_references(self):
        base = create_project("Hello")
        invalid = []
        project = copy.deepcopy(base)
        project["tracks"][0]["segments"][0]["id"] = "storyboard"
        invalid.append(project)
        project = copy.deepcopy(base)
        project["global_asset_ids"] = ["missing"]
        invalid.append(project)
        project = copy.deepcopy(base)
        project["tracks"][0]["enabled"] = 1
        invalid.append(project)
        project = copy.deepcopy(base)
        project["captions"] = [{"start": 0, "duration": -1, "text": "bad"}]
        invalid.append(project)
        invalid.extend([{}, [], {"version": True}])
        for project in invalid:
            with self.subTest(project=project), self.assertRaises(ValueError):
                validate_project(project)

    def test_path_escape_and_symlinks(self):
        root = self.folder / "assets"
        root.mkdir()
        outside = self.folder / "outside.png"
        outside.write_bytes(b"fixture")
        project = create_project("Hello")
        project["assets"] = [{"id": "outside", "name": "outside.png",
                              "path": str(outside), "kind": "image"}]
        with self.assertRaises(ValueError):
            validate_project(project, root)
        link = root / "escape.png"
        link.symlink_to(outside)
        project["assets"][0]["path"] = str(link)
        with self.assertRaises(ValueError):
            validate_project(project, root)
        with self.assertRaises(ValueError):
            import_assets(root)
        project["assets"][0]["path"] = "https://example.com/image.png"
        with self.assertRaises(ValueError):
            validate_project(project)

    def test_invalid_json(self):
        path = self.folder / "bad.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_project(path)
        path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_project(path)


if __name__ == "__main__":
    unittest.main()
