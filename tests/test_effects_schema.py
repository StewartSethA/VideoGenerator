import copy
import tempfile
import unittest
from pathlib import Path

from videogenerator.project import create_project, validate_project


class EffectsSchemaTests(unittest.TestCase):
    def test_default_segment_has_transform_and_effects(self):
        project = create_project("Scene")
        segment = project["tracks"][0]["segments"][0]
        self.assertEqual(segment["transform"]["scale"], 1)
        self.assertEqual(segment["transform"]["opacity"], 1)
        self.assertEqual(segment["effects"], [])

    def test_transform_and_supported_effects_validate(self):
        project = create_project("Scene")
        segment = project["tracks"][0]["segments"][0]
        segment["transform"] = {"x":20,"y":-10,"scale":1.2,"rotation":5,"opacity":0.7}
        segment["effects"] = [
            {"type":"grayscale"},
            {"type":"blur","radius":2},
            {"type":"sharpen","amount":1.5},
            {"type":"eq","brightness":0.1,"contrast":1.1,"saturation":0.9},
        ]
        validate_project(project)

    def test_invalid_effect_or_transform_rejected(self):
        base = create_project("Scene")
        cases = [
            ("transform", {"x":0,"y":0,"scale":0,"rotation":0,"opacity":1}),
            ("transform", {"x":0,"y":0,"scale":1,"rotation":0,"opacity":2}),
            ("effects", [{"type":"unknown"}]),
            ("effects", [{"type":"blur","radius":-1}]),
        ]
        for field, value in cases:
            with self.subTest(field=field,value=value):
                project=copy.deepcopy(base)
                project["tracks"][0]["segments"][0][field]=value
                with self.assertRaises(ValueError):
                    validate_project(project)

    def test_audio_mastering_settings_validate(self):
        project=create_project("Scene")
        project["settings"]["audio"]={"normalize":True,"target_lufs":-14,"true_peak":-1,"lra":11}
        validate_project(project)
        project["settings"]["audio"]["target_lufs"]=20
        with self.assertRaises(ValueError):
            validate_project(project)


if __name__=="__main__":
    unittest.main()
