import unittest

from videogenerator.operations import split_segment, crossfade
from videogenerator.project import create_project, validate_project


class OperationTests(unittest.TestCase):
    def test_split_preserves_source_timing(self):
        project=create_project("Scene",duration=4)
        segment=project["tracks"][0]["segments"][0]
        segment["source_start"]=2
        result=split_segment(project,"storyboard","scene-1",1.5)
        segments=result["tracks"][0]["segments"]
        self.assertEqual(len(segments),2)
        self.assertAlmostEqual(segments[0]["duration"],1.5)
        self.assertAlmostEqual(segments[1]["start"],1.5)
        self.assertAlmostEqual(segments[1]["duration"],2.5)
        self.assertAlmostEqual(segments[1]["source_start"],3.5)
        validate_project(result)

    def test_crossfade_creates_overlap_and_fades(self):
        project=create_project("First. Second.",duration=2)
        result=crossfade(project,"storyboard","scene-1","scene-2",0.5)
        left,right=result["tracks"][0]["segments"]
        self.assertEqual(left["fade_out"],0.5)
        self.assertEqual(right["fade_in"],0.5)
        self.assertEqual(right["start"],1.5)
        validate_project(result)

    def test_invalid_edit_is_rejected_without_mutation(self):
        project=create_project("Scene",duration=2)
        with self.assertRaises(ValueError):
            split_segment(project,"storyboard","scene-1",0)
        self.assertEqual(len(project["tracks"][0]["segments"]),1)
        with self.assertRaises(ValueError):
            crossfade(project,"storyboard","scene-1","missing",0.2)


if __name__=="__main__":
    unittest.main()
