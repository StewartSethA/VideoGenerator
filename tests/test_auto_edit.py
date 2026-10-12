import unittest
from unittest.mock import patch

from videogenerator.editing import select_highlights


class AutoEditSelectionTests(unittest.TestCase):
    def test_highlight_selector_prefers_information_over_fillers(self):
        words=[
            {"start":0.0,"duration":0.5,"text":"um"},
            {"start":0.5,"duration":0.5,"text":"uh"},
            {"start":1.0,"duration":0.8,"text":"important"},
            {"start":1.8,"duration":0.8,"text":"result."},
            {"start":4.0,"duration":0.5,"text":"you"},
            {"start":4.5,"duration":0.5,"text":"know"},
            {"start":5.0,"duration":0.8,"text":"another"},
            {"start":5.8,"duration":0.8,"text":"finding."},
        ]
        keep=select_highlights(words,target_duration=3.2)
        self.assertIn(2,keep)
        self.assertIn(3,keep)
        self.assertNotIn(0,keep)
        self.assertNotIn(1,keep)

    def test_selector_preserves_original_order_and_respects_budget(self):
        words=[
            {"start":i,"duration":0.7,"text":text}
            for i,text in enumerate(["alpha","beta.","filler","filler.","gamma","delta."])
        ]
        keep=select_highlights(words,target_duration=2.8)
        self.assertEqual(keep,sorted(keep))
        self.assertLessEqual(sum(words[i]["duration"] for i in keep),2.8+1e-9)

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            select_highlights([{"start":0,"duration":1,"text":"x"}],target_duration=0)


if __name__=="__main__":
    unittest.main()
