import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from videogenerator.project import create_project
from videogenerator.workflow import add_generation


class TakeProvenanceTests(unittest.TestCase):
    def test_generated_take_remembers_storyboard_source(self):
        project=create_project("Scene",duration=1,width=64,height=64)
        def fake_generate(backend,prompt,output,**kwargs):
            output.write_bytes(b"x")
            return output
        with tempfile.TemporaryDirectory() as tmp, \
             patch("videogenerator.workflow.ensure_models"), \
             patch("videogenerator.backends.generate",side_effect=fake_generate):
            result=add_generation(project,"scene-1","sdxl",Path(tmp),seed=3)
        generation=result["tracks"][-1]["segments"][0]["generation"]
        self.assertEqual(generation["source_segment_id"],"scene-1")


if __name__=="__main__":
    unittest.main()
