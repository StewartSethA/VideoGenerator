import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from videogenerator.project import create_project
from videogenerator.render import render


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderEffectsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def ffmpeg(self,*args):
        return subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-threads","1",*args],
                              check=True,capture_output=True).stdout

    def test_grayscale_effect_changes_real_pixels(self):
        source=self.root/"red.mp4"
        self.ffmpeg("-f","lavfi","-i","color=red:s=160x90:r=20:d=1","-c:v","libx264",
                    "-pix_fmt","yuv420p",str(source))
        project=create_project("red",duration=1,width=160,height=90,fps=20)
        project["assets"]=[{"id":"red","name":"red.mp4","path":str(source),"kind":"video"}]
        scene=project["tracks"][0]["segments"][0]
        scene["asset_id"]="red"
        scene["effects"]=[{"type":"grayscale"}]
        project["captions"]=[]
        output=render(project,self.root/"gray.mp4")
        pixel=self.ffmpeg("-ss","0.5","-i",str(output),"-frames:v","1","-vf","scale=1:1",
                          "-f","rawvideo","-pix_fmt","rgb24","pipe:1")[:3]
        self.assertLess(max(pixel)-min(pixel),8)

    def test_transform_and_mastering_render_without_invalid_graph(self):
        source=self.root/"source.mp4"
        self.ffmpeg("-f","lavfi","-i","color=blue:s=160x90:r=20:d=1",
                    "-f","lavfi","-i","sine=frequency=440:duration=1",
                    "-c:v","libx264","-c:a","aac","-shortest",str(source))
        project=create_project("source",duration=1,width=160,height=90,fps=20)
        project["assets"]=[{"id":"src","name":"source.mp4","path":str(source),"kind":"video"}]
        scene=project["tracks"][0]["segments"][0]
        scene["asset_id"]="src"
        scene["transform"]={"x":5,"y":-3,"scale":0.8,"rotation":2,"opacity":0.8}
        audio=dict(scene)
        audio["id"]="audio"; audio["role"]="source"; audio["effects"]=[]
        project["tracks"].append({"id":"audio-track","name":"Audio","kind":"audio","enabled":True,
                                  "segments":[audio]})
        project["settings"]["audio"]={"normalize":True,"target_lufs":-14,"true_peak":-1,"lra":11}
        project["captions"]=[]
        output=render(project,self.root/"mastered.mp4")
        info=json.loads(subprocess.run(["ffprobe","-v","error","-show_streams","-of","json",str(output)],
                                      check=True,capture_output=True,text=True).stdout)
        self.assertEqual({s["codec_type"] for s in info["streams"]},{"video","audio"})


if __name__=="__main__":
    unittest.main()
