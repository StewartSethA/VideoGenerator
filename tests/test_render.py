import array
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from videogenerator.project import create_project
from videogenerator.render import render


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RenderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".render-test-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()

    def ffmpeg(self, *args):
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", *args],
            check=True, capture_output=True,
        ).stdout

    def video(self, name, color, tone=440):
        path = self.folder / name
        self.ffmpeg("-f", "lavfi", "-i", f"color={color}:s=160x90:r=20:d=2",
                    "-f", "lavfi", "-i", f"sine=frequency={tone}:sample_rate=48000:duration=2",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path))
        return path

    def info(self, path):
        return json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout)

    def pixel(self, path, time):
        data = self.ffmpeg("-ss", str(time), "-i", str(path), "-frames:v", "1",
                           "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
        return tuple(data[:3])

    def audio(self, path, start, duration):
        data = self.ffmpeg("-i", str(path), "-ss", str(start), "-t", str(duration),
                           "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1")
        samples = array.array("f")
        samples.frombytes(data)
        return samples

    def rms(self, samples):
        return math.sqrt(sum(x * x for x in samples) / len(samples))

    def test_prompt_only_and_safe_caption_text(self):
        project = create_project("A title. Another title.", duration=0.5, width=160, height=90, fps=20)
        project["captions"] = [{"start": 0.25, "duration": 0.5,
                                "text": "O'Brien: [x]; %{eif:1:d} \\\\ café"}]
        output = render(project, self.folder / "title.mp4")
        info = self.info(output)
        self.assertEqual({s["codec_type"] for s in info["streams"]}, {"video", "audio"})
        self.assertAlmostEqual(float(info["format"]["duration"]), 1, delta=0.06)
        self.assertGreater(sum(self.pixel(output, 0.3)), 0)
        self.assertLess(self.rms(self.audio(output, 0, 0.5)), 0.0001)

    def test_absolute_gaps_overlaps_and_later_track_priority(self):
        red = self.video("red.mp4", "red")
        blue = self.video("blue.mp4", "blue")
        project = create_project("Base", duration=1.5, width=160, height=90, fps=20)
        project["assets"] = [
            {"id": "red", "name": red.name, "path": str(red), "kind": "video"},
            {"id": "blue", "name": blue.name, "path": str(blue), "kind": "video"},
        ]
        first = project["tracks"][0]["segments"][0]
        first.update(asset_id="red", start=0.25, duration=1)
        later = copy.deepcopy(first)
        later.update(id="later", asset_id="blue", start=0.5, duration=0.25)
        project["tracks"].append({"id": "overlay", "name": "Overlay", "kind": "video",
                                  "enabled": True, "segments": [later]})
        project["captions"] = [{"start": 1.4, "duration": 0.1, "text": ""}]
        output = render(project, self.folder / "overlap.mp4", self.folder)
        early = self.pixel(output, 0.1)
        middle = self.pixel(output, 0.6)
        restored = self.pixel(output, 0.9)
        late = self.pixel(output, 1.35)
        self.assertLess(max(early), 60)
        self.assertGreater(middle[2], middle[0] + 100)
        self.assertGreater(restored[0], restored[2] + 100)
        self.assertLess(max(late), 60)
        self.assertAlmostEqual(float(self.info(output)["format"]["duration"]), 1.5, delta=0.06)

    def test_speed_source_trim_audio_offset_and_pitch(self):
        self.video("red.mp4", "red", 440)
        project = create_project("red", self.folder, duration=1, width=160, height=90, fps=20)
        visual = project["tracks"][0]["segments"][0]
        visual.update(start=0.25, duration=0.75, source_start=0.2, speed=2)
        audio = project["tracks"][1]["segments"][0]
        audio.update(start=0.25, duration=0.75, source_start=0.2, speed=2)
        output = render(project, self.folder / "speed.mp4")
        self.assertLess(self.rms(self.audio(output, 0.02, 0.15)), 0.002)
        samples = self.audio(output, 0.4, 0.3)
        self.assertGreater(self.rms(samples), 0.04)
        crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:]))
        self.assertAlmostEqual(crossings / (len(samples) / 48000), 440, delta=15)
        self.assertAlmostEqual(float(self.info(output)["format"]["duration"]), 1, delta=0.06)

    def test_image_and_disabled_track(self):
        image = self.folder / "green.png"
        self.ffmpeg("-f", "lavfi", "-i", "color=green:s=80x80",
                    "-frames:v", "1", "-threads", "1", str(image))
        project = create_project("green", self.folder, duration=0.5, width=160, height=90, fps=20)
        project["captions"] = []
        hidden = copy.deepcopy(project["tracks"][0])
        hidden.update(id="hidden", enabled=False)
        hidden["segments"][0].update(id="hidden-segment", duration=30)
        project["tracks"].append(hidden)
        output = render(project, self.folder / "image.mp4")
        pixel = self.pixel(output, 0.2)
        self.assertGreater(pixel[1], pixel[0] + 30)
        self.assertAlmostEqual(float(self.info(output)["format"]["duration"]), 0.5, delta=0.06)

    def test_voice_ducking_volume_and_fades(self):
        music = self.folder / "music.wav"
        voice = self.folder / "voice.wav"
        self.ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                    "-c:a", "pcm_s16le", str(music))
        self.ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                    "-t", "0.5", "-c:a", "pcm_s16le", str(voice))
        project = create_project("Title", duration=1.5, width=160, height=90, fps=20)
        project["assets"] = [
            {"id": "music", "name": music.name, "path": str(music), "kind": "audio"},
            {"id": "voice", "name": voice.name, "path": str(voice), "kind": "audio"},
        ]
        music_segment = copy.deepcopy(project["tracks"][0]["segments"][0])
        music_segment.update(id="music-seg", asset_id="music", role="music", duck="hard",
                             volume=0.5, fade_in=0.2, fade_out=0.2)
        voice_segment = copy.deepcopy(music_segment)
        voice_segment.update(id="voice-seg", asset_id="voice", role="voice",
                             start=0.5, duration=0.5, fade_in=0, fade_out=0, volume=1)
        project["tracks"].extend([
            {"id": "music-track", "name": "Music", "kind": "audio", "enabled": True,
             "segments": [music_segment]},
            {"id": "voice-track", "name": "Voice", "kind": "audio", "enabled": True,
             "segments": [voice_segment]},
        ])
        hard = render(project, self.folder / "hard.mp4")
        before = self.rms(self.audio(hard, 0.25, 0.15))
        during = self.rms(self.audio(hard, 0.65, 0.15))
        self.assertLess(during, before * 0.15)
        self.assertLess(self.rms(self.audio(hard, 0.01, 0.05)), before * 0.4)
        self.assertLess(self.rms(self.audio(hard, 1.43, 0.05)), before * 0.4)
        music_segment["duck"] = "soft"
        soft = render(project, self.folder / "soft.mp4")
        soft_during = self.rms(self.audio(soft, 0.65, 0.15))
        self.assertGreater(soft_during, during * 2)
        self.assertLess(soft_during, before * 0.4)

    def test_visual_fade_and_long_atempo_chain(self):
        self.video("red.mp4", "red")
        project = create_project("red", self.folder, duration=1, width=160, height=90, fps=20)
        project["captions"] = []
        visual = project["tracks"][0]["segments"][0]
        visual.update(fade_in=0.4, fade_out=0.4)
        audio = project["tracks"][1]["segments"][0]
        audio.update(speed=4, duration=0.4)
        output = render(project, self.folder / "fade.mp4")
        self.assertLess(self.pixel(output, 0.05)[0], self.pixel(output, 0.5)[0] * 0.5)
        self.assertLess(self.pixel(output, 0.9)[0], self.pixel(output, 0.5)[0] * 0.5)

    def test_renamed_playlists_cannot_load_nested_media(self):
        for name, content in (
            ("network.mp4", "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
             "http://127.0.0.1:9/media.ts\n#EXT-X-ENDLIST\n"),
            ("local.mp4", "ffconcat version 1.0\nfile '/etc/passwd'\n"),
        ):
            with self.subTest(name=name):
                asset = self.folder / name
                asset.write_text(content, encoding="utf-8")
                project = create_project("Playlist", duration=0.2, width=160, height=90, fps=20)
                project["assets"] = [
                    {"id": "playlist", "name": name, "path": str(asset), "kind": "video"},
                ]
                project["tracks"][0]["segments"][0]["asset_id"] = "playlist"
                with self.assertRaisesRegex(RuntimeError, "whitelist|Invalid data"):
                    render(project, self.folder / "rejected.mp4", self.folder)
                self.assertFalse((self.folder / "rejected.mp4").exists())
                audio_track = copy.deepcopy(project["tracks"][0])
                audio_track.update(id="playlist-audio", kind="audio")
                audio_track["segments"][0]["id"] = "playlist-audio-segment"
                project["tracks"].append(audio_track)
                with self.assertRaisesRegex(RuntimeError, "whitelist|Invalid data"):
                    render(project, self.folder / "rejected.mp4", self.folder)

    def test_export_uses_output_filesystem_from_unrelated_readonly_cwd(self):
        cwd = self.folder / "readonly"
        cwd.mkdir()
        cwd.chmod(0o555)
        self.addCleanup(cwd.chmod, 0o755)
        destination = self.folder / "exports" / "output.mp4"
        source = Path(__file__).resolve().parents[1] / "src"
        script = """
import tempfile
from pathlib import Path
from unittest.mock import patch
from videogenerator.project import create_project
from videogenerator.render import render
import sys
output = Path(sys.argv[1])
original = tempfile.TemporaryDirectory
def checked(*args, **kwargs):
    assert Path(kwargs['dir']) == output.parent.resolve()
    assert Path(kwargs['dir']) != Path.cwd()
    return original(*args, **kwargs)
with patch('videogenerator.render.tempfile.TemporaryDirectory', side_effect=checked):
    render(create_project('Title', duration=0.2, width=160, height=90, fps=20), output)
"""
        subprocess.run(
            [sys.executable, "-c", script, str(destination)], cwd=cwd,
            env=dict(os.environ, PYTHONPATH=str(source)),
            check=True, capture_output=True, text=True,
        )
        self.assertAlmostEqual(float(self.info(destination)["format"]["duration"]),
                               0.2, delta=0.06)


if __name__ == "__main__":
    unittest.main()
