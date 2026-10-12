import copy
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from videogenerator.editing import _run as run_media
from videogenerator.editing import edit_media, transcribe
from videogenerator.project import validate_project
from videogenerator.render import _INPUT_SAFETY


def word(start, text, duration=0.2):
    return {"start": start, "duration": duration, "text": text}


def disguised_manifests(source):
    payload = source.with_name("payload.wav")
    with wave.open(str(payload), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    return [
        f"ffconcat version 1.0\nfile '{payload}'\n",
        f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{payload}\n#EXT-X-ENDLIST\n",
    ]


class EditingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".editing-test-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name).resolve() / "recording.mp4"
        self.source.write_bytes(b"mock media")
        self.run = self.enterContext(patch("videogenerator.editing._run"))
        self.run.return_value = subprocess.CompletedProcess([], 0, stdout="10.0\n")

    def test_filler_cuts_keep_original_video_and_audio(self):
        transcript = [word(0, "Um,"), word(0.3, "Hello"), word(0.6, "UH!"),
                      word(0.9, "summer"), word(1.2, "humming"), word(1.5, "ermine"),
                      word(1.8, "you"), word(2.1, "know,"), word(2.4, "world.")]
        original = copy.deepcopy(transcript)
        project = edit_media(self.source, transcript)
        self.assertEqual(transcript, original)
        self.assertEqual(project["prompt"], "Hello summer humming ermine world.")
        self.assertEqual(project["name"], project["prompt"])
        self.assertIn("not a semantic summary", project["description"])
        self.assertEqual([t["kind"] for t in project["tracks"]], ["video", "audio"])
        video, audio = (t["segments"] for t in project["tracks"])
        self.assertEqual([s["source_start"] for s in video], [0.3, 0.9, 2.4])
        for visual, sound in zip(video, audio):
            for field in ("start", "duration", "source_start", "asset_id"):
                self.assertEqual(visual[field], sound[field])
            self.assertEqual(sound["volume"], 1)
            self.assertEqual(sound["speed"], 1)
        self.assertEqual([c["text"] for c in project["captions"]],
                         ["Hello", "summer", "humming", "ermine", "world."])
        for caption, expected in zip(project["captions"], [0, 0.2, 0.5, 0.8, 1.0]):
            self.assertAlmostEqual(caption["start"], expected)
        validate_project(project)

    def test_caps_pauses_and_rebases_every_word(self):
        project = edit_media(self.source, [word(1, "One"), word(1.3, "two"),
                                          word(3, "three")], max_pause=0.35)
        segments = project["tracks"][0]["segments"]
        self.assertEqual(len(segments), 2)
        self.assertEqual([s["source_start"] for s in segments], [1, 3])
        self.assertAlmostEqual(segments[0]["duration"], 0.85)
        self.assertAlmostEqual(segments[1]["start"], 0.85)
        for caption, expected in zip(project["captions"], [0, 0.3, 0.85]):
            self.assertAlmostEqual(caption["start"], expected)

    def test_zero_pause_and_disable_filler_removal(self):
        project = edit_media(self.source, [word(0, "um"), word(1, "Hello")],
                             remove_fillers=False, max_pause=0)
        self.assertEqual(project["prompt"], "um Hello")
        self.assertEqual(len(project["tracks"][0]["segments"]), 2)
        self.assertAlmostEqual(project["captions"][1]["start"], 0.2)

    def test_exact_drop_phrases_and_safe_sentence_entries(self):
        transcript = [word(0, "HELLO,"), word(0.3, "world!"), word(0.6, "worldwide"),
                      word(0.9, "keep um here"), word(1.2, "hello world")]
        project = edit_media(self.source, transcript, drop_phrases=["hello world"])
        self.assertEqual(project["prompt"], "worldwide keep um here")
        self.assertEqual([c["text"] for c in project["captions"]],
                         ["worldwide", "keep um here"])

    def test_audio_source_and_settings(self):
        source = self.source.with_suffix(".wav")
        source.write_bytes(b"fixture")
        project = edit_media(source, [word(0, "Hello")], width=320, height=180, fps=30)
        self.assertEqual(project["assets"][0]["kind"], "audio")
        self.assertEqual([t["kind"] for t in project["tracks"]], ["audio"])
        self.assertEqual(project["settings"], {"width": 320, "height": 180, "fps": 30})
        self.assertEqual(project["global_asset_ids"], [])

    def test_probe_uses_shared_input_safety(self):
        edit_media(self.source, [word(0, "Hello")])
        arguments = self.run.call_args.args[0]
        start = arguments.index("-protocol_whitelist")
        self.assertEqual(arguments[start:start + len(_INPUT_SAFETY)], _INPUT_SAFETY)
        self.assertLess(start, arguments.index(str(self.source)))

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is unavailable")
    def test_probe_rejects_renamed_hls_and_concat(self):
        with patch("videogenerator.editing._run", wraps=run_media):
            for manifest in disguised_manifests(self.source):
                self.source.write_text(manifest)
                with self.subTest(manifest=manifest), self.assertRaisesRegex(
                        RuntimeError, "not on whitelist|Invalid data"):
                    edit_media(self.source, [word(0, "Hello")])

    def test_invalid_transcripts_and_bounds(self):
        invalid = [
            [], {}, [None], [word(-1, "bad")], [word(0, "bad", -1)],
            [word(0, "bad", 0)], [word(float("nan"), "bad")],
            [word(0, "bad", float("inf"))], [word(True, "bad")],
            [word(10, "bad")], [word(0, "")], [word(0, None)],
            [word(1, "first"), word(0, "second")],
            [word(0, "first", 1), word(0.9, "second")],
            [{"start": 0, "text": "missing duration"}],
        ]
        for transcript in invalid:
            with self.subTest(transcript=transcript), self.assertRaises(ValueError):
                edit_media(self.source, transcript)
        with self.assertRaisesRegex(ValueError, "No speech remains"):
            edit_media(self.source, [word(0, "UM!")])

    def test_invalid_options_and_probe(self):
        for options in ({"max_pause": -1}, {"max_pause": float("nan")},
                        {"max_pause": True}, {"remove_fillers": "yes"},
                        {"drop_phrases": "hello"}, {"drop_phrases": [None]},
                        {"drop_phrases": ["!!!"]}, {"width": 321}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                edit_media(self.source, [word(0, "Hello")], **options)
        for duration in ("N/A", "nan", "inf", "0", "-1"):
            self.run.return_value.stdout = duration
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                edit_media(self.source, [word(0, "Hello")])

    def test_unsupported_or_missing_source(self):
        with self.assertRaises(ValueError):
            edit_media(self.source.with_name("absent.mp4"), [word(0, "Hello")])
        unsupported = self.source.with_suffix(".txt")
        unsupported.write_text("text")
        with self.assertRaises(ValueError):
            edit_media(unsupported, [word(0, "Hello")])


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".asr-test-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name).resolve() / "recording.mp4"
        self.source.write_bytes(b"mock media")
        self.scratch = self.source.parent / "os-temp"
        self.scratch.mkdir()
        self.enterContext(patch("tempfile.tempdir", str(self.scratch)))
        samples = Mock()
        samples.astype.return_value.__truediv__ = Mock(return_value="samples")
        self.numpy = types.ModuleType("numpy")
        self.numpy.frombuffer = Mock(return_value=samples)
        self.numpy.float32 = "float32"
        self.recognizer = Mock(return_value={
            "chunks": [{"text": " Hello", "timestamp": (0.1, 0.3)},
                       {"text": " world!", "timestamp": (0.5, 0.8)}],
        })
        self.transformers = types.ModuleType("transformers")
        self.transformers.pipeline = Mock(return_value=self.recognizer)
        self.enterContext(patch.dict(sys.modules, {
            "numpy": self.numpy, "transformers": self.transformers,
        }))
        self.run = self.enterContext(patch("videogenerator.editing._run", side_effect=self.convert))
        self.converted = None

    def convert(self, arguments):
        self.converted = Path(arguments[-1])
        with wave.open(str(self.converted), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000)
        return subprocess.CompletedProcess(arguments, 0, stdout="")

    def test_local_audio_conversion_and_allowlisted_pipeline(self):
        transcript = transcribe(self.source)
        self.assertEqual([w["text"] for w in transcript], ["Hello", "world!"])
        self.assertAlmostEqual(transcript[0]["duration"], 0.2)
        self.assertEqual(transcript[1]["start"], 0.5)
        self.transformers.pipeline.assert_called_once_with(
            "automatic-speech-recognition", model="openai/whisper-small", device=-1,
            trust_remote_code=False, model_kwargs={"use_safetensors": True},
        )
        self.recognizer.assert_called_once_with(
            {"array": "samples", "sampling_rate": 16000},
            return_timestamps="word", chunk_length_s=30,
        )
        arguments = self.run.call_args.args[0]
        self.assertEqual(arguments[0], "ffmpeg")
        self.assertIn("pcm_s16le", arguments)
        self.assertIn("16000", arguments)
        start = arguments.index("-protocol_whitelist")
        self.assertEqual(arguments[start:start + len(_INPUT_SAFETY)], _INPUT_SAFETY)
        self.assertLess(start, arguments.index("-i"))
        self.assertFalse(self.converted.exists())

    def test_transcription_from_read_only_working_directory_and_source(self):
        readonly = self.source.parent / "readonly"
        readonly.mkdir(mode=0o555)
        self.source.parent.chmod(0o555)
        factory = self.enterContext(patch(
            "videogenerator.editing.tempfile.TemporaryDirectory",
            wraps=tempfile.TemporaryDirectory,
        ))
        previous = Path.cwd()
        try:
            os.chdir(readonly)
            transcript = transcribe(self.source)
        finally:
            os.chdir(previous)
            self.source.parent.chmod(0o755)
            readonly.chmod(0o755)
        self.assertEqual(transcript[0]["text"], "Hello")
        factory.assert_called_once_with(prefix=".transcription-")
        self.assertEqual(self.converted.parent.parent, self.scratch)
        self.assertFalse(self.converted.exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is unavailable")
    def test_conversion_rejects_renamed_hls_and_concat(self):
        with patch("videogenerator.editing._run", wraps=run_media):
            for manifest in disguised_manifests(self.source):
                self.source.write_text(manifest)
                with self.subTest(manifest=manifest), self.assertRaisesRegex(
                        RuntimeError, "not on whitelist|Invalid data"):
                    transcribe(self.source)
        self.transformers.pipeline.assert_not_called()

    def test_model_allowlist_before_loading(self):
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            transcribe(self.source, "untrusted/model")
        self.transformers.pipeline.assert_not_called()
        self.run.assert_not_called()

    def test_missing_optional_dependencies_are_actionable(self):
        with patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaisesRegex(RuntimeError, "uv sync --extra models"):
                transcribe(self.source)
        self.run.assert_not_called()

    def test_missing_word_end_is_explicit_and_scratch_is_cleaned(self):
        self.recognizer.return_value = {"chunks": [{"text": "hello", "timestamp": (0, None)}]}
        with self.assertRaisesRegex(ValueError, "missing its start or end"):
            transcribe(self.source)
        self.assertFalse(self.converted.exists())

    def test_model_failure_cleans_scratch(self):
        self.recognizer.side_effect = RuntimeError("Model unavailable")
        with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
            transcribe(self.source)
        self.assertFalse(self.converted.exists())

    def test_invalid_model_results(self):
        for result in ({}, {"chunks": []},
                       {"chunks": [None]},
                       {"chunks": [{"text": "bad"}]},
                       {"chunks": [{"text": None, "timestamp": (0, 0.2)}]},
                       {"chunks": [{"text": "bad", "timestamp": (0.4, 0.2)}]},
                       {"chunks": [{"text": "bad", "timestamp": (0, 2)}]},
                       {"chunks": [{"text": "bad", "timestamp": (0, float("nan"))}]},
                       {"chunks": [{"text": "first", "timestamp": (0, 0.5)},
                                   {"text": "second", "timestamp": (0.4, 0.8)}]}):
            self.recognizer.return_value = result
            with self.subTest(result=result), self.assertRaises(ValueError):
                transcribe(self.source)


if __name__ == "__main__":
    unittest.main()
