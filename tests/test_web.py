import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from videogenerator.web import EditorServer


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = EditorServer(Path(self.tmp.name), 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def request(self, path, data=None, headers=None):
        request = Request(self.base + path, data=data, headers=headers or {})
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except HTTPError as response:
            return response.code, response.read()

    def post(self, path, value):
        return self.request(path, json.dumps(value).encode(), {"X-Editor-Token": self.server.token})

    def test_state_and_static_ui(self):
        status, data = self.request("/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["token"], self.server.token)
        self.assertEqual(self.request("/")[0], 200)
        self.assertEqual(self.request("/ui.js")[0], 200)

    def test_csrf_origin_and_rebinding_protection(self):
        self.assertEqual(self.request("/api/create", b"{}")[0], 403)
        self.assertEqual(self.request("/api/state", headers={"Host": "attacker.invalid"})[0], 403)
        self.assertEqual(self.request("/api/create", b"{}", {
            "X-Editor-Token": self.server.token, "Origin": "http://attacker.invalid"})[0], 403)

    def test_create_save_and_outside_asset_rejected(self):
        status, data = self.post("/api/create", {"prompt": "One. Two.", "duration": 1})
        self.assertEqual(status, 200)
        project = json.loads(data)
        self.assertEqual(len(project["tracks"][0]["segments"]), 2)
        project["assets"] = [{"id": "outside", "name": "secret", "kind": "image", "path": "/etc/passwd"}]
        self.assertEqual(self.post("/api/project", {"project": project})[0], 400)
        self.assertEqual(self.request("/media/../../etc/passwd")[0], 404)

    def test_uploads_unique_ids_and_range_requests(self):
        headers = {"X-Editor-Token": self.server.token}
        # Import is extension-based; decoding is deferred to FFmpeg on export.
        ids = []
        for name in ("scene.png", "scene.png"):
            status, data = self.request(f"/api/upload?name={name}", b"abcdef", headers)
            self.assertEqual(status, 201, data)
            ids.append(json.loads(data)["id"])
        self.assertEqual(len(set(ids)), 2)
        status, data = self.request(f"/media/{ids[0]}", headers={"Range": "bytes=1-3"})
        self.assertEqual((status, data), (206, b"bcd"))
        self.assertEqual(self.request("/api/upload?name=../bad.png", b"abc", headers)[0], 400)

    def test_bad_payload_and_upload_limits(self):
        self.assertEqual(self.post("/api/create", [])[0], 400)
        self.assertEqual(self.post("/api/create", {"prompt": "Text", "duration": "nan"})[0], 400)
        self.assertEqual(self.post("/api/job", {"action": "unknown"})[0], 400)
