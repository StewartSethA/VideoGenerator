import copy
import json
import mimetypes
import re
import secrets
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import backends
from .project import create_project, import_assets, load_project, save_project, validate_project
from .render import render
from .workflow import add_generation, ensure_models, estimate_plan


class EditorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, workspace: Path, port: int = 7860):
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "assets").mkdir(exist_ok=True)
        self.project_path = self.workspace / "project.json"
        self.project = load_project(self.project_path) if self.project_path.exists() else create_project("Your video")
        validate_project(self.project, asset_root=self.workspace)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.jobs = {}
        self.active = False
        super().__init__(("127.0.0.1", port), EditorHandler)

    def persist(self, project):
        validate_project(project, asset_root=self.workspace)
        save_project(project, self.project_path)
        self.project = project

    def start_job(self, action: str, payload: dict):
        if self.active:
            raise ValueError("A job is already running; wait before editing or starting another")
        project = copy.deepcopy(self.project)
        if action == "regenerate":
            if payload["backend"] != "auto":
                backends.get_backend(payload["backend"])
        elif action == "download":
            backends.get_backend(payload["backend"])
        elif action != "render":
            raise ValueError("Unknown job type")
        identifier = uuid.uuid4().hex
        self.active = True
        if len(self.jobs) >= 20:
            self.jobs.pop(next(iter(self.jobs)))
        self.jobs[identifier] = {"id": identifier, "status": "running", "action": action}

        def work():
            try:
                if action == "render":
                    staging = self.workspace / f".export-{identifier}.mp4"
                    try:
                        render(project, staging, asset_root=self.workspace)
                        staging.replace(self.workspace / "export.mp4")
                    finally:
                        staging.unlink(missing_ok=True)
                    result = "/media/export"
                elif action == "download":
                    ensure_models(payload["backend"])
                    result = backends.download(payload["backend"])
                else:
                    generated = add_generation(
                        project, payload["segment"], payload["backend"],
                        self.workspace / "generated", int(payload.get("seed", 0)),
                        payload.get("input_asset_id"), bool(payload.get("enable", False)),
                    )
                    with self.lock:
                        self.persist(generated)
                    result = generated["tracks"][-1]["id"]
                with self.lock:
                    self.jobs[identifier].update(status="done", result=str(result))
            except Exception as error:
                with self.lock:
                    self.jobs[identifier].update(status="error", error=str(error))
            finally:
                with self.lock:
                    self.active = False

        threading.Thread(target=work, daemon=True).start()
        return self.jobs[identifier].copy()


class EditorHandler(BaseHTTPRequestHandler):
    server: EditorServer

    def log_message(self, format, *args):
        pass

    def headers_ok(self):
        host = self.headers.get("Host", "")
        port = self.server.server_port
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            self.json_response({"error": "Invalid Host"}, 403)
            return False
        if self.command == "POST":
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                self.json_response({"error": "Cross-origin request rejected"}, 403)
                return False
            if not secrets.compare_digest(self.headers.get("X-Editor-Token", ""), self.server.token):
                self.json_response({"error": "Editor token required"}, 403)
                return False
        return True

    def common_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self'; media-src 'self'; object-src 'none'; "
                         "base-uri 'none'; frame-ancestors 'none'")

    def json_response(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def file_response(self, path: Path, content_type: str | None = None):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if not match or not any(match.groups()):
                self.json_response({"error": "Invalid range"}, 416)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                start = max(0, size - int(last))
            if start > end or start >= size:
                self.json_response({"error": "Range outside file"}, 416)
                return
            status = 206
        self.send_response(status)
        self.common_headers()
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        if not self.headers_ok():
            return
        route = urlsplit(self.path).path
        try:
            if route == "/api/state":
                with self.server.lock:
                    state = {"project": copy.deepcopy(self.server.project),
                             "token": self.server.token, "jobs": copy.deepcopy(self.server.jobs),
                             "busy": self.server.active}
                state["backends"] = backends.catalog()
                state["backends"].insert(0, {
                    "id": "auto", "name": "Automatic routing", "kind": "video",
                    "inputs": ["image"], "generative": True,
                    "license": "Depends on selected backend",
                })
                state["hardware"] = backends.hardware()
                self.json_response(state)
            elif route.startswith("/media/"):
                identifier = unquote(route.removeprefix("/media/"))
                with self.server.lock:
                    asset = next((a for a in self.server.project["assets"] if a["id"] == identifier), None)
                path = self.server.workspace / "export.mp4" if identifier == "export" else Path(asset["path"]) if asset else None
                if path is None or not path.resolve().is_relative_to(self.server.workspace) or not path.is_file():
                    self.json_response({"error": "Media not found"}, 404)
                    return
                self.file_response(path)
            elif route in {"/", "/ui.js", "/style.css"}:
                name = "index.html" if route == "/" else route[1:]
                self.file_response(Path(__file__).parent / "static" / name)
            else:
                self.json_response({"error": "Not found"}, 404)
        except (ValueError, OSError) as error:
            self.json_response({"error": str(error)}, 400)

    def read_body(self, limit):
        length = int(self.headers.get("Content-Length", "-1"))
        if length < 0 or length > limit:
            raise ValueError(f"Content-Length must be between 0 and {limit}")
        self.connection.settimeout(30)
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("Incomplete request body")
        return data

    def do_POST(self):
        if not self.headers_ok():
            return
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/upload":
                name = parse_qs(parsed.query).get("name", [""])[0]
                if not name or name != Path(name).name or name in {".", ".."} or "\\" in name or "\x00" in name:
                    raise ValueError("Upload name must be a filename")
                suffix = Path(name).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
                    raise ValueError("Unsupported media extension")
                data = self.read_body(128 * 1024 * 1024)
                with self.server.lock:
                    if self.server.active:
                        raise ValueError("Wait for the active job before uploading")
                    path = self.server.workspace / "assets" / f"{uuid.uuid4().hex[:8]}-{name}"
                    path.write_bytes(data)
                    try:
                        asset = next(a for a in import_assets(path.parent) if Path(a["path"]) == path)
                        asset["id"] = uuid.uuid4().hex
                        project = copy.deepcopy(self.server.project)
                        project["assets"].append(asset)
                        self.server.persist(project)
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
                self.json_response(asset, 201)
                return
            payload = json.loads(self.read_body(1024 * 1024))
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
            with self.server.lock:
                if self.server.active:
                    raise ValueError("Wait for the active job before editing")
                if parsed.path == "/api/create":
                    project = create_project(
                        payload["prompt"], self.server.workspace / "assets",
                        payload.get("script"), float(payload.get("duration", 4)),
                        int(payload.get("width", 1280)), int(payload.get("height", 720)),
                        int(payload.get("fps", 24)),
                    )
                    self.server.persist(project)
                    result = project
                elif parsed.path == "/api/project":
                    self.server.persist(payload["project"])
                    result = self.server.project
                elif parsed.path == "/api/estimate":
                    validate_project(payload.get("project", self.server.project), asset_root=self.server.workspace)
                    result = estimate_plan(payload.get("project", self.server.project), payload.get("backend"))
                elif parsed.path == "/api/job":
                    result = self.server.start_job(payload["action"], payload)
                else:
                    self.json_response({"error": "Not found"}, 404)
                    return
            self.json_response(result)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.json_response({"error": str(error)}, 400)


def serve(workspace: Path, port: int = 7860, open_browser: bool = True):
    with EditorServer(workspace, port) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"VideoGenerator editor: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
