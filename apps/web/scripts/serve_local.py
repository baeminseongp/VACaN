"""Serve the dashboard plus two explicitly allow-listed local CAVIAR videos."""

from __future__ import annotations

import argparse
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


WEB_ROOT = Path(__file__).resolve().parents[1]
DIST = WEB_ROOT / "dist"
MEDIA_ROOT = WEB_ROOT.parents[1] / "data/processed/caviar/walkbyshop1/media"
MEDIA = {
    "/local-media/caviar/corridor.mp4": MEDIA_ROOT / "corridor.mp4",
    "/local-media/caviar/front.mp4": MEDIA_ROOT / "front.mp4",
}


def byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    if not value or not value.startswith("bytes=") or "," in value:
        return None
    start_text, end_text = value[6:].split("-", 1)
    if not start_text:
        length = min(int(end_text), size)
        return size - length, size - 1
    start = int(start_text)
    end = min(int(end_text), size - 1) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("unsatisfiable range")
    return start, end


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        media = MEDIA.get(urlsplit(self.path).path)
        if media is None:
            super().do_HEAD()
        else:
            self.send_media(media, head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        media = MEDIA.get(urlsplit(self.path).path)
        if media is None:
            super().do_GET()
        else:
            self.send_media(media, head_only=False)

    def send_media(self, path: Path, head_only: bool) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Run make web-media first")
            return
        size = path.stat().st_size
        try:
            selected = byte_range(self.headers.get("Range"), size)
        except (ValueError, TypeError):
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = selected or (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if selected:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only:
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not DIST.is_dir():
        raise SystemExit("Missing apps/web/dist; run make web-build first")
    missing = [str(path) for path in MEDIA.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing browser media; run make web-media first:\n" + "\n".join(missing))
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"VACaN local dashboard: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
