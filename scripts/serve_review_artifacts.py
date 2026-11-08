"""Loopback static fixture server for the pre-existing dataset-review browser suites.

Only public benchmark data and two catalog review directories are served. Private
projects, floor plans, inventory, model caches and arbitrary repository files are denied.
"""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = [
    ROOT / "data",
    ROOT / "reports/phase6d2-data-quality",
    ROOT / "reports/phase6d3-dataset-expansion",
]


class ReviewHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = Path(self.translate_path(self.path)).resolve()
        if not any(path.is_relative_to(root.resolve()) for root in ALLOWED) or not path.is_file():
            self.send_error(404)
            return
        super().do_GET()


if __name__ == "__main__":
    ThreadingHTTPServer(
        ("127.0.0.1", 8770), partial(ReviewHandler, directory=str(ROOT))
    ).serve_forever()
