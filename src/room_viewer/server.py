"""Loopback-only Three.js viewer, geometry API, projects and private owned inventory."""
import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from .service import ROOT, edit_scene, load_preset, presentation


class Handler(SimpleHTTPRequestHandler):
    def respond(self, value, status=200):
        content = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/workflow":
            try:
                from .workflow import dashboard
                self.respond(dashboard(self.server.projects,self.server.inventory))
            except (ValueError, OSError, KeyError) as exc:
                self.respond({"error":str(exc)},400)
        elif path.startswith("/api/replacements/"):
            try:
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] == "candidates":
                    self.respond(self.server.replacements.candidates(parts[2]))
                else:
                    raise ValueError("Unknown replacement route")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)
        elif path.startswith("/api/budget/"):
            try:
                parts = path.strip("/").split("/")
                if len(parts) != 3:
                    raise ValueError("Unknown financial route")
                self.respond(self.server.budget.get(parts[2]))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)
        elif path.startswith("/api/transfer/"):
            try:
                parts = path.strip("/").split("/")
                transfer = self.server.transfer
                if len(parts) == 3:
                    self.respond(transfer.public(transfer.read(parts[2])))
                elif len(parts) == 4 and parts[3] == "scene":
                    from .home_transfer import room_scene
                    room_id = parse_qs(urlsplit(self.path).query)["room_id"][0]
                    self.respond(presentation(room_scene(transfer.read(parts[2])["home_transfer"], room_id)))
                elif (len(parts) == 5 and parts[3] == "assets") or (len(parts) == 4 and parts[3] == "export"):
                    file = transfer.asset(parts[2],parts[4]) if len(parts) == 5 else None
                    data = file.read_bytes() if file else transfer.export(parts[2])
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip" if file is None else "image/png" if file.suffix == ".png" else "image/jpeg" if file.suffix == ".jpg" else "application/pdf")
                    self.send_header("Cache-Control", "private, no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    raise ValueError("Unknown destination route")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)
        elif path == "/api/inventory" or path.startswith("/api/inventory/"):
            try:
                parts = path.strip("/").split("/")
                library = self.server.inventory
                if path == "/api/inventory":
                    self.respond(library.listing())
                elif path == "/api/inventory/export":
                    self.respond(library.export())
                elif len(parts) == 5 and parts[3] == "photos":
                    file = library.asset(parts[2], parts[4])
                    data = file.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png" if file.suffix == ".png" else "image/jpeg")
                    self.send_header("Cache-Control", "private, no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    raise ValueError("Unknown inventory route")
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
        elif path.startswith("/api/catalog/images/"):
            try:
                from .product_matching import catalog_asset
                file = catalog_asset(unquote(path.rsplit("/", 1)[1]))
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if file.suffix == ".png" else "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)
        elif path.startswith("/api/projects"):
            try:
                parts = path.strip("/").split("/")
                store = self.server.projects
                if len(parts) == 2:
                    self.respond(store.listing())
                elif len(parts) == 3:
                    self.respond(store.public(store.read(parts[2])))
                elif len(parts) == 6 and parts[3] == "assets":
                    file = store.asset(parts[2], parts[4], parts[5])
                    data = file.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg" if file.suffix == ".jpg" else "image/png" if file.suffix == ".png" else "application/octet-stream")
                    self.send_header("Cache-Control", "private, no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif len(parts) == 4 and parts[3] == "export":
                    data = store.export(parts[2])
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", 'attachment; filename="room-project.zip"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    raise ValueError("Unknown project route")
            except (ValueError, KeyError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)
        elif path.startswith("/api/scenes/"):
            try:
                self.respond(load_preset(path.rsplit("/", 1)[1]))
            except (ValueError, KeyError, OSError) as exc:
                self.respond({"error": str(exc)}, 400)
        else:
            super().do_GET()

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                raise ValueError("Only same-origin local requests are accepted")
            parsed = urlsplit(self.path)
            parts = parsed.path.strip("/").split("/")
            if parsed.path == "/api/transfer/import" or (len(parts) == 4 and parts[:2] == ["api", "transfer"] and parts[3] == "floorplan"):
                size = int(self.headers.get("Content-Length",0))
                if not 0 < size <= (64_000_000 if parsed.path.endswith("/import") else 15_000_000):
                    raise ValueError("Destination upload exceeds size limits")
                raw = self.rfile.read(size)
                if parsed.path.endswith("/import"):
                    self.respond(self.server.transfer.import_bundle(raw))
                else:
                    query = parse_qs(parsed.query)
                    self.respond(self.server.transfer.upload(parts[2],raw,query.get("name",["floorplan"])[0],self.headers.get("X-Local-Rights") == "confirmed",int(query.get("page",[0])[0])))
                return
            if len(parts) == 4 and parts[:2] == ["api", "inventory"] and parts[3] == "photos":
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 15_000_000:
                    raise ValueError("Inventory photo must be smaller than 15 MB")
                self.respond(self.server.inventory.upload_photo(parts[2], self.rfile.read(size), self.headers.get("X-Local-Rights") == "confirmed"))
                return
            if len(parts) == 6 and parts[:2] == ["api", "inventory"] and parts[3] == "documents" and parts[5] == "extract":
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 3_000_000:
                    raise ValueError("Use a PDF smaller than 3 MB")
                self.respond(self.server.inventory.extract_document(parts[2], parts[4], self.rfile.read(size), self.headers.get("X-Local-Rights") == "confirmed"))
                return
            if len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "photos":
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 15_000_000:
                    raise ValueError("Choose a room photograph smaller than 15 MB")
                query = parse_qs(parsed.query)
                self.respond(self.server.projects.upload(parts[2], self.rfile.read(size), query.get("name", ["room-photo"])[0], self.headers.get("X-Local-Rights") == "confirmed", query.get("replace", [None])[0]))
                return
            if parsed.path == "/api/projects/import":
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 64_000_000:
                    raise ValueError("Project ZIP must be smaller than 64 MB")
                self.respond(self.server.projects.import_bundle(self.rfile.read(size)))
                return
            if self.headers.get_content_type() != "application/json":
                raise ValueError("JSON requests required")
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 8_000_000:
                raise ValueError("JSON body must be between 1 byte and 8 MB")
            def invalid(value):
                raise ValueError("Nonfinite JSON values are not accepted")
            body = json.loads(self.rfile.read(size), parse_constant=invalid)
            if len(parts) == 4 and parts[:2] == ["api", "replacements"] and parts[3] in ("action", "preview"):
                result = self.server.replacements.preview(parts[2],body) if parts[3] == "preview" else self.server.replacements.action(parts[2],body)
            elif len(parts) == 4 and parts[:2] == ["api", "budget"] and parts[3] == "action":
                result = self.server.budget.action(parts[2],body)
            elif parsed.path == "/api/transfer":
                result = self.server.transfer.create(body)
            elif len(parts) == 4 and parts[:2] == ["api", "transfer"] and parts[3] == "action":
                result = self.server.transfer.action(parts[2],body)
            elif parsed.path == "/api/inventory/action":
                result = self.server.inventory.action(body)
            elif parsed.path == "/api/inventory/import":
                result = self.server.inventory.import_data(body)
            elif parsed.path == "/api/projects":
                result = self.server.projects.create(body.get("name", "Untitled room"))
            elif len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "action":
                result = self.server.projects.action(parts[2], body)
            elif len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "process":
                result = self.server.jobs.submit(parts[2], body["photo_id"], body.get("retry_failed") is True)
            elif len(parts) == 4 and parts[:2] == ["api", "projects"] and parts[3] == "match-products":
                result = self.server.product_matching.submit(parts[2], body)
            elif self.path == "/api/edit":
                result = edit_scene(body["scene"], body["action"])
            elif self.path == "/api/import":
                result = presentation(body["scene"])
            else:
                raise ValueError("Unknown API endpoint")
            self.respond(result)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, OSError) as exc:
            self.respond({"error": str(exc)}, 400)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT / "viewer")))
    from .photo_jobs import PhotoJobs
    from .photo_projects import ProjectStore
    server.projects = ProjectStore()
    from .inventory import InventoryStore
    server.inventory = InventoryStore(server.projects)
    from .home_transfer import TransferStore
    server.transfer = TransferStore(server.projects, server.inventory)
    from .furniture_budget import BudgetStore
    server.budget = BudgetStore(server.transfer)
    from .replacements import ReplacementStore
    server.replacements = ReplacementStore(server.transfer)
    server.jobs = PhotoJobs(server.projects)
    from .product_matching import ProductMatching
    server.product_matching = ProductMatching(server.projects, server.jobs)
    print(f"Furniture AI prototype: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
