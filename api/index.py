"""Public Vercel API: live Codeforces contest data and frozen RL policy."""
from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from cf_pilot.live_portal import UpstreamError, get_live_portal

PAYLOAD = Path(__file__).resolve().parent / "_private"


class handler(BaseHTTPRequestHandler):
    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > 4_000_000:
            status, body = 413, b'{"error":"Contest response exceeds the demo size limit."}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def dispatch(self, body=None):
        query = parse_qs(urlsplit(self.path).query)
        route = query.get("route", [""])[0].strip("/")
        if route not in ({"snapshot", "detail"} if body is None else {"review"}):
            return self.reply(404, {"error": "Endpoint not found"})
        portal = get_live_portal(PAYLOAD, query.get("contestId", [None])[0])
        snapshot_id = query.get("snapshotId", [None])[0] if body is None else body.get("snapshot_id")
        if route != "snapshot" and snapshot_id != portal.snapshot["snapshot_id"]:
            return self.reply(409, {"error": "Contest data changed since loading. Load the contest again before reviewing."})
        if body is None and route == "snapshot":
            return self.reply(200, portal.public_snapshot())
        if body is None and route == "detail":
            row = portal.by_id.get(query.get("sessionId", [""])[0])
            return self.reply(200 if row else 404, row or {"error": "Session not found"})
        result = portal.infer()
        result["snapshot_id"] = portal.snapshot["snapshot_id"]
        return self.reply(200, result)

    def do_GET(self):
        self.process_api_request()

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin != f"{self.headers.get('X-Forwarded-Proto', 'https')}://{self.headers.get('Host', '')}":
            return self.reply(403, {"error": "Request origin was not accepted"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 4096:
                return self.reply(413, {"error": "Request is too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, UnicodeDecodeError):
            return self.reply(400, {"error": "Invalid request"})
        self.process_api_request(body)

    def process_api_request(self, body=None):
        try:
            self.dispatch(body)
        except UpstreamError as error:
            self.reply(502, {"error": str(error)})
        except ValueError as error:
            self.reply(400, {"error": str(error)})
        except (OSError, KeyError, TypeError):
            self.reply(500, {"error": "Contest data could not be processed. Try reloading."})
