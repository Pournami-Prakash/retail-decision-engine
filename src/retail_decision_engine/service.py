from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .operational import score_decision_request


class DecisionHandler(BaseHTTPRequestHandler):
    server_version = "RetailDecisionEngine/0.2"

    def _json(self, status: int, body: dict[str, object]) -> None:
        payload = json.dumps(body, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(
                200,
                {
                    "service": "healthy",
                    "decision_policy": "blocked_until_release_gates_pass",
                },
            )
            return
        self._json(404, {"status": "not_found", "reasons": ["unknown_endpoint"]})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/decisions":
            self._json(404, {"status": "not_found", "reasons": ["unknown_endpoint"]})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid_content_length")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request_must_be_an_object")
            result = score_decision_request(request)
            status = 200 if result["status"] == "review_required" else 422
            audit = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "path": self.path,
                "status": result["status"],
                "reasons": result["reasons"],
            }
            print(json.dumps(audit, separators=(",", ":")), flush=True)
            self._json(status, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"status": "invalid", "reasons": [str(exc)]})

    def log_message(self, format: str, *args: object) -> None:
        return


def serve_decisions(host: str = "127.0.0.1", port: int = 8080) -> None:
    with ThreadingHTTPServer((host, port), DecisionHandler) as server:
        server.serve_forever()
