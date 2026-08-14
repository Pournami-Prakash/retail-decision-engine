from __future__ import annotations

import http.server
import socketserver
from pathlib import Path


PORT = 8765
DIRECTORY = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    with ReusableTCPServer(("127.0.0.1", PORT), Handler) as server:
        print(f"Decision lab available at http://127.0.0.1:{PORT}")
        server.serve_forever()


if __name__ == "__main__":
    main()
