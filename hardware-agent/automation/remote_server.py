"""
Remote HTTP サーバー

ehr_controller のフローを HTTP API として公開する。
別マシンの remote_client/client.py から呼び出すことができる。

使用方法:
    python -m automation.remote_server
    # または
    ./scripts/start_remote_server.sh

環境変数:
    REMOTE_SERVER_PORT      ポート番号 (デフォルト: 8765)
    REMOTE_SERVER_API_KEY   Bearer トークン (未設定なら認証なし・LAN 内限定想定)

エンドポイント:
    GET  /health                ヘルスチェック
    POST /run/copy-prev-rx      --copy-prev-rx フロー
    POST /run/open-note         --open-note フロー
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
from datetime import datetime

PORT = int(os.environ.get("REMOTE_SERVER_PORT", "8765"))
API_KEY = os.environ.get("REMOTE_SERVER_API_KEY", "")

_run_lock = threading.Lock()
_is_running = False


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {self.address_string()} {format % args}", flush=True)

    def _check_auth(self) -> bool:
        if not API_KEY:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {API_KEY}"

    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, data: bytes | str) -> None:
        if isinstance(data, str):
            data = data.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._check_auth():
            self._json(401, {"error": "unauthorized"})
            return
        if self.path == "/run/copy-prev-rx":
            self._run_ehr_controller("--copy-prev-rx")
        elif self.path == "/run/open-note":
            self._run_ehr_controller("--open-note")
        else:
            self._json(404, {"error": "not found"})

    def _run_ehr_controller(self, flag: str) -> None:
        global _is_running
        with _run_lock:
            if _is_running:
                self._json(409, {"error": "already running"})
                return
            _is_running = True

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env = os.environ.copy()
            env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                [sys.executable, "-m", "automation.ehr_controller", flag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=project_root,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self._chunk(line)
            proc.wait()
            self._chunk(f"[exit {proc.returncode}]\n")
            self._chunk(b"")  # chunked transfer encoding terminator
        except Exception as e:
            try:
                self._chunk(f"[server error] {e}\n")
                self._chunk(f"[exit 1]\n")
                self._chunk(b"")
            except Exception:
                pass
        finally:
            with _run_lock:
                _is_running = False


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main() -> None:
    server = _ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"Remote server listening on 0.0.0.0:{PORT}", flush=True)
    if API_KEY:
        print("認証: Bearer トークン有効", flush=True)
    else:
        print("認証: なし (LAN 内限定)", flush=True)
    print("停止するには Ctrl+C を押してください。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("サーバー終了", flush=True)


if __name__ == "__main__":
    main()
