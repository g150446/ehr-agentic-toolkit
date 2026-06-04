#!/usr/bin/env python3
"""
EHR remote client

ehr_controller のフローをネットワーク越しに呼び出す CLI。
このファイルは別 Mac にコピーして単体で使用できる。

必要な依存関係:
    pip install requests

使用方法:
    python client.py --host <サーバーIP> [--port 8765] [--api-key KEY] --last-prescription
    python client.py --host <サーバーIP> [--port 8765] [--api-key KEY] --open-note

    # ヘルスチェック
    python client.py --host 192.168.x.x --health
"""

from __future__ import annotations

import argparse
import sys

try:
    import requests
except ImportError:
    print("requests が必要です: pip install requests", file=sys.stderr)
    sys.exit(1)


def health_check(host: str, port: int) -> int:
    url = f"http://{host}:{port}/health"
    try:
        resp = requests.get(url, timeout=5)
        print(resp.json())
        return 0 if resp.status_code == 200 else 1
    except requests.exceptions.ConnectionError:
        print(f"エラー: サーバーに接続できません ({host}:{port})", file=sys.stderr)
        return 1


def _run_flow(host: str, port: int, api_key: str, endpoint: str) -> int:
    url = f"http://{host}:{port}{endpoint}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with requests.post(url, headers=headers, stream=True, timeout=(10, None)) as resp:
            if resp.status_code == 409:
                print("エラー: サーバーで別の処理が実行中です", file=sys.stderr)
                return 1
            if resp.status_code == 401:
                print("エラー: 認証失敗 (--api-key を確認してください)", file=sys.stderr)
                return 1
            if resp.status_code != 200:
                print(f"エラー: HTTP {resp.status_code}", file=sys.stderr)
                return 1

            exit_code = 0
            for line in resp.iter_lines(decode_unicode=True):
                if line.startswith("[exit ") and line.endswith("]"):
                    try:
                        exit_code = int(line[6:-1])
                    except ValueError:
                        pass
                else:
                    print(line, flush=True)
            return exit_code

    except requests.exceptions.ConnectionError:
        print(f"エラー: サーバーに接続できません ({host}:{port})", file=sys.stderr)
        return 1
    except requests.exceptions.Timeout:
        print("エラー: 接続タイムアウト", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n中断されました", file=sys.stderr)
        return 1


def run_last_prescription(host: str, port: int, api_key: str) -> int:
    return _run_flow(host, port, api_key, "/run/last-prescription")


def run_open_note(host: str, port: int, api_key: str) -> int:
    return _run_flow(host, port, api_key, "/run/open-note")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EHR remote client — ehr_controller をネットワーク越しに呼び出す"
    )
    parser.add_argument("--host", required=True, help="サーバーの IP アドレス")
    parser.add_argument("--port", type=int, default=8765, help="ポート番号 (デフォルト: 8765)")
    parser.add_argument("--api-key", default="", help="Bearer トークン (サーバーで設定している場合)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--last-prescription", action="store_true", help="最終処方フローを実行")
    group.add_argument("--open-note", action="store_true", help="メモ帳を開く")
    group.add_argument("--health", action="store_true", help="サーバーのヘルスチェック")
    args = parser.parse_args()

    if args.health:
        sys.exit(health_check(args.host, args.port))
    elif args.last_prescription:
        sys.exit(run_last_prescription(args.host, args.port, args.api_key))
    elif args.open_note:
        sys.exit(run_open_note(args.host, args.port, args.api_key))


if __name__ == "__main__":
    main()
