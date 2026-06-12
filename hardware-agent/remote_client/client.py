#!/usr/bin/env python3
"""
EHR remote client

ehr_controller のフローをネットワーク越しに呼び出す CLI。
このファイルは別 Mac にコピーして単体で使用できる。

必要な依存関係:
    pip install requests

使用方法:
    # 対話モード (--host のみ指定)
    python client.py --host <サーバーIP>

    # 一発実行
    python client.py --host <サーバーIP> [--port 8765] [--api-key KEY] --copy-prev-rx
    python client.py --host <サーバーIP> [--port 8765] [--api-key KEY] --open-note

    # 対話モード (--host のみ): "copy prev rx", "open note", "open test", "exit"
    python client.py --host <サーバーIP>

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


def run_copy_prev_rx(host: str, port: int, api_key: str) -> int:
    return _run_flow(host, port, api_key, "/run/copy-prev-rx")


def run_open_note(host: str, port: int, api_key: str) -> int:
    return _run_flow(host, port, api_key, "/run/open-note")


def run_care_plan(host: str, port: int, api_key: str) -> int:
    return _run_flow(host, port, api_key, "/run/care-plan")


_COMMANDS: dict[str, str] = {
    "copy prev rx": "/run/copy-prev-rx",
    "open note":    "/run/open-note",
    "open test":    "/run/open-test",
    "care plan":    "/run/care-plan",
}


def interactive_mode(host: str, port: int, api_key: str) -> int:
    print(f"EHR remote client — {host}:{port}")
    print("コマンド: " + ", ".join(f"'{k}'" for k in _COMMANDS) + ", 'exit'")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します")
            return 0
        if not line:
            continue
        cmd = line.lower()
        if cmd in ("exit", "quit"):
            return 0
        endpoint = _COMMANDS.get(cmd)
        if endpoint is None:
            print(f"不明なコマンド: {line!r}  (使用可能: {', '.join(_COMMANDS)})")
            continue
        _run_flow(host, port, api_key, endpoint)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EHR remote client — ehr_controller をネットワーク越しに呼び出す"
    )
    parser.add_argument("--host", required=True, help="サーバーの IP アドレス")
    parser.add_argument("--port", type=int, default=8765, help="ポート番号 (デフォルト: 8765)")
    parser.add_argument("--api-key", default="", help="Bearer トークン (サーバーで設定している場合)")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--copy-prev-rx", action="store_true", help="前回処方フローを実行")
    group.add_argument("--open-note", action="store_true", help="メモ帳を開く")
    group.add_argument("--care-plan", action="store_true", help="療養計画書フローを実行")
    group.add_argument("--health", action="store_true", help="サーバーのヘルスチェック")
    args = parser.parse_args()

    if args.health:
        sys.exit(health_check(args.host, args.port))
    elif args.copy_prev_rx:
        sys.exit(run_copy_prev_rx(args.host, args.port, args.api_key))
    elif args.open_note:
        sys.exit(run_open_note(args.host, args.port, args.api_key))
    elif args.care_plan:
        sys.exit(run_care_plan(args.host, args.port, args.api_key))
    else:
        sys.exit(interactive_mode(args.host, args.port, args.api_key))


if __name__ == "__main__":
    main()
