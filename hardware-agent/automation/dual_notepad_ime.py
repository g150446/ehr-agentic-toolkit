"""
指定したテキストファイルの内容を、2つのメモ帳を使って IME 変換しながら入力する。

動作概要:
    1. 変換用メモ帳と貼り付け用メモ帳を開く
    2. 両方のメモ帳を Win+Up で最大化する
    3. テキストファイルを1行ずつ読み込む
    4. 変換用メモ帳で日本語入力（IME 変換）
    5. Ctrl+A → Ctrl+X で切り取り
    6. Alt+Tab で貼り付け用メモ帳に切り替え
    7. Ctrl+V → Enter で貼り付け
    8. Alt+Tab で変換用メモ帳に戻る

前提条件:
    - BLE サーバーが起動していること
        python -m automation.ble_server
    - omlx VLM サーバーが起動していること（日本語 IME 変換に必要）
        ./scripts/start_servers.sh
    - HDMI キャプチャデバイスが接続されていること（IME モード検出に使用）

実行例:
    # 基本的な使い方
    python -m automation.dual_notepad_ime input.txt

    # venv 経由で実行
    ./venv/bin/python -m automation.dual_notepad_ime input.txt

    # 相対パス / 絶対パスどちらも指定可能
    python -m automation.dual_notepad_ime data/notes.txt
    python -m automation.dual_notepad_ime /Users/kazami/projects/ehr/toolkit-dev/hardware-agent/test/input.txt

注意事項:
    - 実行中に他のウィンドウを最前面にしないでください。
      Alt+Tab は「直前のウィンドウ」に切り替える動作を前提としています。
    - 日本語変換には omlx VLM サーバーが必要です。サーバーが起動していない場合、
      漢字変換せずにひらがな/アルファベットのまま入力される可能性があります。
    - 空行は貼り付け用メモ帳に改行のみ追加します。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from automation.ehr_input import (
    _configure_runtime,
    _wait_for_ble_connected,
    type_japanese_sentence,
)
from automation.ble_client import BLEClient


def _open_notepad(client: BLEClient) -> None:
    """Win + note + Enter でメモ帳を開く。"""
    client.switch_to_keyboard_mode()
    time.sleep(0.3)
    ok = client.press_key("win")
    print(f"press_key(win) -> {'OK' if ok else 'NG'}")
    time.sleep(1.0)
    ok = client.type_text("note")
    print(f"type_text(note) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    ok = client.press_key("enter")
    print(f"press_key(enter) -> {'OK' if ok else 'NG'}")
    time.sleep(3.0)


def _maximize_window(client: BLEClient) -> None:
    """Win+Up ショートカットで最前面ウィンドウを最大化する。"""
    client.switch_to_keyboard_mode()
    time.sleep(0.3)
    ok = client.press_key("win_up")
    print(f"press_key(win_up) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)


def _clear_notepad(client: BLEClient) -> None:
    """IME 未確定をキャンセルし、末尾から Backspace 200 回でクリアする。"""
    ok = client.press_key("escape")
    print(f"press_key(escape) -> {'OK' if ok else 'NG'}")
    time.sleep(0.3)
    ok = client.press_key("ctrl_end")
    print(f"press_key(ctrl_end) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    ok = client.type_text("\x08" * 200)
    print(f"type_text(backspace x200) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)


def _cut_all(client: BLEClient) -> None:
    """Ctrl+A → Ctrl+X で全選択・切り取り。"""
    ok = client.press_key("ctrl_a")
    print(f"press_key(ctrl_a) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    ok = client.press_key("ctrl_x")
    print(f"press_key(ctrl_x) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)


def _paste_and_enter(client: BLEClient) -> None:
    """Ctrl+V → Enter で貼り付けと改行。"""
    ok = client.press_key("ctrl_v")
    print(f"press_key(ctrl_v) -> {'OK' if ok else 'NG'}")
    time.sleep(1.0)
    ok = client.press_key("enter")
    print(f"press_key(enter) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)


def _alt_tab(client: BLEClient, label: str = "") -> None:
    ok = client.alt_tab()
    print(f"alt_tab{' (' + label + ')' if label else ''} -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)


def _prepare_notepads(client: BLEClient) -> None:
    """変換用・貼り付け用の2つのメモ帳を開き、両方をクリアして変換用メモ帳にフォーカスを戻す。"""
    print("=== 変換用メモ帳を開きます ===")
    _open_notepad(client)
    print("=== 変換用メモ帳を最大化 ===")
    _maximize_window(client)
    print("=== 貼り付け用メモ帳を開きます ===")
    _open_notepad(client)
    print("=== 貼り付け用メモ帳を最大化 ===")
    _maximize_window(client)

    # 貼り付け用が最前面なので、Alt+Tab で変換用に戻る
    print("=== 変換用メモ帳にフォーカスを移動 ===")
    _alt_tab(client, "to conversion")

    print("=== 変換用メモ帳をクリア ===")
    _clear_notepad(client)

    print("=== 貼り付け用メモ帳にフォーカスを移動 ===")
    _alt_tab(client, "to result")

    print("=== 貼り付け用メモ帳をクリア ===")
    _clear_notepad(client)

    print("=== 変換用メモ帳にフォーカスを戻す ===")
    _alt_tab(client, "to conversion")


def _process_line(
    client: BLEClient,
    line: str,
    is_first_line: bool,
    current_mode: Optional[str],
) -> Optional[str]:
    """1行を変換用メモ帳に入力し、貼り付け用メモ帳に貼り付ける。"""
    print(f"\n--- 行入力 ---")
    print(f"内容: {line!r}")

    # 変換用メモ帳に入力（初回のみクリア）
    current_mode = type_japanese_sentence(
        line,
        clear_field=is_first_line,
        _current_mode=current_mode,
    )
    time.sleep(0.5)

    # キーボードモードに切り替えて全選択・切り取り
    client.switch_to_keyboard_mode()
    time.sleep(0.5)
    _cut_all(client)

    # 貼り付け用メモ帳に切り替えて貼り付け
    _alt_tab(client, "to result")
    _paste_and_enter(client)

    # 変換用メモ帳に戻る
    _alt_tab(client, "to conversion")

    return current_mode


def _process_empty_line(client: BLEClient) -> None:
    """空行の場合は貼り付け用メモ帳に改行のみ追加する。"""
    print("\n--- 空行: 改行を追加 ---")
    _alt_tab(client, "to result")
    ok = client.press_key("enter")
    print(f"press_key(enter) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)
    _alt_tab(client, "to conversion")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="テキストファイルの内容を2つのメモ帳で IME 変換しながら入力する。",
    )
    parser.add_argument("file", help="入力するテキストファイルのパス")
    args = parser.parse_args(argv)

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"[ERROR] ファイルが見つかりません: {file_path}", file=sys.stderr)
        return 1

    text = file_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    print(f"ファイル読込完了: {file_path} ({len(lines)} 行)")

    # omlx を既定のエンジンとして設定
    _configure_runtime(omlx=True)

    client = _wait_for_ble_connected()
    _prepare_notepads(client)

    current_mode: Optional[str] = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            _process_empty_line(client)
        else:
            current_mode = _process_line(
                client,
                line,
                is_first_line=(i == 0),
                current_mode=current_mode,
            )

    print("\n=== 完了 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
