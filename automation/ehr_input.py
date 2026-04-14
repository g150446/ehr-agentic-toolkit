"""
EHR field input automation.

Captures the current HDMI screen, finds a labeled input field,
and types text into it via BLE (ESP32) mouse/keyboard control.

Uses the same AsyncBLERunner pattern as ble_test_cli.py to ensure
identical BLE event-loop behaviour on macOS CoreBluetooth.
"""

import base64
import json
import cv2
import re
import socket
import tempfile
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from automation.config import load_config
from automation.screen_analyzer import (
    capture_screen,
    load_ocr_reader,
    run_ocr,
)
from automation.gui_image_analyzer import find_textbox_right_of_label
from automation.ble_client import BLEClient
from automation.local_segmentation import segment_japanese_text_locally
from automation.mlx_vlm_segmentation import (
    MlxVlmSegmentationError,
    segment_japanese_text_with_mlx_vlm,
)


MLX_VLM_IME_URL = os.getenv(
    "MLX_VLM_IME_URL",
    os.getenv("MLX_VLM_SEGMENTATION_URL", "http://localhost:8181/v1/chat/completions"),
)
MLX_VLM_IME_MODEL = os.getenv(
    "MLX_VLM_IME_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
)
MLX_VLM_IME_TIMEOUT = float(
    os.getenv("MLX_VLM_IME_TIMEOUT", os.getenv("MLX_VLM_SEGMENTATION_TIMEOUT", "120"))
)

_TEXT_NORMALIZATION_MAP = str.maketrans({
    "（": "(",
    "）": ")",
    "％": "%",
    "：": ":",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
    "，": ",",
    "．": ".",
    "　": " ",
})
_MULTI_CHAR_REPLACEMENTS = {
    "→": "->",
    "⇒": "=>",
    "〜": "~",
}
_ASCII_SPECIAL_KEYS = {
    "\n": "enter",
    "\t": "tab",
    "[": "lbracket",
    "]": "rbracket",
    "(": "lparen",
    ")": "rparen",
    "%": "percent",
    ":": "colon",
}
_JP_PUNCTUATION = {
    "、": ",",
    "。": ".",
}


def _load_ocr_engine(config):
    return load_ocr_reader(config.ocr_languages, config.ocr_use_gpu)


def _wait_for_ble_connected(timeout: float = 70.0) -> BLEClient:
    """
    BLE サーバーが起動して BLE デバイスへ接続済みになるまで待機する。

    BLE サーバーは切断後 60 秒で自動再接続する。その間 is_server_running() が
    False を返すため、タイムアウトまでポーリングして待機する。

    Args:
        timeout: 最大待機秒数（デフォルト 70 秒 = 60 秒再接続サイクル + 余裕）

    Returns:
        接続済み BLEClient インスタンス

    Raises:
        RuntimeError: タイムアウトまでに接続が確立されなかった場合
    """
    client = BLEClient()
    if client.is_server_running():
        return client

    print(f"BLE 未接続。最大 {timeout:.0f} 秒待機します（サーバー再接続中の可能性があります）...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2.0)
        if client.is_server_running():
            print("BLE 接続を確認しました。")
            return client
        remaining = deadline - time.monotonic()
        print(f"  BLE 接続待機中... (残り {remaining:.0f} 秒)")

    raise RuntimeError(
        "BLE サーバーが起動していないか、BLE デバイスへの接続がタイムアウトしました。\n"
        "  python -m automation.ble_server  を先に別ターミナルで実行してください"
    )


def _request_ocr_results(frame, config) -> list[tuple]:
    return run_ocr(_load_ocr_engine(config), frame)


def _resolve_text_argument(raw: str) -> str:
    """Resolve a CLI text argument, loading file contents when given a path."""
    candidate_path = Path(raw)
    if not candidate_path.is_file():
        return raw

    try:
        text = candidate_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"テキストファイルを UTF-8 として読めませんでした: {candidate_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"テキストファイルを読めませんでした: {candidate_path}") from exc

    resolved = text.rstrip("\r\n")
    print(f"テキストファイル読込: {candidate_path}")
    return resolved


def _input_resolved_text(text: str) -> None:
    """Route already-resolved text through the existing input pipeline."""
    if _is_japanese(text):
        if len(text) <= 4 and not any(ch in text for ch in "をにはがでも") and not _is_ascii_only(text):
            romaji = _kanji_to_romaji(text)
            print(f"IME変換: {romaji} → {text}")
            type_kanji_via_ime(romaji, text)
        else:
            type_japanese_sentence(text)
        return

    print(f"英語入力: {text!r}")
    _type_english_text(text)


def _normalize_text_for_typing(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").translate(_TEXT_NORMALIZATION_MAP)
    for src, dest in _MULTI_CHAR_REPLACEMENTS.items():
        normalized = normalized.replace(src, dest)
    return normalized


def _classify_input_char(ch: str) -> str:
    if ch == "\n":
        return "newline"
    if ch in _JP_PUNCTUATION:
        return "jp_punct"
    if ord(ch) < 128:
        return "ascii"
    if _is_japanese(ch):
        return "japanese"
    return "ascii"


def _tokenize_text_for_input(text: str) -> list[dict[str, str]]:
    normalized = _normalize_text_for_typing(text)
    tokens: list[dict[str, str]] = []
    buffer: list[str] = []
    current_kind: Optional[str] = None

    def flush_buffer() -> None:
        nonlocal buffer, current_kind
        if buffer:
            tokens.append({"kind": current_kind or "ascii", "text": "".join(buffer)})
            buffer = []
            current_kind = None

    for ch in normalized:
        kind = _classify_input_char(ch)
        if kind in {"newline", "jp_punct"}:
            flush_buffer()
            tokens.append({"kind": kind, "text": ch})
            continue
        if current_kind != kind:
            flush_buffer()
            current_kind = kind
        buffer.append(ch)

    flush_buffer()
    return tokens


def _segment_japanese_with_default_vlm(text: str) -> list[dict[str, str]]:
    try:
        raw_content, segments = segment_japanese_text_with_mlx_vlm(text)
        if "".join(segment["text"] for segment in segments) != text:
            raise MlxVlmSegmentationError(
                f"Qwen分割結果が元テキストを保持していません: source={text!r} segments={segments!r}"
            )
        if _should_fallback_to_local_segmentation(segments):
            raise MlxVlmSegmentationError(
                f"Qwen分割結果が IME 候補を不安定化させる粒度です: source={text!r} segments={segments!r}"
            )
        normalized_segments = []
        for segment in segments:
            segment_text = segment["text"]
            romaji = _kanji_to_romaji(segment_text)
            normalized_segments.append({"text": segment_text, "romaji": romaji})
        print(f"Qwen分割結果: {raw_content}")
        print(f"Qwen分割補正後: {normalized_segments}")
        return normalized_segments
    except MlxVlmSegmentationError as exc:
        print(f"Qwen分割失敗 → ローカル分割へフォールバック: {exc}")
        summary, segments = segment_japanese_text_locally(text)
        print(f"ローカル分割サマリ: {summary}")
        return segments


def _is_single_kanji(text: str) -> bool:
    return len(text) == 1 and _has_kanji(text)


def _is_hiragana_only(text: str) -> bool:
    return bool(text) and all("\u3040" <= ch <= "\u309f" for ch in text)


def _should_fallback_to_local_segmentation(segments: list[dict[str, str]]) -> bool:
    consecutive_single_kanji = 0
    for index, segment in enumerate(segments):
        text = segment["text"]
        if _is_single_kanji(text):
            consecutive_single_kanji += 1
            if consecutive_single_kanji >= 2:
                return True
            if index + 1 < len(segments) and _is_hiragana_only(segments[index + 1]["text"]):
                return True
        else:
            consecutive_single_kanji = 0
    return False


def _segment_text_for_input(text: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for segment in _iter_segments_for_input(text):
        segments.append(segment)
    return segments


def _iter_segments_for_input(text: str):
    for token in _tokenize_text_for_input(text):
        kind = token["kind"]
        value = token["text"]
        if kind == "japanese":
            for segment in _segment_japanese_with_default_vlm(value):
                yield segment
        elif kind == "jp_punct":
            yield {"text": value, "romaji": _JP_PUNCTUATION[value]}
        elif kind == "newline":
            yield {"text": "\n", "romaji": "<enter>"}
        else:
            yield {"text": value, "romaji": value}


def _type_ascii_text_precisely(client: BLEClient, text: str) -> None:
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        chunk = "".join(buffer)
        ok = client.type_text(chunk)
        print(f"type:{chunk} -> {'OK' if ok else 'NG'}")
        buffer.clear()

    for ch in text:
        key_name = _ASCII_SPECIAL_KEYS.get(ch)
        if key_name is None:
            buffer.append(ch)
            continue
        flush_buffer()
        ok = client.press_key(key_name)
        print(f"key:{key_name} -> {'OK' if ok else 'NG'}")
    flush_buffer()


def input_text_to_field(
    input_text: str = "tesuto",
    label: str = "フリガナ"
) -> None:
    """
    Find a labeled input field on the HDMI screen and type text into it.

    Args:
        input_text: Text to type into the field.
        label: Label text to search for (finds textbox to its right).
    """
    config = load_config(skip_password=True)
    # Use full-image OCR so label text like "フリガナ" is found even when YOLO
    # doesn't detect its surrounding region as a UI element.
    config.detection_mode = 'ocr'

    # 1. Capture frame from HDMI device
    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    # 2. Save to temp file for analysis
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)
    print(f"スクリーンショット保存: {tmp_path}")

    try:
        # 3. Find textbox to the right of the label
        print(f"「{label}」ラベルの右にあるテキストボックスを検索中...")
        # y_tolerance=10: 「フリガナ」行と「生年月日」行の間隔が約24pxのため、
        # デフォルトの30pxでは下の行の「年」を誤検出する。10pxに絞ることで
        # テキストなしの場合はエッジ検出にフォールバックし正しいボックスを検出する。
        coords = find_textbox_right_of_label(tmp_path, label, config, y_tolerance=10)
        if coords is None:
            raise RuntimeError(f"「{label}」ラベルの右にテキストボックスが見つかりませんでした")

        x, y = coords
        print(f"テキストボックス座標: ({x}, {y})")

    finally:
        os.unlink(tmp_path)

    # 4. BLE operations — delegate to ble_server.py (must be running beforehand)
    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(x, y)
    print(f"moveto ({x}, {y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click -> {'OK' if ok else 'NG'}")

    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")

    ok = client.type_text(input_text)
    print(f"type:{input_text} -> {'OK' if ok else 'NG'}")

    ok = client.press_key("enter")
    print(f"key:enter -> {'OK' if ok else 'NG'}")

    print("完了")


def open_test_patient_chart() -> None:
    """
    テスト患者のカルテを開く。

    以下の手順を自動実行する:
    0. 「患者検索」タブをOCRで検出してクリック → 患者検索画面を前面に出す
    1. フリガナ欄に「tesuto」と入力して Enter → 患者一覧を表示
    2. 0.5 秒待ってから Enter → 先頭患者を選択してカルテを開く
    3. 2 秒待ってから Enter → 表示直後のダイアログを閉じる

    ble_server.py が事前に起動済みであること。
    """
    # Step 0: 「患者検索」タブをクリックして患者検索画面を前面に出す
    config = load_config(skip_password=True)
    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    results = _request_ocr_results(frame, config)

    tab_x: Optional[int] = None
    tab_y: Optional[int] = None
    for bbox, text, conf in results:
        if "患者検索" in text:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            tab_x = int(sum(xs) / len(xs))
            tab_y = int(sum(ys) / len(ys))
            print(f"「患者検索」検出: {text!r} at ({tab_x}, {tab_y}), conf={conf:.2f}")
            break

    if tab_x is None or tab_y is None:
        print("「患者検索」タブが検出できませんでした（既に選択済みと判断）。スキップします。")
    else:
        client = _wait_for_ble_connected()

        ok = client.switch_to_mouse_mode()
        print(f"mode:mouse -> {'OK' if ok else 'NG'}")
        ok = client.move_mouse_to_position(tab_x, tab_y)
        print(f"moveto ({tab_x}, {tab_y}) -> {'OK' if ok else 'NG'}")
        ok = client.click()
        print(f"click -> {'OK' if ok else 'NG'}")

        print("「患者検索」タブをクリックしました。タブ切替を待機中 (0.5秒)...")
        time.sleep(0.5)

    # Step 1: フリガナ欄に「tesuto」と入力して Enter → 患者一覧を表示させる
    input_text_to_field(input_text="tesuto", label="フリガナ")

    client = _wait_for_ble_connected()
    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")

    # Step 2: 患者一覧が表示されるまで待ってから Enter で先頭患者を選択
    print("患者一覧の表示を待機中 (0.5秒)...")
    time.sleep(0.5)
    ok = client.press_key("enter")
    print(f"key:enter (select patient) -> {'OK' if ok else 'NG'}")

    # Step 3: ダイアログを閉じるため 2 秒待って Enter
    print("ダイアログの表示を待機中 (2秒)...")
    time.sleep(2.0)
    ok = client.press_key("enter")
    print(f"key:enter (dialog close) -> {'OK' if ok else 'NG'}")

    # カルテが完全に開くまで待機
    print("カルテ表示を待機中 (2秒)...")
    time.sleep(2.0)

    print("完了")


def close_record() -> None:
    """
    画面右上の「取り消し[F9]」ボタンをクリックしてカルテを閉じる。

    OCR で「取り消し」テキストを検出し、その座標にマウスを移動してクリックする。
    ble_server.py が事前に起動済みであること。
    """
    config = load_config(skip_password=True)

    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    results = _request_ocr_results(frame, config)

    # 「取り消し」テキストを含む結果を検索
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    for bbox, text, conf in results:
        if "取消" in text:
            # bbox は [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] 形式
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            target_x = int(sum(xs) / len(xs))
            target_y = int(sum(ys) / len(ys))
            print(f"「取消」検出: {text!r} at ({target_x}, {target_y}), conf={conf:.2f}")
            break

    if target_x is None or target_y is None:
        raise RuntimeError("「取消」ボタンが画面上に見つかりませんでした")

    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(target_x, target_y)
    print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click -> {'OK' if ok else 'NG'}")

    print("完了")


def click_history(date_str: str) -> None:
    """
    過去カルテ列から指定日付のエントリを検出してクリックする。

    automation.mlx_vlm_history と同じ OCR→候補抽出→VLM 判定の経路を使用する。

    Args:
        date_str: 日付文字列 (yyyymmdd 形式, 例: "20190502")
    """
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"日付は yyyymmdd 形式で指定してください: {date_str!r}")

    config = load_config(skip_password=True)
    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    try:
        from automation.mlx_vlm_history import MlxVlmHistoryError, find_history_date_in_image

        ocr_languages = getattr(config, "ocr_languages", ["ja", "en"])
        coords = find_history_date_in_image(
            date_str,
            frame,
            languages=ocr_languages,
        )
    except MlxVlmHistoryError as exc:
        raise RuntimeError(f"過去カルテ日付の検出に失敗しました: {exc}") from exc

    if coords is None:
        print(f"日付 {date_str} のエントリが画面上に見つかりませんでした")
        return

    target_x, target_y = coords

    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(target_x, target_y)
    print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click -> {'OK' if ok else 'NG'}")

    print("完了")


EDIT_BUTTON_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "match_templates",
    "edit_button.jpg",
)


def _find_button_with_template(
    frame: np.ndarray,
    template_path: str,
    threshold: float = 0.7,
) -> Optional[tuple[int, int]]:
    """OpenCV テンプレートマッチングでボタン中心座標を返す。"""
    tmpl = cv2.imread(template_path)
    if tmpl is None:
        raise RuntimeError(f"テンプレート画像を読み込めません: {template_path}")
    result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"テンプレートマッチング スコア: {max_val:.3f} ({template_path})")
    if max_val < threshold:
        return None
    th, tw = tmpl.shape[:2]
    return (max_loc[0] + tw // 2, max_loc[1] + th // 2)


def edit_history(date_str: str) -> None:
    """過去カルテの日付エントリをクリックし、修正ボタンをクリックする。

    Args:
        date_str: 日付文字列 (yyyymmdd 形式, 例: "20190502")
    """
    click_history(date_str)
    time.sleep(1.0)

    config = load_config(skip_password=True)
    print("修正ボタンを探しています...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    coords = _find_button_with_template(frame, EDIT_BUTTON_TEMPLATE)
    if coords is None:
        raise RuntimeError("修正ボタンが画面上に見つかりませんでした")

    target_x, target_y = coords
    print(f"修正ボタン: ({target_x}, {target_y})")

    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(target_x, target_y)
    print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click -> {'OK' if ok else 'NG'}")

    print("完了")


def _find_ime_candidate_region(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    画面から IME 変換候補の反転表示ブロック（黒背景＋白文字）を検出して切り出す。

    Windows IME は Space キー押下後、選択中の変換候補を黒背景・白文字で反転表示する。
    この特徴を利用して変換候補領域のみを切り出し、OCR の誤検知を防ぐ。

    Returns:
        検出した候補ブロックの画像（色反転済み）。見つからない場合は None。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 暗い領域（黒背景）を検出: ピクセル値 30 以下を白に
    _, dark_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # ノイズ除去: 小さな孤立点を消す
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # IME 候補ウィンドウのおよそのサイズ範囲でフィルタ
        if w < 20 or w > 800 or h < 12 or h > 100:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    roi = frame[y:y + h, x:x + w]
    # 白文字を黒文字に反転して OCR しやすくする
    return cv2.bitwise_not(roi)


def _find_changed_region(base: np.ndarray, current: np.ndarray) -> Optional[np.ndarray]:
    """
    2 フレームの差分から変化した矩形領域を切り出す（フォールバック用）。

    IME 候補ブロックが検出できない場合、入力前後の差分で変化領域を特定する。

    Returns:
        変化した領域の画像。見つからない場合は None。
    """
    diff = cv2.absdiff(base, current)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 最大の変化領域を返す
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    # 最小サイズフィルタ
    if w < 10 or h < 10:
        return None
    return current[y:y + h, x:x + w]


def _encode_frame_as_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("IME候補画像の PNG エンコードに失敗しました")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _extract_vlm_text_content(result: dict) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"mlx_vlm応答に content がありません: {result!r}") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        merged = "".join(parts).strip()
        if merged:
            return merged
    raise RuntimeError(f"mlx_vlm応答の content 形式が不正です: {content!r}")


def _parse_ime_candidate_response(content: str) -> Optional[str]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("candidate")
            if candidate is None:
                return None
            if isinstance(candidate, str):
                return candidate.strip() or None

    quoted = re.search(r'"candidate"\s*:\s*"([^"]+)"', content)
    if quoted:
        return quoted.group(1).strip() or None
    return content.strip() or None


def _read_ime_candidate_with_vlm(frame: np.ndarray, target_kanji: str) -> Optional[str]:
    prompt = (
        "この画像は Windows IME の変換候補の一部です。"
        "現在選択されている候補1件だけを正確に読み取ってください。"
        "読めない場合は null を返してください。"
        "JSONのみで回答してください。"
        ' 形式: {"candidate": "候補文字列"} または {"candidate": null}'
        f" 目標文字列: {target_kanji}"
    )
    payload = {
        "model": MLX_VLM_IME_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _encode_frame_as_data_url(frame)}},
            ],
        }],
        "stream": False,
    }
    req = urllib.request.Request(
        MLX_VLM_IME_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=MLX_VLM_IME_TIMEOUT) as resp:
            result = json.loads(resp.read())
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"IME候補確認の mlx_vlm リクエストが {MLX_VLM_IME_TIMEOUT:g} 秒でタイムアウトしました"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"IME候補確認の mlx_vlm 接続に失敗しました: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("IME候補確認の mlx_vlm 応答 JSON を解析できませんでした") from exc

    content = _extract_vlm_text_content(result)
    return _parse_ime_candidate_response(content)


def type_kanji_via_ime(
    romaji: str,
    target_kanji: str,
    max_attempts: int = 5,
    wait_sec: float = 0.5,
) -> None:
    """
    ローマ字を入力し、IME 変換で目的の漢字を確定させる。

    手順:
    1. ローマ字を type_text で入力する
    2. Space キーで変換候補を呼び出す
    3. HDMIキャプチャ → IME反転ブロック検出 → OCR で候補テキストを確認
    4. target_kanji と一致すれば Enter で確定
    5. 一致しなければ Space でサイクルして繰り返す（最大 max_attempts 回）

    Args:
        romaji: 入力するローマ字（例: "haien"）
        target_kanji: 確定したい漢字（例: "肺炎"）
        max_attempts: Space サイクルの最大試行回数
        wait_sec: Space 押下後に候補が表示されるまでの待機秒数

    Raises:
        RuntimeError: BLE サーバー未起動、またはキャプチャ失敗の場合
        ValueError: max_attempts 回試行しても target_kanji が見つからない場合
    """
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    # ベースフレームをキャプチャ（差分検出のフォールバック用）
    print("ベースフレームをキャプチャ中...")
    base_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if base_frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    # ローマ字入力
    print(f"ローマ字入力: {romaji}")
    ok = client.type_text(romaji)
    print(f"type:{romaji} -> {'OK' if ok else 'NG'}")
    time.sleep(0.3)  # IMEがローマ字処理するまで待機

    print("[Space] 変換候補を表示...")
    ok = client.press_key("space")
    print(f"key:space -> {'OK' if ok else 'NG'}")
    time.sleep(wait_sec)

    for attempt in range(1, max_attempts + 1):
        # フレームキャプチャ
        frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame is None:
            raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

        # IME 反転ブロック（黒背景白文字）を検出して OCR
        roi = _find_ime_candidate_region(frame)
        source = "IME反転ブロック"
        ocr_results = []
        combined = ""
        vlm_candidate: Optional[str] = None

        if roi is not None:
            try:
                vlm_candidate = _read_ime_candidate_with_vlm(roi, target_kanji)
                print(f"  [試行{attempt}] {source} Qwen読取: {vlm_candidate!r}")
            except RuntimeError as exc:
                print(f"  [試行{attempt}] {source} Qwen読取失敗: {exc}")
            ocr_results = _request_ocr_results(roi, config)
            texts = [text for (_, text, _) in ocr_results]
            combined = "".join(texts)
            # 日本語文字（漢字・ひらがな・カタカナ）が含まれない場合は誤検出とみなす
            if vlm_candidate is None and not any("\u3040" <= ch <= "\u9fff" for ch in combined):
                print(f"  [試行{attempt}] IME反転ブロック OCR結果に日本語なし ({combined!r}) → フレーム差分でフォールバック")
                roi = None

        if roi is None:
            roi = _find_changed_region(base_frame, frame)
            source = "差分領域"
            if roi is not None:
                try:
                    vlm_candidate = _read_ime_candidate_with_vlm(roi, target_kanji)
                    print(f"  [試行{attempt}] {source} Qwen読取: {vlm_candidate!r}")
                except RuntimeError as exc:
                    print(f"  [試行{attempt}] {source} Qwen読取失敗: {exc}")
                ocr_results = _request_ocr_results(roi, config)
                texts = [text for (_, text, _) in ocr_results]
                combined = "".join(texts)

        if roi is None:
            print(f"  [試行{attempt}] 変化領域も未検出。少し待って再試行...")
            time.sleep(0.3)
            continue

        texts = [text for (_, text, _) in ocr_results]
        print(f"  [試行{attempt}] {source} OCR結果: {texts!r} → 結合: {combined!r}")

        if vlm_candidate == target_kanji or _ime_candidate_matches(target_kanji, combined):
            print(f"  「{target_kanji}」を確認 → Enter で確定")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            print("完了")
            return

        if attempt < max_attempts:
            print(f"  「{target_kanji}」は未確認。Space で次候補へ進みます...")
            ok = client.press_key("space")
            print(f"key:space -> {'OK' if ok else 'NG'}")
            time.sleep(wait_sec)

    raise ValueError(f"IME候補を {max_attempts} 回確認しましたが「{target_kanji}」を確定できませんでした")


def _ime_candidate_matches(target: str, combined: str) -> bool:
    """IME候補OCRにターゲット文字列が完全一致で含まれているか確認する。"""
    return target in combined


def _is_ascii_only(text: str) -> bool:
    """文字列が ASCII 文字のみで構成されているか判定する。"""
    return all(ord(ch) < 128 for ch in text)


def detect_ime_mode(frame: np.ndarray, config=None) -> Optional[str]:
    """
    スクリーン右下の IME フローティングウィンドウを OCR して現在の入力モードを判定する。

    Windows IME は画面右下（タスクバー付近）に現在のモードを示す小さなインジケーターを
    表示する。ひらがなモードでは「あ」、英数字モードでは「A」が表示される。

    Args:
        frame: HDMI キャプチャフレーム（BGR numpy 配列）
        config: AppConfig。None の場合はデフォルト設定を使用。

    Returns:
        'japanese': ひらがな入力モード（「あ」が検出された）
        'english':  英数字入力モード（「A」が検出され日本語文字なし）
        None:       判定不能
    """
    h, w = frame.shape[:2]
    # IME インジケーターは画面下部に存在する（タスクバー高さ 80px、全幅でスキャン）
    roi = frame[max(0, h - 80):h, :]

    results = _request_ocr_results(roi, config)
    texts = "".join(text for (_, text, _) in results)
    print(f"  [IME検出] OCR結果: {texts!r}")

    # ひらがな「あ」が検出されればひらがなモード（最優先）
    if "あ" in texts:
        return "japanese"
    # 英数字モードのインジケーター: 半角「A」または全角「Ａ」（U+FF21）
    # 否定フィルターは使わない — 時計・日付の OCR ノイズに CJK 文字が混入するため
    if "A" in texts or "\uff21" in texts:
        return "english"
    return None


def toggle_ime(client: "BLEClient") -> None:
    """半角/全角キーを送って IME モードをトグルする。"""
    print("  [IME切替] 半角/全角 を送信")
    client.press_key("zenkaku")
    time.sleep(0.3)  # IME 切替の反映を待つ


def ensure_ime_mode(
    target_mode: str,
    client: "BLEClient",
    current_mode: Optional[str],
) -> Optional[str]:
    """
    current_mode が target_mode と異なる場合に半角/全角でトグルし、新しいモードを返す。

    画面キャプチャは行わない。呼び出し元が開始時に1回だけ detect_ime_mode() で
    モードを取得し、以降はこの関数の戻り値でトラッキングする設計。

    Args:
        target_mode: 目標モード ('japanese' または 'english')
        client: BLEClient インスタンス
        current_mode: 現在の IME モード。None の場合は判定不能として扱う。

    Returns:
        切替後の（または変更なしの）IME モード文字列。
        current_mode が None の場合は None を返す（トグルしない）。
    """
    if current_mode is None:
        print(f"  [IME切替] モード不明 → 切替をスキップ（{target_mode} を期待）")
        return None
    if current_mode == target_mode:
        print(f"  [IME切替] {current_mode} → 変更不要")
        return current_mode
    toggle_ime(client)
    print(f"  [IME切替] {current_mode} → {target_mode}")
    return target_mode


def _kanji_to_romaji(text: str) -> str:
    """漢字・かな文字列をヘボン式ローマ字に変換する。"""
    import pykakasi
    kks = pykakasi.kakasi()
    return "".join(item["hepburn"] for item in kks.convert(text))


def _is_japanese(text: str) -> bool:
    """文字列に日本語文字（漢字・ひらがな・カタカナ）が含まれるか判定する。"""
    return any(
        "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"
        for ch in text
    )


def _has_kanji(text: str) -> bool:
    """文字列に漢字（CJK統合漢字）が含まれるか判定する。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _segment_japanese_locally(text: str) -> list:
    """
    sudachipy + pykakasi を使って日本語テキストを IME 変換単位（文節）に分割する。

    Returns:
        [{"text": "肺炎", "romaji": "haien"}, {"text": "に", "romaji": "ni"}, ...]
    """
    summary, segments = segment_japanese_text_locally(text)
    print(f"分割サマリ: {summary}")
    return segments


def type_japanese_sentence(text: str) -> None:
    """
    日本語・英語混在文を文節単位で入力する。

    sudachipy + pykakasi で文節分割し、各文節の種類に応じて処理する:
    - ASCII のみ（英単語・数字・記号）: 英数字モードで直接入力
    - 漢字を含む: ひらがなモードで IME 変換（type_kanji_via_ime）
    - ひらがな・カタカナのみ: ひらがなモードでローマ字 + Enter
    - 句読点（、。）: ひらがなモードで IME 変換キー送信

    IME モードの切替には半角/全角キー（key:zenkaku）を使用する。
    各文節の前にスクリーンキャプチャで現在のモードを確認し、必要な場合のみ切替える。

    Args:
        text: 入力するテキスト（日本語・英語混在可）
    """
    print(f"文節分割中 (Qwen優先): {text!r}")
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    # 開始時に1回だけ IME モードを検出し、以降は内部変数でトラッキングする
    print("現在の IME モードを検出中...")
    init_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    current_mode: Optional[str] = None
    if init_frame is not None:
        current_mode = detect_ime_mode(init_frame, config)
    print(f"初期 IME モード: {current_mode!r}")

    for seg in _iter_segments_for_input(text):
        seg_text = seg["text"]
        seg_romaji = seg["romaji"]
        print(f"\n--- 文節: {seg_text!r} ({seg_romaji}) ---")

        if seg_text == "\n":
            current_mode = ensure_ime_mode("english", client, current_mode)
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")

        elif seg_text in ("、", "。"):
            # 句読点: ひらがなモードで IME が自動変換するキー（,/.）を送る。
            # 「。」はIMEの変換バッファに残るため Enter で確定が必要。
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  句読点入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            if seg_text == "。":
                ok = client.press_key("enter")
                print(f"key:enter -> {'OK' if ok else 'NG'}")

        elif _is_ascii_only(seg_text):
            # ASCII のみ（英単語・数字・記号）: 英数字モードで直接入力（IME 変換不要）
            current_mode = ensure_ime_mode("english", client, current_mode)
            print(f"  英数字直接入力: {seg_text!r}")
            _type_ascii_text_precisely(client, seg_text)

        elif _has_kanji(seg_text):
            # 漢字を含む文節: ひらがなモードで IME 変換候補を確認してから確定
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            type_kanji_via_ime(seg_romaji, seg_text)

        else:
            # ひらがな・カタカナのみ: ひらがなモードでローマ字を直接入力して Enter で確定
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  ひらがな直接入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")

    print("\n文章入力完了")


def _type_english_text(text: str) -> None:
    """
    英語テキストを英数字モードで直接入力する。

    IME を英数字モードに切替えてからテキストを送信する。
    Enter は送らない（呼び出し元がフィールド確定を制御する）。

    Args:
        text: 入力する英数字文字列
    """
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    current_mode = detect_ime_mode(frame, config)
    ensure_ime_mode("english", client, current_mode)

    print(f"英語入力: {text!r}")
    _type_ascii_text_precisely(client, _normalize_text_for_typing(text))
    print("完了")


def _run_cli(args: list[str]) -> int:
    """CLI entry point for manual EHR input automation."""
    if not args:
        # 引数なし: デフォルト動作（後方互換）
        open_test_patient_chart()
        return 0

    if len(args) == 1 and args[0] == "open test":
        # "open test" のみ → テスト患者カルテを開く
        open_test_patient_chart()
        return 0

    if len(args) == 1 and args[0] == "close record":
        # "close record" → 「取り消し[F9]」ボタンをクリックしてカルテを閉じる
        close_record()
        return 0

    if len(args) == 1 and args[0].startswith("click history "):
        # "click history yyyymmdd" → 過去カルテ列の指定日付をクリック
        date_str = args[0][len("click history "):].strip()
        click_history(date_str)
        return 0

    if len(args) >= 2 and args[0] == "click history":
        # "click history" "yyyymmdd" (2引数) → 過去カルテ列の指定日付をクリック
        click_history(args[1])
        return 0

    if len(args) == 1 and args[0].startswith("edit history "):
        # "edit history yyyymmdd" → 過去カルテ日付クリック後に修正ボタンをクリック
        date_str = args[0][len("edit history "):].strip()
        edit_history(date_str)
        return 0

    if len(args) >= 2 and args[0] == "edit history":
        # "edit history" "yyyymmdd" (2引数) → 過去カルテ日付クリック後に修正ボタンをクリック
        edit_history(args[1])
        return 0

    if len(args) == 1:
        text = _resolve_text_argument(args[0])
        _input_resolved_text(text)
        return 0

    if len(args) >= 2 and args[0] == "open test":
        # 第一引数が "open test"、第二引数がテキスト → カルテ開いてから入力
        text = _resolve_text_argument(args[1])
        print(f"テスト患者カルテを開いてから入力: {text!r}")
        open_test_patient_chart()
        _input_resolved_text(text)
        return 0

    print("使い方:")
    print('  python -m automation.ehr_input                         # テスト患者カルテを開く')
    print('  python -m automation.ehr_input "open test"             # テスト患者カルテを開く')
    print('  python -m automation.ehr_input "close record"          # 取り消し[F9]ボタンをクリックしてカルテを閉じる')
    print('  python -m automation.ehr_input "click history 20190502"  # 過去カルテの指定日付をクリック')
    print('  python -m automation.ehr_input "edit history 20190502"   # 過去カルテ日付クリック後に修正ボタンをクリック')
    print('  python -m automation.ehr_input 肺炎                    # IME変換のみ')
    print('  python -m automation.ehr_input note.txt                # テキストファイル内容を入力')
    print('  python -m automation.ehr_input "COVID-19の検査"        # 日英混在入力')
    print('  python -m automation.ehr_input tesuto                  # 英語直接入力')
    print('  python -m automation.ehr_input "open test" 肺炎        # カルテを開いてからIME変換')
    print('  python -m automation.ehr_input "open test" note.txt    # カルテを開いてからファイル内容を入力')
    print('  python -m automation.ehr_input "open test" "MRI所見"   # カルテを開いてから混在入力')
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    return _run_cli(args)


if __name__ == '__main__':
    import sys

    sys.exit(main())
