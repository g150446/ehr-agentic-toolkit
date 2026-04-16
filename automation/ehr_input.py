"""
EHR field input automation.

Captures the current HDMI screen, finds a labeled input field,
and types text into it via BLE (ESP32) mouse/keyboard control.

Uses the same AsyncBLERunner pattern as ble_test_cli.py to ensure
identical BLE event-loop behaviour on macOS CoreBluetooth.
"""

import json
import cv2
import re
import tempfile
import os
import time
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
from automation.mlx_vlm_ime import (
    MlxVlmImeError,
    detect_ime_mode_from_typed_a,
    read_inline_candidate_context,
    read_inline_candidate_roi,
    read_popup_candidates,
    read_popup_candidates_numbered,
    suggest_ime_helper_word,
)



_TEXT_NORMALIZATION_MAP = str.maketrans({
    # 全角括弧・記号 → 半角 ASCII（形状がほぼ同一のもの）
    "（": "(",
    "）": ")",
    "％": "%",
    "：": ":",
    "；": ";",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
    "，": ",",
    "．": ".",
    "　": " ",
    # 全角 ASCII 記号 (U+FF01–U+FF5E) → 半角
    "！": "!",
    "＂": '"',
    "＃": "#",
    "＄": "$",
    "＆": "&",
    "＊": "*",
    "＋": "+",
    "－": "-",
    "／": "/",
    "＜": "<",
    "＝": "=",
    "＞": ">",
    "？": "?",
    "＠": "@",
    "＼": "\\",
    "＾": "^",
    "＿": "_",
    "｀": "`",
    "｜": "|",
    "～": "~",
    # スマートクォート → ストレートクォート
    "\u2018": "'",   # '
    "\u2019": "'",   # '
    "\u201c": '"',   # "
    "\u201d": '"',   # "
    # ダッシュ類 → ハイフン
    "\u2010": "-",   # ‐ (hyphen)
    "\u2013": "-",   # – (en dash)
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
    "+": "plus",
}
_JP_PUNCTUATION = {
    "、": ",",
    "。": ".",
}

# 特殊日本語記号 → IME で入力するときの読み（ローマ字）
_JP_SYMBOL_IME_READINGS: dict[str, str] = {
    "※": "kome",
    "〒": "yuubin",
    "〇": "maru",
    "〆": "shime",
    "♪": "onpu",
    "★": "hoshi",
    "☆": "hoshi",
}

# BLE キーボード (ASCII のみ) では直接送信できない非 ASCII 文字の代替マッピング。
# /μL → /uL のように医療文書で慣例的に使われる ASCII 表記で代替する。
_CHAR_ASCII_FALLBACK: dict[str, str] = {
    "μ": "u",   # マイクロ記号 → ASCII u (/μL → /uL)
    "°": "",    # 度記号 → 省略（℃ は直接入力されるため通常不要）
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


def _input_resolved_text(
    text: str, windows_version: str = "windows7", clear_field: bool = False
) -> None:
    """Route already-resolved text through the existing input pipeline."""
    if _is_japanese(text):
        if len(text) <= 4 and not any(ch in text for ch in "をにはがでも") and not _is_ascii_only(text):
            romaji = _kanji_to_romaji(text)
            print(f"IME変換: {romaji} → {text}")
            type_kanji_via_ime(romaji, text, windows_version=windows_version, clear_field=clear_field)
        else:
            type_japanese_sentence(text, windows_version=windows_version, clear_field=clear_field)
        return

    print(f"英語入力: {text!r}")
    _type_english_text(text, windows_version=windows_version, clear_field=clear_field)


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


# セグメント単位の分割オーバーライド: Qwen/ローカル分割で1トークンにまとまる語を
# 複数の IME 入力単位に強制分割する。IME の候補ポップアップが単体で出にくい語に対して使う。
_SEGMENT_OVERRIDES: dict[str, list[dict[str, str]]] = {
    "歳頃": [{"text": "歳", "romaji": "sai"}, {"text": "頃", "romaji": "koro"}],
    "吸入剤": [{"text": "吸入", "romaji": "kyuunyuu"}, {"text": "剤", "romaji": "zai"}],
    "過膨張": [{"text": "過膨張", "romaji": "kabouchou"}],
    # 肺野 (はいや) は標準 IME 辞書にない医療用語 → 肺(hai)+野(ya) に分割
    "肺野": [{"text": "肺", "romaji": "hai"}, {"text": "野", "romaji": "ya"}],
    # 認めるが → 認める(mitomeru) + が(ga) に分割
    "認めるが": [{"text": "認める", "romaji": "mitomeru"}, {"text": "が", "romaji": "ga"}],
    # 動脈血ガス → 動脈血(doumyakuchi) + ガス(gasu) に分割
    "動脈血ガス": [{"text": "動脈血", "romaji": "doumyakuchi"}, {"text": "ガス", "romaji": "gasu"}],
    # 動脈血 単体でもオーバーライド
    "動脈血": [{"text": "動脈血", "romaji": "doumyakuchi"}],
    # ・ (中黒/ナカテン) : JIS キーボードでは '/' キーで入力する。
    # カタカナ変換パス (F7) で処理するため romaji は '/' とする。
    "・": [{"text": "・", "romaji": "/"}],
    # ソル・コーテフ (Solu-Cortef) : カタカナ医薬品名、中黒を含む。
    # '/' = ・、'-' = ー (長音符) として romaji を組む。
    "ソル・コーテフ": [{"text": "ソル・コーテフ", "romaji": "soru/ko-tefu"}],
}


def _expand_segment_overrides(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    """_SEGMENT_OVERRIDES に登録されたトークンを複数サブセグメントに展開する。"""
    result: list[dict[str, str]] = []
    for seg in segments:
        if seg["text"] in _SEGMENT_OVERRIDES:
            result.extend(_SEGMENT_OVERRIDES[seg["text"]])
        else:
            result.append(seg)
    return result


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
        normalized_segments = _expand_segment_overrides(normalized_segments)
        print(f"Qwen分割結果: {raw_content}")
        print(f"Qwen分割補正後: {normalized_segments}")
        return normalized_segments
    except MlxVlmSegmentationError as exc:
        print(f"Qwen分割失敗 → ローカル分割へフォールバック: {exc}")
        summary, segments = segment_japanese_text_locally(text)
        print(f"ローカル分割サマリ: {summary}")
        return _expand_segment_overrides(segments)


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


def _presplit_japanese_for_overrides(text: str) -> list[str]:
    """テキストを _SEGMENT_OVERRIDES のキーで事前分割する。

    例: "動脈血ガス分析" + _SEGMENT_OVERRIDES["動脈血ガス"] →
        ["動脈血ガス", "分析"]
    """
    keys = sorted(_SEGMENT_OVERRIDES.keys(), key=len, reverse=True)
    chunks: list[str] = [text]
    for key in keys:
        new_chunks: list[str] = []
        for chunk in chunks:
            if chunk in _SEGMENT_OVERRIDES:
                # already an override key; keep as-is
                new_chunks.append(chunk)
                continue
            if key in chunk:
                parts = chunk.split(key)
                for i, part in enumerate(parts):
                    if part:
                        new_chunks.append(part)
                    if i < len(parts) - 1:
                        new_chunks.append(key)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
    return [c for c in chunks if c]


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
            for chunk in _presplit_japanese_for_overrides(value):
                if chunk in _SEGMENT_OVERRIDES:
                    print(f"[事前オーバーライド] {chunk!r} → {_SEGMENT_OVERRIDES[chunk]}")
                    yield from _SEGMENT_OVERRIDES[chunk]
                else:
                    for segment in _segment_japanese_with_default_vlm(chunk):
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


_MATCH_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "match_templates",
)
EDIT_BUTTON_TEMPLATE = os.path.join(_MATCH_TEMPLATES_DIR, "edit_button.jpg")


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

    # 暗い領域（黒背景または紺色背景）を検出: ピクセル値 80 以下を白に
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # ノイズ除去: 小さな孤立点を消す
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh = frame.shape[0]
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # IME 候補ウィンドウのおよそのサイズ範囲でフィルタ
        if w < 20 or w > 800 or h < 12 or h > 100:
            continue
        # タスクバー・スタートボタン等の画面下部を除外（下15%以内）
        if y > fh * 0.85:
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


def _find_changed_region(
    base: np.ndarray,
    current: np.ndarray,
    exclude_bottom_px: int = 220,
) -> Optional[np.ndarray]:
    """
    2 フレームの差分から変化した矩形領域を切り出す（フォールバック用）。

    IME 候補ブロックが検出できない場合、入力前後の差分で変化領域を特定する。
    タスクバー・IMEツールバー等の画面下部ノイズを除外するため、
    exclude_bottom_px より下の領域は差分検索から除外する。

    Returns:
        変化した領域の画像。見つからない場合は None。
    """
    h = base.shape[0]
    search_end = max(h - exclude_bottom_px, h // 2)  # 最低でも上半分は含める

    diff = cv2.absdiff(base[:search_end], current[:search_end])
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 最大の変化領域を返す
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h_ = cv2.boundingRect(largest)
    # 最小サイズフィルタ
    if w < 32 or h_ < 32:
        return None
    return current[y:y + h_, x:x + w]


def _save_debug_image(frame: np.ndarray, name: str) -> None:
    """デバッグ用: フレームを captures/ に保存する（失敗しても無視）。"""
    try:
        captures_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captures"
        )
        os.makedirs(captures_dir, exist_ok=True)
        # 特殊文字をアンダースコアに置換してファイル名を安全にする
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = os.path.join(captures_dir, f"debug_{safe_name}.png")
        cv2.imwrite(path, frame)
        print(f"  [debug] ROI保存: {path}")
    except Exception:  # noqa: BLE001
        pass


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
    # Handle Python-dict-style single-quoted responses (e.g. {'candidate': '通'})
    single_quoted = re.search(r"'candidate'\s*:\s*'([^']+)'", content)
    if single_quoted:
        return single_quoted.group(1).strip() or None
    # Last resort: return None rather than raw content (which would cause false positives)
    return None


def _read_ime_candidate_with_vlm(frame: np.ndarray, target_kanji: str) -> Optional[str]:
    # NOTE: Do NOT include target_kanji in the prompt — it causes the VLM to hallucinate
    # the expected kanji even when a different candidate (e.g. 官房 vs 感冒) is highlighted.
    try:
        return read_inline_candidate_roi(frame)
    except MlxVlmImeError as exc:
        raise RuntimeError(f"IME候補確認の mlx_vlm 呼び出しが失敗しました: {exc}") from exc


def _read_ime_inline_candidate_fullframe(frame: np.ndarray, target_kanji: str) -> Optional[str]:
    """全画面フレームからインライン変換候補を読み取る（ROI検出失敗時のフォールバック）。

    フレームを中央帯にクロップしてから mlx_vlm サーバーに送信する。
    """
    try:
        return read_inline_candidate_context(frame)
    except MlxVlmImeError as exc:
        raise RuntimeError(f"インライン候補フルフレーム mlx_vlm 呼び出しが失敗しました: {exc}") from exc


def type_kanji_via_ime(
    romaji: str,
    target_kanji: str,
    max_attempts: int = 1,
    wait_sec: float = 0.5,
    windows_version: str = "windows7",
    clear_field: bool = False,
    _current_ime_mode: Optional[str] = None,
    _no_helper_fallback: bool = False,
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
        windows_version: Windows バージョン（"windows7" または "windows10"）—Win10 固有の動作制御に使用
        clear_field: True の場合、入力前に Backspace を 50 回送信してフィールドをクリアする
        _current_ime_mode: 呼び出し元が既に検出した IME モード。指定時は内部で再検出しない。
        _no_helper_fallback: True の場合、ヘルパー単語フォールバックを使わない（再帰防止用）。

    Raises:
        RuntimeError: BLE サーバー未起動、またはキャプチャ失敗の場合
        ValueError: max_attempts 回試行しても target_kanji が見つからない場合
    """
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    if clear_field:
        print("フィールドをクリア中 (Backspace x50)...")
        for _ in range(50):
            client.press_key("backspace")
        time.sleep(0.3)

    # IME をひらがな入力モードに確保してからローマ字入力
    # _current_ime_mode が渡された場合は呼び出し元が既に切替済みなので再検出しない
    if _current_ime_mode is None:
        current_mode = detect_ime_mode(client, config, windows_version=windows_version)
        ensure_ime_mode("japanese", client, current_mode)
    else:
        # 呼び出し元が既に Japanese モードを確保済み
        print(f"  [IME] 呼び出し元が {_current_ime_mode!r} を確認済み → 再検出スキップ")

    # ローマ字入力
    print(f"ローマ字入力: {romaji}")
    ok = client.type_text(romaji)
    print(f"type:{romaji} -> {'OK' if ok else 'NG'}")
    time.sleep(0.3)  # IMEがローマ字処理するまで待機

    # Space #1 の前にフレームをキャプチャ（インライン変換との差分検出用）
    pre_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if pre_frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    # 1回目の Space: インライン変換（ひらがな→第1候補に変わる、ポップアップなし）
    print("[Space] インライン変換 (第1候補)...")
    ok = client.press_key("space")
    print(f"key:space -> {'OK' if ok else 'NG'}")
    time.sleep(wait_sec)

    # 第1候補をキャプチャしてOCR確認
    base_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if base_frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    # Windows 10 では下線表示のインライン変換はVLM/OCRで読み取れないためスキップ。
    # Windows 7 のみ反転ブロック検出 + VLM/OCR を試みる。
    skip_inline = (windows_version == "windows10")

    inline_vlm: Optional[str] = None
    inline_combined = ""
    if not skip_inline:
        # インライン候補を検出: ROI検出を試み、非日本語なら失敗とみなしてポップアップへ
        inline_roi = _find_ime_candidate_region(base_frame)
        if inline_roi is None:
            inline_roi = _find_changed_region(pre_frame, base_frame)
        if inline_roi is not None:
            _save_debug_image(inline_roi, f"ime_inline_{target_kanji}")
            try:
                inline_vlm = _read_ime_candidate_with_vlm(inline_roi, target_kanji)
                print(f"  [第1候補] Qwen読取(ROI): {inline_vlm!r}")
            except RuntimeError as exc:
                print(f"  [第1候補] Qwen読取失敗: {exc}")
            ocr_results_inline = _request_ocr_results(inline_roi, config)
            inline_combined = "".join(t for (_, t, _) in ocr_results_inline)
            print(f"  [第1候補] OCR結合: {inline_combined!r}")
        else:
            print(f"  [第1候補] 変化領域を検出できませんでした")

        # インラインROIでVLM/OCRが両方失敗した場合、フルフレームVLMでフォールバック
        # ただし ROI が取得できなかった場合はスキップ（タイムアウトでIME状態が崩れるのを防ぐ）
        inline_fullframe_vlm: Optional[str] = None
        if (inline_roi is not None
                and not _ime_candidate_matches(target_kanji, inline_vlm or "")
                and not _ime_candidate_matches(target_kanji, inline_combined)):
            try:
                inline_fullframe_vlm = _read_ime_inline_candidate_fullframe(base_frame, target_kanji)
                print(f"  [第1候補] Qwen読取(全フレーム): {inline_fullframe_vlm!r}")
            except RuntimeError as exc:
                print(f"  [第1候補] Qwen全フレーム読取失敗: {exc}")

        if (_ime_candidate_matches(target_kanji, inline_vlm or "")
                or _ime_candidate_matches(target_kanji, inline_combined)
                or _ime_fullframe_exact_match(target_kanji, inline_fullframe_vlm or "")):
            print(f"  「{target_kanji}」第1候補を確認 → Enter で確定")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            print("完了")
            return
    else:
        print(f"  [Win10] インライン読取スキップ → ポップアップへ直行")

    # 第1候補が不一致（または Win10 でスキップ）→ Space #2 でポップアップを開く
    print(f"  第1候補が「{target_kanji}」と不一致。ポップアップを開きます...")
    ok = client.press_key("space")
    print(f"key:space (popup) -> {'OK' if ok else 'NG'}")
    time.sleep(max(wait_sec, 1.0))  # ポップアップが完全に表示されるまで待機

    # ポップアップ直後の候補リストをVLM（全フレーム）で解析してターゲットの位置を特定
    popup_init_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if popup_init_frame is not None:
        # 全フレームをデバッグ用に保存
        _save_debug_image(popup_init_frame, f"ime_popup_{target_kanji}")

        # 番号付き候補リストを1回のVLMコールで取得（番号とテキストの両方）
        numbered: list[tuple[int, str]] = []
        try:
            numbered = read_popup_candidates_numbered(popup_init_frame, debug_name=target_kanji)
            print(f"  ポップアップ候補解析(VLM番号付き): {numbered}")
        except Exception as exc:
            print(f"  VLM番号付き候補読取失敗: {exc}")

        # --- 句読点幅選択ポップアップ（全/半）や記号バリエーションポップアップを誤検出した場合 ---
        _WIDTH_ONLY_TEXTS = {"全", "半", "全角", "半角"}
        def _is_symbol_variation_popup(candidates: list) -> bool:
            if not candidates:
                return False
            if all(t in _WIDTH_ONLY_TEXTS for _, t in candidates):
                return True
            # VLM が "+[半]", "+[全]", "、[全]" 等と読み取った場合もシンボルポップアップ
            return all(("[半]" in t or "[全]" in t) for _, t in candidates if t)

        if numbered and _is_symbol_variation_popup(numbered):
            print(f"  [警告] 記号バリエーションポップアップを検出 → Escape で閉じてリトライします")
            client.press_key("escape")
            time.sleep(0.5)
            # IME 組成をキャンセル（誤入力された記号を削除）
            client.press_key("escape")
            time.sleep(0.3)
            # コミット済み余分記号を削除
            client.press_key("backspace")
            time.sleep(0.2)
            # ローマ字を再入力してポップアップを開く
            ok = client.type_text(romaji)
            print(f"type:{romaji} (リトライ) -> {'OK' if ok else 'NG'}")
            time.sleep(0.5)
            ok = client.press_key("space")
            print(f"key:space (インライン変換リトライ) -> {'OK' if ok else 'NG'}")
            time.sleep(wait_sec)
            ok = client.press_key("space")
            print(f"key:space (リトライ popup) -> {'OK' if ok else 'NG'}")
            time.sleep(max(wait_sec, 1.0))
            retry_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if retry_frame is not None:
                try:
                    numbered = read_popup_candidates_numbered(retry_frame)
                    print(f"  リトライ後候補: {numbered}")
                except Exception as exc:
                    print(f"  リトライVLM失敗: {exc}")

        # --- 完全一致 ---
        exact_match = _find_best_candidate_match(target_kanji, numbered)

        if exact_match is not None:
            display_num, matched_text = exact_match
            print(f"  [VLM一致] 「{target_kanji}」→ 表示番号 {display_num} (読取: {matched_text!r})")
            if display_num <= 9:
                print(f"  「{target_kanji}」→ type:{display_num}")
                client.send_command(f"type:{display_num}")
                time.sleep(wait_sec)
                ok = client.press_key("enter")
                print(f"key:enter -> {'OK' if ok else 'NG'}")
                print("完了")
            else:
                spaces_needed = display_num - 2
                print(f"  「{target_kanji}」→ Space×{spaces_needed} + Enter (表示番号 {display_num})")
                for _ in range(spaces_needed):
                    client.press_key("space")
                    time.sleep(wait_sec)
                ok = client.press_key("enter")
                print(f"key:enter -> {'OK' if ok else 'NG'}")
                print("完了")
            return
        print(f"  候補リストに「{target_kanji}」なし → プレフィックス一致を試みます")

        # --- プレフィックス一致: 候補の中にターゲットの前半部分があれば選択して残りを変換 ---
        prefix_match: Optional[tuple[int, str]] = None
        for n, c in numbered:
            if c and 0 < len(c) < len(target_kanji) and target_kanji.startswith(c):
                prefix_match = (n, c)
                break

        if prefix_match is not None:
            display_num, prefix_text = prefix_match
            remaining_target = target_kanji[len(prefix_text):]
            print(f"  [プレフィックス一致] 「{prefix_text}」を候補で発見（表示番号: {display_num}、残り: 「{remaining_target}」）")
            # プレフィックス候補を選択
            if display_num <= 9:
                print(f"  「{prefix_text}」→ type:{display_num}")
                client.send_command(f"type:{display_num}")
            else:
                spaces_needed = display_num - 2
                for _ in range(spaces_needed):
                    client.press_key("space")
                    time.sleep(wait_sec)
                client.press_key("enter")
            time.sleep(max(wait_sec, 2.0))  # ポップアップが閉じるまで待機

            # プレフィックス選択後の画面を確認（デバッグ用）
            pre_space_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if pre_space_frame is not None:
                _save_debug_image(pre_space_frame, f"ime_popup_{target_kanji}_after_prefix")

            # 右矢印で次のセグメントに移動してからポップアップを開く
            # （Space を押すと現在のセグメントのポップアップが再表示されるため）
            print(f"  次セグメント「{remaining_target}」へ移動...")
            ok = client.press_key("right")
            print(f"key:right (次セグメント移動) -> {'OK' if ok else 'NG'}")
            time.sleep(0.5)
            print(f"  次セグメント「{remaining_target}」のポップアップを開きます...")
            ok = client.press_key("space")
            print(f"key:space (次セグメント) -> {'OK' if ok else 'NG'}")
            time.sleep(max(wait_sec, 2.0))

            rem_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if rem_frame is None:
                client.press_key("enter")
                print("完了（残りフレーム取得失敗 → Enterで確定）")
                return
            _save_debug_image(rem_frame, f"ime_popup_{target_kanji}_remaining")
            try:
                rem_numbered = read_popup_candidates_numbered(rem_frame)
                print(f"  残りポップアップ候補: {rem_numbered}")
                rem_match = _find_best_candidate_match(remaining_target, rem_numbered)
                if rem_match is not None:
                    rem_display, matched_rem = rem_match
                    print(f"  「{remaining_target}」→ 表示番号 {rem_display} (読取: {matched_rem!r})")
                    if rem_display <= 9:
                        print(f"  「{remaining_target}」→ type:{rem_display}")
                        client.send_command(f"type:{rem_display}")
                        time.sleep(wait_sec)
                    else:
                        spaces_needed = rem_display - 2
                        for _ in range(spaces_needed):
                            client.press_key("space")
                            time.sleep(wait_sec)
                        ok = client.press_key("enter")
                        print(f"key:enter -> {'OK' if ok else 'NG'}")
                    print("完了")
                    return
                print(f"  残りポップアップに「{remaining_target}」なし → Enterで仮確定")
            except Exception as exc:
                print(f"  残りポップアップVLM読取失敗: {exc}")
            client.press_key("enter")
            print("完了（残り部分はEnterで確定）")
            return

        # セグメント拡張 (Shift+Right) で再試行 — 現在は無効化
        # print(f"  プレフィックス一致なし → Shift+Right でセグメント拡張を試みます")
        # for ext in range(4):
        #     client.press_key("escape")
        #     time.sleep(0.3)
        #     client.press_key("shift_right")
        #     time.sleep(0.3)
        #     ok = client.press_key("space")
        #     print(f"  [拡張{ext+1}] Escape+Shift+Right+Space -> {'OK' if ok else 'NG'}")
        #     time.sleep(max(wait_sec, 1.0))
        #
        #     ext_frame = capture_screen(
        #         device_index=config.capture_device_index,
        #         width=config.capture_width,
        #         height=config.capture_height,
        #     )
        #     if ext_frame is None:
        #         continue
        #     try:
        #         ext_numbered = read_popup_candidates_numbered(ext_frame)
        #         print(f"  [拡張{ext+1}] 候補: {ext_numbered}")
        #         ext_match = _find_best_candidate_match(target_kanji, ext_numbered)
        #         if ext_match is not None:
        #             display_num, c = ext_match
        #             if display_num <= 9:
        #                 print(f"  「{target_kanji}」→ type:{display_num}")
        #                 client.send_command(f"type:{display_num}")
        #                 time.sleep(wait_sec)
        #             else:
        #                 spaces_needed = display_num - 2
        #                 for _ in range(spaces_needed):
        #                     client.press_key("space")
        #                     time.sleep(wait_sec)
        #                 ok = client.press_key("enter")
        #                 print(f"key:enter -> {'OK' if ok else 'NG'}")
        #             print("完了")
        #             return
        #     except Exception as exc:
        #         print(f"  [拡張{ext+1}] VLM読取失敗: {exc}")
        # print(f"  セグメント拡張でも「{target_kanji}」なし → 試行ループへ")

    # --- ヘルパー単語フォールバック ---
    # ポップアップ候補にターゲットが存在しない場合、Qwen3 に「目標漢字を含む
    # 一般的な日本語単語」を提案してもらい、変換後に余分な文字を削除して得る。
    if not _no_helper_fallback:
        print(f"  ヘルパー単語フォールバックを試みます...")
        if _try_helper_word_fallback(client, config, target_kanji, wait_sec, windows_version, romaji=romaji):
            return

    for attempt in range(1, max_attempts + 1):
        # フレームキャプチャ
        frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame is None:
            raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

        vlm_candidate: Optional[str] = None

        # 全フレームをVLMに渡して現在ハイライトされている候補を読取
        try:
            vlm_candidate = _read_ime_candidate_with_vlm(frame, target_kanji)
            print(f"  [試行{attempt}] 全フレーム Qwen読取: {vlm_candidate!r}")
        except RuntimeError as exc:
            print(f"  [試行{attempt}] 全フレーム Qwen読取失敗: {exc}")

        confirmed = _ime_candidate_matches(target_kanji, vlm_candidate or "")

        if confirmed:
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

    # 全候補を確認できなかった → Esc×1 でひらがな状態に戻してから Enter で確定
    # Windows MS-IME: ポップアップ状態から Esc×1 = ひらがなに戻る
    print(f"  ⚠️ IME候補を {max_attempts} 回確認しましたが「{target_kanji}」を確定できませんでした")
    print("  → Esc×1 でひらがなに戻して確定します（誤変換防止）")
    client.press_key("escape")
    time.sleep(0.2)
    client.press_key("enter")


def _try_helper_word_fallback(
    client: "BLEClient",
    config,
    target_kanji: str,
    wait_sec: float,
    windows_version: str,
    romaji: str = "",
) -> bool:
    """ヘルパー単語フォールバック: Qwen3にヘルパー単語を問い合わせ、
    変換後に余分な文字を削除して目標漢字を入力する。

    IMEポップアップ候補に目標漢字が現れない場合のラストリゾート。
    Qwen3 に「目標漢字を含む一般的な日本語単語」を提案してもらい、
    その単語を変換確定後に余分な末尾文字を削除することで目標漢字を得る。

    例: target_kanji="過膨張"
        Qwen3 が {"word": "過去", "romaji": "kako", "covered_prefix": "過", "delete_count": 1} を提案
        → "kako" を入力 → "過去" に変換確定 → Backspace × 1 で "去" 削除 → "過" 確定済み
        → 残り "膨張" を type_kanji_via_ime で入力

    Args:
        client: BLEClient インスタンス
        config: キャプチャ設定
        target_kanji: 変換しようとしている漢字/語句
        wait_sec: 変換待機秒数
        windows_version: Windows バージョン

    Returns:
        True: ヘルパー単語アプローチで全体の入力が成功した
        False: 失敗（呼び出し元が次のフォールバックへ進む）
    """
    # suggest_ime_helper_word は対象の最初の1文字を渡す
    target_char = target_kanji[0]
    print(f"  [ヘルパー単語] 「{target_char}」のヘルパー単語をQwen3に問い合わせ中...")
    suggestions = suggest_ime_helper_word(target_char)
    if not suggestions:
        print(f"  [ヘルパー単語] 提案なし → フォールバック失敗")
        return False

    for idx, suggestion in enumerate(suggestions):
        helper_word = suggestion["word"]
        # backspace_count はヘルパー単語長 - 対象文字長から確定的に計算する（Qwen3の提案値は使わない）
        backspace_count = len(helper_word) - len(target_char)
        # ヘルパー単語確定後に残る先頭部分: word[:-backspace_count] or word全体
        covered_prefix = helper_word[:-backspace_count] if backspace_count > 0 else helper_word
        # ヘルパー単語のローマ字は _kanji_to_romaji で計算
        helper_romaji = _kanji_to_romaji(helper_word)
        print(
            f"  [ヘルパー単語] 提案{idx+1}/{len(suggestions)}: {helper_word!r} "
            f"(romaji={helper_romaji!r}), covers={covered_prefix!r}, backspace={backspace_count}"
        )

        # 現在の IME 状態をキャンセル (Escape×2: popup→inline→完全解除)
        print("  [ヘルパー単語] Escape で IME 状態をキャンセル...")
        client.press_key("escape")
        time.sleep(0.3)
        client.press_key("escape")
        time.sleep(0.3)
        # 注意: Escape×2 により IME 組成バッファは完全に解除されるため
        # 追加の Backspace は不要（確定済みテキストを誤削除する恐れがある）

        # ヘルパー単語のローマ字を入力
        print(f"  [ヘルパー単語] ローマ字入力: {helper_romaji!r}")
        ok = client.type_text(helper_romaji)
        print(f"  type:{helper_romaji} -> {'OK' if ok else 'NG'}")
        time.sleep(0.3)

        # Space でインライン変換 → さらに Space でポップアップを開く
        ok = client.press_key("space")
        print(f"  key:space (インライン変換) -> {'OK' if ok else 'NG'}")
        time.sleep(wait_sec)
        ok = client.press_key("space")
        print(f"  key:space (ポップアップ) -> {'OK' if ok else 'NG'}")
        time.sleep(max(wait_sec, 1.0))

        # ポップアップ候補を読取
        helper_frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if helper_frame is None:
            print("  [ヘルパー単語] フレーム取得失敗 → 次の提案を試みます")
            client.press_key("escape")
            time.sleep(0.2)
            client.press_key("escape")
            continue

        _save_debug_image(helper_frame, f"ime_helper_{target_kanji}_{helper_word}")

        helper_numbered: list[tuple[int, str]] = []
        try:
            helper_numbered = read_popup_candidates_numbered(
                helper_frame, debug_name=f"helper_{target_kanji}"
            )
            print(f"  [ヘルパー単語] ポップアップ候補: {helper_numbered}")
        except Exception as exc:
            print(f"  [ヘルパー単語] 候補読取失敗: {exc}")

        # ヘルパー単語を候補から検索
        helper_match = _find_best_candidate_match(helper_word, helper_numbered)
        if helper_match is None:
            print(f"  [ヘルパー単語] 「{helper_word}」が候補に見つかりません → 次の提案を試みます")
            client.press_key("escape")
            time.sleep(0.2)
            client.press_key("escape")
            time.sleep(0.3)
            # 注意: Escape×2 により IME 組成バッファは完全に解除されるため追加 Backspace は不要
            continue

        display_num, matched_text = helper_match
        print(f"  [ヘルパー単語] 「{helper_word}」→ 表示番号 {display_num} (読取: {matched_text!r})")

        # ヘルパー単語を選択して確定
        if display_num <= 9:
            print(f"  [ヘルパー単語] type:{display_num}")
            client.send_command(f"type:{display_num}")
            time.sleep(wait_sec)
        else:
            spaces_needed = display_num - 2
            for _ in range(spaces_needed):
                client.press_key("space")
                time.sleep(wait_sec)
            client.press_key("enter")
            time.sleep(wait_sec)

        # 余分な文字を削除
        if backspace_count > 0:
            print(f"  [ヘルパー単語] Backspace × {backspace_count} で余分な文字を削除...")
            for _ in range(backspace_count):
                client.press_key("backspace")
                time.sleep(0.15)

        # 数字キー選択（display_num <= 9）の場合、候補は未確定状態のため Enter で確定する。
        # Space 循環＋Enter（display_num > 9）の場合は Enter で既にコミット済みなので不要。
        if display_num <= 9:
            print(f"  [ヘルパー単語] Enter で確定...")
            ok = client.press_key("enter")
            print(f"  key:enter (helper confirm) -> {'OK' if ok else 'NG'}")
            time.sleep(0.3)

        print(f"  [ヘルパー単語] 「{covered_prefix}」の入力完了")

        # 残りの文字列を入力
        remaining = target_kanji[len(covered_prefix):]
        if remaining:
            print(f"  [ヘルパー単語] 残り「{remaining}」を続けて入力します...")
            remaining_romaji = _kanji_to_romaji(remaining)
            # ESP32がBackspace処理を完了するまで待機してから次のセグメントへ
            time.sleep(1.0)
            # _no_helper_fallback=True で無限再帰を防ぐ
            type_kanji_via_ime(
                remaining_romaji,
                remaining,
                wait_sec=wait_sec,
                windows_version=windows_version,
                _current_ime_mode="japanese",
                _no_helper_fallback=True,
            )

        return True

    print(f"  [ヘルパー単語] 全提案が失敗 → フォールバック失敗")
    return False



    """mlx_vlm を使って IME ポップアップ候補リスト全体を読み取る。

    ポップアップ領域を自動検出してクロップしてから Qwen3-VL に送信する。

    Returns:
        候補文字列のリスト（表示順、0-indexed）。失敗時は空リスト。
    """
    return read_popup_candidates(frame)


def _extract_ime_candidates(ocr_texts: list[str]) -> list[str]:
    """OCR テキストリストから IME ポップアップ候補を抽出する（ノイズ除去）。

    入力フィールドのコンテキストノイズ（数字・記号・アルファベット混在テキスト）を除去し、
    純粋な日本語文字のみで構成される候補のみを返す。
    """
    result = []
    for text in ocr_texts:
        text = text.strip()
        if not text:
            continue
        # 純粋な日本語文字（ひらがな U+3040-、カタカナ U+30A0-、漢字 U+4E00-）のみ 1-6 文字
        if all("\u3040" <= ch <= "\u9fff" for ch in text) and 1 <= len(text) <= 6:
            result.append(text)
    return result


def _ime_candidate_matches(target: str, combined: str) -> bool:
    """IME候補OCRにターゲット文字列が含まれているか確認する。

    OCR ノイズを 1 文字まで許容する: 先頭文字が一致し、残りの差異が 1 文字以内の
    部分文字列も一致とみなす（例: 感冒 → 感昌 のような視覚的誤読への対策）。
    """
    if target in combined:
        return True
    n = len(target)
    if n < 2:
        return False
    for i in range(len(combined) - n + 1):
        substr = combined[i:i + n]
        if substr[0] == target[0]:
            mismatches = sum(a != b for a, b in zip(substr, target))
            if mismatches <= 1:
                return True
    return False


def _find_best_candidate_match(
    target: str, numbered: list[tuple[int, str]]
) -> Optional[tuple[int, str]]:
    """番号付き候補リストからターゲットに最もよく一致する候補を返す。

    完全一致を優先し、なければファジーマッチを試みる。
    例: target="痛", candidates=[(1,'通'),(2,'疼痛'),(5,'痛')] → (5, '痛')
    """
    # First pass: exact match (including fuzzy OCR noise, same length)
    for n, c in numbered:
        if c == target:
            return (n, c)
    # Second pass: fuzzy same-length match (OCR noise tolerance)
    for n, c in numbered:
        if len(c) == len(target) and _ime_candidate_matches(target, c):
            return (n, c)
    # Third pass: general substring/fuzzy match (e.g., target is part of candidate)
    for n, c in numbered:
        if _ime_candidate_matches(target, c):
            return (n, c)
    return None


def _ime_fullframe_exact_match(target: str, fullframe_result: str) -> bool:
    """フルフレームVLM結果との厳格な一致判定。

    フルフレームVLMは前後の入力済み文字列まで含めて返すことがあるため、
    部分一致は使わず、完全一致のみを許容する。
    """
    if not fullframe_result or not target:
        return False
    return fullframe_result.strip() == target



def _is_ascii_only(text: str) -> bool:
    """文字列が ASCII 文字のみで構成されているか判定する。"""
    return all(ord(ch) < 128 for ch in text)


def _is_katakana_only(text: str) -> bool:
    """文字列がカタカナ（長音符含む）のみで構成されているか判定する。"""
    return bool(text) and all(
        "\u30a0" <= ch <= "\u30ff" or ch == "\u30fc"  # カタカナ + 長音符ー
        for ch in text
    )


def detect_ime_mode(
    client: "BLEClient",
    config=None,
    windows_version: str = "windows7",
) -> Optional[str]:
    """
    'a' を1文字入力し、Qwen3-VL でスクリーンを読んで IME モードを検出する。

    英語入力モードなら 'a' が、日本語（ひらがな）入力モードなら「あ」が表示される。
    VLM で判定後に Backspace で入力した文字を削除する。

    Args:
        client: BLEClient インスタンス（キー送信に使用）
        config: キャプチャ設定（None の場合は load_config() で取得）
        windows_version: 未使用（互換性維持のため残す）

    Returns:
        'japanese': ひらがな入力モード
        'english':  英数字入力モード
        None:       判定不能
    """
    if config is None:
        config = load_config(skip_password=True)

    # 'a' を1文字入力してIMEの反応を確認
    client.type_text("a")
    time.sleep(0.4)

    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )

    result: Optional[str] = None
    if frame is not None:
        result = detect_ime_mode_from_typed_a(frame)
        print(f"  [IME検出/VLM] 結果: {result!r}")
    else:
        print("  [IME検出] キャプチャ失敗")

    # 入力した 'a' または 'あ' を削除する
    if result == "japanese":
        # 日本語モード: IME の未確定「あ」を Escape でキャンセル（確実）
        client.press_key("escape")
        time.sleep(0.25)
        # 念のため Backspace も送る（Escape が効かない場合への保険）
        client.press_key("backspace")
        time.sleep(0.2)
    elif result == "english":
        # 英語モード: 'a' を Backspace で削除
        client.press_key("backspace")
        time.sleep(0.2)
    else:
        # 判定不能: Escape と Backspace 両方送る
        client.press_key("escape")
        time.sleep(0.15)
        client.press_key("backspace")
        time.sleep(0.2)

    return result


def toggle_ime(client: "BLEClient") -> None:
    """半角/全角キーを送って IME モードをトグルする。"""
    print("  [IME切替] 半角/全角 を送信")
    client.press_key("zenkaku")
    time.sleep(0.5)  # IME 切替の反映を待つ


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
    # 医療用語など pykakasi が誤読みするケースの手動オーバーライド
    _ROMAJI_OVERRIDES: dict[str, str] = {
        "鼻汁": "bijuu",
        "咳嗽": "gaisou",
        "嘔吐": "outo",
        "浮腫": "fushuu",
        "倦怠": "kentai",
        "痙攣": "keiren",
        "蕁麻疹": "jinmashin",
        "喀痰": "kakutan",
        "喘鳴": "zenmei",
        "哮喘": "kozen",
        "喘息": "zensoku",
        "膿胸": "noukyo",
        "胸水": "kyousui",
        "肺炎": "haien",
    }
    if text in _ROMAJI_OVERRIDES:
        return _ROMAJI_OVERRIDES[text]
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


def type_japanese_sentence(text: str, windows_version: str = "windows7", clear_field: bool = False) -> None:
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
        windows_version: Windows バージョン（"windows7" または "windows10"）—Win10 固有の動作制御に使用
        clear_field: True の場合、入力前に Backspace を 50 回送信してフィールドをクリアする
    """
    print(f"文節分割中 (Qwen優先): {text!r}")
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    if clear_field:
        # 入力前にフィールドをクリア（Backspace を一括送信）
        print("フィールドをクリア中 (Backspace x50)...")
        # 個別 press_key×50 は BLE で遅いため type_text でバッチ送信
        client.type_text("\x08" * 50)
    # 開始時に1回だけ IME モードを検出し、以降は内部変数でトラッキングする
    print("現在の IME モードを検出中...")
    current_mode: Optional[str] = detect_ime_mode(client, config, windows_version=windows_version)
    # 検出失敗(None)の場合: 前回の中断でポップアップが残っている可能性があるため
    # Escape を複数回送ってクリアし、再検出する
    if current_mode is None:
        print("  [IME回復] モード不明 → Escape×3 でクリアして再検出します")
        for _ in range(3):
            client.press_key("escape")
            time.sleep(0.2)
        time.sleep(0.5)
        current_mode = detect_ime_mode(client, config, windows_version=windows_version)
        print(f"  [IME回復] 再検出結果: {current_mode!r}")
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
            # Windows 10 では「、」「。」ともに変換候補ポップアップが表示されるため
            # Enter で確定が必要。Windows 7 では「。」のみ必要。
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  句読点入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            if seg_text == "。" or windows_version == "windows10":
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
            type_kanji_via_ime(
                seg_romaji, seg_text,
                windows_version=windows_version,
                _current_ime_mode=current_mode,
            )
            # Enter確定後、WindowsのIMEが処理を完了するまで待機（タイミング競合防止）
            time.sleep(0.25)

        elif _is_katakana_only(seg_text):
            # カタカナのみ: ひらがなモードでローマ字入力後 F7 でカタカナ変換して Enter
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  カタカナ直接入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            ok = client.press_key("f7")  # 全角カタカナに変換
            print(f"key:f7 -> {'OK' if ok else 'NG'}")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")

        elif seg_text in ("「", "」", "『", "』"):
            # 日本語括弧: ひらがなモードで lbracket/rbracket キーを押す
            # JIS キーボードの日本語モードでは [ → 「、] → 」 になる
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            key_name = "lbracket" if seg_text in ("「", "『") else "rbracket"
            ok = client.press_key(key_name)
            print(f"key:{key_name} ({seg_text}) -> {'OK' if ok else 'NG'}")

        elif any(ch in _JP_SYMBOL_IME_READINGS or ch in _CHAR_ASCII_FALLBACK for ch in seg_text):
            # ※ などの特殊記号、または μ などの ASCII 代替文字を含む: 文字単位で処理
            # ASCII 部分は英数字モードで直接入力、特殊記号は IME で読み変換
            for ch in seg_text:
                if ch in _JP_SYMBOL_IME_READINGS:
                    reading = _JP_SYMBOL_IME_READINGS[ch]
                    current_mode = ensure_ime_mode("japanese", client, current_mode)
                    print(f"  [特殊記号] {ch!r} を IME読み {reading!r} で入力")
                    type_kanji_via_ime(
                        reading, ch,
                        windows_version=windows_version,
                        _current_ime_mode=current_mode,
                    )
                elif ch in _CHAR_ASCII_FALLBACK:
                    # μ など BLE 経由で直接送れない文字は ASCII 代替文字で入力
                    fallback = _CHAR_ASCII_FALLBACK[ch]
                    if fallback:
                        current_mode = ensure_ime_mode("english", client, current_mode)
                        print(f"  [ASCII代替] {ch!r} → {fallback!r}")
                        _type_ascii_text_precisely(client, fallback)
                    else:
                        print(f"  [スキップ] {ch!r} (ASCII代替なし)")
                elif _is_ascii_only(ch):
                    current_mode = ensure_ime_mode("english", client, current_mode)
                    _type_ascii_text_precisely(client, ch)
                else:
                    current_mode = ensure_ime_mode("japanese", client, current_mode)
                    ok = client.type_text(ch)
                    if ok:
                        client.press_key("enter")

        else:
            # ひらがな・カタカナのみ: ひらがなモードでローマ字を直接入力して Enter で確定
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  ひらがな直接入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            # Enter確定後、WindowsのIMEが処理を完了するまで待機
            time.sleep(0.15)

    print("\n文章入力完了")


def _type_english_text(text: str, windows_version: str = "windows7", clear_field: bool = False) -> None:
    """
    英語テキストを英数字モードで直接入力する。

    IME を英数字モードに切替えてからテキストを送信する。
    Enter は送らない（呼び出し元がフィールド確定を制御する）。

    Args:
        text: 入力する英数字文字列
        windows_version: Windows バージョン（"windows7" または "windows10"）—Win10 固有の動作制御に使用
        clear_field: True の場合、入力前に Backspace を 50 回送信してフィールドをクリアする
    """
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    if clear_field:
        print("フィールドをクリア中 (Backspace x50)...")
        for _ in range(50):
            client.press_key("backspace")
        time.sleep(0.3)

    current_mode = detect_ime_mode(client, config, windows_version=windows_version)
    ensure_ime_mode("english", client, current_mode)

    print(f"英語入力: {text!r}")
    _type_ascii_text_precisely(client, _normalize_text_for_typing(text))
    print("完了")


def _print_usage() -> None:
    """使い方を表示する。"""
    print("EHR Input Automation")
    print()
    print("使い方:")
    print("  python -m automation.ehr_input [オプション] [コマンド/テキスト]")
    print()
    print("オプション:")
    print("  --win10              Windows 10 モードで実行（カンマ後 Enter、インライン変換スキップ等）")
    print("  --clear              入力前に Backspace を 50 回送信してフィールドをクリアする")
    print("  --help, -h, help     このヘルプを表示")
    print()
    print("コマンド:")
    print("  (引数なし)                         テスト患者カルテを開く")
    print("  open test                          テスト患者カルテを開く")
    print("  close record                       取り消し[F9]ボタンをクリックしてカルテを閉じる")
    print("  click history <yyyymmdd>           過去カルテの指定日付をクリック")
    print("  edit history <yyyymmdd>            過去カルテ日付クリック後に修正ボタンをクリック")
    print("  detect                             IMEモードを検出して表示")
    print()
    print("テキスト入力:")
    print("  <テキスト>                         テキストを入力（日本語はIME変換）")
    print("  <ファイル.txt>                     テキストファイルの内容を入力")
    print("  open test <テキスト/ファイル>      カルテを開いてからテキスト入力")
    print()
    print("例:")
    print('  python -m automation.ehr_input 肺炎')
    print('  python -m automation.ehr_input --win10 肺炎')
    print('  python -m automation.ehr_input --clear 肺炎')
    print('  python -m automation.ehr_input --win10 --clear 肺炎')
    print('  python -m automation.ehr_input "COVID-19の検査"')
    print('  python -m automation.ehr_input note.txt')
    print('  python -m automation.ehr_input "open test" 肺炎')
    print('  python -m automation.ehr_input --win10 "open test" "MRI所見"')
    print('  python -m automation.ehr_input "click history 20190502"')


def _run_cli(args: list[str]) -> int:
    """CLI entry point for manual EHR input automation."""
    # --win10 / --clear フラグを先頭で抽出（残りの args はそのまま処理）
    win10 = "--win10" in args
    clear_field = "--clear" in args
    if win10 or clear_field:
        args = [a for a in args if a not in ("--win10", "--clear")]
    windows_version = "windows10" if win10 else "windows7"

    if not args:
        # 引数なし: デフォルト動作（後方互換）
        open_test_patient_chart()
        return 0

    if len(args) == 1 and args[0] in ("help", "--help", "-h"):
        _print_usage()
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
        # Named key shortcut: "esc", "backspace", "enter", "space", "tab", "delete",
        # "f6", "f7", "f8", or generic "key:xxx" syntax → press the key directly.
        _NAMED_KEYS = {"esc", "escape", "backspace", "enter", "return", "space", "tab",
                       "delete", "f6", "f7", "f8", "f9", "up", "down", "left", "right"}
        if text.lower() in _NAMED_KEYS or text.lower().startswith("key:"):
            key_name = text[4:] if text.lower().startswith("key:") else text.lower()
            client = _wait_for_ble_connected()
            ok = client.press_key(key_name)
            print(f"key:{key_name} -> {'OK' if ok else 'NG'}")
            return 0
        try:
            _input_resolved_text(text, windows_version=windows_version, clear_field=clear_field)
        except MlxVlmSegmentationError as exc:
            print(f"mlx_vlm文節分割エラー: {exc}")
            print("omlxサーバーの動作確認: curl -s -H 'Authorization: Bearer omlxkey' http://localhost:8000/v1/models")
            return 1
        return 0

    if len(args) >= 2 and args[0] == "open test":
        # 第一引数が "open test"、第二引数がテキスト → カルテ開いてから入力
        text = _resolve_text_argument(args[1])
        print(f"テスト患者カルテを開いてから入力: {text!r}")
        open_test_patient_chart()
        try:
            _input_resolved_text(text, windows_version=windows_version, clear_field=clear_field)
        except MlxVlmSegmentationError as exc:
            print(f"mlx_vlm文節分割エラー: {exc}")
            print("omlxサーバーの動作確認: curl -s -H 'Authorization: Bearer omlxkey' http://localhost:8000/v1/models")
            return 1
        return 0

    _print_usage()
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    return _run_cli(args)


if __name__ == '__main__':
    import sys

    sys.exit(main())
