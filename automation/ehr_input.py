"""
EHR field input automation.

Captures the current HDMI screen, finds a labeled input field,
and types text into it via BLE (ESP32) mouse/keyboard control.

Uses the same AsyncBLERunner pattern as ble_test_cli.py to ensure
identical BLE event-loop behaviour on macOS CoreBluetooth.
"""

import cv2
import tempfile
import os
import time
from typing import Optional

import numpy as np

from automation.config import load_config
from automation.screen_analyzer import capture_screen, load_rapidocr_reader, run_ocr
from automation.gui_image_analyzer import find_textbox_right_of_label
from automation.ble_client import BLEClient


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
    client = BLEClient()
    if not client.is_server_running():
        raise RuntimeError(
            "BLE サーバーが起動していません。\n"
            "  python -m automation.ble_server  を先に別ターミナルで実行してください"
        )

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
    1. フリガナ欄に「tesuto」と入力して Enter → 患者一覧を表示
    2. 0.5 秒待ってから Enter → 先頭患者を選択してカルテを開く
    3. 1 秒待ってから Enter → 表示直後のダイアログを閉じる

    ble_server.py が事前に起動済みであること。
    """
    # Step 1: フリガナ欄に「tesuto」と入力して Enter → 患者一覧を表示させる
    input_text_to_field(input_text="tesuto", label="フリガナ")

    client = BLEClient()
    if not client.is_server_running():
        raise RuntimeError(
            "BLE サーバーが起動していません。\n"
            "  python -m automation.ble_server  を先に別ターミナルで実行してください"
        )
    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")

    # Step 2: 患者一覧が表示されるまで待ってから Enter で先頭患者を選択
    print("患者一覧の表示を待機中 (0.5秒)...")
    time.sleep(0.5)
    ok = client.press_key("enter")
    print(f"key:enter (select patient) -> {'OK' if ok else 'NG'}")

    # Step 3: ダイアログを閉じるため 1 秒待って Enter
    print("ダイアログの表示を待機中 (1秒)...")
    time.sleep(1.0)
    ok = client.press_key("enter")
    print(f"key:enter (dialog close) -> {'OK' if ok else 'NG'}")

    # カルテが完全に開くまで待機
    print("カルテ表示を待機中 (2秒)...")
    time.sleep(2.0)

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

    client = BLEClient()
    if not client.is_server_running():
        raise RuntimeError(
            "BLE サーバーが起動していません。\n"
            "  python -m automation.ble_server  を先に別ターミナルで実行してください"
        )

    ocr_reader = load_rapidocr_reader()

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

    for attempt in range(1, max_attempts + 1):
        print(f"[試行 {attempt}/{max_attempts}] Space キーで変換候補を表示...")
        ok = client.press_key("space")
        print(f"key:space -> {'OK' if ok else 'NG'}")
        time.sleep(wait_sec)

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

        if roi is None:
            print("  IME反転ブロック未検出 → フレーム差分でフォールバック")
            roi = _find_changed_region(base_frame, frame)
            source = "差分領域"

        if roi is None:
            print("  変化領域も未検出。候補がまだ表示されていない可能性があります")
            continue

        # OCR 実行
        ocr_results = run_ocr(ocr_reader, roi)
        texts = [text for (_, text, _) in ocr_results]
        combined = "".join(texts)
        print(f"  {source} OCR結果: {texts!r} → 結合: {combined!r}")

        if target_kanji in combined:
            print(f"  「{target_kanji}」を確認 → Enter で確定")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            print("完了")
            return

        print(f"  「{target_kanji}」は未確認（候補: {combined!r}）")

    # 全試行失敗
    raise ValueError(
        f"{max_attempts} 回試行しましたが「{target_kanji}」の変換候補が確認できませんでした。"
        " IME設定や変換候補を確認してください。"
    )


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


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if not args:
        # 引数なし: デフォルト動作（後方互換）
        open_test_patient_chart()
    elif len(args) == 1 and _is_japanese(args[0]):
        # 第一引数が日本語 → IME 変換のみ
        kanji = args[0]
        romaji = _kanji_to_romaji(kanji)
        print(f"IME変換: {romaji} → {kanji}")
        type_kanji_via_ime(romaji, kanji)
    elif len(args) >= 2 and args[0] == "open test" and _is_japanese(args[1]):
        # 第一引数が "open test"、第二引数が日本語 → カルテ開いてから IME 変換
        kanji = args[1]
        romaji = _kanji_to_romaji(kanji)
        print(f"テスト患者カルテを開いてから IME変換: {romaji} → {kanji}")
        open_test_patient_chart()
        type_kanji_via_ime(romaji, kanji)
    else:
        print("使い方:")
        print("  python -m automation.ehr_input                     # テスト患者カルテを開く")
        print("  python -m automation.ehr_input 肺炎                # IME変換のみ")
        print('  python -m automation.ehr_input "open test" 肺炎   # カルテを開いてからIME変換')
        sys.exit(1)
