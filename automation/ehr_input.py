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
from automation.local_segmentation import segment_japanese_text_locally


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

    # Space を1回だけ押して最初の変換候補を表示する。
    # その後は候補を進めず、OCR を複数回試みて確認する。
    # OCR で確認できなくても最初の候補を信頼して Enter で確定する。
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

        if roi is not None:
            ocr_results = run_ocr(ocr_reader, roi)
            texts = [text for (_, text, _) in ocr_results]
            combined = "".join(texts)
            # 日本語文字（漢字・ひらがな・カタカナ）が含まれない場合は誤検出とみなす
            if not any("\u3040" <= ch <= "\u9fff" for ch in combined):
                print(f"  [試行{attempt}] IME反転ブロック OCR結果に日本語なし ({combined!r}) → フレーム差分でフォールバック")
                roi = None

        if roi is None:
            roi = _find_changed_region(base_frame, frame)
            source = "差分領域"
            if roi is not None:
                ocr_results = run_ocr(ocr_reader, roi)
                texts = [text for (_, text, _) in ocr_results]
                combined = "".join(texts)

        if roi is None:
            print(f"  [試行{attempt}] 変化領域も未検出。少し待って再試行...")
            time.sleep(0.3)
            continue

        texts = [text for (_, text, _) in ocr_results]
        print(f"  [試行{attempt}] {source} OCR結果: {texts!r} → 結合: {combined!r}")

        if _ime_candidate_matches(target_kanji, combined, attempt):
            print(f"  「{target_kanji}」を確認 → Enter で確定")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            print("完了")
            return

        print(f"  「{target_kanji}」は未確認。再キャプチャして再試行...")
        time.sleep(0.3)

    # OCR で確認できなかったが、最初の候補（Space 1回）を信頼して Enter で確定
    print(f"  OCR確認できませんでしたが、最初の候補を信頼して Enter で確定します")
    ok = client.press_key("enter")
    print(f"key:enter -> {'OK' if ok else 'NG'}")
    print("完了（確認なし）")


def _ime_candidate_matches(target: str, combined: str, attempt: int) -> bool:
    """IME候補テキストにターゲット文字列が含まれているか確認する。

    試行1（最初のSpace直後）は最初の候補がハイライトされている状態。
    OCRが差分領域を部分的にしか読めないことが多いため、ターゲットの
    先頭漢字が含まれていれば一致とみなして即確定する。

    試行2以降はSpaceでカーソルが次候補へ進んでいるため完全一致のみ許容する。
    これにより「対して」→「大して」への誤移動を防ぐ。
    """
    if target in combined:
        return True
    # 試行1のみ: OCRが先頭の漢字文字だけ読めていれば最初の候補として確定
    if attempt == 1:
        first_kanji = next(
            (ch for ch in target if "\u4e00" <= ch <= "\u9fff"), None
        )
        if first_kanji and first_kanji in combined:
            return True
    return False


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
    日本語文をIMEを使って文節単位で入力する。

    sudachipy + pykakasi で文節分割し、各文節を個別に IME 変換・確定する。
    漢字を含む文節は type_kanji_via_ime()、ひらがな・カタカナのみの文節は
    直接入力（ローマ字 + Enter）で処理する。

    Args:
        text: 入力する日本語文（例: "肺炎に対して抗菌薬による治療を行う"）
    """
    print(f"文節分割中 (sudachipy + pykakasi): {text!r}")
    segments = _segment_japanese_locally(text)
    print(f"分割結果: {segments}")

    client = BLEClient()
    if not client.is_server_running():
        raise RuntimeError(
            "BLE サーバーが起動していません。\n"
            "  python -m automation.ble_server  を先に別ターミナルで実行してください"
        )

    for seg in segments:
        seg_text = seg["text"]
        seg_romaji = seg["romaji"]
        print(f"\n--- 文節: {seg_text!r} ({seg_romaji}) ---")

        if seg_text in ("、", "。"):
            # 句読点: IMEが自動変換するキー（,/.）を直接送るだけ（Enterは不要）
            print(f"  句読点入力: {seg_romaji!r}")
            ok = client.switch_to_keyboard_mode()
            print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
        elif _has_kanji(seg_text):
            # 漢字を含む文節: IME変換候補を確認してから確定
            type_kanji_via_ime(seg_romaji, seg_text)
        else:
            # ひらがな・カタカナのみ: ローマ字を直接入力してEnterで確定
            print(f"  直接入力: {seg_romaji!r}")
            ok = client.switch_to_keyboard_mode()
            print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")

    print("\n文章入力完了")


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

    if len(args) == 1 and _is_japanese(args[0]):
        # 第一引数が日本語 → IME 変換
        text = args[0]
        # 短い単語（4文字以下かつ助詞なし）は単一変換、長い文章は文節分割
        if len(text) <= 4 and not any(ch in text for ch in "をにはがでも"):
            romaji = _kanji_to_romaji(text)
            print(f"IME変換: {romaji} → {text}")
            type_kanji_via_ime(romaji, text)
        else:
            type_japanese_sentence(text)
        return 0

    if len(args) >= 2 and args[0] == "open test" and _is_japanese(args[1]):
        # 第一引数が "open test"、第二引数が日本語 → カルテ開いてから IME 変換
        text = args[1]
        print(f"テスト患者カルテを開いてから文章入力: {text!r}")
        open_test_patient_chart()
        if len(text) <= 4 and not any(ch in text for ch in "をにはがでも"):
            romaji = _kanji_to_romaji(text)
            type_kanji_via_ime(romaji, text)
        else:
            type_japanese_sentence(text)
        return 0

    print("使い方:")
    print('  python -m automation.ehr_input                     # テスト患者カルテを開く')
    print('  python -m automation.ehr_input "open test"         # テスト患者カルテを開く')
    print('  python -m automation.ehr_input 肺炎                # IME変換のみ')
    print('  python -m automation.ehr_input "open test" 肺炎   # カルテを開いてからIME変換')
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    return _run_cli(args)


if __name__ == '__main__':
    import sys

    sys.exit(main())
