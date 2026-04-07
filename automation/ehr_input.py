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

from automation.config import load_config
from automation.screen_analyzer import capture_screen
from automation.gui_image_analyzer import find_textbox_right_of_label, find_first_patient_row
from automation.ble_client import BLEClient


def input_text_to_field(
    input_text: str = "テスト",
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


def select_first_patient() -> None:
    """
    患者一覧画面で先頭の患者行をダブルクリックして選択する。

    ble_server.py が事前に起動済みであること。
    """
    config = load_config(skip_password=True)
    config.detection_mode = 'ocr'

    # 1. スクリーンキャプチャ
    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)
    print(f"スクリーンショット保存: {tmp_path}")

    try:
        # 2. 先頭患者行の座標を取得
        coords = find_first_patient_row(tmp_path, config)
        if coords is None:
            raise RuntimeError("患者一覧の先頭行が見つかりませんでした")
    finally:
        os.unlink(tmp_path)

    x, y = coords
    print(f"先頭患者行の座標: ({x}, {y})")

    # 3. BLE サーバー経由でダブルクリック
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

    ok = client.double_click()
    print(f"double_click -> {'OK' if ok else 'NG'}")

    print("完了")


if __name__ == '__main__':
    # Step 1: フリガナ欄に「テスト」と入力して Enter → 患者一覧を表示させる
    input_text_to_field(input_text="テスト", label="フリガナ")
    # Step 2: 先頭患者行をダブルクリックして選択
    select_first_patient()
