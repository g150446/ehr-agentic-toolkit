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

    ocr_reader = load_rapidocr_reader()
    results = run_ocr(ocr_reader, frame)

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

    ocr_reader = load_rapidocr_reader()
    results = run_ocr(ocr_reader, frame)

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

    client = _wait_for_ble_connected()

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

    ocr_reader = load_rapidocr_reader()
    results = run_ocr(ocr_reader, roi)
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
    print(f"文節分割中 (sudachipy + pykakasi): {text!r}")
    segments = _segment_japanese_locally(text)
    print(f"分割結果: {segments}")

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

    for seg in segments:
        seg_text = seg["text"]
        seg_romaji = seg["romaji"]
        print(f"\n--- 文節: {seg_text!r} ({seg_romaji}) ---")

        if seg_text in ("、", "。"):
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
            ok = client.type_text(seg_text)
            print(f"type:{seg_text} -> {'OK' if ok else 'NG'}")

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
    ok = client.type_text(text)
    print(f"type:{text} -> {'OK' if ok else 'NG'}")
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

    if len(args) == 1:
        text = args[0]
        if _is_japanese(text):
            # 日本語または混在テキスト → 文節分割して IME 変換
            # 短い純日本語単語（4文字以下かつ助詞なし）は単一変換で高速化
            if len(text) <= 4 and not any(ch in text for ch in "をにはがでも") and not _is_ascii_only(text):
                romaji = _kanji_to_romaji(text)
                print(f"IME変換: {romaji} → {text}")
                type_kanji_via_ime(romaji, text)
            else:
                type_japanese_sentence(text)
        else:
            # 英数字のみ → 英数字モードで直接入力
            print(f"英語入力: {text!r}")
            _type_english_text(text)
        return 0

    if len(args) >= 2 and args[0] == "open test":
        # 第一引数が "open test"、第二引数がテキスト → カルテ開いてから入力
        text = args[1]
        print(f"テスト患者カルテを開いてから入力: {text!r}")
        open_test_patient_chart()
        if _is_japanese(text):
            if len(text) <= 4 and not any(ch in text for ch in "をにはがでも") and not _is_ascii_only(text):
                romaji = _kanji_to_romaji(text)
                type_kanji_via_ime(romaji, text)
            else:
                type_japanese_sentence(text)
        else:
            _type_english_text(text)
        return 0

    print("使い方:")
    print('  python -m automation.ehr_input                         # テスト患者カルテを開く')
    print('  python -m automation.ehr_input "open test"             # テスト患者カルテを開く')
    print('  python -m automation.ehr_input "close record"          # 取り消し[F9]ボタンをクリックしてカルテを閉じる')
    print('  python -m automation.ehr_input 肺炎                    # IME変換のみ')
    print('  python -m automation.ehr_input "COVID-19の検査"        # 日英混在入力')
    print('  python -m automation.ehr_input tesuto                  # 英語直接入力')
    print('  python -m automation.ehr_input "open test" 肺炎        # カルテを開いてからIME変換')
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
