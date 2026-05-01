"""
過去カルテ領域のテキストを VLM で読み取るツール。

HDMI キャプチャで取得した患者カルテ画面を解析し、
左から2番目の「過去カルテ」領域を切り出して VLM に渡し、
記載内容を構造化（JSON）して標準出力に表示する。

実行方法:
  python -m automation.ehr_reader --omlx
  python -m automation.ehr_reader --omlx gemma-4-26b-a4b-it-4bit
  python -m automation.ehr_reader --omlx --scroll        # 読み取り後にスクロール (デフォルト3回)
  python -m automation.ehr_reader --omlx --scroll 5      # スクロール回数指定
  python -m automation.ehr_reader --scroll-only           # スクロールのみテスト (デフォルト3回)
  python -m automation.ehr_reader --scroll-only 5         # スクロール回数指定
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from automation.config import load_config
from automation.screen_analyzer import capture_screen as _capture_screen_hdmi
from automation.mlx_vlm_ime import (
    _encode_image_data_url,
    _find_gray_divider_candidates,
    MLX_VLM_IME_API_KEY,
    MLX_VLM_IME_MODEL,
    MLX_VLM_IME_TIMEOUT,
    MLX_VLM_IME_URL,
)
from automation.ble_client import BLEClient

_OMLX_CHAT_URL = "http://localhost:8000/v1/chat/completions"
_OMLX_DEFAULT_MODEL = "gemma-4-26b-a4b-it-4bit"
_OMLX_API_KEY = "penguin"

_CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def _save_debug_frame(frame: np.ndarray, name: str) -> str:
    """デバッグ用フレームを captures/ に保存する。"""
    _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _CAPTURES_DIR / f"ehr_reader_{name}_{ts}.png"
    cv2.imwrite(str(path), frame)
    print(f"  [debug] 保存: {path}")
    return str(path)


def _measure_divider_thicknesses(
    frame: np.ndarray,
    candidates: list[int],
) -> list[tuple[int, int]]:
    """各候補線の太さを測定し、(x座標, 太さ) のリストを返す。"""
    h, w = frame.shape[:2]
    y1 = int(h * 0.05)
    y2 = int(h * 0.95)
    band = frame[y1:y2, :]
    b = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    r = band[:, :, 2].astype(np.int16)
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    value = ((b + g + r) / 3.0)
    mask = ((spread <= 14) & (value >= 120) & (value <= 235)).astype(np.uint8) * 255

    result = []
    for x in candidates:
        left = max(0, x - 15)
        right = min(w, x + 15)
        col_strength = np.count_nonzero(mask[:, left:right], axis=0)
        threshold = max(int((y2 - y1) * 0.45), 40)
        active = col_strength >= threshold
        if active.any():
            indices = np.where(active)[0]
            thickness = int(indices[-1] - indices[0] + 1)
        else:
            thickness = 0
        result.append((x, thickness))
    return result


def _select_thick_gray_dividers(
    frame: np.ndarray,
    *,
    num_lines: int = 3,
    min_gap: int = 80,
) -> list[int]:
    """太いグレーの縦線を優先して num_lines 本を選ぶ。"""
    gray_candidates = _find_gray_divider_candidates(frame)
    if not gray_candidates:
        return []

    thicknesses = _measure_divider_thicknesses(frame, gray_candidates)
    thicknesses.sort(key=lambda t: t[1], reverse=True)

    selected = []
    for x, thickness in thicknesses:
        if thickness == 0:
            continue
        if all(abs(x - sx) >= min_gap for sx in selected):
            selected.append(x)
        if len(selected) >= num_lines:
            break

    return sorted(selected)


def _detect_all_dividers(
    frame: np.ndarray,
    *,
    debug: bool = False,
) -> Optional[list[int]]:
    """画面の太いグレーの縦線を3本検出する。

    Returns:
        検出された3本の x 座標（左から昇順）、または検出失敗時 None
    """
    h, w = frame.shape[:2]
    accepted = _select_thick_gray_dividers(frame, num_lines=3, min_gap=80)

    if debug:
        overlay = frame.copy()
        gray_candidates = _find_gray_divider_candidates(frame)
        for x in gray_candidates:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 215, 255), 1)
        if accepted:
            for x in accepted:
                cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        _save_debug_frame(overlay, "dividers")

    if len(accepted) < 3:
        return None
    return accepted


def _extract_past_chart_region(
    frame: np.ndarray,
    dividers: list[int],
    *,
    debug: bool = False,
) -> np.ndarray:
    """過去カルテ領域（左から2番目のパネル）を切り出す。

    3本の区切り線 d[0], d[1], d[2] に対し、
    パネル0: 0 〜 d[0]
    パネル1: d[0] 〜 d[1]  ← 過去カルテ（対象）
    パネル2: d[1] 〜 d[2]
    パネル3: d[2] 〜 右端
    """
    x_start = dividers[0]
    x_end = dividers[1]
    cropped = frame[:, x_start:x_end]

    if debug:
        overlay = frame.copy()
        h = overlay.shape[0]
        for x in dividers:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        cv2.rectangle(overlay, (x_start, 0), (x_end, h - 1), (0, 0, 255), 2)
        _save_debug_frame(overlay, "past_chart_region")
        _save_debug_frame(cropped, "past_chart_crop")

    return cropped


def _read_past_chart_with_vlm(
    cropped: np.ndarray,
    *,
    model: str,
    url: str,
    api_key: str,
    timeout: float,
) -> str:
    """VLM に過去カルテ領域の画像を渡してテキストを読み取る。

    max_tokens を 4096 に設定し、長いカルテ内容でも途中で切れないようにする。
    """
    data_url = _encode_image_data_url(cropped, debug_name="vlm_past_chart")

    prompt = (
        "### 指示\n"
        "添付された画像は電子カルテシステムの「過去カルテ」領域のスクリーンショットです。\n"
        "画像の内容を読み取り、日付ごとに整理された診療録データ（JSON形式）を作成してください。\n\n"
        "### 処理のガイドライン\n"
        "1. **情報の抽出**:\n"
        "   - 画像から「日付」と、それに対応する診療録の本文をすべて抽出してください。\n"
        "   - 本文は[S][O][A][P]などの記号も含め、改行を維持したまま一つのテキストブロックとして扱ってください。\n"
        "   - 日付が明記されていないセクションがある場合、同じ画像内や前後の文脈から正しい日付を特定して紐付けてください。\n\n"
        "2. **画像間の重複対応**:\n"
        "   - 複数の画像間で内容が重複している場合、それらを二重に登録せず、一つの自然な文章として統合してください。\n\n"
        "3. **不要な情報の除外**:\n"
        "   - ヘッダー、ナビゲーションバー、スクロールバー、ページネーション、タブ名などのUI要素は除外してください。\n"
        "   - 過去カルテの診療録本文以外のテキスト（画面タイトル、メニュー項目、ボタン名など）は含めないでください。\n"
        "   - システム管理用テキスト（例：「deletedAt=N0」「isDeleted=false」など）は除外してください。\n\n"
        "4. **出力フォーマット**:\n"
        "   - 必ず以下の構造のJSON形式のみを出力してください。\n"
        "[\n"
        "  {\n"
        '    "date": "YYYY年MM月DD日(曜日)",\n'
        '    "content": "抽出された本文テキスト（改行を含む）"\n'
        "  }\n"
        "]\n\n"
        "### 出力\n"
        "抽出・統合が完了したJSONデータのみを出力してください。"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 4096,
    }

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())

    return result["choices"][0]["message"]["content"]


def _parse_vlm_response(raw: str) -> list[dict]:
    """VLM の応答から JSON 配列を抽出・パースする。"""
    content = re.sub(r"```json\s*", "", raw, flags=re.DOTALL).strip()
    content = re.sub(r"```\s*$", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"VLM 応答から JSON 配列を抽出できませんでした: {raw!r}")

    json_str = content[start:end]
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"VLM 応答の JSON を解析できませんでした: {json_str!r}") from exc

    if not isinstance(result, list):
        raise ValueError(f"VLM 応答が配列ではありません: {result!r}")
    return result


def _wait_for_ble_connected(timeout: float = 70.0) -> BLEClient:
    """BLE サーバーが起動して BLE デバイスへ接続済みになるまで待機する。"""
    client = BLEClient()
    if client.is_server_running():
        return client

    print(f"BLE 未接続。最大 {timeout:.0f} 秒待機します...")
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


def _scroll_past_chart_down(
    *,
    dividers: list[int],
    screen_width: int,
    screen_height: int,
    scroll_count: int = 1,
) -> None:
    """過去カルテ領域の右下をクリックし、指定回数だけ上にスクロールする。

    基本スクロール量（15単位）を scroll_count 回、0.5秒間隔で送信する。
    """
    client = _wait_for_ble_connected()

    # 右下のクリック座標（端から少し内側、左に50px・上に100pxずらす）
    click_x = dividers[1] - 70
    click_y = screen_height - 150

    # マウスモードに切り替え
    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    # クリック位置へ移動
    ok = client.move_mouse_to_position(click_x, click_y)
    print(f"moveto ({click_x}, {click_y}) -> {'OK' if ok else 'NG'}")

    # クリックしてフォーカスを過去カルテ領域に合わせる
    ok = client.click()
    print(f"click (past chart focus) -> {'OK' if ok else 'NG'}")
    print(f"過去カルテ領域右下をクリック: ({click_x}, {click_y})")

    base_scroll = 1
    for i in range(scroll_count):
        ok = client.scroll(-base_scroll)
        print(f"scroll:-{base_scroll} ({i + 1}/{scroll_count}) -> {'OK' if ok else 'NG'}")
        if i < scroll_count - 1:
            time.sleep(0.5)
    print(f"スクロール: {base_scroll} 単位 × {scroll_count} 回（上方向）")


def _build_runtime_config(
    *,
    omlx_model: Optional[str] = None,
) -> dict[str, str]:
    """VLM ランタイム設定を構築する。"""
    model = omlx_model or _OMLX_DEFAULT_MODEL
    return {
        "url": _OMLX_CHAT_URL,
        "model": model,
        "api_key": _OMLX_API_KEY,
    }


def _parse_cli_options(args: list[str]) -> tuple[bool, Optional[str], bool, int, bool]:
    """CLI オプションをパースする。"""
    omlx = False
    omlx_model: Optional[str] = None
    do_scroll = False
    scroll_count = 1
    scroll_only = False
    filtered_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--omlx":
            omlx = True
            next_index = index + 1
            if next_index < len(args) and not args[next_index].startswith("--"):
                omlx_model = args[next_index]
                index += 1
            else:
                omlx_model = _OMLX_DEFAULT_MODEL
        elif arg.startswith("--omlx="):
            omlx = True
            _, _, omlx_model = arg.partition("=")
            if not omlx_model:
                omlx_model = _OMLX_DEFAULT_MODEL
        elif arg == "--scroll":
            do_scroll = True
            next_index = index + 1
            if next_index < len(args) and not args[next_index].startswith("--"):
                try:
                    scroll_count = int(args[next_index])
                    index += 1
                except ValueError:
                    scroll_count = 3
            else:
                scroll_count = 3
        elif arg.startswith("--scroll="):
            do_scroll = True
            _, _, val = arg.partition("=")
            if val:
                try:
                    scroll_count = int(val)
                except ValueError:
                    scroll_count = 3
            else:
                scroll_count = 3
        elif arg == "--scroll-only":
            scroll_only = True
            next_index = index + 1
            if next_index < len(args) and not args[next_index].startswith("--"):
                try:
                    scroll_count = int(args[next_index])
                    index += 1
                except ValueError:
                    scroll_count = 3
            else:
                scroll_count = 3
        elif arg.startswith("--scroll-only="):
            scroll_only = True
            _, _, val = arg.partition("=")
            if val:
                try:
                    scroll_count = int(val)
                except ValueError:
                    scroll_count = 3
            else:
                scroll_count = 3
        elif arg.startswith("--"):
            raise RuntimeError(f"不明なオプション: {arg}")
        else:
            filtered_args.append(arg)
        index += 1
    return omlx, omlx_model, do_scroll, scroll_count, scroll_only


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        omlx, omlx_model, do_scroll, scroll_count, scroll_only = _parse_cli_options(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not omlx and not scroll_only:
        print("[ERROR] --omlx または --scroll-only のいずれかのオプションが必要です", file=sys.stderr)
        print("使用例: python -m automation.ehr_reader --omlx", file=sys.stderr)
        print("       python -m automation.ehr_reader --scroll-only", file=sys.stderr)
        return 1

    runtime = _build_runtime_config(omlx_model=omlx_model) if omlx else {"model": "", "url": "", "api_key": ""}
    if omlx:
        print(f"VLM ランタイム: {runtime['model']} ({runtime['url']})")

    config = load_config(skip_password=True)
    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = _capture_screen_hdmi(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        print("[ERROR] HDMIキャプチャデバイスからフレームを取得できませんでした", file=sys.stderr)
        return 1

    _save_debug_frame(frame, "full_screen")
    print("画面を解析中...")

    dividers = _detect_all_dividers(frame, debug=True)
    if dividers is None:
        print("[ERROR] 患者カルテ画面の区切り線（太いグレーの縦線3本）を検出できませんでした", file=sys.stderr)
        return 1

    print(f"区切り線検出: x={dividers}")
    print(f"  パネル0 (左端): 0 〜 {dividers[0]}")
    print(f"  パネル1 (過去カルテ): {dividers[0]} 〜 {dividers[1]}")
    print(f"  パネル2: {dividers[1]} 〜 {dividers[2]}")
    print(f"  パネル3 (右端): {dividers[2]} 〜 {frame.shape[1]}")

    if scroll_only:
        print(f"\nスクロールのみモード: 1 単位 × {scroll_count} 回（上方向）")
        _scroll_past_chart_down(
            dividers=dividers,
            screen_width=config.capture_width,
            screen_height=config.capture_height,
            scroll_count=scroll_count,
        )
        return 0

    print("\n過去カルテ領域を切り出し中...")
    past_chart = _extract_past_chart_region(frame, dividers, debug=True)
    print(f"切り出しサイズ: {past_chart.shape[1]}x{past_chart.shape[0]} px")

    print("\nVLM で過去カルテの内容を読み取り中...")
    raw_response = _read_past_chart_with_vlm(
        past_chart,
        model=runtime["model"],
        url=runtime["url"],
        api_key=runtime["api_key"],
        timeout=MLX_VLM_IME_TIMEOUT,
    )

    print(f"\n--- VLM 生応答 ---\n{raw_response}\n--- 終了 ---\n")

    try:
        structured = _parse_vlm_response(raw_response)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("--- 構造化出力 (JSON) ---")
    print(json.dumps(structured, ensure_ascii=False, indent=2))
    print("--- 終了 ---")

    if structured:
        print(f"\n過去カルテ {len(structured)} 件を読み取りました。")
    else:
        print("\n過去カルテにテキストが見つかりませんでした。")

    if do_scroll:
        print(f"\n過去カルテ領域をスクロール中...")
        _scroll_past_chart_down(
            dividers=dividers,
            screen_width=config.capture_width,
            screen_height=config.capture_height,
            scroll_count=scroll_count,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
