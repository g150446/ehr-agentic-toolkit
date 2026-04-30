"""
過去カルテ領域のテキストを VLM で読み取るツール。

HDMI キャプチャで取得した患者カルテ画面を解析し、
左から2番目の「過去カルテ」領域を切り出して VLM に渡し、
記載内容を構造化（JSON）して標準出力に表示する。

実行方法:
  python -m automation.ehr_reader
  python -m automation.ehr_reader --omlx
  python -m automation.ehr_reader --omlx gemma-4-26b-a4b-it-4bit
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
    _cluster_x_positions,
    _encode_image_data_url,
    _find_gray_divider_candidates,
    _find_hough_divider_candidates,
    _select_divider_group,
    MLX_VLM_IME_API_KEY,
    MLX_VLM_IME_MODEL,
    MLX_VLM_IME_TIMEOUT,
    MLX_VLM_IME_URL,
)

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


def _detect_all_dividers(
    frame: np.ndarray,
    *,
    debug: bool = False,
) -> Optional[list[int]]:
    """画面の4本の縦方向区切り線を検出する。

    Returns:
        検出された4本の x 座標（左から昇順）、または検出失敗時 None
    """
    h, w = frame.shape[:2]
    gray_candidates = _find_gray_divider_candidates(frame)
    hough_candidates = _find_hough_divider_candidates(frame)
    clustered = _cluster_x_positions(gray_candidates + hough_candidates)
    accepted = _select_divider_group(clustered, width=w)

    if debug and clustered:
        overlay = frame.copy()
        for x in clustered:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 215, 255), 1)
        if accepted:
            for x in accepted:
                cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        _save_debug_frame(overlay, "dividers")

    if not accepted:
        return None
    return accepted


def _extract_past_chart_region(
    frame: np.ndarray,
    dividers: list[int],
    *,
    debug: bool = False,
) -> np.ndarray:
    """過去カルテ領域（左から2番目のパネル）を切り出す。

    4本の区切り線 d[0]..d[3] に対し、
    パネル0: 0 〜 d[0]
    パネル1: d[0] 〜 d[1]  ← 過去カルテ（対象）
    パネル2: d[1] 〜 d[2]
    パネル3: d[2] 〜 d[3]
    パネル4: d[3] 〜 右端
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


def _parse_cli_options(args: list[str]) -> tuple[bool, Optional[str]]:
    """CLI オプションをパースする。"""
    omlx = False
    omlx_model: Optional[str] = None
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
        elif arg.startswith("--"):
            raise RuntimeError(f"不明なオプション: {arg}")
        else:
            filtered_args.append(arg)
        index += 1
    return omlx, omlx_model


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        omlx, omlx_model = _parse_cli_options(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not omlx:
        print("[ERROR] --omlx オプションが必要です", file=sys.stderr)
        print("使用例: python -m automation.ehr_reader --omlx", file=sys.stderr)
        return 1

    runtime = _build_runtime_config(omlx_model=omlx_model)
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
        print("[ERROR] 患者カルテ画面の区切り線（4本の縦線）を検出できませんでした", file=sys.stderr)
        return 1

    print(f"区切り線検出: x={dividers}")
    print(f"  パネル0 (左端): 0 〜 {dividers[0]}")
    print(f"  パネル1 (過去カルテ): {dividers[0]} 〜 {dividers[1]}")
    print(f"  パネル2: {dividers[1]} 〜 {dividers[2]}")
    print(f"  パネル3: {dividers[2]} 〜 {dividers[3]}")
    print(f"  パネル4 (右端): {dividers[3]} 〜 {frame.shape[1]}")

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
