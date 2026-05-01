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
  python -m automation.ehr_reader --letter                # letter_icon.png を検出してクリック
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
from automation.screen_analyzer import capture_screen as _capture_screen_hdmi, load_ocr_reader, run_ocr_word_split
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


def _extract_ocr_text(image: np.ndarray) -> str:
    """EasyOCRで画像からテキストを抽出し、整形して返す。"""
    reader = load_ocr_reader(languages=['ja', 'en'], use_gpu=False)
    results = run_ocr_word_split(reader, image)
    
    # Y座標でソートして上から下に並べる
    lines = []
    for bbox, text, conf in results:
        if conf < 0.3:  # 低信頼度はスキップ
            continue
        y = min(p[1] for p in bbox)
        lines.append((y, text))
    
    lines.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in lines)


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
    *,
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
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
    mask = ((spread <= spread_max) & (value >= value_min) & (value <= 235)).astype(np.uint8) * 255

    result = []
    for x in candidates:
        left = max(0, x - 15)
        right = min(w, x + 15)
        col_strength = np.count_nonzero(mask[:, left:right], axis=0)
        threshold = max(int((y2 - y1) * coverage_ratio), 40)
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
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
) -> list[int]:
    """太いグレーの縦線を優先して num_lines 本を選ぶ。"""
    gray_candidates = _find_gray_divider_candidates(
        frame,
        spread_max=spread_max,
        value_min=value_min,
        coverage_ratio=coverage_ratio,
    )
    if not gray_candidates:
        return []

    thicknesses = _measure_divider_thicknesses(
        frame,
        gray_candidates,
        spread_max=spread_max,
        value_min=value_min,
        coverage_ratio=coverage_ratio,
    )
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


def _try_detect_dividers(
    frame: np.ndarray,
    *,
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
    debug: bool = False,
) -> Optional[list[int]]:
    """指定パラメータで区切り線を検出し、3本見つかれば座標を返す。"""
    accepted = _select_thick_gray_dividers(
        frame,
        num_lines=3,
        min_gap=80,
        spread_max=spread_max,
        value_min=value_min,
        coverage_ratio=coverage_ratio,
    )

    if debug:
        h = frame.shape[0]
        overlay = frame.copy()
        gray_candidates = _find_gray_divider_candidates(
            frame,
            spread_max=spread_max,
            value_min=value_min,
            coverage_ratio=coverage_ratio,
        )
        for x in gray_candidates:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 215, 255), 1)
        if accepted:
            for x in accepted:
                cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        _save_debug_frame(overlay, "dividers")

    if len(accepted) >= 3:
        return accepted
    return None


def _detect_all_dividers(
    frame: np.ndarray,
    *,
    debug: bool = False,
) -> Optional[list[int]]:
    """画面の太いグレーの縦線を3本検出する（2段階フォールバック）。

    Returns:
        検出された3本の x 座標（左から昇順）、または検出失敗時 None
    """
    # 第1段階: 通常パラメータ
    print("  区切り線検出 第1段階（通常パラメータ: spread≤14, value≥120, カバレッジ45%）...")
    accepted = _try_detect_dividers(
        frame, spread_max=14, value_min=120, coverage_ratio=0.45, debug=debug,
    )
    if accepted is not None:
        print(f"    第1段階で成功: {accepted[:3]}")
        return accepted[:3]

    # 第2段階: 緩和パラメータ
    print("  区切り線検出 第2段階（緩和パラメータ: spread≤25, value≥100, カバレッジ30%）...")
    accepted = _try_detect_dividers(
        frame, spread_max=25, value_min=100, coverage_ratio=0.30, debug=debug,
    )
    if accepted is not None:
        print(f"    第2段階で成功: {accepted[:3]}")
        return accepted[:3]

    print("    検出失敗")
    return None


def _is_frame_unchanged(prev: np.ndarray, curr: np.ndarray, *, diff_threshold: int = 15, max_diff_ratio: float = 0.005) -> bool:
    """2フレームを比較し、変化が無視できるレベルなら True を返す。"""
    if prev.shape != curr.shape:
        return False
    diff = cv2.absdiff(prev, curr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    significant_pixels = np.count_nonzero(gray > diff_threshold)
    total_pixels = gray.size
    ratio = significant_pixels / total_pixels
    print(f"    画面変化率: {ratio*100:.3f}% ({significant_pixels}/{total_pixels} 画素)")
    return ratio < max_diff_ratio


def _detect_edit_button_bottom(
    frame: np.ndarray,
    x_start: int,
    x_end: int,
    *,
    threshold: float = 0.7,
) -> int | None:
    """過去カルテパネル内の edit_button.jpg（ペンアイコン）を検出し、下側一致位置の下端 y 座標を返す。"""
    template_path = (
        Path(__file__).resolve().parent.parent
        / "match_templates"
        / "edit_button.jpg"
    )
    if not template_path.exists():
        print(f"[WARNING] テンプレート画像が見つかりません: {template_path}")
        return None

    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        print(f"[WARNING] テンプレート画像の読み込みに失敗しました: {template_path}")
        return None

    roi = frame[:, x_start:x_end]
    if roi.size == 0:
        return None

    # BGR でテンプレートマッチング
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]

    # threshold 以上の全ピークを取得
    loc = np.where(result >= threshold)
    points = list(zip(*loc[::-1]))  # (x, y)

    if not points:
        return None

    # 下側（y が最大）の一致位置を選ぶ
    best_x, best_y = max(points, key=lambda p: p[1])
    bottom_y = best_y + h

    print(f"  edit_button 検出: ({best_x + x_start}, {best_y})〜({best_x + x_start + w}, {bottom_y}) (score={result[best_y, best_x]:.3f})")
    return bottom_y


def _find_letter_icon(
    frame: np.ndarray,
    *,
    threshold: float = 0.7,
) -> tuple[int, int] | None:
    """画面全体から letter_icon.png を検出し、一致矩形の中心座標 (x, y) を返す。"""
    template_path = (
        Path(__file__).resolve().parent.parent
        / "match_templates"
        / "letter_icon.png"
    )
    if not template_path.exists():
        print(f"[WARNING] テンプレート画像が見つかりません: {template_path}")
        return None

    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"[WARNING] テンプレート画像の読み込みに失敗しました: {template_path}")
        return None

    # アルファチャンネルがある場合は透過背景を白に置き換えて BGR に変換
    if template.shape[2] == 4:
        b, g, r, a = cv2.split(template)
        alpha = a.astype(np.float32) / 255.0
        white_bg = np.full_like(b, 255, dtype=np.uint8)
        b = (b.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        template = cv2.merge([b, g, r])

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        print(f"  letter_icon 未検出 (最高スコア {max_val:.3f} < {threshold})")
        return None

    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2
    top_y = max_loc[1]
    print(f"  letter_icon 検出: ({max_loc[0]}, {max_loc[1]})〜({max_loc[0] + w}, {max_loc[1] + h}) 中心=({cx}, {cy}) (score={max_val:.3f})")
    return (cx, cy, top_y)


def _click_letter_icon(
    click_x: int,
    click_y: int,
) -> bool:
    """letter_icon の中心座標にマウスを移動してクリックする。"""
    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(click_x, click_y)
    print(f"moveto ({click_x}, {click_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click (letter_icon) -> {'OK' if ok else 'NG'}")
    return ok


def _find_text_position_ocr(
    panel2: np.ndarray,
    x_offset: int,
    search_text: str = "退院時要約",
    *,
    min_confidence: float = 0.3,
) -> tuple[int, int] | None:
    """パネル2画像を EasyOCR で認識し、指定テキストの画面全体座標の中心を返す。

    search_text の部分一致でテキストを検索する。
    検出できなければ None を返す。
    """
    reader = load_ocr_reader(languages=['ja', 'en'], use_gpu=False)
    results = run_ocr_word_split(reader, panel2)

    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        if search_text in text:
            # bbox は [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            # 中心座標を計算し、画面全体座標に変換
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = x_offset + int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            print(f"  OCR '{search_text}' 検出: bbox中心=({cx}, {cy}) text='{text}' conf={conf:.2f}")
            return (cx, cy)

    print(f"  OCR '{search_text}' は見つかりませんでした")
    return None


def _find_target_y_by_ocr(
    frame: np.ndarray,
    keywords: list[str] = None,
    *,
    min_confidence: float = 0.3,
) -> int | None:
    """画面全体を EasyOCR で認識し、指定キーワードの y 座標（中心）を返す。

    keywords は優先順位順。先頭のキーワードから順に検索する。
    検出できなければ None を返す。
    """
    if keywords is None:
        keywords = ["担当医", "医師名"]

    reader = load_ocr_reader(languages=['ja', 'en'], use_gpu=False)
    results = run_ocr_word_split(reader, frame)

    for keyword in keywords:
        for bbox, text, conf in results:
            if conf < min_confidence:
                continue
            if keyword in text:
                ys = [p[1] for p in bbox]
                cy = int(sum(ys) / len(ys))
                print(f"  OCR '{keyword}' 検出: y={cy} text='{text}' conf={conf:.2f}")
                return cy

    print(f"  OCR {keywords} はいずれも見つかりませんでした")
    return None


def _find_word_return_mark_x(
    frame: np.ndarray,
    *,
    threshold: float = 0.7,
) -> int | None:
    """画面全体から word_return_mark.jpg を検出し、マッチ位置のうち x 座標が最も大きい矩形の中心 x 座標を返す。"""
    template_path = (
        Path(__file__).resolve().parent.parent
        / "match_templates"
        / "word_return_mark.jpg"
    )
    if not template_path.exists():
        print(f"[WARNING] テンプレート画像が見つかりません: {template_path}")
        return None

    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        print(f"[WARNING] テンプレート画像の読み込みに失敗しました: {template_path}")
        return None

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]

    # threshold 以上の全ピークを取得
    loc = np.where(result >= threshold)
    points = list(zip(*loc[::-1]))  # [(x, y), ...]

    if not points:
        print(f"  word_return_mark 未検出 (最高スコア {cv2.minMaxLoc(result)[1]:.3f} < {threshold})")
        return None

    # x 座標が最大のものを選ぶ（最も右側）
    best_x, best_y = max(points, key=lambda p: p[0])
    cx = int(best_x + w // 2)
    print(f"  word_return_mark 検出（最右）: ({best_x}, {best_y})〜({best_x + w}, {best_y + h}) 中心x={cx} (score={result[best_y, best_x]:.3f}, total={len(points)})")
    return cx


def _find_word_return_mark_bottom(
    frame: np.ndarray,
    screen_width: int,
    *,
    threshold: float = 0.7,
) -> tuple[int, int] | None:
    """画面全体から word_return_mark.jpg を検出し、y 座標が最も大きい矩形の中心座標を返す。

    検出範囲は x 座標が screen_width * 1/4 から screen_width * 3/4 の間に限定する。
    """
    template_path = (
        Path(__file__).resolve().parent.parent
        / "match_templates"
        / "word_return_mark.jpg"
    )
    if not template_path.exists():
        print(f"[WARNING] テンプレート画像が見つかりません: {template_path}")
        return None

    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template is None:
        print(f"[WARNING] テンプレート画像の読み込みに失敗しました: {template_path}")
        return None

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]

    # threshold 以上の全ピークを取得
    loc = np.where(result >= threshold)
    points = list(zip(*loc[::-1]))  # [(x, y), ...]

    if not points:
        print(f"  word_return_mark 未検出 (最高スコア {cv2.minMaxLoc(result)[1]:.3f} < {threshold})")
        return None

    # x 座標が screen_width * 1/4 から screen_width * 3/4 の範囲内のみを抽出
    x_min = screen_width // 4
    x_max = screen_width * 3 // 4
    filtered = [(x, y) for x, y in points if x_min <= x <= x_max]

    if not filtered:
        print(f"  word_return_mark 未検出 (範囲内: {x_min}〜{x_max} に {len(points)} 件中 0 件)")
        return None

    # y 座標が最大のものを選ぶ（最も下側）
    best_x, best_y = max(filtered, key=lambda p: p[1])
    cx = int(best_x + w // 2)
    cy = int(best_y + h // 2 + 10)
    print(f"  word_return_mark 検出（最下）: ({best_x}, {best_y})〜({best_x + w}, {best_y + h}) 中心=({cx}, {cy}) (score={result[best_y, best_x]:.3f}, total={len(points)}, filtered={len(filtered)})")
    return (cx, cy)


def _save_ocr_overlay(
    panel2: np.ndarray,
    x_offset: int,
    search_text: str = "退院時要約",
    *,
    min_confidence: float = 0.3,
) -> str:
    """OCR結果をバウンディングボックス+テキストラベルでオーバーレイした画像を captures/ に保存する。"""
    reader = load_ocr_reader(languages=['ja', 'en'], use_gpu=False)
    results = run_ocr_word_split(reader, panel2)

    overlay = panel2.copy()
    h, w = overlay.shape[:2]

    # フォントサイズを画像サイズに応じて調整
    font_scale = max(0.4, min(w, h) / 1500)
    thickness = max(1, int(min(w, h) / 500))

    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
        # search_text を含む場合は赤、それ以外は緑
        color = (0, 0, 255) if search_text in text else (0, 255, 0)
        cv2.polylines(overlay, [pts], True, color, thickness)

        # テキストラベルを左上に描画
        label = f"{text} ({conf:.2f})"
        x, y = int(bbox[0][0]), int(bbox[0][1])
        # 背景矩形
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 4, y), color, -1)
        cv2.putText(overlay, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    _CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _CAPTURES_DIR / f"ehr_reader_ocr_overlay_{ts}.png"
    cv2.imwrite(str(path), overlay)
    print(f"  [debug] OCRオーバーレイ保存: {path}")
    return str(path)


def _save_ocr_text_log(
    panel2: np.ndarray,
    *,
    min_confidence: float = 0.3,
) -> str:
    """OCRテキストを logs/ に保存する。"""
    reader = load_ocr_reader(languages=['ja', 'en'], use_gpu=False)
    results = run_ocr_word_split(reader, panel2)

    lines = []
    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        y = min(p[1] for p in bbox)
        lines.append((y, text, conf))

    lines.sort(key=lambda x: x[0])

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _LOGS_DIR / f"ehr_reader_ocr_{ts}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for _, text, conf in lines:
            f.write(f"[{conf:.2f}] {text}\n")

    print(f"  [debug] OCRテキストログ保存: {path}")
    return str(path)


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

    edit_button.jpg（ペンアイコン）を検出し、その下端より下の領域のみを対象とする。
    """
    x_start = dividers[0]
    x_end = dividers[1]

    y_start = _detect_edit_button_bottom(frame, x_start, x_end)
    if y_start is None:
        raise RuntimeError(
            "過去カルテ領域内に edit_button（ペンアイコン）を検出できませんでした。"
        )

    y_start = max(0, y_start)
    cropped = frame[y_start:, x_start:x_end]

    if debug:
        overlay = frame.copy()
        h = overlay.shape[0]
        for x in dividers:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        cv2.rectangle(overlay, (x_start, y_start), (x_end, h - 1), (0, 0, 255), 2)
        _save_debug_frame(overlay, "past_chart_region")
        _save_debug_frame(cropped, "past_chart_crop")

    return cropped


def _read_past_chart_with_vlm(
    cropped: np.ndarray,
    ocr_text: str,
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
        "### EasyOCR認識結果（参考）\n"
        "以下は画像からEasyOCRで抽出したテキストです。レイアウト情報は失われている可能性があるため、画像も併せて確認してください。\n"
        "```\n"
        f"{ocr_text}\n"
        "```\n\n"
        "### 処理のガイドライン\n"
        "1. **情報の抽出（すべての医療情報を漏らさず）**:\n"
        "   - 画像とOCR結果の両方を参考にし、「日付」とそれに対応する診療録の本文を「すべて」抽出してください。\n"
        "   - [S][O][A][P]の各セクションに加え、『血液検査』『生化学検査』『尿検査』『画像検査』『胸部X線』『動脈血ガス』など、明示的にラベル付けされたすべての検査結果・所見を抽出してください。\n"
        "   - 『自由』『自由記載』などのラベルが付いた欄の内容も、診療録の一部として漏らさず抽出してください。\n"
        "   - Vitals、身体所見の間や前後に記載されている検査数値・結果（WBC、CRP、Hbなど）も見逃さずに抽出してください。\n"
        "   - 処方内容、投与薬の詳細（薬剤名・容量・単位・頻度・投与方法など）など、画像内にある「すべての医療情報」を漏らさず含めてください。\n"
        "   - テキストは画像に書かれているものをそのまま、改行を維持して抽出してください。要約や再構成は絶対に行わないでください。\n"
        "   - 画像に存在しない情報は、一切追加・補完・推測しないでください（ハルシネーション防止）。\n"
        "   - 日付が明記されていないセクションがある場合、同じ画像内や前後の文脈から正しい日付を特定して紐付けてください。\n\n"
        "2. **出力フォーマット**:\n"
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


def _read_past_chart_with_vlm_merge(
    cropped: np.ndarray,
    current_json: list[dict],
    ocr_text: str,
    *,
    model: str,
    url: str,
    api_key: str,
    timeout: float,
) -> str:
    """VLM に新しい画像と現在のJSONを渡して統合した診療録データを作成する。"""
    data_url = _encode_image_data_url(cropped, debug_name="vlm_past_chart_merge")

    current_json_str = json.dumps(current_json, ensure_ascii=False, indent=2)

    prompt = (
        "### 指示\n"
        "添付された「新しい画像」の内容を読み取り、以下の【現在のJSONデータ】と統合して、最新の診療録データを作成してください。\n\n"
        "### EasyOCR認識結果（参考）\n"
        "以下は新しい画像からEasyOCRで抽出したテキストです。レイアウト情報は失われている可能性があるため、画像も併せて確認してください。\n"
        "```\n"
        f"{ocr_text}\n"
        "```\n\n"
        "### 現在のJSONデータ\n"
        f"{current_json_str}\n\n"
        "### 統合のルール\n"
        "1. **既存データの保護（絶対に改変しない）**:\n"
        "   - 【現在のJSONデータ】の各 `content` に含まれるテキストは、絶対に改変・要約・再構成・削除しないでください。\n"
        "   - 既存の `content` はそのまま保持し、画像から得られた「追加情報」や「続きの文章」のみを追記・統合してください。\n"
        "   - 既存の文章を短くしたり、別の表現に言い換えたりしないでください。\n\n"
        "2. **内容の同期と結合（すべての医療情報を漏らさず）**:\n"
        "   - 画像とOCR結果の両方を参考にし、日付を確認してください。すでにJSON内にその日付が存在する場合は、画像から得られた情報を既存の `content` の末尾や適切な位置に追記してください。\n"
        "   - [S][O][A][P]の各セクションに加え、『血液検査』『生化学検査』『尿検査』『画像検査』『胸部X線』『動脈血ガス』など、明示的にラベル付けされたすべての検査結果・所見を抽出してください。\n"
        "   - 『自由』『自由記載』などのラベルが付いた欄の内容も、診療録の一部として漏らさず抽出してください。\n"
        "   - Vitals、身体所見の間や前後に記載されている検査数値・結果（WBC、CRP、Hbなど）も見逃さずに抽出してください。\n"
        "   - 処方内容、投与薬の詳細（薬剤名・容量・単位・頻度・投与方法など）など、画像内にある「すべての医療情報」を漏らさず含めてください。\n"
        "   - テキストは画像に書かれているものをそのまま、改行を維持して抽出してください。要約や再構成は絶対に行わないでください。\n"
        "   - 文章が途切れている場合は、自然に繋がるように結合してください。\n\n"
        "3. **新規データの追加**:\n"
        "   - JSONに含まれていない新しい日付の診療録が画像内にある場合は、新しいオブジェクトとして末尾に追加してください。\n\n"
        "4. **ハルシネーション防止**:\n"
        "   - 画像に存在しない情報は、一切追加・補完・推測しないでください。\n"
        "   - 「および酸素療法」など、画像にないフレーズを勝手に挿入しないでください。\n"
        "   - 画像にない薬剤名、検査結果、処置内容は追加しないでください。\n\n"
        "5. **出力フォーマット**:\n"
        "   - 以下の構造を維持したJSON形式のみを出力してください。\n"
        "[\n"
        "  {\n"
        '    "date": "YYYY年MM月DD日(曜日)",\n'
        '    "content": "統合された本文テキスト"\n'
        "  }\n"
        "]\n\n"
        "### 出力\n"
        "統合が完了した最新のJSONデータのみを出力してください。"
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


def _parse_cli_options(args: list[str]) -> tuple[bool, Optional[str], bool, Optional[int], bool, bool, bool]:
    """CLI オプションをパースする。"""
    omlx = False
    omlx_model: Optional[str] = None
    do_scroll = False
    scroll_count: Optional[int] = None
    scroll_only = False
    do_letter = False
    do_summary = False
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
                    scroll_count = None
            else:
                scroll_count = None
        elif arg.startswith("--scroll="):
            do_scroll = True
            _, _, val = arg.partition("=")
            if val:
                try:
                    scroll_count = int(val)
                except ValueError:
                    scroll_count = None
            else:
                scroll_count = None
        elif arg == "--scroll-only":
            scroll_only = True
            next_index = index + 1
            if next_index < len(args) and not args[next_index].startswith("--"):
                try:
                    scroll_count = int(args[next_index])
                    index += 1
                except ValueError:
                    scroll_count = None
            else:
                scroll_count = None
        elif arg.startswith("--scroll-only="):
            scroll_only = True
            _, _, val = arg.partition("=")
            if val:
                try:
                    scroll_count = int(val)
                except ValueError:
                    scroll_count = None
            else:
                scroll_count = None
        elif arg == "--letter":
            do_letter = True
        elif arg == "--summary":
            do_summary = True
        elif arg.startswith("--"):
            raise RuntimeError(f"不明なオプション: {arg}")
        else:
            filtered_args.append(arg)
        index += 1
    return omlx, omlx_model, do_scroll, scroll_count, scroll_only, do_letter, do_summary


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        omlx, omlx_model, do_scroll, scroll_count, scroll_only, do_letter, do_summary = _parse_cli_options(args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if not omlx and not scroll_only and not do_letter and not do_summary:
        print("[ERROR] --omlx または --scroll-only または --letter または --summary のいずれかのオプションが必要です", file=sys.stderr)
        print("使用例: python -m automation.ehr_reader --omlx", file=sys.stderr)
        print("       python -m automation.ehr_reader --scroll-only", file=sys.stderr)
        print("       python -m automation.ehr_reader --letter", file=sys.stderr)
        print("       python -m automation.ehr_reader --summary", file=sys.stderr)
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

    if do_letter:
        print("\nletter_icon.png を画面全体から検索中...")
        letter_pos = _find_letter_icon(frame, threshold=0.7)
        if letter_pos is None:
            print("[ERROR] letter_icon.png が画面内に見つかりませんでした", file=sys.stderr)
            return 1
        click_x, click_y, _ = letter_pos
        print(f"letter_icon クリック: ({click_x}, {click_y})")
        _click_letter_icon(click_x, click_y)

        # ポップアップの位置を幾何学的に計算
        panel2_width = dividers[2] - dividers[1]
        popup_width = panel2_width // 3
        target_x = dividers[2] - popup_width // 2
        target_y = click_y  # letter_icon の中心 y と同じ

        print(f"ポップアップ上段メニュー中心を計算: ({target_x}, {target_y})")

        # カーソルを上段メニュー中心へ移動
        client = _wait_for_ble_connected()
        ok = client.switch_to_mouse_mode()
        print(f"mode:mouse -> {'OK' if ok else 'NG'}")
        ok = client.move_mouse_to_position(target_x, target_y)
        print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

        # 1秒待機後、新しいポップアップが表示されるのを待つ
        print("1.0秒待機（新ポップアップ表示待ち）...")
        time.sleep(1.0)

        # パネル2中央へ移動
        panel2_center_x = dividers[1] + panel2_width // 2
        panel2_center_y = target_y
        print(f"パネル2中央へ移動: ({panel2_center_x}, {panel2_center_y})")
        ok = client.move_mouse_to_position(panel2_center_x, panel2_center_y)
        print(f"moveto ({panel2_center_x}, {panel2_center_y}) -> {'OK' if ok else 'NG'}")

        return 0

    if do_summary:
        print("\nletter_icon.png を画面全体から検索中...")
        letter_pos = _find_letter_icon(frame, threshold=0.7)
        if letter_pos is None:
            print("[ERROR] letter_icon.png が画面内に見つかりませんでした", file=sys.stderr)
            return 1
        click_x, click_y, _ = letter_pos
        print(f"letter_icon クリック: ({click_x}, {click_y})")
        _click_letter_icon(click_x, click_y)

        # ポップアップの位置を幾何学的に計算
        panel2_width = dividers[2] - dividers[1]
        popup_width = panel2_width // 3
        target_x = dividers[2] - popup_width // 2
        target_y = click_y  # letter_icon の中心 y と同じ

        print(f"ポップアップ上段メニュー中心を計算: ({target_x}, {target_y})")

        # カーソルを上段メニュー中心へ移動
        client = _wait_for_ble_connected()
        ok = client.switch_to_mouse_mode()
        print(f"mode:mouse -> {'OK' if ok else 'NG'}")
        ok = client.move_mouse_to_position(target_x, target_y)
        print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

        # 1秒待機後、新しいポップアップが表示されるのを待つ
        print("1.0秒待機（新ポップアップ表示待ち）...")
        time.sleep(1.0)

        # パネル2中央へ移動
        panel2_center_x = dividers[1] + panel2_width // 2
        panel2_center_y = target_y
        print(f"パネル2中央へ移動: ({panel2_center_x}, {panel2_center_y})")
        ok = client.move_mouse_to_position(panel2_center_x, panel2_center_y)
        print(f"moveto ({panel2_center_x}, {panel2_center_y}) -> {'OK' if ok else 'NG'}")

        # 0.5秒待機後、ポップアップをOCRで認識
        print("0.5秒待機（ポップアップ安定）...")
        time.sleep(0.5)

        # クリック後の画面を再キャプチャ（ポップアップが表示された状態）
        print("ポップアップ表示後の画面をキャプチャ中...")
        popup_frame = _capture_screen_hdmi(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if popup_frame is None:
            print("[ERROR] ポップアップ表示後のキャプチャに失敗しました", file=sys.stderr)
            return 1
        _save_debug_frame(popup_frame, "full_screen_with_popup")

        # パネル2を切り出し
        x_start = dividers[1]
        x_end = dividers[2]
        panel2 = popup_frame[:, x_start:x_end]

        print("\nパネル2をEasyOCRで認識中...")
        # OCRテキストログ保存
        _save_ocr_text_log(panel2)
        # OCRオーバーレイ画像保存
        _save_ocr_overlay(panel2, x_offset=x_start, search_text="退院時要約")

        text_pos = _find_text_position_ocr(panel2, x_offset=x_start, search_text="退院時要約")
        if text_pos is None:
            print("[ERROR] '退院時要約' が見つかりませんでした", file=sys.stderr)
            return 1

        text_x, text_y = text_pos
        print(f"'退院時要約' へカーソル移動: ({text_x}, {text_y})")
        ok = client.move_mouse_to_position(text_x, text_y)
        print(f"moveto ({text_x}, {text_y}) -> {'OK' if ok else 'NG'}")

        # 0.5秒待機後にクリック
        print("0.5秒待機...")
        time.sleep(0.5)
        ok = client.click()
        print(f"click (退院時要約) -> {'OK' if ok else 'NG'}")

        # 5.0秒待機（Microsoft Word が開くのを待つ）
        print("5.0秒待機（Word起動待ち）...")
        time.sleep(5.0)

        # キーボードモードに切り替えて Alt+Tab
        ok = client.switch_to_keyboard_mode()
        print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
        ok = client.alt_tab()
        print(f"alt_tab -> {'OK' if ok else 'NG'}")

        # マウスモードに戻す
        ok = client.switch_to_mouse_mode()
        print(f"mode:mouse -> {'OK' if ok else 'NG'}")

        # Word画面を再キャプチャ
        print("\nWord画面をキャプチャ中...")
        word_frame = _capture_screen_hdmi(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if word_frame is None:
            print("[ERROR] Word画面のキャプチャに失敗しました", file=sys.stderr)
            return 1
        _save_debug_frame(word_frame, "word_screen")

        # OCRで "担当医" または "医師名" を検索
        print("Word画面をEasyOCRで認識中...")
        target_y = _find_target_y_by_ocr(word_frame, keywords=["担当医", "医師名"])
        if target_y is None:
            print("[ERROR] '担当医'/'医師名' が見つかりませんでした", file=sys.stderr)
            return 1

        # テンプレートマッチングで word_return_mark の x 座標を検出
        print("word_return_mark.jpg を検索中...")
        target_x = _find_word_return_mark_x(word_frame, threshold=0.7)
        if target_x is None:
            print("[ERROR] 'word_return_mark.jpg' が見つかりませんでした", file=sys.stderr)
            return 1

        print(f"最終カーソル位置: ({target_x}, {target_y})")
        ok = client.move_mouse_to_position(target_x, target_y)
        print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

        # 最も下の word_return_mark を検索（ドラッグ先）
        print("\n最下 word_return_mark を検索中...")
        bottom_pos = _find_word_return_mark_bottom(word_frame, screen_width=config.capture_width, threshold=0.7)
        if bottom_pos is None:
            print("[ERROR] ドラッグ先が見つかりませんでした", file=sys.stderr)
            return 1

        bottom_x, bottom_y = bottom_pos
        print(f"ドラッグ先: ({bottom_x}, {bottom_y})")

        # ドラッグ実行
        ok = client.mouse_down()
        print(f"mouse_down -> {'OK' if ok else 'NG'}")
        time.sleep(0.1)

        ok = client.move_mouse_to_position(bottom_x, bottom_y)
        print(f"moveto ({bottom_x}, {bottom_y}) -> {'OK' if ok else 'NG'}")
        time.sleep(0.1)

        # 2秒待機（ドラッグ範囲を目視確認）
        print("2.0秒待機（ドラッグ範囲確認）...")
        time.sleep(2.0)

        # 先にマウスボタンを離す
        ok = client.mouse_up()
        print(f"mouse_up -> {'OK' if ok else 'NG'}")

        # キーボードモードに切り替えて backspace
        ok = client.switch_to_keyboard_mode()
        print(f"mode:keyboard -> {'OK' if ok else 'NG'}")
        ok = client.press_key("backspace")
        print(f"press_key(backspace) -> {'OK' if ok else 'NG'}")

        # Ctrl+L (左寄せ)
        ok = client.press_key("ctrl_l")
        print(f"press_key(ctrl_l) -> {'OK' if ok else 'NG'}")

        # Windows キー
        ok = client.press_key("win")
        print(f"press_key(win) -> {'OK' if ok else 'NG'}")

        # 1.0秒待機
        print("1.0秒待機...")
        time.sleep(1.0)

        # "note" テキスト入力
        ok = client.type_text("note")
        print(f"type_text(note) -> {'OK' if ok else 'NG'}")

        # 0.5秒待機
        print("0.5秒待機...")
        time.sleep(0.5)

        # Enter 送信
        ok = client.press_key("enter")
        print(f"press_key(enter) -> {'OK' if ok else 'NG'}")

        # 0.5秒待機後、ウィンドウ最大化 (Win+Up)
        print("0.5秒待機...")
        time.sleep(0.5)
        ok = client.press_key("win_up")
        print(f"press_key(win_up) -> {'OK' if ok else 'NG'}")

        return 0

    if scroll_only:
        prev_frame = frame.copy()
        iteration = 0
        max_iterations = scroll_count if scroll_count is not None else 20

        while True:
            iteration += 1
            if iteration > max_iterations:
                if scroll_count is None:
                    print("\n安全上限(20セット)に達しました。自動終了します。")
                break

            print(f"\nスクロールのみモード [セット {iteration}] 過去カルテ領域をスクロール中...")
            _scroll_past_chart_down(
                dividers=dividers,
                screen_width=config.capture_width,
                screen_height=config.capture_height,
                scroll_count=3,
            )

            print("スクロール後の画面をキャプチャ中...")
            frame = _capture_screen_hdmi(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if frame is None:
                print("[WARNING] 再キャプチャに失敗しました。終了します。", file=sys.stderr)
                break

            print("画面変化を確認中...")
            if _is_frame_unchanged(prev_frame, frame):
                print("スクロール後の画面が変化しませんでした。自動終了します。")
                break
            prev_frame = frame.copy()

            _save_debug_frame(frame, f"full_screen_scroll_{iteration}")
            print("画面を解析中...")
            dividers = _detect_all_dividers(frame, debug=True)
            if dividers is None:
                print("[ERROR] スクロール後の画面で区切り線を検出できませんでした", file=sys.stderr)
                break

        return 0

    print("\n過去カルテ領域を切り出し中...")
    try:
        past_chart = _extract_past_chart_region(frame, dividers, debug=True)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"切り出しサイズ: {past_chart.shape[1]}x{past_chart.shape[0]} px")

    print("\nEasyOCR でテキスト抽出中...")
    ocr_text = _extract_ocr_text(past_chart)
    print(f"OCR抽出完了 ({len(ocr_text)} 文字)")

    print("\nVLM で過去カルテの内容を読み取り中...")
    raw_response = _read_past_chart_with_vlm(
        past_chart,
        ocr_text,
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
        prev_frame = frame.copy()
        iteration = 0
        max_iterations = scroll_count if scroll_count is not None else 20

        while True:
            iteration += 1
            if iteration > max_iterations:
                if scroll_count is None:
                    print("\n安全上限(20セット)に達しました。自動終了します。")
                break

            print(f"\n[セット {iteration}] 過去カルテ領域をスクロール中...")
            _scroll_past_chart_down(
                dividers=dividers,
                screen_width=config.capture_width,
                screen_height=config.capture_height,
                scroll_count=3,
            )

            # スクロール後に再キャプチャ
            print("スクロール後の画面をキャプチャ中...")
            frame = _capture_screen_hdmi(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if frame is None:
                print("[WARNING] 再キャプチャに失敗しました。現在の結果で終了します。", file=sys.stderr)
                break

            print("画面変化を確認中...")
            if _is_frame_unchanged(prev_frame, frame):
                print("スクロール後の画面が変化しませんでした。自動終了します。")
                break
            prev_frame = frame.copy()

            _save_debug_frame(frame, f"full_screen_scroll_{iteration}")
            print("画面を解析中...")

            dividers = _detect_all_dividers(frame, debug=True)
            if dividers is None:
                print("[ERROR] スクロール後の画面で区切り線を検出できませんでした", file=sys.stderr)
                break

            print(f"区切り線検出: x={dividers}")
            try:
                past_chart = _extract_past_chart_region(frame, dividers, debug=True)
            except RuntimeError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                break

            print("\nEasyOCR でテキスト抽出中...")
            ocr_text = _extract_ocr_text(past_chart)
            print(f"OCR抽出完了 ({len(ocr_text)} 文字)")

            print(f"\n[セット {iteration}] VLM で統合読み取り中...")
            raw_response = _read_past_chart_with_vlm_merge(
                past_chart,
                structured,
                ocr_text,
                model=runtime["model"],
                url=runtime["url"],
                api_key=runtime["api_key"],
                timeout=MLX_VLM_IME_TIMEOUT,
            )

            print(f"\n--- VLM 生応答 (セット {iteration}) ---\n{raw_response}\n--- 終了 ---\n")

            try:
                structured = _parse_vlm_response(raw_response)
            except ValueError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                break

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
