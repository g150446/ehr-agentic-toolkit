"""omlx VLM サーバーを使った IME 候補読み取りヘルパー。

omlx（OpenAI 互換 API、ポート 8000）に画像を送信し、
IME インライン変換候補やポップアップ候補リストを読み取る。
"""

from __future__ import annotations

import base64
from datetime import datetime
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

MLX_VLM_IME_URL = os.getenv(
    "MLX_VLM_IME_URL",
    os.getenv("MLX_VLM_SEGMENTATION_URL", "http://localhost:8000/v1/chat/completions"),
)
MLX_VLM_IME_MODEL = os.getenv(
    "MLX_VLM_IME_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "Qwen3-VL-8B-Instruct-4bit"),
)
MLX_VLM_IME_API_KEY = os.getenv("MLX_VLM_IME_API_KEY", "omlxkey")
MLX_VLM_IME_TIMEOUT = float(os.getenv("MLX_VLM_IME_TIMEOUT", "90"))
# Shorter timeout for inline candidate reads (ROI/fullframe).
# When the server is slow, we want to quickly fall through to popup mode.
MLX_VLM_INLINE_TIMEOUT = float(os.getenv("MLX_VLM_INLINE_TIMEOUT", "45"))
# Text-only model for tasks that don't require vision (e.g., helper word suggestions).
# Falls back to the same VL model if a dedicated text model is not available.
MLX_VLM_TEXT_MODEL = os.getenv("MLX_VLM_TEXT_MODEL", MLX_VLM_IME_MODEL)
MLX_VLM_TEXT_URL = os.getenv("MLX_VLM_TEXT_URL", MLX_VLM_IME_URL)
MLX_VLM_TEXT_API_KEY = os.getenv("MLX_VLM_TEXT_API_KEY", MLX_VLM_IME_API_KEY)
_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_HTTP_REFERER = os.getenv(
    "OPENROUTER_HTTP_REFERER",
    "https://github.com/g150446/ehr-agentic-toolkit",
)
_OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "EHR Agentic Toolkit")
_OPENROUTER_PROVIDER_ORDER = [
    item.strip()
    for item in os.getenv("OPENROUTER_PROVIDER_ORDER", "io-net").split(",")
    if item.strip()
]
_VLM_REQUEST_COOLDOWN_SEC = 3.0
_last_vlm_response_monotonic: Optional[float] = None


class MlxVlmImeError(RuntimeError):
    """Raised when mlx_vlm IME call fails."""


def describe_runtime(*, url: Optional[str], model: Optional[str], default_kind: str = "VLM") -> str:
    """Return a human-readable backend label for logs."""
    model_name = model or "unknown"
    if url and _OPENROUTER_CHAT_URL in url:
        return f"OpenRouter({model_name})"
    return f"{default_kind}({model_name})"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_MIN_FRAME_HEIGHT = 80
_MIN_FRAME_WIDTH = 200


def _captures_dir() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
    os.makedirs(path, exist_ok=True)
    return path


def _logs_dir() -> str:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(path, exist_ok=True)
    return path


def _safe_debug_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def _save_debug_frame(frame: np.ndarray, *, name: str, prefix: str) -> None:
    path = os.path.join(_captures_dir(), f"{prefix}_{_safe_debug_name(name)}.png")
    cv2.imwrite(path, frame)
    print(f"  [debug] 保存: {path}")


def _wait_for_vlm_cooldown() -> None:
    """Wait long enough so each VLM request starts 3s after the previous response."""
    global _last_vlm_response_monotonic
    if _last_vlm_response_monotonic is None:
        return
    elapsed = time.monotonic() - _last_vlm_response_monotonic
    remaining = _VLM_REQUEST_COOLDOWN_SEC - elapsed
    if remaining <= 0:
        return
    print(f"  [mlx_vlm cooldown] 前回応答から {remaining:.2f} 秒待機します")
    time.sleep(remaining)


def _ensure_min_size(frame: np.ndarray) -> np.ndarray:
    """VLM が処理できる最小サイズに拡大する（アスペクト比維持）。"""
    h, w = frame.shape[:2]
    scale = max(_MIN_FRAME_HEIGHT / h, _MIN_FRAME_WIDTH / w, 1.0)
    if scale <= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _encode_image_data_url(frame: np.ndarray, *, debug_name: str = "") -> str:
    """numpy BGR フレームを data URI 形式の PNG base64 文字列に変換する。"""
    frame = _ensure_min_size(frame)
    if debug_name:
        _save_debug_frame(frame, name=debug_name, prefix="debug_vlm_input")
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise MlxVlmImeError("画像の PNG エンコードに失敗しました")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Geometry helpers (ported from ollama_vlm_ime.py)
# ---------------------------------------------------------------------------

def _cluster_x_positions(xs: list[int], *, max_gap: int = 20) -> list[int]:
    if not xs:
        return []
    xs.sort()
    clusters: list[int] = []
    current: list[int] = [xs[0]]
    for x in xs[1:]:
        if x - current[-1] <= max_gap:
            current.append(x)
        else:
            clusters.append(int(round(sum(current) / len(current))))
            current = [x]
    clusters.append(int(round(sum(current) / len(current))))
    return clusters


def _find_gray_divider_candidates(frame: np.ndarray) -> list[int]:
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
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(25, (y2 - y1) // 6)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))

    col_strength = np.count_nonzero(mask, axis=0)
    threshold = max(int((y2 - y1) * 0.45), 40)
    candidates: list[int] = []
    start: Optional[int] = None
    for x, strength in enumerate(col_strength):
        if strength >= threshold and start is None:
            start = x
        elif strength < threshold and start is not None:
            end = x - 1
            if 1 <= end - start + 1 <= max(18, w // 30):
                candidates.append((start + end) // 2)
            start = None
    if start is not None:
        end = len(col_strength) - 1
        if 1 <= end - start + 1 <= max(18, w // 30):
            candidates.append((start + end) // 2)
    return candidates


def _find_hough_divider_candidates(frame: np.ndarray) -> list[int]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(h * 0.45),
        maxLineGap=20,
    )
    if lines is None:
        return []
    xs: list[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 8:
            continue
        if abs(y2 - y1) < int(h * 0.45):
            continue
        if x1 < 40 or x1 > w - 40:
            continue
        xs.append(int(round((x1 + x2) / 2)))
    return xs


def _select_divider_group(dividers: list[int], *, width: int) -> Optional[list[int]]:
    dividers = [x for x in dividers if 40 < x < width - 40]
    if len(dividers) < 4:
        return None
    if len(dividers) == 4:
        return dividers

    best_group: Optional[list[int]] = None
    best_score: Optional[float] = None
    for start in range(0, len(dividers) - 3):
        group = dividers[start:start + 4]
        gaps = np.diff(group)
        if np.any(gaps < width * 0.05) or np.any(gaps > width * 0.45):
            continue
        score = float(np.std(gaps))
        if best_score is None or score < best_score:
            best_score = score
            best_group = group
    return best_group


def _save_divider_debug_overlay(
    frame: np.ndarray,
    *,
    name: str,
    candidate_dividers: list[int],
    accepted_dividers: Optional[list[int]],
) -> None:
    overlay = frame.copy()
    h = overlay.shape[0]
    for x in candidate_dividers:
        cv2.line(overlay, (x, 0), (x, h - 1), (0, 215, 255), 1)
    if accepted_dividers:
        for x in accepted_dividers:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        x1, x2 = accepted_dividers[1], accepted_dividers[2]
        cv2.rectangle(overlay, (x1, 0), (x2, h - 1), (255, 0, 0), 2)
    _save_debug_frame(overlay, name=name, prefix="debug_panel_detection")


def detect_patient_record_panel3(
    frame: np.ndarray,
    *,
    debug_name: str = "",
) -> Optional[tuple[int, int]]:
    """患者記録画面の第3ペインの x 座標範囲を検出する。"""
    _, w = frame.shape[:2]
    gray_candidates = _find_gray_divider_candidates(frame)
    hough_candidates = _find_hough_divider_candidates(frame)
    clustered = _cluster_x_positions(gray_candidates + hough_candidates)
    accepted = _select_divider_group(clustered, width=w)
    if debug_name:
        _save_divider_debug_overlay(
            frame,
            name=debug_name,
            candidate_dividers=clustered,
            accepted_dividers=accepted,
        )
    if not accepted:
        return None
    return (accepted[1], accepted[2])


def crop_to_input_region(frame: np.ndarray, *, debug_name: str = "") -> np.ndarray:
    """フレームをテキスト入力領域にクロップする（第3ペイン検出 → フォールバック全画面）。"""
    panel = detect_patient_record_panel3(frame, debug_name=debug_name)
    if panel is not None:
        x1, x2 = panel
        return frame[:, x1:x2]
    return frame


def _resize_for_screen_classification(frame: np.ndarray, *, max_side: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    return cv2.resize(
        frame,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _crop_center_band(
    frame: np.ndarray,
    top_ratio: float = 0.25,
    bottom_ratio: float = 0.75,
) -> np.ndarray:
    h = frame.shape[0]
    return frame[int(h * top_ratio):int(h * bottom_ratio), :]


def _find_dark_region_y(frame: np.ndarray) -> Optional[int]:
    """IME 反転表示（暗い背景または Windows 10 青色選択バー）の y 座標を検出する。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 閾値 150: 白背景(>200)を除く暗め・中間色（青選択バーも捕捉）
    _, dark_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh = frame.shape[0]
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or w > 800 or h < 12 or h > 100:
            continue
        # タスクバー除外（下15%）
        if y > fh * 0.85:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (y, h)

    if best is None:
        return None
    y, h = best
    return y + h // 2


def crop_to_ime_popup_by_blue(frame: np.ndarray) -> Optional[np.ndarray]:
    """Windows IME ポップアップの青い選択インジケータを HSV 色検出で特定し、
    そのインジケータを基準にポップアップ全体をクロップして返す。

    - Win10 IME: 選択中の候補に青いハイライトバー（横長: w >> h）
    - Win11 IME: 選択中の候補の左端に青い細い縦バー（縦長: h >> w）
    (標準 Windows アクセント色: RGB 0,120,215 → HSV H≈106, S≈182, V≈215)

    正方形に近いアイコン（スピナー、IME モードボタンなど）は除外する。

    Returns:
        クロップされたポップアップ領域。検出失敗時は None。
    """
    fh, fw = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Windows IME 選択インジケータ (青: H 95-130, S 50+, V 60+)
    lower_blue = np.array([95, 50, 60])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 小ノイズ除去（OPEN は使わない: 選択バーの青画素が疎なため除去されてしまう）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[tuple[int, int, int, int]] = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # IME インジケータの形状チェック:
        #   Win11: 細い縦バー (h >= w * 1.5)
        #   Win10: 横長バー  (w >= h * 2.0)
        # スピナー・モードアイコンなど正方形に近い要素は除外する
        is_vertical_bar = h >= w * 1.5    # Win11 スタイル
        is_horizontal_bar = w >= h * 2.0  # Win10 スタイル
        if not (is_vertical_bar or is_horizontal_bar):
            continue
        # Win11縦バー: 幅は実際 2-9px（環境によって異なる）。誤検出防止のため最大12pxに制限。
        # Win10横バー: 幅30-450px
        if is_vertical_bar and (w < 1 or w > 12):
            continue
        if is_horizontal_bar and (w < 30 or w > 450):
            continue
        if h < 8 or h > 80:
            continue
        # タスクバー・画面上端付近を除外
        # アプリのツールバー (~8%) と タスクバー (~92%) を除外
        if y < fh * 0.08 or y > fh * 0.92:
            continue
        # 画面右端 (5%) を除外: スクロールバーやウィンドウ枠の誤検出防止
        # 左端は除外しない: Notepadなど左端揃えテキスト直下に現れるポップアップを失わないため
        if x > fw * 0.95:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    is_vertical_bar = h >= w * 1.5
    # 選択候補（position 2）の上にある position 1 を確実に含める。
    # position 1 は選択バーの ~30px 上にあるが、行ヘッダーも含め余裕を 65px に設定。
    # 以前の 35px では position 1 が切れて OCR に読み取られなかった。
    # 下方向は最大 8 候補分（各行 ~25px = 200px）+ 余白
    popup_y1 = max(0, y - 65)
    popup_y2 = min(fh, y + h + 230)
    popup_x1 = max(0, x - 6)
    if is_vertical_bar:
        # Win11縦バー: バーは左端にあるのでポップアップ幅分右に広げる
        popup_x2 = min(fw, x + 260)
    else:
        # Win10横バー: バー幅 + マージン
        popup_x2 = min(fw, x + w + 6)

    return frame[popup_y1:popup_y2, popup_x1:popup_x2]


def _crop_popup_region(frame: np.ndarray, *, debug_name: str = "") -> np.ndarray:
    """IME ポップアップ候補リストの領域を切り出す。

    1. 全画面で青い選択バーを HSV 検出して精密クロップ（Win10 IME）
    2. 失敗時は input_region にクロップして暗い領域を検出（Win7 反転）
    3. それも失敗時は中央帯クロップ
    """
    # Win10: 全フレームで青いハイライトバーを精密クロップ
    popup = crop_to_ime_popup_by_blue(frame)
    if popup is not None:
        if debug_name:
            _save_debug_popup(popup, debug_name)
        return popup

    # フォールバック: 入力ペインに限定してグレースケール閾値で検出
    roi = crop_to_input_region(frame, debug_name=f"popup_region_{debug_name or 'current'}")
    center_y = _find_dark_region_y(roi)
    fh = roi.shape[0]

    if center_y is not None:
        y1 = max(0, center_y - 120)
        y2 = min(fh, center_y + 300)
        crop = roi[y1:y2, :]
        if debug_name:
            _save_debug_popup(crop, debug_name)
        return crop

    crop = _crop_center_band(roi, top_ratio=0.08, bottom_ratio=0.85)
    if debug_name:
        _save_debug_popup(crop, debug_name)
    return crop


def _save_debug_popup(frame: np.ndarray, name: str) -> None:
    """ポップアップクロップ画像をデバッグ用に保存する。"""
    _save_debug_frame(frame, name=name, prefix="debug_popup_crop")


def _parse_screen_type_response(content: str) -> str:
    """VLM 応答から helper reset 用の画面種別を抽出する。"""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            screen_type = str(payload.get("screen_type", "")).strip().lower()
            if screen_type in {"patient_record", "notepad", "other"}:
                return screen_type

    normalized = content.lower().replace("-", "_").replace(" ", "_")
    if "patient_record" in normalized or "patient_chart" in normalized:
        return "patient_record"
    if "notepad" in normalized:
        return "notepad"
    return "other"


def classify_helper_reset_screen(frame: np.ndarray, *, debug_name: str = "") -> str:
    """Return patient_record / notepad / other for helper-reset cropping."""
    if detect_patient_record_panel3(frame) is not None:
        return "patient_record"

    resized = _resize_for_screen_classification(frame)
    data_url = _encode_image_data_url(
        resized,
        debug_name=f"{debug_name}_screen_classifier" if debug_name else "screen_classifier",
    )
    prompt = (
        "この HDMI キャプチャ画像の画面種別を分類してください。"
        "候補は patient_record / notepad / other の3つだけです。"
        "patient_record は患者電子カルテの記録画面、notepad は Windows のメモ帳画面、"
        "other はそれ以外です。"
        "JSONのみで答えてください。"
        ' 形式: {"screen_type": "patient_record|notepad|other"}'
    )
    try:
        content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    except MlxVlmImeError as exc:
        print(f"  [helper reset] 画面分類失敗: {exc}")
        return "other"
    screen_type = _parse_screen_type_response(content)
    print(f"  [helper reset] 画面分類: {screen_type}")
    return screen_type


def _find_taskbar_top(frame: np.ndarray) -> Optional[int]:
    """Detect the top edge of a dark Windows taskbar near the bottom of the frame."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_rows = np.count_nonzero(gray < 80, axis=1)
    row_means = gray.mean(axis=1)
    threshold = int(w * 0.6)
    start_y = int(h * 0.8)
    band_start: Optional[int] = None
    band_end: Optional[int] = None
    for y in range(start_y, h):
        if dark_rows[y] >= threshold or (row_means[y] < 245 and dark_rows[y] >= int(w * 0.04)):
            if band_start is None:
                band_start = y
            band_end = y
        elif band_start is not None:
            break
    if band_start is None or band_end is None:
        return None
    if band_end - band_start + 1 < max(6, int(h * 0.006)):
        return None
    return band_start


def _find_notepad_document_top(frame: np.ndarray) -> int:
    """Find the first row that looks like the Notepad document body."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    row_means = gray.mean(axis=1)
    dark_rows = np.count_nonzero(gray < 120, axis=1)
    h, w = gray.shape
    max_scan = max(12, int(h * 0.3))
    window = max(4, int(h * 0.008))
    dark_threshold = max(20, int(w * 0.06))
    bright_threshold = 248

    for y in range(0, max_scan):
        y2 = min(h, y + window)
        if y2 - y < window:
            break
        if float(row_means[y:y2].mean()) >= bright_threshold and int(dark_rows[y:y2].max()) <= dark_threshold:
            return y
    return min(max_scan, int(h * 0.12))


def crop_notepad_document_region(frame: np.ndarray, *, debug_name: str = "") -> np.ndarray:
    """Crop the Notepad document body, excluding the top menu and Windows taskbar."""
    h, w = frame.shape[:2]
    taskbar_top = _find_taskbar_top(frame)
    work = frame[:taskbar_top, :] if taskbar_top is not None else frame

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect: Optional[tuple[int, int, int, int]] = None
    best_area = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < max(120, int(w * 0.2)) or bh < max(80, int(h * 0.15)):
            continue
        if y > int(work.shape[0] * 0.75):
            continue
        if area > best_area:
            best_area = area
            best_rect = (x, y, bw, bh)

    if best_rect is not None:
        x, y, bw, bh = best_rect
        crop = work[y:y + bh, x:x + bw]
    else:
        top = min(work.shape[0], int(work.shape[0] * 0.12))
        left = min(work.shape[1], int(work.shape[1] * 0.04))
        right = max(left + 1, int(work.shape[1] * 0.96))
        crop = work[top:, left:right]

    doc_top = _find_notepad_document_top(crop)
    if 0 < doc_top < crop.shape[0]:
        crop = crop[doc_top:, :]

    if debug_name:
        _save_debug_frame(crop, name=debug_name, prefix="debug_notepad_crop")
    return crop


def crop_helper_reset_region(
    frame: np.ndarray,
    *,
    screen_type: Optional[str] = None,
    debug_name: str = "",
) -> tuple[np.ndarray, str]:
    """Return the comparison crop and resolved screen type for helper reset."""
    resolved_screen_type = screen_type or classify_helper_reset_screen(
        frame,
        debug_name=f"{debug_name}_classify" if debug_name else "",
    )
    if resolved_screen_type == "patient_record":
        cropped = crop_to_input_region(frame, debug_name=f"{debug_name}_panel" if debug_name else "")
    elif resolved_screen_type == "notepad":
        cropped = crop_notepad_document_region(frame, debug_name=f"{debug_name}_notepad" if debug_name else "")
    else:
        cropped = frame

    if cropped.size == 0:
        return frame, resolved_screen_type
    return cropped, resolved_screen_type


# ---------------------------------------------------------------------------
# mlx_vlm API call
# ---------------------------------------------------------------------------

def _call_mlx_vlm_with_content(
    content: list[dict[str, object]],
    prompt: str,
    *,
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    enable_reasoning: bool = False,
    thinking_log: bool = False,
) -> str:
    """omlx VLM サーバーの OpenAI 互換エンドポイントに画像付きリクエストを送信する。"""
    model = model or MLX_VLM_IME_MODEL
    url = url or MLX_VLM_IME_URL
    timeout = MLX_VLM_IME_TIMEOUT if timeout is None else timeout
    api_key = api_key or MLX_VLM_IME_API_KEY

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *content],
            }
        ],
        "stream": False,
        "max_tokens": 256,
    }
    reasoning_requested = False
    if enable_reasoning:
        payload["include_reasoning"] = True
        if url and _OPENROUTER_CHAT_URL in url:
            if _OPENROUTER_PROVIDER_ORDER:
                payload["provider"] = {"order": _OPENROUTER_PROVIDER_ORDER}
        else:
            payload["reasoning"] = {"enabled": True}
        reasoning_requested = True

    def _send_request(request_payload: dict) -> dict:
        global _last_vlm_response_monotonic
        _wait_for_vlm_cooldown()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        if url and _OPENROUTER_CHAT_URL in url:
            headers["HTTP-Referer"] = _OPENROUTER_HTTP_REFERER
            headers["X-Title"] = _OPENROUTER_X_TITLE
        req = urllib.request.Request(
            url,
            data=json.dumps(request_payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        _last_vlm_response_monotonic = time.monotonic()
        return result

    try:
        result = _send_request(payload)
    except urllib.error.HTTPError as exc:
        global _last_vlm_response_monotonic
        reason = exc.read().decode("utf-8", errors="replace")
        _last_vlm_response_monotonic = time.monotonic()
        if enable_reasoning:
            print("  [mlx_vlm thinking] reasoning指定が拒否されたため、reasoningなしで再試行します")
            payload.pop("reasoning", None)
            payload.pop("include_reasoning", None)
            reasoning_requested = False
            try:
                result = _send_request(payload)
            except urllib.error.HTTPError as retry_exc:
                retry_reason = retry_exc.read().decode("utf-8", errors="replace")
                _last_vlm_response_monotonic = time.monotonic()
                raise MlxVlmImeError(
                    f"mlx_vlm IME リクエストが HTTP {retry_exc.code} で失敗しました: {retry_reason}. endpoint={url}"
                ) from retry_exc
        else:
            raise MlxVlmImeError(
                f"mlx_vlm IME リクエストが HTTP {exc.code} で失敗しました: {reason}. endpoint={url}"
            ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise MlxVlmImeError(
            f"mlx_vlm IME リクエストが {timeout:g} 秒でタイムアウトしました"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise MlxVlmImeError(
                f"mlx_vlm IME リクエストが {timeout:g} 秒でタイムアウトしました"
            ) from exc
        raise MlxVlmImeError(
            f"mlx_vlm IME への接続に失敗しました: {reason}. endpoint={url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MlxVlmImeError("mlx_vlm IME 応答の JSON を解析できませんでした") from exc

    if thinking_log:
        _save_thinking_log(
            prompt=prompt,
            image_count=sum(1 for item in content if item.get("type") == "image_url"),
            model=model,
            url=url,
            reasoning_requested=reasoning_requested,
            result=result,
        )

    return _extract_message_text(result, error_prefix="mlx_vlm IME")


def _call_mlx_vlm_with_image(
    image_data_url: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    enable_reasoning: bool = False,
    thinking_log: bool = False,
) -> str:
    return _call_mlx_vlm_with_content(
        [{"type": "image_url", "image_url": {"url": image_data_url}}],
        prompt,
        model=model,
        url=url,
        timeout=timeout,
        api_key=api_key,
        enable_reasoning=enable_reasoning,
        thinking_log=thinking_log,
    )


def _call_mlx_vlm_with_images(
    image_data_urls: list[str],
    prompt: str,
    *,
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
    enable_reasoning: bool = False,
    thinking_log: bool = False,
) -> str:
    return _call_mlx_vlm_with_content(
        [{"type": "image_url", "image_url": {"url": image_url}} for image_url in image_data_urls],
        prompt,
        model=model,
        url=url,
        timeout=timeout,
        api_key=api_key,
        enable_reasoning=enable_reasoning,
        thinking_log=thinking_log,
    )


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _extract_message_text(result: dict, *, error_prefix: str) -> str:
    """OpenAI互換応答から content、なければ reasoning を取り出す。"""
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MlxVlmImeError(
            f"{error_prefix} 応答に content テキストがありません: {result!r}"
        ) from exc

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning_parts: list[str] = []
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        reasoning_parts.append(reasoning.strip())

    details = message.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    reasoning_parts.append(text.strip())

    if reasoning_parts:
        merged: list[str] = []
        for part in reasoning_parts:
            if part not in merged:
                merged.append(part)
        return "\n".join(merged).strip()

    raise MlxVlmImeError(
        f"{error_prefix} 応答に content テキストがありません: {result!r}"
    )


def _extract_reasoning_text(result: dict) -> str:
    """OpenAI互換応答から reasoning / reasoning_details をまとめて取り出す。"""
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""

    parts: list[str] = []
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(reasoning.strip())

    details = message.get("reasoning_details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

    merged: list[str] = []
    for part in parts:
        if part not in merged:
            merged.append(part)
    return "\n\n".join(merged)


def _save_thinking_log(
    *,
    prompt: str,
    image_count: int,
    model: str,
    url: str,
    reasoning_requested: bool,
    result: dict,
) -> None:
    """Save compare-request reasoning details to logs/<timestamp>_thinking.txt."""
    timestamp = datetime.now().strftime("%m%d_%H%M%S_%f")
    path = os.path.join(_logs_dir(), f"{timestamp}_thinking.txt")
    reasoning_text = _extract_reasoning_text(result)
    try:
        message = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        message = {}
    content_text = message.get("content")
    if not isinstance(content_text, str):
        content_text = repr(content_text)
    lines = [
        f"timestamp: {timestamp}",
        f"model: {model}",
        f"url: {url}",
        f"image_count: {image_count}",
        f"reasoning_requested: {reasoning_requested}",
        f"reasoning_present: {bool(reasoning_text.strip())}",
        "",
        "[prompt]",
        prompt,
        "",
        "[content]",
        content_text,
        "",
        "[reasoning]",
        reasoning_text or "<empty>",
        "",
        "[raw_message]",
        json.dumps(message, ensure_ascii=False, indent=2),
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  [mlx_vlm thinking] 保存: {path}")

def _parse_candidate_response(content: str) -> Optional[str]:
    """VLM 応答から単一候補文字列を抽出する。"""
    # <think>...</think> ブロックを除去（Qwen3 の思考出力）
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("candidate")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if candidate is None:
                return None
    sq = re.search(r"'candidate'\s*:\s*'([^']+)'", content)
    if sq:
        return sq.group(1).strip() or None
    return None


def _parse_candidates_response(content: str) -> list[str]:
    """VLM 応答から候補リストを抽出する（ポップアップ用）。"""
    # <think>...</think> ブロックを除去
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    cleaned = re.sub(r"```[a-z]*\n?", "", content).strip().rstrip("`").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            candidates = parsed.get("candidates", [])
            if isinstance(candidates, list):
                result = []
                for c in candidates:
                    c = re.sub(r'^[0-9]+[\s.\-、。]*', '', str(c).strip()).strip()
                    if c:
                        result.append(c)
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_inline_candidate_roi(roi: np.ndarray) -> Optional[str]:
    """インライン変換の ROI 画像から現在の変換候補を読み取る。

    Args:
        roi: IME 反転ブロックの ROI 画像

    Returns:
        読み取った候補文字列、または None
    """
    prompt = (
        "この画像は Windows IME の変換候補ウィンドウです。"
        "黒く反転（ハイライト）されている行の文字列だけを正確に読み取ってください。"
        "前後に余計な句読点や記号を付け足さないでください。"
        "読めない場合は null を返してください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidate": "候補文字列"} または {"candidate": null}'
    )
    data_url = _encode_image_data_url(roi, debug_name="inline_candidate_roi")
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_candidate_response(content)


def read_inline_candidate_context(frame: np.ndarray) -> Optional[str]:
    """全画面フレームから入力フィールド行の変換候補を読み取る（フォールバック用）。

    Args:
        frame: 全画面フレーム

    Returns:
        変換中の文字列のみ、または None
    """
    cropped = _crop_center_band(
        crop_to_input_region(frame, debug_name="inline_candidate_context_panel"),
        top_ratio=0.05,
        bottom_ratio=0.75,
    )
    prompt = (
        "この画像はWindowsの画面の一部（入力フィールド周辺）です。"
        "テキスト入力フィールドの中で、下線または黒背景・白文字で強調表示されている、"
        "現在変換中の文字列（1〜4文字程度の漢字やひらがな）だけを読み取ってください。"
        "前後に入力済みの文字列は含めないでください。変換中の部分だけを返してください。"
        "見つかった場合はその文字列のみを返してください。見つからない場合は null を返してください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidate": "文字列"} または {"candidate": null}'
    )
    data_url = _encode_image_data_url(cropped, debug_name="inline_candidate_context")
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_candidate_response(content)


def read_highlighted_popup_candidate(frame: np.ndarray, *, debug_name: str = "") -> Optional[str]:
    """IME ポップアップ内の現在ハイライトされている候補だけを読み取る。"""
    cropped = _crop_popup_region(frame, debug_name=debug_name)
    prompt = (
        "この画像は Windows 日本語 IME の変換候補ポップアップです。"
        "現在ハイライトされている1行だけの候補文字列を正確に読んでください。"
        "行頭の数字は含めず、候補文字列だけを返してください。"
        "候補以外の入力欄テキストや周辺UIは無視してください。"
        "読めない場合は null を返してください。"
        "JSONのみで回答してください。"
        ' 形式: {"candidate": "候補文字列"} または {"candidate": null}'
    )
    data_url = _encode_image_data_url(
        cropped,
        debug_name=f"highlighted_popup_{debug_name or 'current'}",
    )
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_candidate_response(content)


def read_popup_candidates_ocr(frame: np.ndarray, *, debug_name: str = "") -> list[tuple[int, str]]:
    """OCRを使ってIMEポップアップ候補リストを読み取る。

    VLM（30-37秒）より高速（1-2秒）で安定。
    EasyOCRで全画面をスキャンし、「数字+テキスト」パターンの候補を抽出する。

    Args:
        frame: 全画面フレーム（ポップアップ表示中）

    Returns:
        (display_number, candidate_text) のリスト。失敗時は空リスト。
    """
    # ポップアップ領域をクロップ（失敗時はスキップ）
    cropped = crop_to_ime_popup_by_blue(frame)
    if cropped is None:
        # ポップアップが検出できない = まだ表示されていないか、フレームが古い。
        # フルフレームでのEasyOCRは30秒以上かかるためスキップして空を返す。
        return []

    if debug_name:
        _save_debug_popup(cropped, debug_name)

    # 大きすぎるクロップはポップアップではなく背景 → スキップ
    h, w = cropped.shape[:2]
    if w > 600 or h > 600:
        return []

    try:
        from automation.screen_analyzer import load_ocr_reader, run_ocr
        reader = load_ocr_reader()
        results = run_ocr(reader, cropped)
    except Exception as exc:
        print(f"  [OCR] 読取失敗: {exc}")
        return []

    if not results:
        return []

    # EasyOCR結果をY座標でソート
    items: list[tuple[float, float, str]] = []
    for entry in results:
        if len(entry) == 3:
            bbox, text, _ = entry
        else:
            bbox, text = entry[0], entry[1]
        y = (bbox[0][1] + bbox[2][1]) / 2
        x = (bbox[0][0] + bbox[2][0]) / 2
        items.append((y, x, str(text).strip()))
    items.sort()

    candidates: list[tuple[int, str]] = []
    i = 0
    while i < len(items):
        _, _, text = items[i]

        # パターン1: 単独の数字の次の要素がテキスト（"1", "聴診" と別々のボックスの場合）
        if re.match(r'^\d+$', text):
            num = int(text)
            if 1 <= num <= 9 and i + 1 < len(items):
                _, _, next_text = items[i + 1]
                if next_text and not re.match(r'^\d+$', next_text) and len(next_text) <= 25:
                    candidates.append((num, next_text))
                    i += 2
                    continue

        # パターン2: "1. テキスト" または "1 テキスト" が同一ボックス
        m = re.match(r'^(\d+)[.。\s]+(.+)$', text)
        if m:
            num = int(m.group(1))
            cand = m.group(2).strip()
            if 1 <= num <= 9 and cand and len(cand) <= 25:
                candidates.append((num, cand))
        i += 1

    # 重複除去してソート
    seen: set[int] = set()
    result = []
    for num, text in sorted(candidates):
        if num not in seen:
            seen.add(num)
            result.append((num, text))

    # Windows 11 IME は候補ポップアップの上方に「入力履歴/クラウド候補」パネルを
    # 表示する場合がある。これらは純粋な ASCII ローマ字テキスト（例: 'choushin',
    # 'ni te'）として OCR に読み込まれる。真の IME 候補は日本語文字（漢字・
    # ひらがな・カタカナ）を含むため、ASCII のみのエントリを除外する。
    result = [(n, t) for n, t in result if not t.isascii()]
    return result


def read_popup_candidates_numbered_vlm(frame: np.ndarray, *, debug_name: str = "") -> list[tuple[int, str]]:
    """VLM のみで IME ポップアップ候補リストを番号付き読取する。"""
    cropped = _crop_popup_region(frame, debug_name=debug_name)
    prompt = (
        "この画像はWindows日本語IMEの変換候補ポップアップです。"
        "ポップアップ内の各候補に付いた数字（1〜9）とテキストを正確に読んでください。"
        "各行に「数字. テキスト」の形式で表示されています（例: 1. イン, 2. 院）。"
        "数字とテキストのペアをJSONで返してください。"
        '形式: {"candidates": [{"n": 1, "text": "イン"}, {"n": 2, "text": "院"}, {"n": 3, "text": "員"}, {"n": 4, "text": "咽頭"}]}'
        "候補リスト以外のテキストは含めないでください。JSONのみ返してください。"
    )
    data_url = _encode_image_data_url(
        cropped,
        debug_name=f"popup_numbered_{debug_name or 'current'}",
    )
    try:
        content = _call_mlx_vlm_with_image(data_url, prompt)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        cleaned = re.sub(r"```[a-z]*\n?", "", content).strip().rstrip("`").strip()

        # Try JSON parse first (fix missing commas between objects)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            json_str = m.group()
            # Fix common VLM JSON issue: missing commas between array objects
            json_str = re.sub(r"\}\s*\{", "}, {", json_str)
            try:
                parsed = json.loads(json_str)
                items = parsed.get("candidates", [])
                result = []
                for item in items:
                    n = item.get("n")
                    text = item.get("text", "")
                    if isinstance(n, int) and 1 <= n <= 9 and text:
                        # Sanity check: real IME candidates are short (≤25 chars).
                        # Longer strings indicate Windows Search results leaked in.
                        if len(text) <= 25:
                            result.append((n, text))
                if result:
                    return result
            except json.JSONDecodeError:
                pass  # fall through to regex extraction

        # Regex fallback: extract {"n": X, "text": "Y"} or "n": X ... "text": "Y" patterns
        pairs = re.findall(r'"n"\s*:\s*(\d+)[^}]{0,80}?"text"\s*:\s*"([^"]+)"', cleaned)
        if not pairs:
            # Also try "text": "Y" ... "n": X order
            pairs_rev = re.findall(r'"text"\s*:\s*"([^"]+)"[^}]{0,80}?"n"\s*:\s*(\d+)', cleaned)
            pairs = [(n, text) for text, n in pairs_rev]
        if not pairs:
            natural = re.findall(
                r"(?:Item|Line)\s*\d+.*?number\s+is\s+[`\"]?(\d+)[`\"]?.*?text(?:\s+next\s+to\s+it)?\s+is\s+[`\"]([^`\"\n]+)[`\"]",
                cleaned,
                flags=re.IGNORECASE | re.DOTALL,
            )
            pairs = [(n, text.strip()) for n, text in natural]
        if pairs:
            result = [(int(n), text) for n, text in pairs if 1 <= int(n) <= 9 and text and len(text) <= 25]
            if result:
                return result
    except Exception as exc:
        print(f"  [mlx_vlm 番号付き候補取得] 失敗: {exc}")
    return []


def read_popup_candidates_numbered(frame: np.ndarray, *, debug_name: str = "") -> list[tuple[int, str]]:
    """IME ポップアップ候補リストを番号付きで読み取る。

    OCRを優先しつつ、OCR結果が疎な場合は VLM も試して補完する。
    """
    ocr_result: list[tuple[int, str]] = []
    try:
        ocr_result = read_popup_candidates_ocr(frame, debug_name=debug_name)
        if ocr_result:
            print(f"  [OCR番号付き候補] {ocr_result}")
            if len(ocr_result) >= 3:
                return ocr_result
    except Exception as exc:
        print(f"  [OCR番号付き候補] 失敗: {exc}")

    vlm_result = read_popup_candidates_numbered_vlm(frame, debug_name=debug_name)
    if vlm_result and len(vlm_result) > len(ocr_result):
        return vlm_result
    return ocr_result


def read_popup_candidates(frame: np.ndarray) -> list[str]:
    """全画面フレームから IME ポップアップ候補リスト全体を読み取る。

    Args:
        frame: 全画面フレーム（ポップアップ表示中）

    Returns:
        候補文字列のリスト（表示順）。失敗時は空リスト。
    """
    cropped = _crop_popup_region(frame)
    prompt = (
        "この画像には Windows 日本語IME の変換候補リストが表示されています。"
        "候補リスト（縦に並んだ変換候補のポップアップ）を探し、"
        "1番目（ハイライトまたは選択中の行）から上から下の順に、全ての候補を配列で返してください。"
        "各候補の先頭の数字（1, 2, 3...）は除いて、テキストだけを返してください。"
        "候補リスト以外のテキスト（入力中の文章や画面上の他のテキスト）は含めないでください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidates": ["候補1", "候補2", "候補3", ...]}'
    )
    data_url = _encode_image_data_url(cropped, debug_name="popup_candidates")
    try:
        content = _call_mlx_vlm_with_image(data_url, prompt)
        return _parse_candidates_response(content)
    except MlxVlmImeError as exc:
        print(f"  [mlx_vlm IME候補リスト] 取得失敗: {exc}")
        return []


def _extract_diff_crop(
    pre_frame: np.ndarray,
    post_frame: np.ndarray,
    *,
    min_change_px: int = 12,
    pad: int = 30,
    max_y_fraction: float = 0.82,
) -> Optional[np.ndarray]:
    """2フレームの差分から変化した矩形領域を切り出す。

    'a' キー入力前後のフレームを比較し、新しく追加された文字('a' または「あ」)の
    周辺領域のみを返す。既存テキストの影響を排除するためのヘルパー。

    max_y_fraction: 画面下部（IME クラウド候補等）を除外するための上限 (0〜1)。
    Windows IME のクラウド候補パネルは画面下部に表示されることが多いため、
    その領域を差分検索から除外することで誤検出を防ぐ。

    Returns:
        変化領域の BGR クロップ。変化なし or 小さすぎる場合は None。
    """
    h = min(pre_frame.shape[0], post_frame.shape[0])
    w = min(pre_frame.shape[1], post_frame.shape[1])
    # 画面下部（IMEクラウド候補/ツールバーエリア）を除外
    h_limit = int(h * max_y_fraction)
    diff = cv2.absdiff(pre_frame[:h_limit, :w], post_frame[:h_limit, :w])
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    if bw < min_change_px or bh < min_change_px:
        return None
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(post_frame.shape[1], x + bw + pad)
    y2 = min(post_frame.shape[0], y + bh + pad)
    return post_frame[y1:y2, x1:x2]


def detect_ime_mode_from_typed_a(
    frame: np.ndarray,
    *,
    pre_frame: Optional[np.ndarray] = None,
) -> Optional[str]:
    """'a' を入力直後のスクリーンショットから IME モードを判定する。

    呼び出し元が事前に 'a' を1文字入力し、このスクリーンキャプチャを渡す。
    pre_frame（入力前フレーム）が与えられた場合は差分クロップを使って
    既存テキストに惑わされない精度の高い判定を行う。

    Args:
        frame:     'a' 入力直後の全画面 BGR フレーム
        pre_frame: 'a' 入力前の全画面 BGR フレーム（省略可）

    Returns:
        'japanese': ひらがなモード（「あ」が表示されている）
        'english':  英数字モード（'a' が表示されている）
        None:       判定不能
    """
    # --- 差分クロップ優先 (pre_frame 提供時) ---
    if pre_frame is not None:
        diff_crop = _extract_diff_crop(pre_frame, frame)
        if diff_crop is not None:
            diff_crop = _ensure_min_size(diff_crop)
            # デバッグ用に差分クロップを保存
            _save_debug_popup(diff_crop, "ime_mode_diff")
            data_url = _encode_image_data_url(diff_crop, debug_name="ime_mode_diff")
            prompt = (
                "この画像は 'a' キーを1回押した直後に画面で変化した部分だけを切り抜いたものです。"
                "新しく追加された1文字を特定してください。"
                "平仮名の「あ」（曲線的な日本語文字）であれば 'japanese'、"
                "半角英字の 'a'（単純なラテン文字）であれば 'english' とだけ答えてください。"
                "それ以外の説明は不要です。"
            )
            try:
                content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_IME_TIMEOUT)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                print(f"  [VLM IME検出/diff] 応答: {content!r}")
                if "japanese" in content.lower() or "あ" in content:
                    return "japanese"
                if "english" in content.lower():
                    return "english"
                # 判定できなかった場合はフォールバックへ
                print("  [VLM IME検出/diff] 判定不能 → 全体フレームで再試行")
            except MlxVlmImeError as exc:
                print(f"  [VLM IME検出/diff] 失敗: {exc} → 全体フレームで再試行")
        else:
            print("  [VLM IME検出/diff] 差分なし → 全体フレームで再試行")

    # --- フォールバック: 入力エリア全体を VLM で判定 ---
    cropped = crop_to_input_region(frame, debug_name="ime_mode_typed_a_panel")
    cropped = _ensure_min_size(cropped)
    data_url = _encode_image_data_url(cropped, debug_name="ime_mode_typed_a")
    prompt = (
        "テキスト入力フィールドに 'a' というキーを1回押しました。"
        "直前に入力した最後の1文字だけに注目してください。"
        "その文字が平仮名の「あ」なら 'japanese'、半角の 'a' なら 'english' とだけ答えてください。"
        "既存のテキストは無視して、最後に追加された文字だけを判定してください。"
        "それ以外のテキストや説明は不要です。"
    )
    try:
        content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_IME_TIMEOUT)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        print(f"  [VLM IME検出/typed-a] 応答: {content!r}")
        if "japanese" in content.lower() or "あ" in content:
            return "japanese"
        if "english" in content.lower():
            return "english"
        return None
    except MlxVlmImeError as exc:
        print(f"  [VLM IME検出/typed-a] 失敗: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helper word suggestion (text-only)
# ---------------------------------------------------------------------------

def _call_mlx_vlm_text_only(
    prompt: str,
    *,
    model: Optional[str] = None,
    url: Optional[str] = None,
    timeout: Optional[float] = None,
    api_key: Optional[str] = None,
) -> str:
    """omlx VLM サーバーに画像なしのテキストのみリクエストを送信する。"""
    model = model or MLX_VLM_TEXT_MODEL
    url = url or MLX_VLM_TEXT_URL
    timeout = MLX_VLM_IME_TIMEOUT if timeout is None else timeout
    api_key = api_key or MLX_VLM_TEXT_API_KEY

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except (TimeoutError, socket.timeout) as exc:
        raise MlxVlmImeError(
            f"mlx_vlm テキストリクエストが {timeout:g} 秒でタイムアウトしました"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise MlxVlmImeError(
                f"mlx_vlm テキストリクエストが {timeout:g} 秒でタイムアウトしました"
            ) from exc
        raise MlxVlmImeError(
            f"mlx_vlm への接続に失敗しました: {reason}. endpoint={url}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MlxVlmImeError("mlx_vlm 応答の JSON を解析できませんでした") from exc

    return _extract_message_text(result, error_prefix="mlx_vlm")


def _parse_yes_no_response(content: str) -> Optional[bool]:
    """VLM 応答から yes/no を抽出する。"""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = cleaned.strip("`").strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"yes", '"yes"', "'yes'"}:
        return True
    if lowered in {"no", '"no"', "'no'"}:
        return False
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("answer", "result", "has_composition", "active"):
            value = parsed.get(key)
            if value is True:
                return True
            if value is False:
                return False
            if isinstance(value, str):
                value = value.strip().lower()
                if value == "yes":
                    return True
                if value == "no":
                    return False
    return None


def _parse_helper_reset_state_response(content: str) -> dict[str, bool]:
    """VLM 応答から helper reset 判定 JSON を抽出する。"""
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```[a-z]*\n?", "", cleaned).strip().rstrip("`").strip()
    if not cleaned:
        return {
            "left_context_preserved": False,
            "composition_cleared": False,
            "ready": False,
        }

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {
                "left_context_preserved": False,
                "composition_cleared": False,
                "ready": False,
            }
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "left_context_preserved": False,
                "composition_cleared": False,
                "ready": False,
            }

    if not isinstance(parsed, dict):
        return {
            "left_context_preserved": False,
            "composition_cleared": False,
            "ready": False,
        }

    def _as_bool(value) -> bool:
        if value is True:
            return True
        if value is False or value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "ok"}
        return False

    left_context_preserved = _as_bool(parsed.get("left_context_preserved"))
    composition_cleared = _as_bool(parsed.get("composition_cleared"))
    ready = _as_bool(parsed.get("ready"))
    if "ready" not in parsed:
        ready = left_context_preserved and composition_cleared
    return {
        "left_context_preserved": left_context_preserved,
        "composition_cleared": composition_cleared,
        "ready": ready,
    }


def assess_helper_reset_state(
    frame: np.ndarray,
    *,
    left_context: str = "",
    anchor_text: str = "",
    target_text: str,
) -> dict[str, bool]:
    """Judge whether helper-word precomposition was cleared without harming left context."""
    cropped, screen_type = crop_helper_reset_region(frame, debug_name="helper_reset_panel")
    if screen_type == "patient_record":
        cropped = _crop_center_band(
            cropped,
            top_ratio=0.15,
            bottom_ratio=0.80,
        )
    expected_left = left_context[-8:]
    expected_anchor = anchor_text.strip()[-8:]
    if expected_anchor:
        left_rule = (
            f"入力カーソル直前の確定済みアンカー語として {expected_anchor!r} がそのまま見えており、"
            "その語が欠けたり別の文字に変わったり、全選択/反転されたりしていないこと。"
        )
        composition_rule = (
            f"{expected_anchor!r} の直後に、{target_text!r} に対応する未確定組成、下線、反転候補、"
            "ポップアップ候補、またはそれに準ずる変換中表示や余分な続き文字が、"
            "入力欄内にもう残っていないこと。"
        )
    elif expected_left:
        left_rule = (
            f"入力カーソル直前の確定済み文字列として {expected_left!r} が見えており、"
            "その文字列が欠けたり、全選択/反転されたりしていないこと。"
        )
        composition_rule = (
            f"{target_text!r} に対応する未確定組成、下線、反転候補、ポップアップ候補、"
            "またはそれに準ずる変換中表示が、入力欄内にもう残っていないこと。"
        )
    else:
        left_rule = "左側コンテキスト条件は常に true として扱ってください。"
        composition_rule = (
            f"{target_text!r} に対応する未確定組成、下線、反転候補、ポップアップ候補、"
            "またはそれに準ずる変換中表示が、入力欄内にもう残っていないこと。"
        )
    prompt = (
        "この画像は EHR の入力欄周辺です。"
        f"ヘルパー単語入力前に、変換中だった {target_text!r} を Escape でキャンセルした直後です。"
        "次の2点だけを判定してください。"
        f"1. left_context_preserved: {left_rule}"
        f"2. composition_cleared: {composition_rule}"
        "全入力欄が選択されている、または左側の確定済み文字が壊れている場合は left_context_preserved=false にしてください。"
        "JSONのみで答えてください。"
        ' 形式: {"left_context_preserved": true|false, "composition_cleared": true|false, "ready": true|false}'
    )
    data_url = _encode_image_data_url(cropped, debug_name="helper_reset_state")
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_helper_reset_state_response(content)


def compare_helper_reset_images(
    baseline_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    anchor_text: str = "",
    target_text: str,
    left_context: str = "",
    screen_type: str = "",
) -> bool:
    """Compare helper-reset baseline/current images and return True only when reset is complete."""
    expected_anchor = anchor_text.strip()[-8:]
    expected_left = left_context.strip()[-8:]
    if expected_anchor:
        comparison_rule = (
            f"1枚目の最後の単語は {expected_anchor!r} です。"
            f"2枚目でも最後の単語が同様に {expected_anchor!r} で終わっていれば yes、"
            f"{expected_anchor!r} の後ろに別の文字、未確定文字、下線、反転、候補表示が"
            "1つでも見えれば no と答えてください。"
        )
    elif expected_left:
        comparison_rule = (
            f"1枚目の最後の確定済み文字列は {expected_left!r} です。"
            f"2枚目でも最後が同様に {expected_left!r} で終わっていれば yes、"
            f"{expected_left!r} の後ろに別の文字、未確定文字、下線、反転、候補表示が"
            "1つでも見えれば no と答えてください。"
        )
    else:
        comparison_rule = (
            f"1枚目は正常状態、2枚目は Escape 後です。"
            f"2枚目に {target_text!r} に対応する余分な文字、未確定文字、下線、反転、"
            "候補表示が残っていなければ yes、残っていれば no と答えてください。"
        )
    area_rule = ""
    if screen_type == "notepad":
        area_rule = (
            "画像は Windows Notepad の本文領域です。"
            "メニュー、タイトルバー、タスクバー、余白は無視してください。"
        )
    prompt = (
        "ここでは2枚の別画像が送られます。"
        "1枚目と2枚目を別々の画像として見てください。"
        "1枚目を縦に分割した1枚画像だと思わないでください。"
        f"{area_rule}"
        f"{comparison_rule}"
        "yes または no のみで答えてください。"
    )
    print("  [helper reset][compare] VLMへ比較画像2枚を送信します")
    print(f"  [helper reset][compare][prompt] {prompt}")
    baseline_data_url = _encode_image_data_url(baseline_frame, debug_name="helper_reset_compare_baseline")
    current_data_url = _encode_image_data_url(current_frame, debug_name="helper_reset_compare_current")
    content = _call_mlx_vlm_with_images(
        [baseline_data_url, current_data_url],
        prompt,
        timeout=MLX_VLM_INLINE_TIMEOUT,
        enable_reasoning=True,
        thinking_log=True,
    )
    return _parse_yes_no_response(content) is True


def has_active_ime_composition(frame: np.ndarray) -> bool:
    """入力欄近傍に未確定の IME 組成文字が見えていれば True を返す。"""
    cropped, screen_type = crop_helper_reset_region(frame, debug_name="has_active_composition_panel")
    if screen_type == "patient_record":
        cropped = _crop_center_band(
            cropped,
            top_ratio=0.15,
            bottom_ratio=0.80,
        )
    # 暗い反転/選択領域が物理的に見えない場合は、VLM の hallucination による
    # false positive で追加 Backspace を送らないようにする。
    if _find_dark_region_y(cropped) is None:
        return False
    prompt = (
        "この画像はEHRの入力欄周辺です。"
        "Windows IME の未確定組成文字だけを判定してください。"
        "未確定組成とは、下線付き・反転表示・変換中の候補表示になっている入力中テキストです。"
        "通常の確定済み文字列、ラベル、患者名、一覧表、病名欄の見出しは無視してください。"
        "入力欄の中に未確定組成が見えるなら yes、見えないなら no のみで答えてください。"
    )
    data_url = _encode_image_data_url(cropped, debug_name="has_active_composition")
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_yes_no_response(content) is True


def _collect_helper_words(payload: object) -> list[str]:
    """Qwen の変形 JSON から helper word 候補文字列を抽出する。"""
    words: list[str] = []
    if isinstance(payload, str):
        word = payload.strip()
        if word:
            words.append(word)
        return words
    if isinstance(payload, list):
        for item in payload:
            words.extend(_collect_helper_words(item))
        return words
    if isinstance(payload, dict):
        for key in ("word", "words"):
            if key in payload:
                words.extend(_collect_helper_words(payload[key]))
        return words
    return words


def suggest_ime_helper_word(target: str) -> list[dict]:
    """IME変換が困難な漢字に対してヘルパー単語の候補リストを提案する。

    Qwen3 に対してテキストのみのリクエストを送り、MS-IME の変換辞書に
    確実に存在する一般的な単語を3つ提案してもらう。変換後に余分な末尾文字を
    Backspace で削除することで目標の漢字を得る。

    backspace_count は Qwen3 には計算させず、
    len(word) - len(target) で機械的に算出する。

    例: target="過" → [{"word": "過去", "backspace_count": 1}, ...]

    Args:
        target: IME変換に失敗した漢字（通常は1文字）

    Returns:
        ヘルパー単語情報のリスト（空リストの場合は提案なし）
        各要素のキー:
            word (str): 提案する日本語単語
            backspace_count (int): len(word) - len(target) (算出値)
    """
    prompt = (
        f"{target}、という漢字をIMEで変換して入力したいです。"
        "この漢字を先頭に含む単語またはフレーズで、MS-IMEの変換候補として出現しやすいものを三つ提案して。"
        "ただし、地名・人名など固有名詞と同じ読みの単語は避けてください（例: 「吸収」は「九州」と同音なので不可）。"
        "JSONのみで回答してください。"
        "返答形式は必ず {\"words\":[\"候補1\",\"候補2\",\"候補3\"]} にしてください。"
        "\"word\" や他のキーは使わないでください。説明文やコードフェンスは禁止です。"
    )
    runtime_label = describe_runtime(url=MLX_VLM_TEXT_URL, model=MLX_VLM_TEXT_MODEL, default_kind="Text")
    try:
        raw = _call_mlx_vlm_text_only(prompt, model=MLX_VLM_TEXT_MODEL)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        print(f"  [ヘルパー単語提案] {runtime_label}応答: {raw!r}")

        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`").strip()

        payloads: list[object] = []
        try:
            payloads.append(json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        # まず JSON 配列 [...] を試みる
        if not payloads:
            arr_m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
            if arr_m:
                try:
                    payloads.append(json.loads(arr_m.group()))
                except json.JSONDecodeError:
                    pass

        # 配列/全体JSONが見つからなければ個別の {...} オブジェクトを順に抽出
        if not payloads:
            for obj_m in re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL):
                try:
                    payloads.append(json.loads(obj_m.group()))
                except json.JSONDecodeError:
                    continue

        if not payloads:
            print("  [ヘルパー単語提案] JSONが見つかりません")
            return []

        seen: set[str] = set()
        result = []
        for payload in payloads:
            try:
                words = _collect_helper_words(payload)
            except (TypeError, ValueError):
                continue
            for word in words:
                if word in seen:
                    continue
                seen.add(word)
                backspace_count = len(word) - len(target)
                if backspace_count <= 0:
                    continue
                result.append({"word": word, "backspace_count": backspace_count})

        print(f"  [ヘルパー単語提案] 有効な提案: {result}")
        return result
    except json.JSONDecodeError as exc:
        print(f"  [ヘルパー単語提案] JSON解析失敗: {exc}")
        return []
    except MlxVlmImeError as exc:
        print(f"  [ヘルパー単語提案] {runtime_label}呼び出し失敗: {exc}")
        return []
