"""mlx_vlm server を使った IME 候補読み取りヘルパー。

mlx_vlm.server（OpenAI 互換 API）に画像を送信し、
IME インライン変換候補やポップアップ候補リストを読み取る。
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

MLX_VLM_IME_URL = os.getenv(
    "MLX_VLM_IME_URL",
    os.getenv("MLX_VLM_SEGMENTATION_URL", "http://localhost:8181/v1/chat/completions"),
)
MLX_VLM_IME_MODEL = os.getenv(
    "MLX_VLM_IME_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit"),
)
MLX_VLM_IME_TIMEOUT = float(os.getenv("MLX_VLM_IME_TIMEOUT", "90"))
# Shorter timeout for inline candidate reads (ROI/fullframe).
# When the server is slow, we want to quickly fall through to popup mode.
MLX_VLM_INLINE_TIMEOUT = float(os.getenv("MLX_VLM_INLINE_TIMEOUT", "15"))


class MlxVlmImeError(RuntimeError):
    """Raised when mlx_vlm IME call fails."""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_MIN_FRAME_HEIGHT = 80
_MIN_FRAME_WIDTH = 200


def _ensure_min_size(frame: np.ndarray) -> np.ndarray:
    """VLM が処理できる最小サイズに拡大する（アスペクト比維持）。"""
    h, w = frame.shape[:2]
    scale = max(_MIN_FRAME_HEIGHT / h, _MIN_FRAME_WIDTH / w, 1.0)
    if scale <= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _encode_image_data_url(frame: np.ndarray) -> str:
    """numpy BGR フレームを data URI 形式の PNG base64 文字列に変換する。"""
    frame = _ensure_min_size(frame)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise MlxVlmImeError("画像の PNG エンコードに失敗しました")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Geometry helpers (ported from ollama_vlm_ime.py)
# ---------------------------------------------------------------------------

def detect_patient_record_panel3(frame: np.ndarray) -> Optional[tuple[int, int]]:
    """患者記録画面の第3ペインの x 座標範囲を検出する。"""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=200,
        minLineLength=h // 3,
        maxLineGap=30,
    )
    if lines is None:
        return None

    vert_xs: list[int] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 75:
            vert_xs.append(int(x1))

    if not vert_xs:
        return None

    vert_xs.sort()
    clusters: list[int] = []
    current: list[int] = [vert_xs[0]]
    for x in vert_xs[1:]:
        if x - current[-1] < 30:
            current.append(x)
        else:
            clusters.append(int(np.mean(current)))
            current = [x]
    clusters.append(int(np.mean(current)))

    main_dividers = [x for x in clusters if 50 < x < w - 50]
    if len(main_dividers) < 3:
        return None

    return (main_dividers[1], main_dividers[2])


def crop_to_input_region(frame: np.ndarray) -> np.ndarray:
    """フレームをテキスト入力領域にクロップする（第3ペイン検出 → フォールバック全画面）。"""
    panel = detect_patient_record_panel3(frame)
    if panel is not None:
        x1, x2 = panel
        return frame[:, x1:x2]
    return frame


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
        # Win11縦バー: 幅3px程度まで許容。Win10横バー: 幅30-450px
        if w < 3 or w > 450 or h < 8 or h > 80:
            continue
        # タスクバー・画面上端付近を除外
        if y < fh * 0.04 or y > fh * 0.92:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if best is None:
        return None

    x, y, w, h = best
    is_vertical_bar = h >= w * 1.5
    # 選択候補の上にある候補行は最大 1 行分のみ許可（ツールバーを除外するため浅めに切る）
    # 下方向は最大 8 候補分（各行 ~25px = 200px）+ 余白
    popup_y1 = max(0, y - 35)
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
    roi = crop_to_input_region(frame)
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
    import os
    captures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
    os.makedirs(captures_dir, exist_ok=True)
    path = os.path.join(captures_dir, f"debug_popup_crop_{name}.png")
    cv2.imwrite(path, frame)
    print(f"  [debug] ポップアップクロップ保存: {path}")


# ---------------------------------------------------------------------------
# mlx_vlm API call
# ---------------------------------------------------------------------------

def _call_mlx_vlm_with_image(
    image_data_url: str,
    prompt: str,
    *,
    model: str = MLX_VLM_IME_MODEL,
    url: str = MLX_VLM_IME_URL,
    timeout: float = MLX_VLM_IME_TIMEOUT,
) -> str:
    """mlx_vlm.server の OpenAI 互換エンドポイントに画像付きリクエストを送信する。"""
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
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

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MlxVlmImeError(
            f"mlx_vlm IME 応答に content テキストがありません: {result!r}"
        ) from exc

    if not isinstance(content, str):
        raise MlxVlmImeError(f"mlx_vlm IME 応答に content テキストがありません: {result!r}")
    return content.strip()


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

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
    data_url = _encode_image_data_url(roi)
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_candidate_response(content)


def read_inline_candidate_context(frame: np.ndarray) -> Optional[str]:
    """全画面フレームから入力フィールド行の変換候補を読み取る（フォールバック用）。

    Args:
        frame: 全画面フレーム

    Returns:
        変換中の文字列のみ、または None
    """
    cropped = _crop_center_band(crop_to_input_region(frame), top_ratio=0.05, bottom_ratio=0.75)
    prompt = (
        "この画像はWindowsの画面の一部（入力フィールド周辺）です。"
        "テキスト入力フィールドの中で、下線または黒背景・白文字で強調表示されている、"
        "現在変換中の文字列（1〜4文字程度の漢字やひらがな）だけを読み取ってください。"
        "前後に入力済みの文字列は含めないでください。変換中の部分だけを返してください。"
        "見つかった場合はその文字列のみを返してください。見つからない場合は null を返してください。"
        "JSONのみで回答してください（余分なコメントや思考過程は不要）。"
        ' 形式: {"candidate": "文字列"} または {"candidate": null}'
    )
    data_url = _encode_image_data_url(cropped)
    content = _call_mlx_vlm_with_image(data_url, prompt, timeout=MLX_VLM_INLINE_TIMEOUT)
    return _parse_candidate_response(content)


def read_popup_candidates_numbered(frame: np.ndarray, *, debug_name: str = "") -> list[tuple[int, str]]:
    """IME ポップアップ候補リストを番号付きで読み取る（1回のVLMコールで番号+テキストを取得）。

    VLM に番号とテキストの両方を返すよう指示し、表示番号の信頼性を高める。
    `read_popup_candidates()` + `read_candidate_display_number()` の2コール相当を1コールで実現。

    Args:
        frame: 全画面フレーム（ポップアップ表示中）

    Returns:
        (display_number, candidate_text) のリスト。表示番号順。失敗時は空リスト。
    """
    cropped = _crop_popup_region(frame, debug_name=debug_name)
    prompt = (
        "この画像はWindows日本語IMEの変換候補ポップアップです。"
        "ポップアップ内の各候補に付いた数字（1〜9）とテキストを正確に読んでください。"
        "各行に「数字. テキスト」の形式で表示されています（例: 1. イン, 2. 院）。"
        "数字とテキストのペアをJSONで返してください。"
        '形式: {"candidates": [{"n": 1, "text": "イン"}, {"n": 2, "text": "院"}, {"n": 3, "text": "員"}, {"n": 4, "text": "咽頭"}]}'
        "候補リスト以外のテキストは含めないでください。JSONのみ返してください。"
    )
    data_url = _encode_image_data_url(cropped)
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
        if pairs:
            result = [(int(n), text) for n, text in pairs if 1 <= int(n) <= 9 and text and len(text) <= 25]
            if result:
                return result
    except Exception as exc:
        print(f"  [mlx_vlm 番号付き候補取得] 失敗: {exc}")
    return []


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
    data_url = _encode_image_data_url(cropped)
    try:
        content = _call_mlx_vlm_with_image(data_url, prompt)
        return _parse_candidates_response(content)
    except MlxVlmImeError as exc:
        print(f"  [mlx_vlm IME候補リスト] 取得失敗: {exc}")
        return []


def detect_ime_mode_from_typed_a(frame: np.ndarray) -> Optional[str]:
    """'a' を入力直後のスクリーンショットから IME モードを判定する。

    呼び出し元が事前に 'a' を1文字入力し、このスクリーンキャプチャを渡す。
    VLM でテキスト入力エリアを見て、'a'（英語モード）または「あ」（日本語モード）
    が表示されているかを判定する。

    Args:
        frame: 'a' 入力直後の全画面 BGR フレーム

    Returns:
        'japanese': ひらがなモード（「あ」が表示されている）
        'english':  英数字モード（'a' が表示されている）
        None:       判定不能
    """
    cropped = crop_to_input_region(frame)
    cropped = _ensure_min_size(cropped)
    data_url = _encode_image_data_url(cropped)
    prompt = (
        "テキスト入力フィールドに 'a' というキーを1回押しました。"
        "画像の入力エリアに何が表示されていますか？"
        "英語入力モードなら半角の 'a' が、日本語入力モードなら平仮名の「あ」が表示されます。"
        "'a' が表示されているなら 'english'、「あ」が表示されているなら 'japanese' とだけ答えてください。"
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
    model: str = MLX_VLM_IME_MODEL,
    url: str = MLX_VLM_IME_URL,
    timeout: float = MLX_VLM_IME_TIMEOUT,
) -> str:
    """mlx_vlm.server に画像なしのテキストのみリクエストを送信する。"""
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
        "max_tokens": 256,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MlxVlmImeError(
            f"mlx_vlm 応答に content テキストがありません: {result!r}"
        ) from exc

    if not isinstance(content, str):
        raise MlxVlmImeError(f"mlx_vlm 応答に content テキストがありません: {result!r}")
    return content.strip()


def suggest_ime_helper_word(target: str) -> list[dict]:
    """IME変換が困難な漢字に対してヘルパー単語の候補リストを提案する。

    Qwen3 に対してテキストのみのリクエストを送り、MS-IME の変換辞書に
    確実に存在する一般的な単語を3つ提案してもらう。変換後に余分な末尾文字を
    Backspace で削除することで目標の漢字を得る。

    例: target="過" → [{"word": "過去", "backspace_count": 1}, ...]

    Args:
        target: IME変換に失敗した漢字（通常は1文字）

    Returns:
        ヘルパー単語情報のリスト（空リストの場合は提案なし）
        各要素のキー:
            word (str): 提案する日本語単語
            backspace_count (int): 単語確定後に Backspace で削除する文字数
    """
    prompt = (
        f"{target}、という漢字をIMEで変換して入力したいです。"
        "この漢字を先頭に含む単語またはフレーズで、変換候補として出現しやすいものを三つ提案して。"
        "その単語を選んだ際に、求める漢字のみを残すための、バックスペースの個数も出力して。"
        "json形式で答えのみ出力して。keyは\"word\",\"backspace_count\""
    )
    try:
        raw = _call_mlx_vlm_text_only(prompt)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        print(f"  [ヘルパー単語提案] Qwen3応答: {raw!r}")

        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`").strip()

        # まず JSON 配列 [...] を試みる
        items: list = []
        arr_m = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if arr_m:
            try:
                parsed = json.loads(arr_m.group())
                if isinstance(parsed, list):
                    items = parsed
            except json.JSONDecodeError:
                pass

        # 配列が見つからなければ個別の {...} オブジェクトを順に抽出
        if not items:
            for obj_m in re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL):
                try:
                    items.append(json.loads(obj_m.group()))
                except json.JSONDecodeError:
                    continue

        if not items:
            print("  [ヘルパー単語提案] JSONが見つかりません")
            return []

        result = []
        for item in items:
            word = item.get("word", "")
            backspace_count = item.get("backspace_count", 0)
            if not isinstance(word, str) or not word:
                continue
            if not isinstance(backspace_count, int) or backspace_count < 0:
                continue
            # backspace_count は len(word) - len(target) でなければならない
            # Qwen3 が誤った値を返すことがあるため強制補正する
            expected_backspace = len(word) - len(target)
            if expected_backspace <= 0:
                # helper word がターゲットより短い or 同じ: 不適切なのでスキップ
                continue
            if backspace_count != expected_backspace:
                print(
                    f"  [ヘルパー単語提案] backspace_count補正: {word!r} "
                    f"{backspace_count} → {expected_backspace}"
                )
                backspace_count = expected_backspace
            result.append({"word": word, "backspace_count": backspace_count})

        print(f"  [ヘルパー単語提案] 有効な提案: {result}")
        return result
    except json.JSONDecodeError as exc:
        print(f"  [ヘルパー単語提案] JSON解析失敗: {exc}")
        return []
    except MlxVlmImeError as exc:
        print(f"  [ヘルパー単語提案] Qwen3呼び出し失敗: {exc}")
        return []
