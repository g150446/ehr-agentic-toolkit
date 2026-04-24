#!/usr/bin/env python3
"""
detect_ime_popup.py

IME漢字変換候補ポップアップを画面キャプチャから検出・切り取りするスクリプト。
オプションでOpenRouter経由のVLMに送信し、変換候補一覧を番号付きで取得できます。

--vlm を指定すると、切り取り後に EasyOCR でテキスト認識 → OCR結果をプロンプトに
含めて VLM（google/gemma-4-31b-it）に送信し、補正済み候補一覧を JSON で出力します。

Usage:
    python scripts/detect_ime_popup.py captures/popup1.png.jpg
    python scripts/detect_ime_popup.py captures/*.jpg --vlm
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# automation モジュールをインポート可能にする
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import cv2
import numpy as np

from automation.mlx_vlm_ime import (
    _call_mlx_vlm_with_image,
    _encode_image_data_url,
    MlxVlmImeError,
)
from automation.screen_analyzer import load_ocr_reader, run_ocr

_VLM_MODEL = "google/gemma-4-31b-it"
_VLM_URL = "https://openrouter.ai/api/v1/chat/completions"

# EasyOCR reader は遅延初期化・キャッシュ
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = load_ocr_reader(languages=["ja", "en"], use_gpu=False)
    return _ocr_reader


def _run_easyocr_on_roi(roi: np.ndarray) -> list[tuple[str, float]]:
    """
    切り取ったポップアップ画像を EasyOCR で認識し、
    (テキスト, confidence) のリストを返す。
    Y座標でソート済み。
    """
    reader = _get_ocr_reader()
    results = run_ocr(reader, roi)
    # results: List[([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence)]
    # Y座標（バウンディングボックスの左上 y）でソート
    sorted_results = sorted(results, key=lambda r: r[0][0][1])
    return [(text, conf) for (_, text, conf) in sorted_results]


def _build_vlm_prompt_with_ocr(ocr_texts: list[str]) -> str:
    """
    OCR結果を埋め込んだ VLM プロンプトを生成。
    """
    ocr_lines = "\n".join(f"- {t}" for t in ocr_texts)
    prompt = (
        "この画像はWindows日本語IMEの変換候補ポップアップです。\n"
        "OCRで読み取った結果は以下の通りです（順不同・誤認識を含む）：\n"
        f"{ocr_lines}\n\n"
        "画像を確認して、上記OCR結果を補正し、ポップアップ内に表示されている\n"
        "正確な変換候補一覧を数字1〜9付きでJSON形式で返してください。\n"
        "誤認識（例:「買」が「貫」になっている等）があれば修正してください。\n"
        "候補が存在しない番号は含めないでください。JSONのみ返してください。\n\n"
        '形式: {"candidates": [{"n": 1, "text": "買"}, {"n": 2, "text": "科"}, {"n": 3, "text": "か"}, {"n": 4, "text": "課"}]}'
    )
    return prompt


def _find_edge_by_brightness_gradient(
    gray: np.ndarray,
    start_y: int,
    direction: int,
    x1: int,
    x2: int,
    max_steps: int = 20,
    step_size: int = 5,
    edge_threshold: int = 15,
) -> int:
    """
    輝度勾配でエッジ（境界）を探す。
    direction=+1 で下方向、-1 で上方向。
    """
    h = gray.shape[0]
    cx = (x1 + x2) // 2
    prev_val = int(gray[start_y, cx])

    for i in range(1, max_steps + 1):
        y = start_y + direction * i * step_size
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        curr_val = int(gray[y, cx])
        if abs(curr_val - prev_val) > edge_threshold:
            return y
        prev_val = curr_val

    y = start_y + direction * max_steps * step_size
    return max(0, min(h - 1, y))


def _score_candidate_by_white_rows(
    gray: np.ndarray, x: int, y: int, w: int, h: int
) -> int:
    """
    水色選択行候補の上方向に、白背景行がいくつ続くか数える。
    """
    step = max(h + 4, 12)
    white_rows = 0
    miss = 0
    x1 = max(0, x - 5)
    x2 = min(gray.shape[1], x + w + 5)
    for i in range(1, 12):
        check_y = y - i * step
        if check_y < 0:
            break
        line_slice = gray[check_y : check_y + step, x1:x2]
        if line_slice.size == 0:
            continue
        white_ratio = np.sum(line_slice > 180) / line_slice.size
        if white_ratio > 0.60:
            white_rows += 1
            miss = 0
        else:
            miss += 1
            if miss >= 2:
                break
    return white_rows


def detect_by_blue_line(image: np.ndarray) -> tuple[np.ndarray, tuple] | tuple[None, None]:
    """
    水色選択行を検出し、ポップアップ全体を切り取る。
    """
    fh, fw = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_cyan = np.array([75, 20, 150])
    upper_cyan = np.array([105, 255, 255])
    mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 50 or w > 500:
            continue
        if h < 10 or h > 100:
            continue
        if y < fh * 0.05 or y > fh * 0.95:
            continue
        if x > fw * 0.95:
            continue
        if w < h * 1.5:
            continue
        if w > h * 8:
            continue
        area = w * h
        candidates.append((x, y, w, h, area))

    if not candidates:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best = None
    best_score = -1

    for x, y, w, h, area in candidates:
        white_rows = _score_candidate_by_white_rows(gray, x, y, w, h)
        score = white_rows * 10 + (1 if 1000 < area < 20000 else 0)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None or best_score < 5:
        return None, None

    x, y, w, h = best

    roi_x1 = max(0, x - 10)
    roi_x2 = min(fw, x + w + 10)

    roi_y1 = _find_edge_by_brightness_gradient(
        gray, y, direction=-1, x1=roi_x1, x2=roi_x2, max_steps=20, step_size=max(h // 2, 8)
    )
    roi_y1 = max(0, roi_y1 - 2)
    min_y1 = max(0, y - int(h * 4))
    if roi_y1 > min_y1:
        roi_y1 = min_y1

    roi_y2 = _find_edge_by_brightness_gradient(
        gray, y + h, direction=+1, x1=roi_x1, x2=roi_x2, max_steps=20, step_size=max(h // 2, 8)
    )
    roi_y2 = min(fh, roi_y2 + 2)
    min_y2 = min(fh, y + h + int(h * 3))
    if roi_y2 < min_y2:
        roi_y2 = min_y2

    if roi_x2 - roi_x1 < 60 or roi_y2 - roi_y1 < 40:
        return None, None

    if roi_y2 - roi_y1 > h * 12:
        roi_y1 = max(0, y - int(h * 2))
        roi_y2 = min(fh, roi_y1 + int(h * 10))

    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    return roi, (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)


def _extract_text_rows(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    画像から「文字っぽい小さな暗い輪郭」を抽出し、水平テキスト行のリストを返す。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 80 < area < 6000 and 6 < w < 180 and 8 < h < 70:
            text_boxes.append((x, y, w, h))

    text_boxes.sort(key=lambda b: (b[1], b[0]))

    rows = []
    if not text_boxes:
        return rows

    current = [text_boxes[0]]
    for box in text_boxes[1:]:
        if abs(box[1] - current[-1][1]) <= max(20, int(current[-1][3] * 0.8)):
            current.append(box)
        else:
            xs = [b[0] for b in current]
            ys = [b[1] for b in current]
            x2s = [b[0] + b[2] for b in current]
            y2s = [b[1] + b[3] for b in current]
            rows.append((min(xs), min(ys), max(x2s), max(y2s)))
            current = [box]

    if current:
        xs = [b[0] for b in current]
        ys = [b[1] for b in current]
        x2s = [b[0] + b[2] for b in current]
        y2s = [b[1] + b[3] for b in current]
        rows.append((min(xs), min(ys), max(x2s), max(y2s)))

    return rows


def detect_by_white_rectangle(image: np.ndarray) -> tuple[np.ndarray, tuple] | tuple[None, None]:
    h_img, w_img = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 80 or h < 40 or w > 600 or h > 500:
            continue
        aspect = w / max(h, 1)
        if aspect > 6 or aspect < 0.3:
            continue
        if w * h > (h_img * w_img) * 0.5:
            continue
        candidates.append((x, y, w, h))

    best_roi = None
    best_rect = None
    best_score = -1

    for x, y, w, h in candidates:
        roi = image[y:y + h, x:x + w]
        rows = _extract_text_rows(roi)
        if len(rows) >= 3:
            score = len(rows)
            if score > best_score:
                best_score = score
                best_roi = roi
                best_rect = (x, y, w, h)

    if best_roi is not None:
        return best_roi, best_rect
    return None, None


def detect_by_text_clustering(image: np.ndarray) -> tuple[np.ndarray, tuple] | tuple[None, None]:
    h_img, w_img = image.shape[:2]
    rows = _extract_text_rows(image)
    if len(rows) < 3:
        return None, None

    row_centers = [(r[0], r[1], r[2], r[3], (r[1] + r[3]) // 2) for r in rows]

    blocks = []
    current = [row_centers[0]]
    for row in row_centers[1:]:
        prev = current[-1]
        cy = row[4]
        prev_cy = prev[4]
        if cy - prev_cy <= 55:
            current.append(row)
        else:
            if len(current) >= 3:
                xs = [r[0] for r in current]
                ys = [r[1] for r in current]
                x2s = [r[2] for r in current]
                y2s = [r[3] for r in current]
                blocks.append((min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys), len(current)))
            current = [row]

    if len(current) >= 3:
        xs = [r[0] for r in current]
        ys = [r[1] for r in current]
        x2s = [r[2] for r in current]
        y2s = [r[3] for r in current]
        blocks.append((min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys), len(current)))

    if not blocks:
        return None, None

    best = max(blocks, key=lambda b: b[4])
    x, y, w, h, row_cnt = best

    if w < 80 or h < 40 or w > 600 or h > 500:
        return None, None

    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    roi = image[y1:y2, x1:x2]
    return roi, (x1, y1, x2 - x1, y2 - y1)


def detect_ime_popup(image_path: Path) -> tuple[np.ndarray, tuple] | tuple[None, None]:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] 画像の読み込みに失敗: {image_path}", file=sys.stderr)
        return None, None

    roi, rect = detect_by_blue_line(img)
    if roi is not None:
        return roi, rect

    roi, rect = detect_by_white_rectangle(img)
    if roi is not None:
        return roi, rect

    roi, rect = detect_by_text_clustering(img)
    if roi is not None:
        return roi, rect

    return None, None


def _parse_vlm_candidates(content: str) -> list[tuple[int, str]]:
    """
    VLM応答文字列から番号付き候補リストを抽出する。
    """
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    cleaned = content
    # Remove markdown code fences
    cleaned = re.sub(r"```[a-zA-Z0-9]*\n", "", cleaned)
    cleaned = re.sub(r"```", "", cleaned).strip()

    # Try JSON parse first
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        json_str = m.group()
        json_str = re.sub(r"\}\s*\{", "}, {", json_str)
        try:
            parsed = json.loads(json_str)
            items = parsed.get("candidates", [])
            result = []
            for item in items:
                n = item.get("n")
                text = item.get("text", "")
                if isinstance(n, int) and 1 <= n <= 9 and text and len(text) <= 25:
                    result.append((n, text))
            if result:
                return result
        except json.JSONDecodeError:
            pass

    # Regex fallback
    pairs = re.findall(r'"n"\s*:\s*(\d+)[^}]{0,80}?"text"\s*:\s*"([^"]+)"', cleaned)
    if not pairs:
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
    return []


def read_candidates_with_vlm_ocr(roi: np.ndarray) -> tuple[list[tuple[str, float]], list[tuple[int, str]]]:
    """
    EasyOCR でテキスト認識 → OCR結果をプロンプトに含めて VLM に送信。

    Returns:
        (ocr_raw_results, vlm_candidates)
        ocr_raw_results: List[(text, confidence)]
        vlm_candidates:  List[(n, text)]
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY 環境変数が設定されていません。")

    # 1. EasyOCR で認識
    ocr_results = _run_easyocr_on_roi(roi)
    ocr_texts = [t for t, _ in ocr_results]

    # 2. VLM プロンプトに OCR 結果を含める
    data_url = _encode_image_data_url(roi)
    prompt = _build_vlm_prompt_with_ocr(ocr_texts)

    try:
        content = _call_mlx_vlm_with_image(
            data_url,
            prompt,
            model=_VLM_MODEL,
            url=_VLM_URL,
            api_key=api_key,
            timeout=90,
        )
        vlm_candidates = _parse_vlm_candidates(content)
        return ocr_results, vlm_candidates
    except MlxVlmImeError as exc:
        print(f"  [VLM] 候補取得失敗: {exc}", file=sys.stderr)
        return ocr_results, []


def main():
    parser = argparse.ArgumentParser(
        description="IME漢字変換候補ポップアップを検出して切り取り、capturesフォルダに保存します。"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="対象の画像ファイルパス（複数指定可）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="captures",
        help="切り取った画像の保存先ディレクトリ（デフォルト: captures）",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        default="ime_popup_",
        help="出力ファイル名の接頭辞（デフォルト: ime_popup_）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="検出矩形を元画像に描画したデバッグ画像も保存する",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help=(
            "切り取ったポップアップを EasyOCR で認識し、その結果をプロンプトに"
            "含めて VLM に送信。補正済み候補一覧を JSON で出力する"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved_count = 0

    for image_path in args.images:
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[SKIP] ファイルが存在しません: {image_path}", file=sys.stderr)
            continue

        print(f"処理中: {image_path}", file=sys.stderr)
        roi, rect = detect_ime_popup(image_path)

        record: dict = {
            "file": str(image_path),
            "popup_rect": None,
            "ocr_raw": [],
            "candidates": [],
        }

        if roi is not None:
            out_name = f"{args.prefix}{image_path.name}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), roi)
            print(f"  -> 保存: {out_path}  (矩形: {rect})", file=sys.stderr)
            saved_count += 1

            record["popup_rect"] = list(rect)

            if args.debug and rect is not None:
                img = cv2.imread(str(image_path))
                x, y, w, h = rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                debug_name = f"{args.prefix}debug_{image_path.name}"
                cv2.imwrite(str(output_dir / debug_name), img)
                print(f"     デバッグ: {output_dir / debug_name}", file=sys.stderr)

            if args.vlm:
                ocr_results, vlm_candidates = read_candidates_with_vlm_ocr(roi)
                record["ocr_raw"] = [
                    {"text": text, "confidence": round(float(conf), 4)}
                    for text, conf in ocr_results
                ]
                record["candidates"] = [{"n": n, "text": text} for n, text in vlm_candidates]
        else:
            print(f"  -> ポップアップ未検出", file=sys.stderr)

        results.append(record)

    print(file=sys.stderr)
    print(f"完了: {saved_count}/{len(args.images)} 件を検出・保存しました。", file=sys.stderr)

    if args.vlm:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
