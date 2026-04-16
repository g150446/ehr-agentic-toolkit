"""History date finder backed by EasyOCR + local mlx_vlm.server.

Runs full-image EasyOCR to collect coordinate anchors for date-like rows, then
asks a local multimodal MLX VLM to read the visible history dates in top-to-
bottom order from the screenshot itself. EasyOCR data is used only to estimate
clickable positions and row ordering.

Usage:
    # Start mlx_vlm server first:
    #   bash scripts/start_mlx_vlm_server.sh qwen
    #
    # Test against a captured image:
    python -m automation.mlx_vlm_history captures/history.jpg 20260312
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
from automation.screen_analyzer import load_ocr_reader, run_ocr


MLX_VLM_HISTORY_URL = os.getenv(
    "MLX_VLM_HISTORY_URL",
    "http://localhost:8000/v1/chat/completions",
)
MLX_VLM_HISTORY_MODEL = os.getenv(
    "MLX_VLM_HISTORY_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "Qwen3.5-9B-MLX-4bit"),
)
MLX_VLM_HISTORY_API_KEY = os.getenv("MLX_VLM_HISTORY_API_KEY", "omlxkey")
MLX_VLM_HISTORY_TIMEOUT = float(os.getenv("MLX_VLM_HISTORY_TIMEOUT", "120"))


class MlxVlmHistoryError(RuntimeError):
    """Raised when VLM history finding fails or returns unusable data."""


def _date_matches_text(text: str, year: int, month: int, day: int) -> bool:
    """OCRテキストが指定年月日を含むか正規表現で検証する。

    OCRの既知誤読パターンを許容:
    - 1桁月に "1" または "日" が前置される (4月→14月, 4月→日4月)
    - 数字 "8" が漢字 "日" と誤認識される
    - 1桁日の先頭ゼロが省略される
    - 10-19日の先頭 "1" が重複する (10日→110日)
    """
    def _dp(d: str) -> str:
        return '[8日]' if d == '8' else re.escape(d)

    # 1桁月: 前に "0"（ゼロ埋め03月）"1"（誤読14月）"日"（誤読日4月）が付く場合を許容
    month_p = f'[01日]?{month}' if month < 10 else str(month)
    if day < 10:
        day_p = f'0?{_dp(str(day))}'
    else:
        day_s = f'{day:02d}'
        day_core = ''.join(_dp(d) for d in day_s)
        day_p = f'1?{day_core}' if day_s.startswith('1') else day_core
    pattern = re.compile(f'{year}年{month_p}月{day_p}日')
    return bool(pattern.search(text))


def _build_candidates(ocr_results: list) -> list[tuple[int, str, int, int]]:
    """Collect coordinate anchors for rows likely to contain history dates.

    Returns list of (list_index, text, cx, cy).
    The OCR text is retained for local debugging, but downstream VLM prompting
    uses only candidate coordinates so date recognition comes from the image.
    """
    candidates = []
    for idx, (bbox, text, conf) in enumerate(ocr_results):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int((min(xs) + max(xs)) / 2)
        cy = int((min(ys) + max(ys)) / 2)

        # Exclude left column (簡略化履歴一覧), right-side notes, and the header area.
        if cx < 200 or cx > 700 or cy < 250:
            continue

        marker_count = sum(marker in text for marker in ("年", "月", "日"))
        digit_count = sum(ch.isdigit() for ch in text)

        # Keep broad "date-like" anchors; VLM decides the actual date from image.
        if marker_count >= 2 or ("年" in text and digit_count >= 4) or digit_count >= 6:
            candidates.append((idx, text, cx, cy))

    return candidates


def _build_prompt(date_str: str, candidate_count: int) -> str:
    """Build the multimodal prompt for ordered history-date reading."""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # ゼロ埋め・ゼロなし両方の表記を提示する
    month_fmt = f"{month:02d}月 または {month}月"
    day_fmt = f"{day:02d}日 または {day}日"

    return (
        f"以下は電子カルテの過去カルテ欄を切り出した画像です。\n"
        f"画像を見て、過去カルテ欄に表示されている日付の一覧を上から下へ順番どおりに作成してください。\n"
        f"EasyOCR 候補数は {candidate_count} 件ですが、OCR 文字は信用しなくてよいです。\n\n"
        f"【探す日付】{year}年 {month_fmt} {day_fmt}\n"
        f"（ゼロ埋めの有無は問いません。例: 3月2日 と 03月02日 は同じ日付です）\n\n"
        f"ルール:\n"
        f"- 日付の読み取りは画像だけを根拠にしてください。\n"
        f"- 時刻や本文は無視し、各行の日付だけを 1 つずつ返してください。\n"
        f"- OCR由来の画像なので一部欠けやノイズがあっても、画像上で最も自然に読める日付へ補正してください。\n"
        f"- 探す日付と一致しない近い日付へ寄せてはいけません。\n"
        f"- 日付が 1 件も読めない場合は空配列を返してください。\n"
        f"JSON のみで回答してください。形式は {{\"dates\": [\"2026年4月8日\", \"2026年4月7日\"]}} です。"
    )


def _encode_image_data_url(image) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise MlxVlmHistoryError("VLM送信用の画像エンコードに失敗しました")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_history_crop(image, candidates: list[tuple[int, str, int, int]]):
    height, width = image.shape[:2]
    xs = [cx for _, _, cx, _ in candidates]
    ys = [cy for _, _, _, cy in candidates]

    x1 = max(min(xs) - 160, 0)
    x2 = min(max(xs) + 320, width)
    y1 = max(min(ys) - 140, 0)
    y2 = min(max(ys) + 80, height)

    if x2 <= x1 or y2 <= y1:
        raise MlxVlmHistoryError("過去カルテ欄の画像切り出しに失敗しました")

    return image[y1:y2, x1:x2].copy()


def _extract_response_content(result: dict) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MlxVlmHistoryError(f"VLM応答の解析に失敗しました: {result!r}") from exc

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        joined = "".join(text_parts).strip()
        if joined:
            return joined

    raise MlxVlmHistoryError(f"VLM応答に content テキストが含まれていません: {result!r}")


def _parse_ordered_dates(content: str) -> Optional[list[str]]:
    stripped = content.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        dates = parsed.get("dates")
        if isinstance(dates, list) and all(isinstance(item, str) for item in dates):
            normalized = [_normalize_vlm_date(item) for item in dates]
            return [item for item in normalized if item is not None]

    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        normalized = [_normalize_vlm_date(item) for item in parsed]
        return [item for item in normalized if item is not None]

    matches = re.findall(r"\b(20\d{6})\b", stripped)
    if matches:
        return matches
    return None


def _normalize_vlm_date(text: str) -> Optional[str]:
    direct = re.search(r"\b(20\d{6})\b", text)
    if direct:
        return direct.group(1)

    match = re.search(r"(20\d{2})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})", text)
    if not match:
        return None

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}{month:02d}{day:02d}"


def _run_full_image_ocr(image, languages: list[str] | None = None) -> list[tuple]:
    reader = load_ocr_reader(languages or ["ja", "en"])
    return run_ocr(reader, image)


def find_history_date_with_vlm(
    date_str: str,
    ocr_results: list,
    *,
    image=None,
    model: str = MLX_VLM_HISTORY_MODEL,
    url: str = MLX_VLM_HISTORY_URL,
    timeout: float = MLX_VLM_HISTORY_TIMEOUT,
    api_key: str = MLX_VLM_HISTORY_API_KEY,
) -> Optional[Tuple[int, int]]:
    """Use local mlx_vlm to identify which OCR segment matches the target date.

    Args:
        date_str:    Target date in yyyymmdd format (e.g. "20260312").
        ocr_results: List of (bbox, text, conf) from run_ocr().
        image:       Source screenshot for multimodal fallback.
        model:       mlx_vlm model identifier.
        url:         Chat completion endpoint served by mlx_vlm.server.
        timeout:     Request timeout in seconds.

    Returns:
        (cx, cy) pixel coordinates of the matched entry, or None if not found.

    Raises:
        MlxVlmHistoryError: On network / response parsing failures.
    """
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    all_candidates = _build_candidates(ocr_results)
    if not all_candidates:
        raise MlxVlmHistoryError("OCR候補が見つかりませんでした（過去カルテ列にテキストなし）")

    print(f"全候補数: {len(all_candidates)} セグメント")
    for pos, (_, text, cx, cy) in enumerate(all_candidates):
        print(f"  [{pos}] ({cx},{cy}) {text!r}")

    if image is None:
        regex_matched = [
            (pos, text, cx, cy)
            for pos, (_, text, cx, cy) in enumerate(all_candidates)
            if _date_matches_text(text, year, month, day)
        ]
        print(f"正規表現一致: {len(regex_matched)} 件")
        if len(regex_matched) == 1:
            _, text, cx, cy = regex_matched[0]
            print(f"正規表現で一意特定: {text!r} at ({cx},{cy})")
            return (cx, cy)
        if len(regex_matched) > 1:
            _, text, cx, cy = min(regex_matched, key=lambda item: (item[3], item[2]))
            print(f"複数一致のため最上段を採用: {text!r} at ({cx},{cy})")
            return (cx, cy)
        raise MlxVlmHistoryError("画像なしでは対象日付を一意に判定できませんでした")

    print("画像優先モード: 日付一覧の読取りは VLM に委譲します")
    vlm_candidates = sorted(all_candidates, key=lambda item: (item[3], item[2]))
    prompt = _build_prompt(date_str, len(vlm_candidates))
    history_crop = _build_history_crop(image, vlm_candidates)

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "Read the visible history dates from the image and reply with JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(history_crop)}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 128,
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
            response_body = resp.read()
    except (TimeoutError, socket.timeout) as exc:
        raise MlxVlmHistoryError(
            f"VLMへのリクエストが {timeout:g} 秒でタイムアウトしました。"
            f" endpoint={url} model={model}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise MlxVlmHistoryError(
                f"VLMへのリクエストが {timeout:g} 秒でタイムアウトしました。"
                f" endpoint={url} model={model}"
            ) from exc
        raise MlxVlmHistoryError(
            f"VLMへの接続に失敗しました: {reason}. endpoint={url} model={model}"
        ) from exc

    try:
        result = json.loads(response_body)
        content = _extract_response_content(result)
    except json.JSONDecodeError as exc:
        raise MlxVlmHistoryError(
            f"VLM応答の解析に失敗しました: {response_body!r}"
        ) from exc

    print(f"VLM応答: {content!r}")

    ordered_dates = _parse_ordered_dates(content)
    if ordered_dates is None:
        print(f"⚠️ VLM応答から日付一覧を抽出できませんでした: {content!r}")
        return None

    print(f"VLM日付一覧: {ordered_dates}")
    if not ordered_dates:
        print("VLM: 該当エントリなし")
        return None

    try:
        rank = ordered_dates.index(date_str)
    except ValueError:
        print(f"VLM日付一覧に対象日付 {date_str} がありません")
        return None

    if rank >= len(vlm_candidates):
        raise MlxVlmHistoryError(
            f"VLM日付順位 {rank} に対応するOCR候補がありません "
            f"(dates={len(ordered_dates)}, candidates={len(vlm_candidates)})"
        )

    _, text, cx, cy = vlm_candidates[rank]
    print(f"VLM特定: rank[{rank}] {text!r} at ({cx},{cy})")
    return (cx, cy)


def find_history_date_in_image(
    date_str: str,
    image,
    *,
    languages: list[str] | None = None,
    model: str = MLX_VLM_HISTORY_MODEL,
    url: str = MLX_VLM_HISTORY_URL,
    timeout: float = MLX_VLM_HISTORY_TIMEOUT,
) -> Optional[Tuple[int, int]]:
    """Run full-image EasyOCR, then identify the target history date."""
    ocr_results = _run_full_image_ocr(image, languages)
    return find_history_date_with_vlm(
        date_str,
        ocr_results,
        image=image,
        model=model,
        url=url,
        timeout=timeout,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for testing against a saved image."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("使い方:")
        print("  python -m automation.mlx_vlm_history <image_path> <yyyymmdd>")
        print("  python -m automation.mlx_vlm_history captures/history.jpg 20260312")
        return 1

    image_path, date_str = args[0], args[1]

    if not Path(image_path).exists():
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return 1

    if len(date_str) != 8 or not date_str.isdigit():
        print(f"❌ 日付は yyyymmdd 形式で指定してください: {date_str!r}")
        return 1

    print(f"画像: {image_path}")
    print(f"対象日付: {date_str}")
    print(f"endpoint: {MLX_VLM_HISTORY_URL}")
    print(f"model: {MLX_VLM_HISTORY_MODEL}")
    print(f"timeout: {MLX_VLM_HISTORY_TIMEOUT:g}秒\n")

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 画像を読み込めません: {image_path}")
        return 1

    print("OCR実行中...")
    try:
        ocr_results = _run_full_image_ocr(image, ["ja", "en"])
    except Exception as exc:
        print(f"❌ OCR エラー: {exc}")
        return 1
    print(f"OCR結果: {len(ocr_results)} セグメント\n")

    try:
        coords = find_history_date_with_vlm(date_str, ocr_results, image=image)
    except MlxVlmHistoryError as exc:
        print(f"❌ エラー: {exc}")
        return 1

    if coords:
        x, y = coords
        print(f"\n✅ クリック座標: ({x}, {y})")
    else:
        print("\n❌ 該当エントリが見つかりませんでした")

    # --- テンプレートマッチング（修正ボタン） ---
    template_path = Path(__file__).parent.parent / "match_templates" / "edit_button.jpg"
    if template_path.exists():
        print(f"\nテンプレートマッチング: {template_path}")
        tmpl = cv2.imread(str(template_path))
        if tmpl is not None:
            match_result = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
            print(f"最高スコア: {max_val:.3f}")
            if max_val >= 0.7:
                th, tw = tmpl.shape[:2]
                cx = max_loc[0] + tw // 2
                cy = max_loc[1] + th // 2
                print(f"✅ 修正ボタン検出: ({cx}, {cy})")
            else:
                print(f"❌ 修正ボタン未検出 (スコア {max_val:.3f} < 0.7)")
        else:
            print(f"❌ テンプレート画像を読み込めません: {template_path}")
    else:
        print(f"⚠️ テンプレートファイルが見つかりません: {template_path}")

    return 0 if coords else 1


if __name__ == "__main__":
    sys.exit(main())
