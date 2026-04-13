"""History date finder backed by EasyOCR + local mlx_vlm.server.

Runs full-image EasyOCR to collect date-like candidates, then asks a local
multimodal MLX VLM to pick the correct candidate using both:

- the screenshot itself
- the EasyOCR candidate list of (date text, position)

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
    "http://localhost:8181/v1/chat/completions",
)
MLX_VLM_HISTORY_MODEL = os.getenv(
    "MLX_VLM_HISTORY_MODEL",
    os.getenv("MLX_VLM_SERVER_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
)
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
    """Filter OCR results to date entries likely in the 過去カルテ column.

    Returns list of (list_index, text, cx, cy).
    Only includes segments that contain ALL of 年・月・日 (full date pattern),
    and excludes the 簡略化履歴一覧 column (x < 200) and far-right menu (x > 1500).
    """
    candidates = []
    for idx, (bbox, text, conf) in enumerate(ocr_results):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = int((min(xs) + max(xs)) / 2)
        cy = int((min(ys) + max(ys)) / 2)

        # Exclude left column (簡略化履歴一覧) and far-right menu
        if cx < 200 or cx > 1500:
            continue

        # Only include segments containing 年・月・日 all three (full date pattern)
        if "年" in text and "月" in text and "日" in text:
            candidates.append((idx, text, cx, cy))

    return candidates


def _build_prompt(date_str: str, candidates: list[tuple[int, str, int, int]]) -> str:
    """Build the multimodal VLM prompt for identifying the target date entry."""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    entries = "\n".join(
        f"  [{pos}] x={cx}, y={cy}: {text!r}"
        for pos, (_, text, cx, cy) in enumerate(candidates)
    )

    # ゼロ埋め・ゼロなし両方の表記を提示する
    month_fmt = f"{month:02d}月 または {month}月"
    day_fmt = f"{day:02d}日 または {day}日"

    return (
        f"以下は電子カルテ画面の画像と、EasyOCR が抽出した日付候補リストです。\n"
        f"画像を見て文字を読み取り、OCR誤認識を補正しながら選んでください。\n"
        f"座標は候補リストにだけ書かれています。\n\n"
        f"【探す日付】{year}年 {month_fmt} {day_fmt}\n"
        f"（ゼロ埋めの有無は問いません。例: 3月2日 と 03月02日 は同じ日付です）\n\n"
        f"候補リスト:\n"
        f"{entries}\n\n"
        f"ルール:\n"
        f"- 画像と候補リストの両方を見て、探す日付と同じ年・月・日の候補番号のみを返してください。\n"
        f"- OCR文字列が少し壊れていても、画像上で同じ日付だと読めるなら選んでください。\n"
        f"- 年・月・日のいずれか一つでも違う候補は選ばないでください。\n"
        f"- 探す日付と一致する候補が一つもない場合は、必ず -1 を返してください。\n"
        f"- 近い日付や似た日付を選ぶことは禁止です。完全一致のみ有効です。\n"
        f"数字のみで回答してください（例: 3 または -1）。"
    )


def _encode_image_data_url(image) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise MlxVlmHistoryError("VLM送信用の画像エンコードに失敗しました")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


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


def _parse_candidate_index(content: str) -> Optional[int]:
    stripped = content.strip()
    patterns = (
        r"^(-?\d+)$",
        r"^\[\s*(-?\d+)\s*\]$",
        r"^候補\s*\[\s*(-?\d+)\s*\]$",
    )
    for pattern in patterns:
        m = re.fullmatch(pattern, stripped)
        if m:
            return int(m.group(1))
    return None


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

    # Step 1: 正規表現で日付が一致する候補に絞り込む
    regex_matched = [
        (pos, text, cx, cy)
        for pos, (_, text, cx, cy) in enumerate(all_candidates)
        if _date_matches_text(text, year, month, day)
    ]
    print(f"正規表現一致: {len(regex_matched)} 件")

    if len(regex_matched) == 1:
        # 1件のみ → VLM 不要、直接採用
        _, text, cx, cy = regex_matched[0]
        print(f"正規表現で一意特定: {text!r} at ({cx},{cy})")
        return (cx, cy)

    if len(regex_matched) > 1:
        _, text, cx, cy = min(regex_matched, key=lambda item: (item[3], item[2]))
        print(f"複数一致のため最上段を採用: {text!r} at ({cx},{cy})")
        return (cx, cy)

    # Step 2: 0件 → VLM に問い合わせる
    if image is None:
        raise MlxVlmHistoryError("VLM fallback には元画像が必要です")

    vlm_candidates = regex_matched if regex_matched else all_candidates
    prompt = _build_prompt(date_str, [(0, text, cx, cy) for _, text, cx, cy in vlm_candidates])

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_url(image)}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 16,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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

    pos_idx = _parse_candidate_index(content)
    if pos_idx is None:
        print(f"⚠️ VLM応答から候補番号を厳密抽出できませんでした: {content!r}")
        return None

    if pos_idx == -1:
        print("VLM: 該当エントリなし")
        return None

    if pos_idx < 0 or pos_idx >= len(vlm_candidates):
        raise MlxVlmHistoryError(
            f"VLM応答の番号 {pos_idx} が候補範囲外です (0-{len(vlm_candidates)-1}): {content!r}"
        )

    _, text, cx, cy = vlm_candidates[pos_idx]

    # VLM の選択を正規表現で最終検証
    if not _date_matches_text(text, year, month, day):
        print(f"⚠️ VLMが選んだ候補 {text!r} は {date_str} と一致しません（無効）")
        return None

    print(f"VLM特定: {text!r} at ({cx},{cy})")
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
