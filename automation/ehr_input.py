"""
EHR field input automation.

Captures the current HDMI screen, finds a labeled input field,
and types text into it via BLE (ESP32) mouse/keyboard control.

Uses the same AsyncBLERunner pattern as ble_test_cli.py to ensure
identical BLE event-loop behaviour on macOS CoreBluetooth.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import json
import cv2
import re
import shlex
import tempfile
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from automation.config import load_config
from automation.screen_analyzer import (
    capture_screen as _capture_screen_hdmi,
    load_ocr_reader,
    run_ocr,
)
from automation.gui_image_analyzer import find_textbox_right_of_label
from automation.ble_client import BLEClient
from automation.local_segmentation import (
    _katakana_to_romaji,
    segment_japanese_text_locally,
)
from automation.mlx_vlm_segmentation import (
    MlxVlmSegmentationError,
    segment_japanese_text_with_mlx_vlm,
)
from automation.mlx_vlm_ime import (
    MlxVlmImeError,
    detect_ime_mode_from_typed_a,
    has_active_ime_composition,
    read_highlighted_popup_candidate,
    read_inline_candidate_context,
    read_inline_candidate_roi,
    read_popup_candidates_numbered,
    suggest_ime_helper_word,
)
from automation import mlx_vlm_ime, mlx_vlm_segmentation



_TEXT_NORMALIZATION_MAP = str.maketrans({
    # 全角括弧・記号 → 半角 ASCII（形状がほぼ同一のもの）
    "（": "(",
    "）": ")",
    "％": "%",
    "：": ":",
    "；": ";",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
    "，": ",",
    "．": ".",
    "　": " ",
    # 全角 ASCII 記号 (U+FF01–U+FF5E) → 半角
    "！": "!",
    "＂": '"',
    "＃": "#",
    "＄": "$",
    "＆": "&",
    "＊": "*",
    "＋": "+",
    "－": "-",
    "／": "/",
    "＜": "<",
    "＝": "=",
    "＞": ">",
    "？": "?",
    "＠": "@",
    "＼": "\\",
    "＾": "^",
    "＿": "_",
    "｀": "`",
    "｜": "|",
    "～": "~",
    # スマートクォート → ストレートクォート
    "\u2018": "'",   # '
    "\u2019": "'",   # '
    "\u201c": '"',   # "
    "\u201d": '"',   # "
    # ダッシュ類 → ハイフン
    "\u2010": "-",   # ‐ (hyphen)
    "\u2013": "-",   # – (en dash)
})
_MULTI_CHAR_REPLACEMENTS = {
    "→": "->",
    "⇒": "=>",
    "〜": "~",
}
_ASCII_SPECIAL_KEYS = {
    "\n": "enter",
    "\t": "tab",
    "[": "lbracket",
    "]": "rbracket",
    "(": "lparen",
    ")": "rparen",
    "%": "percent",
    ":": "colon",
    "+": "plus",
}
_JP_PUNCTUATION = {
    "、": ",",
    "。": ".",
}

# 特殊日本語記号 → IME で入力するときの読み（ローマ字）
_JP_SYMBOL_IME_READINGS: dict[str, str] = {
    "※": "kome",
    "〒": "yuubin",
    "〇": "maru",
    "〆": "shime",
    "♪": "onpu",
    "★": "hoshi",
    "☆": "hoshi",
}

# BLE キーボード (ASCII のみ) では直接送信できない非 ASCII 文字の代替マッピング。
# /μL → /uL のように医療文書で慣例的に使われる ASCII 表記で代替する。
_CHAR_ASCII_FALLBACK: dict[str, str] = {
    "μ": "u",   # マイクロ記号 → ASCII u (/μL → /uL)
    "°": "",    # 度記号 → 省略（℃ は直接入力されるため通常不要）
}

_OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_RUNTIME_OPTIONS = SimpleNamespace(
    openrouter_model=None,
)
_DEFAULT_SEGMENTATION_RUNTIME = {
    "url": mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL,
    "model": mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL,
    "api_key": mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY,
}
_DEFAULT_IME_RUNTIME = {
    "url": mlx_vlm_ime.MLX_VLM_IME_URL,
    "model": mlx_vlm_ime.MLX_VLM_IME_MODEL,
    "api_key": mlx_vlm_ime.MLX_VLM_IME_API_KEY,
}
_DEFAULT_IME_TEXT_RUNTIME = {
    "url": mlx_vlm_ime.MLX_VLM_TEXT_URL,
    "model": mlx_vlm_ime.MLX_VLM_TEXT_MODEL,
    "api_key": mlx_vlm_ime.MLX_VLM_TEXT_API_KEY,
}
_RUN_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class _TeeStream:
    """Mirror writes to both the terminal stream and the run log file."""

    def __init__(self, console_stream, log_stream) -> None:
        self._console_stream = console_stream
        self._log_stream = log_stream

    def write(self, data: str) -> int:
        self._console_stream.write(data)
        self._log_stream.write(data)
        return len(data)

    def flush(self) -> None:
        self._console_stream.flush()
        self._log_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console_stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._console_stream, "encoding", "utf-8")


def _build_run_log_path(now: Optional[datetime] = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%m%d_%H%M")
    candidate = _RUN_LOGS_DIR / f"{timestamp}.txt"
    suffix = 2
    while candidate.exists():
        candidate = _RUN_LOGS_DIR / f"{timestamp}_{suffix}.txt"
        suffix += 1
    return candidate


@contextmanager
def _capture_run_output(
    log_path: Optional[Path] = None,
    *,
    stdout=None,
    stderr=None,
):
    """Tee stdout/stderr to a timestamped per-run log file."""
    import sys

    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    log_path = log_path or _build_run_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_stdout = _TeeStream(stdout, log_file)
        tee_stderr = _TeeStream(stderr, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            yield log_path


def _parse_cli_options(args: list[str]) -> tuple[list[str], dict[str, object]]:
    """Split raw CLI args into positional arguments and normalized option state."""
    clear_field = False
    openrouter_model: Optional[str] = None
    filtered_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--clear":
            clear_field = True
        elif arg == "--openrouter":
            if index + 1 >= len(args):
                raise RuntimeError("--openrouter の後にモデル名を指定してください")
            openrouter_model = args[index + 1]
            index += 1
        elif arg.startswith("--"):
            raise RuntimeError(f"不明なオプション: {arg}")
        else:
            filtered_args.append(arg)
        index += 1

    option_summary = {
        "clear_field": clear_field,
        "openrouter_model": openrouter_model,
    }
    return filtered_args, option_summary


def _build_run_log_header(
    executable_path: str,
    raw_args: list[str],
    positional_args: list[str],
    option_summary: dict[str, object],
) -> str:
    raw_executable = executable_path or "automation.ehr_input"
    raw_argv = [raw_executable, *raw_args]
    executable_name = Path(raw_executable).name or raw_executable
    lines = [
        "=== ehr_input invocation ===",
        f"executable: {executable_name}",
        f"command_line: {shlex.join(raw_argv)}",
        f"argv: {json.dumps(raw_argv, ensure_ascii=False)}",
        f"parsed_options: {json.dumps(option_summary, ensure_ascii=False, sort_keys=True)}",
        f"positional_args: {json.dumps(positional_args, ensure_ascii=False)}",
    ]
    return "\n".join(lines)




def _configure_runtime(*, openrouter_model: Optional[str] = None) -> None:
    _RUNTIME_OPTIONS.openrouter_model = openrouter_model

    mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL = _DEFAULT_SEGMENTATION_RUNTIME["url"]
    mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL = _DEFAULT_SEGMENTATION_RUNTIME["model"]
    mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY = _DEFAULT_SEGMENTATION_RUNTIME["api_key"]
    mlx_vlm_ime.MLX_VLM_IME_URL = _DEFAULT_IME_RUNTIME["url"]
    mlx_vlm_ime.MLX_VLM_IME_MODEL = _DEFAULT_IME_RUNTIME["model"]
    mlx_vlm_ime.MLX_VLM_IME_API_KEY = _DEFAULT_IME_RUNTIME["api_key"]
    mlx_vlm_ime.MLX_VLM_TEXT_URL = _DEFAULT_IME_TEXT_RUNTIME["url"]
    mlx_vlm_ime.MLX_VLM_TEXT_MODEL = _DEFAULT_IME_TEXT_RUNTIME["model"]
    mlx_vlm_ime.MLX_VLM_TEXT_API_KEY = _DEFAULT_IME_TEXT_RUNTIME["api_key"]

    if openrouter_model:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("--openrouter を使うには OPENROUTER_API_KEY 環境変数が必要です")
        mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL = _OPENROUTER_CHAT_URL
        mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL = openrouter_model
        mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_API_KEY = api_key
        mlx_vlm_ime.MLX_VLM_IME_URL = _OPENROUTER_CHAT_URL
        mlx_vlm_ime.MLX_VLM_IME_MODEL = openrouter_model
        mlx_vlm_ime.MLX_VLM_IME_API_KEY = api_key
        mlx_vlm_ime.MLX_VLM_TEXT_URL = _OPENROUTER_CHAT_URL
        mlx_vlm_ime.MLX_VLM_TEXT_MODEL = openrouter_model
        mlx_vlm_ime.MLX_VLM_TEXT_API_KEY = api_key


def _runtime_label(*, url: str, model: str, default_kind: str = "VLM") -> str:
    return mlx_vlm_ime.describe_runtime(url=url, model=model, default_kind=default_kind)


def _segmentation_runtime_label() -> str:
    return _runtime_label(
        url=mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_URL,
        model=mlx_vlm_segmentation.MLX_VLM_SEGMENTATION_MODEL,
        default_kind="VLM",
    )


def _ime_vision_runtime_label() -> str:
    return _runtime_label(
        url=mlx_vlm_ime.MLX_VLM_IME_URL,
        model=mlx_vlm_ime.MLX_VLM_IME_MODEL,
        default_kind="VLM",
    )


def _ime_text_runtime_label() -> str:
    return _runtime_label(
        url=mlx_vlm_ime.MLX_VLM_TEXT_URL,
        model=mlx_vlm_ime.MLX_VLM_TEXT_MODEL,
        default_kind="Text",
    )


def capture_screen(*, device_index: int, width: int, height: int, **kwargs):
    return _capture_screen_hdmi(
        device_index=device_index,
        width=width,
        height=height,
        **kwargs,
    )


def _load_ocr_engine(config):
    return load_ocr_reader(config.ocr_languages, config.ocr_use_gpu)


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


def _capture_frame(config):
    return capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )


def _request_ocr_results(frame, config) -> list[tuple]:
    return run_ocr(_load_ocr_engine(config), frame)


def _resolve_text_argument(raw: str) -> str:
    """Resolve a CLI text argument, loading file contents when given a path."""
    candidate_path = Path(raw)
    if not candidate_path.is_file():
        return raw

    try:
        text = candidate_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"テキストファイルを UTF-8 として読めませんでした: {candidate_path}") from exc
    except OSError as exc:
        raise RuntimeError(f"テキストファイルを読めませんでした: {candidate_path}") from exc

    resolved = text.rstrip("\r\n")
    print(f"テキストファイル読込: {candidate_path}")
    return resolved


def _input_resolved_text(
    text: str, clear_field: bool = False
) -> None:
    """Route already-resolved text through the existing input pipeline."""
    if _is_japanese(text):
        if _is_hiragana_only(text) or _is_katakana_only(text):
            type_japanese_sentence(text, clear_field=clear_field)
        elif len(text) <= 4 and not any(ch in text for ch in "をにはがでも") and not _is_ascii_only(text):
            romaji = _kanji_to_romaji(text)
            print(f"IME変換: {romaji} → {text}")
            type_kanji_via_ime(romaji, text, clear_field=clear_field)
        else:
            type_japanese_sentence(text, clear_field=clear_field)
        return

    print(f"英語入力: {text!r}")
    _type_english_text(text, clear_field=clear_field)


def _normalize_text_for_typing(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").translate(_TEXT_NORMALIZATION_MAP)
    for src, dest in _MULTI_CHAR_REPLACEMENTS.items():
        normalized = normalized.replace(src, dest)
    return normalized


def _classify_input_char(ch: str) -> str:
    if ch == "\n":
        return "newline"
    if ch in _JP_PUNCTUATION:
        return "jp_punct"
    if ord(ch) < 128:
        return "ascii"
    if _is_japanese(ch):
        return "japanese"
    return "ascii"


def _tokenize_text_for_input(text: str) -> list[dict[str, str]]:
    normalized = _normalize_text_for_typing(text)
    tokens: list[dict[str, str]] = []
    buffer: list[str] = []
    current_kind: Optional[str] = None

    def flush_buffer() -> None:
        nonlocal buffer, current_kind
        if buffer:
            tokens.append({"kind": current_kind or "ascii", "text": "".join(buffer)})
            buffer = []
            current_kind = None

    for ch in normalized:
        kind = _classify_input_char(ch)
        if kind in {"newline", "jp_punct"}:
            flush_buffer()
            tokens.append({"kind": kind, "text": ch})
            continue
        if current_kind != kind:
            flush_buffer()
            current_kind = kind
        buffer.append(ch)

    flush_buffer()
    return tokens


# セグメント単位の分割オーバーライド: Qwen/ローカル分割で1トークンにまとまる語を
# 複数の IME 入力単位に強制分割する。IME の候補ポップアップが単体で出にくい語に対して使う。
_SEGMENT_OVERRIDES: dict[str, list[dict[str, str]]] = {
    "歳頃": [{"text": "歳", "romaji": "sai"}, {"text": "頃", "romaji": "koro"}],
    "吸入剤": [{"text": "吸入", "romaji": "kyuunyuu"}, {"text": "剤", "romaji": "zai"}],
    "過膨張": [{"text": "過", "romaji": "ka"}, {"text": "膨張", "romaji": "bouchou"}],
    # 肺野 (はいや) は標準 IME 辞書にない医療用語 → 肺(hai)+野(ya) に分割
    "肺野": [{"text": "肺", "romaji": "hai"}, {"text": "野", "romaji": "ya"}],
    # 認めるが → 認める(mitomeru) + が(ga) に分割
    "認めるが": [{"text": "認める", "romaji": "mitomeru"}, {"text": "が", "romaji": "ga"}],
    # 動脈血ガス → 動脈血(doumyakuketsu) + ガス(gasu) に分割
    "動脈血ガス": [{"text": "動脈血", "romaji": "doumyakuketsu"}, {"text": "ガス", "romaji": "gasu"}],
    # 動脈血 単体でもオーバーライド
    "動脈血": [{"text": "動脈血", "romaji": "doumyakuketsu"}],
    # ・ (中黒/ナカテン) : JIS キーボードでは '/' キーで入力する。
    # カタカナ変換パス (F7) で処理するため romaji は '/' とする。
    "・": [{"text": "・", "romaji": "/"}],
    # ソル・コーテフ (Solu-Cortef) : カタカナ医薬品名、中黒を含む。
    # '/' = ・、'-' = ー (長音符) として romaji を組む。
    "ソル・コーテフ": [{"text": "ソル・コーテフ", "romaji": "soru/ko-tefu"}],
}


def _expand_segment_overrides(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    """_SEGMENT_OVERRIDES に登録されたトークンを複数サブセグメントに展開する。"""
    result: list[dict[str, str]] = []
    for seg in segments:
        if seg["text"] in _SEGMENT_OVERRIDES:
            result.extend(_SEGMENT_OVERRIDES[seg["text"]])
        else:
            result.append(seg)
    return result


def _validate_vlm_romaji(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    """VLM が返したローマ字を pykakasi で検証し、音節重複やリーク文字を修正する。

    漢字を含むセグメントのみ対象。VLM のローマ字が pykakasi より長い場合のみ
    修正する（音節重複 e.g. kokikisei→kokisei や隣接セグメントからの文字リーク
    e.g. kyuukise→kyuuki を検出するヒューリスティック）。
    pykakasi が医学用語で誤変換する場合（VLM が同等以下の長さ）は VLM を信頼する。
    """
    corrected: list[dict[str, str]] = []
    for seg in segments:
        text = seg["text"]
        if not _has_kanji(text):
            corrected.append(seg)
            continue
        expected = _kanji_to_romaji(text)
        if seg["romaji"] != expected and len(seg["romaji"]) > len(expected):
            print(f"[ローマ字補正] {text!r}: VLM={seg['romaji']!r} → pykakasi={expected!r}")
            corrected.append({"text": text, "romaji": expected})
        else:
            corrected.append(seg)
    return corrected


def _segment_japanese_with_default_vlm(text: str) -> list[dict[str, str]]:
    runtime_label = _segmentation_runtime_label()
    try:
        raw_content, segments = segment_japanese_text_with_mlx_vlm(text)
        if "".join(segment["text"] for segment in segments) != text:
            raise MlxVlmSegmentationError(
                f"{runtime_label}分割結果が元テキストを保持していません: source={text!r} segments={segments!r}"
            )
        if _should_fallback_to_local_segmentation(segments):
            raise MlxVlmSegmentationError(
                f"{runtime_label}分割結果が IME 候補を不安定化させる粒度です: source={text!r} segments={segments!r}"
            )
        normalized_segments = [
            {"text": seg["text"], "romaji": seg["romaji"]}
            for seg in segments
        ]
        normalized_segments = _expand_segment_overrides(normalized_segments)
        normalized_segments = _validate_vlm_romaji(normalized_segments)
        print(f"{runtime_label}分割結果: {raw_content}")
        print(f"{runtime_label}分割補正後: {normalized_segments}")
        return normalized_segments
    except MlxVlmSegmentationError as exc:
        print(f"{runtime_label}分割失敗 → ローカル分割へフォールバック: {exc}")
        summary, segments = segment_japanese_text_locally(text)
        print(f"ローカル分割サマリ: {summary}")
        return _expand_segment_overrides(segments)


def _is_single_kanji(text: str) -> bool:
    return len(text) == 1 and _has_kanji(text)


def _is_hiragana_only(text: str) -> bool:
    return bool(text) and all("\u3040" <= ch <= "\u309f" for ch in text)


def _should_fallback_to_local_segmentation(segments: list[dict[str, str]]) -> bool:
    consecutive_single_kanji = 0
    for index, segment in enumerate(segments):
        text = segment["text"]
        if _is_single_kanji(text):
            consecutive_single_kanji += 1
            if consecutive_single_kanji >= 2:
                return True
            if index + 1 < len(segments) and _is_hiragana_only(segments[index + 1]["text"]):
                return True
        else:
            consecutive_single_kanji = 0
    return False


def _presplit_japanese_for_overrides(text: str) -> list[str]:
    """テキストを _SEGMENT_OVERRIDES のキーで事前分割する。

    例: "動脈血ガス分析" + _SEGMENT_OVERRIDES["動脈血ガス"] →
        ["動脈血ガス", "分析"]
    """
    keys = sorted(_SEGMENT_OVERRIDES.keys(), key=len, reverse=True)
    chunks: list[str] = [text]
    for key in keys:
        new_chunks: list[str] = []
        for chunk in chunks:
            if chunk in _SEGMENT_OVERRIDES:
                # already an override key; keep as-is
                new_chunks.append(chunk)
                continue
            if key in chunk:
                parts = chunk.split(key)
                for i, part in enumerate(parts):
                    if part:
                        new_chunks.append(part)
                    if i < len(parts) - 1:
                        new_chunks.append(key)
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
    return [c for c in chunks if c]


def _segment_text_for_input(text: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    for segment in _iter_segments_for_input(text):
        segments.append(segment)
    return segments


def _iter_segments_for_input(text: str):
    for token in _tokenize_text_for_input(text):
        kind = token["kind"]
        value = token["text"]
        if kind == "japanese":
            for chunk in _presplit_japanese_for_overrides(value):
                if chunk in _SEGMENT_OVERRIDES:
                    print(f"[事前オーバーライド] {chunk!r} → {_SEGMENT_OVERRIDES[chunk]}")
                    yield from _SEGMENT_OVERRIDES[chunk]
                else:
                    for segment in _segment_japanese_with_default_vlm(chunk):
                        yield segment
        elif kind == "jp_punct":
            yield {"text": value, "romaji": _JP_PUNCTUATION[value]}
        elif kind == "newline":
            yield {"text": "\n", "romaji": "<enter>"}
        else:
            yield {"text": value, "romaji": value}


def _type_ascii_text_precisely(client: BLEClient, text: str) -> None:
    buffer: list[str] = []

    def flush_buffer() -> None:
        if not buffer:
            return
        chunk = "".join(buffer)
        ok = client.type_text(chunk)
        print(f"type:{chunk} -> {'OK' if ok else 'NG'}")
        buffer.clear()

    for ch in text:
        key_name = _ASCII_SPECIAL_KEYS.get(ch)
        if key_name is None:
            buffer.append(ch)
            continue
        flush_buffer()
        ok = client.press_key(key_name)
        print(f"key:{key_name} -> {'OK' if ok else 'NG'}")
    flush_buffer()


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

    results = _request_ocr_results(frame, config)

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

    results = _request_ocr_results(frame, config)

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


def click_history(date_str: str) -> None:
    """
    過去カルテ列から指定日付のエントリを検出してクリックする。

    automation.mlx_vlm_history と同じ OCR→候補抽出→VLM 判定の経路を使用する。

    Args:
        date_str: 日付文字列 (yyyymmdd 形式, 例: "20190502")
    """
    if len(date_str) != 8 or not date_str.isdigit():
        raise ValueError(f"日付は yyyymmdd 形式で指定してください: {date_str!r}")

    config = load_config(skip_password=True)
    print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    try:
        from automation.mlx_vlm_history import MlxVlmHistoryError, find_history_date_in_image

        ocr_languages = getattr(config, "ocr_languages", ["ja", "en"])
        coords = find_history_date_in_image(
            date_str,
            frame,
            languages=ocr_languages,
        )
    except MlxVlmHistoryError as exc:
        raise RuntimeError(f"過去カルテ日付の検出に失敗しました: {exc}") from exc

    if coords is None:
        print(f"日付 {date_str} のエントリが画面上に見つかりませんでした")
        return

    target_x, target_y = coords

    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(target_x, target_y)
    print(f"moveto ({target_x}, {target_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click -> {'OK' if ok else 'NG'}")

    print("完了")


_MATCH_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "match_templates",
)
EDIT_BUTTON_TEMPLATE = os.path.join(_MATCH_TEMPLATES_DIR, "edit_button.jpg")


def _find_button_with_template(
    frame: np.ndarray,
    template_path: str,
    threshold: float = 0.7,
) -> Optional[tuple[int, int]]:
    """OpenCV テンプレートマッチングでボタン中心座標を返す。"""
    tmpl = cv2.imread(template_path)
    if tmpl is None:
        raise RuntimeError(f"テンプレート画像を読み込めません: {template_path}")
    result = cv2.matchTemplate(frame, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    print(f"テンプレートマッチング スコア: {max_val:.3f} ({template_path})")
    if max_val < threshold:
        return None
    th, tw = tmpl.shape[:2]
    return (max_loc[0] + tw // 2, max_loc[1] + th // 2)


def edit_history(date_str: str) -> None:
    """過去カルテの日付エントリをクリックし、修正ボタンをクリックする。

    Args:
        date_str: 日付文字列 (yyyymmdd 形式, 例: "20190502")
    """
    click_history(date_str)
    time.sleep(1.0)

    config = load_config(skip_password=True)
    print("修正ボタンを探しています...")
    frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    coords = _find_button_with_template(frame, EDIT_BUTTON_TEMPLATE)
    if coords is None:
        raise RuntimeError("修正ボタンが画面上に見つかりませんでした")

    target_x, target_y = coords
    print(f"修正ボタン: ({target_x}, {target_y})")

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

    # 暗い領域（黒背景または紺色背景）を検出: ピクセル値 80 以下を白に
    _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # ノイズ除去: 小さな孤立点を消す
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh = frame.shape[0]
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # IME 候補ウィンドウのおよそのサイズ範囲でフィルタ
        if w < 20 or w > 800 or h < 12 or h > 100:
            continue
        # タスクバー・スタートボタン等の画面下部を除外（下15%以内）
        if y > fh * 0.85:
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


def _find_changed_region(
    base: np.ndarray,
    current: np.ndarray,
    exclude_bottom_px: int = 220,
) -> Optional[np.ndarray]:
    """
    2 フレームの差分から変化した矩形領域を切り出す（フォールバック用）。

    IME 候補ブロックが検出できない場合、入力前後の差分で変化領域を特定する。
    タスクバー・IMEツールバー等の画面下部ノイズを除外するため、
    exclude_bottom_px より下の領域は差分検索から除外する。

    Returns:
        変化した領域の画像。見つからない場合は None。
    """
    h = base.shape[0]
    search_end = max(h - exclude_bottom_px, h // 2)  # 最低でも上半分は含める

    diff = cv2.absdiff(base[:search_end], current[:search_end])
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 最大の変化領域を返す
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h_ = cv2.boundingRect(largest)
    # 最小サイズフィルタ
    if w < 32 or h_ < 32:
        return None
    return current[y:y + h_, x:x + w]


def _save_debug_image(frame: np.ndarray, name: str) -> None:
    """デバッグ用: フレームを captures/ に保存する（失敗しても無視）。"""
    try:
        captures_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "captures"
        )
        os.makedirs(captures_dir, exist_ok=True)
        # 特殊文字をアンダースコアに置換してファイル名を安全にする
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = os.path.join(captures_dir, f"debug_{safe_name}.png")
        cv2.imwrite(path, frame)
        print(f"  [debug] ROI保存: {path}")
    except Exception:  # noqa: BLE001
        pass


def _extract_vlm_text_content(result: dict) -> str:
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"mlx_vlm応答に content がありません: {result!r}") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        merged = "".join(parts).strip()
        if merged:
            return merged
    raise RuntimeError(f"mlx_vlm応答の content 形式が不正です: {content!r}")


def _parse_ime_candidate_response(content: str) -> Optional[str]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = payload.get("candidate")
            if candidate is None:
                return None
            if isinstance(candidate, str):
                return candidate.strip() or None

    quoted = re.search(r'"candidate"\s*:\s*"([^"]+)"', content)
    if quoted:
        return quoted.group(1).strip() or None
    # Handle Python-dict-style single-quoted responses (e.g. {'candidate': '通'})
    single_quoted = re.search(r"'candidate'\s*:\s*'([^']+)'", content)
    if single_quoted:
        return single_quoted.group(1).strip() or None
    # Last resort: return None rather than raw content (which would cause false positives)
    return None


def _read_ime_candidate_with_vlm(frame: np.ndarray, target_kanji: str) -> Optional[str]:
    # NOTE: Do NOT include target_kanji in the prompt — it causes the VLM to hallucinate
    # the expected kanji even when a different candidate (e.g. 官房 vs 感冒) is highlighted.
    try:
        return read_inline_candidate_roi(frame)
    except MlxVlmImeError as exc:
        raise RuntimeError(f"IME候補確認の mlx_vlm 呼び出しが失敗しました: {exc}") from exc


def _read_ime_inline_candidate_fullframe(frame: np.ndarray, target_kanji: str) -> Optional[str]:
    """全画面フレームからインライン変換候補を読み取る（ROI検出失敗時のフォールバック）。

    フレームを中央帯にクロップしてから mlx_vlm サーバーに送信する。
    """
    try:
        return read_inline_candidate_context(frame)
    except MlxVlmImeError as exc:
        raise RuntimeError(f"インライン候補フルフレーム mlx_vlm 呼び出しが失敗しました: {exc}") from exc


def type_kanji_via_ime(
    romaji: str,
    target_kanji: str,
    max_attempts: int = 1,
    wait_sec: float = 0.5,
    clear_field: bool = False,
    _current_ime_mode: Optional[str] = None,
    _no_helper_fallback: bool = False,
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
        clear_field: True の場合、入力前に Backspace を 50 回送信してフィールドをクリアする
        _current_ime_mode: 呼び出し元が既に検出した IME モード。指定時は内部で再検出しない。
        _no_helper_fallback: True の場合、ヘルパー単語フォールバックを使わない（再帰防止用）。

    Raises:
        RuntimeError: BLE サーバー未起動、またはキャプチャ失敗の場合
        ValueError: max_attempts 回試行しても target_kanji が見つからない場合
    """
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    if clear_field:
        print("フィールドをクリア中 (Backspace x50)...")
        for _ in range(50):
            client.press_key("backspace")
        time.sleep(0.3)

    # IME をひらがな入力モードに確保してからローマ字入力
    # _current_ime_mode が渡された場合は呼び出し元が既に切替済みなので再検出しない
    if _current_ime_mode is None:
        current_mode = detect_ime_mode(client, config)
        ensure_ime_mode("japanese", client, current_mode)
    else:
        # 呼び出し元が既に Japanese モードを確保済み
        print(f"  [IME] 呼び出し元が {_current_ime_mode!r} を確認済み → 再検出スキップ")

    # キャプチャ不能な状態でローマ字入力を始めると、生の入力が EHR に残ってしまう。
    # 先に1フレーム取得できることを確認してから IME 入力を開始する。
    pre_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if pre_frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    # ローマ字入力
    print(f"ローマ字入力: {romaji}")
    ok = client.type_text(romaji)
    print(f"type:{romaji} -> {'OK' if ok else 'NG'}")
    time.sleep(0.3)  # IMEがローマ字処理するまで待機

    # 1回目の Space: インライン変換（ひらがな→第1候補に変わる、ポップアップなし）
    print("[Space] インライン変換 (第1候補)...")
    ok = client.press_key("space")
    print(f"key:space -> {'OK' if ok else 'NG'}")
    time.sleep(wait_sec)

    # 第1候補をキャプチャしてOCR確認
    base_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if base_frame is None:
        raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

    # 反転ブロック検出 + VLM/OCR でインライン変換候補を確認する。
    inline_vlm: Optional[str] = None
    inline_combined = ""
    vision_runtime_label = _ime_vision_runtime_label()
    # インライン候補を検出: ROI検出を試み、非日本語なら失敗とみなしてポップアップへ
    inline_roi = _find_ime_candidate_region(base_frame)
    if inline_roi is None:
        inline_roi = _find_changed_region(pre_frame, base_frame)
    if inline_roi is not None:
        _save_debug_image(inline_roi, f"ime_inline_{target_kanji}")
        try:
            inline_vlm = _read_ime_candidate_with_vlm(inline_roi, target_kanji)
            print(f"  [第1候補] {vision_runtime_label}読取(ROI): {inline_vlm!r}")
        except RuntimeError as exc:
            print(f"  [第1候補] {vision_runtime_label}読取失敗: {exc}")
        ocr_results_inline = _request_ocr_results(inline_roi, config)
        inline_combined = "".join(t for (_, t, _) in ocr_results_inline)
        print(f"  [第1候補] OCR結合: {inline_combined!r}")
    else:
        print(f"  [第1候補] 変化領域を検出できませんでした")

    # インラインROIでVLM/OCRが両方失敗した場合、フルフレームVLMでフォールバック
    # ただし ROI が取得できなかった場合はスキップ（タイムアウトでIME状態が崩れるのを防ぐ）
    inline_fullframe_vlm: Optional[str] = None
    if (inline_roi is not None
            and not _ime_candidate_matches(target_kanji, inline_vlm or "")
            and not _ime_candidate_matches(target_kanji, inline_combined)):
        try:
            inline_fullframe_vlm = _read_ime_inline_candidate_fullframe(base_frame, target_kanji)
            print(f"  [第1候補] {vision_runtime_label}読取(全フレーム): {inline_fullframe_vlm!r}")
        except RuntimeError as exc:
            print(f"  [第1候補] {vision_runtime_label}全フレーム読取失敗: {exc}")

    if (_ime_candidate_matches(target_kanji, inline_vlm or "")
            or _ime_candidate_matches(target_kanji, inline_combined)
            or _ime_fullframe_exact_match(target_kanji, inline_fullframe_vlm or "")):
        print(f"  「{target_kanji}」第1候補を確認 → Enter で確定")
        ok = client.press_key("enter")
        print(f"key:enter -> {'OK' if ok else 'NG'}")
        print("完了")
        return

    # 第1候補が不一致 → Space #2 でポップアップを開く
    print(f"  第1候補が「{target_kanji}」と不一致。ポップアップを開きます...")
    ok = client.press_key("space")
    print(f"key:space (popup) -> {'OK' if ok else 'NG'}")
    time.sleep(max(wait_sec, 1.0))  # ポップアップが完全に表示されるまで待機

    # ポップアップ直後の候補リストをVLM（全フレーム）で解析してターゲットの位置を特定
    popup_init_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if popup_init_frame is not None:
        # 全フレームをデバッグ用に保存
        _save_debug_image(popup_init_frame, f"ime_popup_{target_kanji}")

        # 番号付き候補リストを1回のVLMコールで取得（番号とテキストの両方）
        numbered: list[tuple[int, str]] = []
        try:
            numbered = _read_popup_candidates_with_fallback(
                target_kanji,
                popup_init_frame,
                debug_name=target_kanji,
            )
            print(f"  ポップアップ候補解析(VLM番号付き): {numbered}")
        except Exception as exc:
            print(f"  VLM番号付き候補読取失敗: {exc}")

        # --- 句読点幅選択ポップアップ（全/半）や記号バリエーションポップアップを誤検出した場合 ---
        _WIDTH_ONLY_TEXTS = {"全", "半", "全角", "半角"}
        def _is_symbol_variation_popup(candidates: list) -> bool:
            if not candidates:
                return False
            if all(t in _WIDTH_ONLY_TEXTS for _, t in candidates):
                return True
            # VLM が "+[半]", "+[全]", "、[全]" 等と読み取った場合もシンボルポップアップ
            return all(("[半]" in t or "[全]" in t) for _, t in candidates if t)

        if numbered and _is_symbol_variation_popup(numbered):
            print(f"  [警告] 記号バリエーションポップアップを検出 → Escape で閉じてリトライします")
            client.press_key("escape")
            time.sleep(0.5)
            # IME 組成をキャンセル（誤入力された記号を削除）
            client.press_key("escape")
            time.sleep(0.3)
            # コミット済み余分記号を削除
            client.press_key("backspace")
            time.sleep(0.2)
            # ローマ字を再入力してポップアップを開く
            ok = client.type_text(romaji)
            print(f"type:{romaji} (リトライ) -> {'OK' if ok else 'NG'}")
            time.sleep(0.5)
            ok = client.press_key("space")
            print(f"key:space (インライン変換リトライ) -> {'OK' if ok else 'NG'}")
            time.sleep(wait_sec)
            ok = client.press_key("space")
            print(f"key:space (リトライ popup) -> {'OK' if ok else 'NG'}")
            time.sleep(max(wait_sec, 1.0))
            retry_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if retry_frame is not None:
                try:
                    numbered = read_popup_candidates_numbered(retry_frame)
                    print(f"  リトライ後候補: {numbered}")
                except Exception as exc:
                    print(f"  リトライVLM失敗: {exc}")

        # --- 完全一致 ---
        exact_match = _find_best_candidate_match(target_kanji, numbered)

        if exact_match is not None:
            display_num, matched_text = exact_match
            print(f"  [VLM一致] 「{target_kanji}」→ 表示番号 {display_num} (読取: {matched_text!r})")
            if display_num <= 9:
                print(f"  「{target_kanji}」→ type:{display_num}")
                client.send_command(f"type:{display_num}")
                time.sleep(max(wait_sec, 0.3))
                ok = client.press_key("enter")
                print(f"key:enter -> {'OK' if ok else 'NG'}")
                print("完了")
            else:
                spaces_needed = display_num - 2
                print(f"  「{target_kanji}」→ Space×{spaces_needed} + Enter (表示番号 {display_num})")
                for _ in range(spaces_needed):
                    client.press_key("space")
                    time.sleep(max(wait_sec, 0.3))
                ok = client.press_key("enter")
                print(f"key:enter -> {'OK' if ok else 'NG'}")
                print("完了")
            return
        print(f"  候補リストに「{target_kanji}」なし → プレフィックス一致を試みます")

        # --- プレフィックス一致: 候補の中にターゲットの前半部分があれば選択して残りを変換 ---
        prefix_match: Optional[tuple[int, str]] = None
        for n, c in numbered:
            if c and 0 < len(c) < len(target_kanji) and target_kanji.startswith(c):
                prefix_match = (n, c)
                break

        if prefix_match is not None:
            display_num, prefix_text = prefix_match
            remaining_target = target_kanji[len(prefix_text):]
            print(f"  [プレフィックス一致] 「{prefix_text}」を候補で発見（表示番号: {display_num}、残り: 「{remaining_target}」）")
            # プレフィックス候補を選択
            if display_num <= 9:
                print(f"  「{prefix_text}」→ type:{display_num}")
                client.send_command(f"type:{display_num}")
            else:
                spaces_needed = display_num - 2
                for _ in range(spaces_needed):
                    client.press_key("space")
                    time.sleep(wait_sec)
                client.press_key("enter")
            time.sleep(max(wait_sec, 2.0))  # ポップアップが閉じるまで待機

            # プレフィックス選択後の画面を確認（デバッグ用）
            pre_space_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if pre_space_frame is not None:
                _save_debug_image(pre_space_frame, f"ime_popup_{target_kanji}_after_prefix")

            # 右矢印で次のセグメントに移動してからポップアップを開く
            # （Space を押すと現在のセグメントのポップアップが再表示されるため）
            print(f"  次セグメント「{remaining_target}」へ移動...")
            ok = client.press_key("right")
            print(f"key:right (次セグメント移動) -> {'OK' if ok else 'NG'}")
            time.sleep(0.5)
            print(f"  次セグメント「{remaining_target}」のポップアップを開きます...")
            ok = client.press_key("space")
            print(f"key:space (次セグメント) -> {'OK' if ok else 'NG'}")
            time.sleep(max(wait_sec, 2.0))

            rem_frame = capture_screen(
                device_index=config.capture_device_index,
                width=config.capture_width,
                height=config.capture_height,
            )
            if rem_frame is None:
                _fallback_remaining_after_prefix(
                    client,
                    config,
                    remaining_target=remaining_target,
                    wait_sec=wait_sec,
                )
                print("完了（残りフレーム取得失敗 → 残りを再入力）")
                return
            _save_debug_image(rem_frame, f"ime_popup_{target_kanji}_remaining")
            try:
                rem_numbered = _read_popup_candidates_with_fallback(
                    remaining_target,
                    rem_frame,
                    debug_name=f"{target_kanji}_remaining",
                )
                print(f"  残りポップアップ候補: {rem_numbered}")
                rem_match = _find_best_candidate_match(remaining_target, rem_numbered)
                if rem_match is not None:
                    rem_display, matched_rem = rem_match
                    print(f"  「{remaining_target}」→ 表示番号 {rem_display} (読取: {matched_rem!r})")
                    if rem_display <= 9:
                        print(f"  「{remaining_target}」→ type:{rem_display}")
                        client.send_command(f"type:{rem_display}")
                        time.sleep(max(wait_sec, 0.3))
                        ok = client.press_key("enter")
                        print(f"key:enter -> {'OK' if ok else 'NG'}")
                    else:
                        spaces_needed = rem_display - 2
                        for _ in range(spaces_needed):
                            client.press_key("space")
                            time.sleep(max(wait_sec, 0.3))
                        ok = client.press_key("enter")
                        print(f"key:enter -> {'OK' if ok else 'NG'}")
                    print("完了")
                    return
                _fallback_remaining_after_prefix(
                    client,
                    config,
                    remaining_target=remaining_target,
                    wait_sec=wait_sec,
                )
                print("完了（残り部分を再入力）")
            except Exception as exc:
                print(f"  残りポップアップVLM読取失敗: {exc}")
                _fallback_remaining_after_prefix(
                    client,
                    config,
                    remaining_target=remaining_target,
                    wait_sec=wait_sec,
                )
            print("完了（残り部分を再入力）")
            return

        # セグメント拡張 (Shift+Right) で再試行 — 現在は無効化
        # print(f"  プレフィックス一致なし → Shift+Right でセグメント拡張を試みます")
        # for ext in range(4):
        #     client.press_key("escape")
        #     time.sleep(0.3)
        #     client.press_key("shift_right")
        #     time.sleep(0.3)
        #     ok = client.press_key("space")
        #     print(f"  [拡張{ext+1}] Escape+Shift+Right+Space -> {'OK' if ok else 'NG'}")
        #     time.sleep(max(wait_sec, 1.0))
        #
        #     ext_frame = capture_screen(
        #         device_index=config.capture_device_index,
        #         width=config.capture_width,
        #         height=config.capture_height,
        #     )
        #     if ext_frame is None:
        #         continue
        #     try:
        #         ext_numbered = read_popup_candidates_numbered(ext_frame)
        #         print(f"  [拡張{ext+1}] 候補: {ext_numbered}")
        #         ext_match = _find_best_candidate_match(target_kanji, ext_numbered)
        #         if ext_match is not None:
        #             display_num, c = ext_match
        #             if display_num <= 9:
        #                 print(f"  「{target_kanji}」→ type:{display_num}")
        #                 client.send_command(f"type:{display_num}")
        #                 time.sleep(wait_sec)
        #             else:
        #                 spaces_needed = display_num - 2
        #                 for _ in range(spaces_needed):
        #                     client.press_key("space")
        #                     time.sleep(wait_sec)
        #                 ok = client.press_key("enter")
        #                 print(f"key:enter -> {'OK' if ok else 'NG'}")
        #             print("完了")
        #             return
        #     except Exception as exc:
        #         print(f"  [拡張{ext+1}] VLM読取失敗: {exc}")
        # print(f"  セグメント拡張でも「{target_kanji}」なし → 試行ループへ")

    # --- ヘルパー単語フォールバック ---
    # ポップアップ候補にターゲットが存在しない場合、テキストモデルに「目標漢字を含む
    # 一般的な日本語単語」を提案してもらい、変換後に余分な文字を削除して得る。
    if not _no_helper_fallback:
        print(f"  ヘルパー単語フォールバックを試みます...")
        if _try_helper_word_fallback(client, config, target_kanji, wait_sec, romaji=romaji):
            return

    for attempt in range(1, max_attempts + 1):
        # フレームキャプチャ
        frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame is None:
            raise RuntimeError("HDMIキャプチャデバイスからフレームを取得できませんでした")

        vlm_candidate: Optional[str] = None

        # 全フレームをVLMに渡して現在ハイライトされている候補を読取
        try:
            vlm_candidate = _read_ime_candidate_with_vlm(frame, target_kanji)
            print(f"  [試行{attempt}] 全フレーム {vision_runtime_label}読取: {vlm_candidate!r}")
        except RuntimeError as exc:
            print(f"  [試行{attempt}] 全フレーム {vision_runtime_label}読取失敗: {exc}")

        confirmed = _ime_candidate_matches(target_kanji, vlm_candidate or "")

        if confirmed:
            print(f"  「{target_kanji}」を確認 → Enter で確定")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            print("完了")
            return

        if attempt < max_attempts:
            print(f"  「{target_kanji}」は未確認。Space で次候補へ進みます...")
            ok = client.press_key("space")
            print(f"key:space -> {'OK' if ok else 'NG'}")
            time.sleep(wait_sec)

    # 全候補を確認できなかった → ひらがなフォールバック（改行なし）
    # 残留する組成バッファを Esc で確実にキャンセルしてから、
    # romaji をひらがなとして再入力し Right Arrow で確定する（Enter は改行になるため使わない）
    print(f"  ⚠️ IME候補を {max_attempts} 回確認しましたが「{target_kanji}」を確定できませんでした")
    print(f"  → ひらがなフォールバック: {romaji!r} を再入力して Right Arrow で確定します")
    for _ in range(2):
        client.press_key("escape")  # ポップアップ/インライン/ひらがなをキャンセル
        time.sleep(0.15)
    ok = client.type_text(romaji)
    print(f"  type:{romaji} (ひらがなフォールバック) -> {'OK' if ok else 'NG'}")
    time.sleep(0.3)
    ok = client.press_key("right")  # 改行なしで組成を確定
    print(f"  key:right (ひらがな確定) -> {'OK' if ok else 'NG'}")


def _text_to_hiragana_len(text: str) -> int:
    """漢字/かな文字列をひらがなに変換して文字数を返す (IME組成バッファのサイズ推定)。"""
    import pykakasi
    kks = pykakasi.kakasi()
    hira = "".join(item["hira"] for item in kks.convert(text))
    return max(len(hira), 1)


# ── ローマ字→ひらがな文字数 変換テーブル ──
# IME のローマ字入力で生成されるひらがなの *文字数* をマッピングする。
# 長音記号 '-' はひらがなの「ー」(1文字)に対応する。
_ROMAJI_KANA_COUNTS: dict[str, int] = {}

def _build_romaji_kana_table() -> dict[str, int]:
    """ローマ字→ひらがな文字数のマッピングテーブルを構築する。"""
    t: dict[str, int] = {}
    # 3文字コンボ (拗音: 2ひらがな文字を生成)
    for prefix in ("ky", "gy", "sh", "ch", "ny", "hy", "by", "py", "my", "ry", "jy", "dy", "ty"):
        for vowel in ("a", "u", "o"):
            t[prefix + vowel] = 2
    # shi/chi/tsu → 1ひらがな文字
    t["shi"] = 1
    t["chi"] = 1
    t["tsu"] = 1
    # 2文字コンボ (基本 CV)
    for c in ("k", "g", "s", "z", "t", "d", "n", "h", "b", "p", "m", "r", "w", "y", "v"):
        for v in ("a", "i", "u", "e", "o"):
            if c + v not in t:
                t[c + v] = 1
    # ja/ju/jo → 2ひらがな文字 (じゃ/じゅ/じょ)
    for combo in ("ja", "ju", "jo"):
        t[combo] = 2
    # fa/fi/fu/fe/fo → 2ひらがな文字 (ふぁ/ふぃ/ふ/ふぇ/ふぉ), ただし fu → 1
    t["fu"] = 1
    t["fa"] = 2
    t["fi"] = 2
    t["fe"] = 2
    t["fo"] = 2
    # 1文字母音
    for v in ("a", "i", "u", "e", "o"):
        t[v] = 1
    # 長音記号
    t["-"] = 1
    return t

_ROMAJI_KANA_COUNTS = _build_romaji_kana_table()


def _romaji_to_hiragana_len(romaji: str) -> int:
    """ローマ字文字列から IME が生成するひらがなの文字数を計算する。

    pykakasi の漢字→ひらがな変換 (_text_to_hiragana_len) は熟語の読みが
    実際にタイプしたローマ字と異なる場合がある
    (例: 静注 → pykakasi は「じょうちゅう」(6文字)、実際は「せいちゅう」(5文字))。
    この関数はローマ字文字列を直接解析して正確なひらがな文字数を返す。
    """
    s = romaji.lower()
    count = 0
    i = 0
    while i < len(s):
        # 促音: 同じ子音の連続 (nn を除く。nn は「ん」)
        if (
            i + 1 < len(s)
            and s[i] == s[i + 1]
            and s[i].isalpha()
            and s[i] not in "aiueon"
        ):
            count += 1  # っ
            i += 1
            continue

        # 「ん」: n + 非母音/非y (n' も含む)
        if s[i] == "n":
            # n' → ん
            if i + 1 < len(s) and s[i + 1] == "'":
                count += 1
                i += 2
                continue
            # nn → ん
            if i + 1 < len(s) and s[i + 1] == "n":
                count += 1
                i += 2
                continue
            # 末尾の n → ん
            if i + 1 >= len(s):
                count += 1
                i += 1
                continue
            # n + 子音 (母音/y 以外) → ん
            if s[i + 1] not in "aiueoy":
                count += 1
                i += 1
                continue

        # テーブル検索: 3文字 → 2文字 → 1文字 の最長一致
        matched = False
        for length in (3, 2, 1):
            chunk = s[i:i + length]
            if chunk in _ROMAJI_KANA_COUNTS:
                count += _ROMAJI_KANA_COUNTS[chunk]
                i += length
                matched = True
                break

        if not matched:
            # 不明文字 (数字、句読点など) → 1文字として扱う
            count += 1
            i += 1

    return max(count, 1)


def _has_ime_composition(frame: np.ndarray) -> bool:
    """フレームに IME 組成テキスト（変換中・ひらがな）がアクティブなら True を返す。

    専用の yes/no 判定を使い、通常テキストやラベルの誤検出を避ける。
    エラー時は False（組成なし）として扱い処理を続行する。
    """
    try:
        return has_active_ime_composition(frame)
    except MlxVlmImeError:
        return False


def _cancel_ime_popup_safe(
    client: "BLEClient",
    text: str,
    wait: float = 0.15,
    config=None,
    romaji: str = "",
) -> None:
    """IMEポップアップをコミットなしでキャンセルし、組成バッファをクリアする。

    観察された Windows 7 MS-IME 動作:
      Esc×1 → ポップアップを閉じ、インライン変換状態に戻る
      F6    → インライン候補をひらがな組成に正規化する
      Backspace×N → ひらがな組成を1文字ずつ削除

    重要: インライン変換状態で Esc を再度押すと、一部の IME（Windows 7 等）では
    インライン候補がそのまま確定（コミット）される。また、インライン状態から
    Backspace だけでは変換が元に戻るだけで文字が削除されない場合がある。
    F6 でひらがな組成に正規化してから Backspace で削除する。

    Args:
        client: BLEClient インスタンス
        text: IME組成バッファに対応する漢字/かな文字列 (ひらがな文字数の計算に使用)
        wait: キー操作間の待機秒数
        config: キャプチャ設定 (指定時はVLM後確認ループを実行)
        romaji: 入力に使用したローマ字 (指定時はより正確なひらがな文字数を計算)
    """
    if romaji:
        hira_len = _romaji_to_hiragana_len(romaji)
    else:
        hira_len = _text_to_hiragana_len(text)

    # Step 1: Esc でポップアップを閉じる（インライン変換状態へ）
    client.press_key("escape")
    time.sleep(wait)

    # Step 2: F6 でインライン候補をひらがな組成に正規化する。
    # これにより「化」等のインライン候補がひらがな「か」に戻り、
    # Esc による誤確定や BS の不安定な動作を回避できる。
    # 既にひらがな組成の場合 F6 は無害（ひらがなのまま）。
    client.press_key("f6")
    time.sleep(0.25)

    # Step 3: ひらがな組成を Backspace で削除
    # IME がマルチセグメント変換状態にある場合、BS がセグメント単位で動作し
    # ひらがな文字数分の固定 BS では過削除（確定済みテキスト侵食）が発生しうる。
    # VLM が使える場合は各 BS 前に組成を確認し、消え次第停止する。
    if config is not None:
        time.sleep(0.4)
        first_frame = _capture_frame(config)
        vlm_sees = first_frame is not None and _has_ime_composition(first_frame)
        if vlm_sees:
            # VLM が組成を確認 → ガード付きループ（過削除を防止）
            client.press_key("backspace")
            time.sleep(0.12)
            for _ in range(hira_len + 1):
                time.sleep(0.15)
                frame = _capture_frame(config)
                if frame is None or not _has_ime_composition(frame):
                    break
                client.press_key("backspace")
                time.sleep(0.12)
        else:
            # VLM が Esc+F6 直後の組成を検出できない場合（偽陰性の可能性）
            # hira_len-1 の固定 BS で過削除リスクを最小化しつつ未削除を抑える。
            # 残り最大 1 文字は呼び出し元の VLM ガードが処理する。
            conservative = max(hira_len - 1, 1)
            print(f"  [cancel_safe] VLM偽陰性 → 控えめBS×{conservative} (hira_len={hira_len})")
            for _ in range(conservative):
                client.press_key("backspace")
                time.sleep(0.12)
    else:
        for _ in range(hira_len):
            client.press_key("backspace")
            time.sleep(0.12)

def _clear_pending_ime_composition(
    client: "BLEClient",
    config,
    *,
    max_backspaces: int,
) -> None:
    """helper 再入力前に未確定組成が残っていれば消し切る。

    VLM が組成を検出する間だけ Backspace を送る。
    組成を 1 回でも検出・削除した場合のみ末尾 Esc で IME 状態をリセットする。
    VLM が組成なしと判定した場合は Esc を送らない（Esc がインライン候補を
    確定してしまう Windows 7 MS-IME の動作を回避するため）。
    """
    if config is None:
        return
    cleared_any = False
    for _ in range(max_backspaces):
        time.sleep(0.15)
        frame = _capture_frame(config)
        if frame is None or not _has_ime_composition(frame):
            break
        print("  [helper cleanup] 未確定組成が残っているため Backspace")
        client.press_key("backspace")
        time.sleep(0.12)
        cleared_any = True
    if cleared_any:
        client.press_key("escape")
        time.sleep(0.1)


def _cleanup_after_helper_backspace(
    client: "BLEClient",
    config,
    *,
    helper_word: str,
    backspace_count: int,
) -> None:
    """Clear any IME residue left after trimming a helper-word commit."""
    if backspace_count <= 0:
        return
    _clear_pending_ime_composition(
        client,
        config,
        max_backspaces=max(_text_to_hiragana_len(helper_word) + 2, 4),
    )


def _fallback_remaining_after_prefix(
    client: "BLEClient",
    config,
    *,
    remaining_target: str,
    wait_sec: float,
) -> None:
    """Cancel an ambiguous remaining popup and re-enter the rest safely."""
    print(f"  残りポップアップに「{remaining_target}」なし → ポップアップをキャンセルして再入力")
    _cancel_ime_popup_safe(client, remaining_target, config=config)
    _clear_pending_ime_composition(
        client,
        config,
        max_backspaces=max(_text_to_hiragana_len(remaining_target) + 2, 4),
    )
    time.sleep(0.3)
    type_kanji_via_ime(
        _kanji_to_romaji(remaining_target),
        remaining_target,
        wait_sec=wait_sec,
        _current_ime_mode="japanese",
    )


def _normalize_helper_popup_candidates(
    helper_word: str,
    helper_romaji: str,
    numbered: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """ヘルパー単語用にポップアップ候補を正規化する。

    VLM が入力欄の既存テキストを候補一覧の先頭に混ぜることがあるため、
    helper_word と同じ読み、または helper_word に照合可能な候補だけを残し、
    上から順に 1..N へ振り直す。
    """
    filtered: list[str] = []
    seen: set[str] = set()
    for _, candidate in numbered:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        keep = _ime_candidate_matches(helper_word, candidate)
        if not keep:
            try:
                keep = _kanji_to_romaji(candidate) == helper_romaji
            except Exception:
                keep = False
        if not keep:
            continue
        seen.add(candidate)
        filtered.append(candidate)
    if not filtered:
        return numbered
    return [(index + 1, candidate) for index, candidate in enumerate(filtered)]


def _has_prefix_candidate(target_kanji: str, numbered: list[tuple[int, str]]) -> bool:
    return any(
        candidate and 0 < len(candidate) < len(target_kanji) and target_kanji.startswith(candidate)
        for _, candidate in numbered
    )


def _merge_numbered_candidates(
    preferred: list[tuple[int, str]],
    fallback: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    merged: dict[int, str] = {}
    for source in (fallback, preferred):
        for display_num, candidate in source:
            if not isinstance(display_num, int) or display_num <= 0:
                continue
            candidate = candidate.strip()
            if not candidate:
                continue
            merged[display_num] = candidate
    return sorted(merged.items())


def _popup_cycle_budget(numbered: list[tuple[int, str]]) -> int:
    max_display_num = max((display_num for display_num, _ in numbered), default=0)
    if not numbered:
        return 6
    return max(max_display_num, len(numbered), 9)


def _read_popup_candidates_with_fallback(
    target_kanji: str,
    frame: np.ndarray,
    *,
    debug_name: str = "",
) -> list[tuple[int, str]]:
    """OCR が疎い/不正確なときは VLM 結果へフォールバックして候補見落としを減らす。"""
    numbered = read_popup_candidates_numbered(frame, debug_name=debug_name)
    if not numbered:
        return numbered

    if len(numbered) >= 3 and (
        _find_best_candidate_match(target_kanji, numbered) is not None
        or _has_prefix_candidate(target_kanji, numbered)
    ):
        return numbered

    try:
        vlm_only = mlx_vlm_ime.read_popup_candidates_numbered_vlm(frame, debug_name=debug_name)
    except Exception as exc:
        print(f"  [VLM補完] 候補再読取失敗: {exc}")
        return numbered

    if not vlm_only:
        return numbered

    merged = _merge_numbered_candidates(vlm_only, numbered)
    print(f"  [VLM補完] 候補再読取: {vlm_only}")
    if (
        _find_best_candidate_match(target_kanji, merged) is not None
        or _has_prefix_candidate(target_kanji, merged)
        or len(merged) > len(numbered)
        or len(numbered) < 3
    ):
        return merged
    return numbered


def _read_helper_popup_candidates(
    helper_word: str,
    frame: np.ndarray,
    *,
    debug_name: str = "",
) -> list[tuple[int, str]]:
    """Helper fallback is accuracy-first: read VLM first, then supplement with OCR."""
    vlm_numbered: list[tuple[int, str]] = []
    try:
        vlm_numbered = mlx_vlm_ime.read_popup_candidates_numbered_vlm(frame, debug_name=debug_name)
        if vlm_numbered:
            print(f"  [ヘルパー単語] VLM優先候補: {vlm_numbered}")
    except Exception as exc:
        print(f"  [ヘルパー単語] VLM候補読取失敗: {exc}")

    ocr_numbered: list[tuple[int, str]] = []
    try:
        ocr_numbered = mlx_vlm_ime.read_popup_candidates_ocr(frame, debug_name=debug_name)
        if ocr_numbered:
            print(f"  [ヘルパー単語] OCR候補: {ocr_numbered}")
    except Exception as exc:
        print(f"  [ヘルパー単語] OCR候補読取失敗: {exc}")

    merged = _merge_numbered_candidates(vlm_numbered, ocr_numbered)
    if merged and merged != vlm_numbered and merged != ocr_numbered:
        print(f"  [ヘルパー単語] VLM優先マージ候補: {merged}")
    return merged or vlm_numbered or ocr_numbered


def _try_helper_word_fallback(
    client: "BLEClient",
    config,
    target_kanji: str,
    wait_sec: float,
    romaji: str = "",
) -> bool:
    """ヘルパー単語フォールバック: テキストモデルにヘルパー単語を問い合わせ、
    変換後に余分な文字を削除して目標漢字を入力する。

    IMEポップアップ候補に目標漢字が現れない場合のラストリゾート。
    テキストモデルに「目標漢字を含む一般的な日本語単語」を提案してもらい、
    その単語を変換確定後に余分な末尾文字を削除することで目標漢字を得る。

    例: target_kanji="過膨張"
        テキストモデルが {"word": "過去", "romaji": "kako", "covered_prefix": "過", "delete_count": 1} を提案
        → "kako" を入力 → "過去" に変換確定 → Backspace × 1 で "去" 削除 → "過" 確定済み
        → 残り "膨張" を type_kanji_via_ime で入力

    Args:
        client: BLEClient インスタンス
        config: キャプチャ設定
        target_kanji: 変換しようとしている漢字/語句
        wait_sec: 変換待機秒数

    Returns:
        True: ヘルパー単語アプローチで全体の入力が成功した
        False: 失敗（呼び出し元が次のフォールバックへ進む）
    """
    # suggest_ime_helper_word は対象の最初の1文字を渡す
    target_char = target_kanji[0]
    # エントリー状態: target_kanji に対応するIMEポップアップが開いている
    # コミットなしでキャンセルしてから問い合わせに入る。問い合わせ待ちの間に
    # 誤候補（例: 「化」）を残したままにしないため、先に IME 組成を必ずクリアする。
    print("  [ヘルパー単語] IMEポップアップをキャンセル (Esc + Backspace×N)...")
    if romaji:
        _hira_len = _romaji_to_hiragana_len(romaji)
    else:
        _hira_len = _text_to_hiragana_len(target_kanji)
    _cancel_ime_popup_safe(client, target_kanji, wait=0.3, config=config, romaji=romaji)
    # _cancel_ime_popup_safe 内で固定 BS + VLM ガード済み。追加の安全ネットとして
    # VLM 確認のみ実施（budget を hira_len に合わせ、残存組成を確実に除去する）。
    _clear_pending_ime_composition(
        client,
        config,
        max_backspaces=max(_hira_len, 3),
    )

    text_runtime_label = _ime_text_runtime_label()
    print(f"  [ヘルパー単語] 「{target_char}」のヘルパー単語を{text_runtime_label}に問い合わせ中...")
    suggestions = suggest_ime_helper_word(target_char)
    if not suggestions:
        print(f"  [ヘルパー単語] 提案なし → フォールバック失敗")
        return False

    for idx, suggestion in enumerate(suggestions):
        helper_word = suggestion["word"]
        # backspace_count はヘルパー単語長 - 対象文字長から確定的に計算する（モデル提案値は使わない）
        backspace_count = len(helper_word) - len(target_char)
        # ヘルパー単語確定後に残る先頭部分: word[:-backspace_count] or word全体
        covered_prefix = helper_word[:-backspace_count] if backspace_count > 0 else helper_word
        # ヘルパー単語のローマ字は _kanji_to_romaji で計算
        helper_romaji = _kanji_to_romaji(helper_word)
        print(
            f"  [ヘルパー単語] 提案{idx+1}/{len(suggestions)}: {helper_word!r} "
            f"(romaji={helper_romaji!r}), covers={covered_prefix!r}, backspace={backspace_count}"
        )

        # ヘルパー単語のローマ字を入力
        print(f"  [ヘルパー単語] ローマ字入力: {helper_romaji!r}")
        ok = client.type_text(helper_romaji)
        print(f"  type:{helper_romaji} -> {'OK' if ok else 'NG'}")
        time.sleep(0.3)

        # Space でインライン変換 → さらに Space でポップアップを開く
        ok = client.press_key("space")
        print(f"  key:space (インライン変換) -> {'OK' if ok else 'NG'}")
        time.sleep(wait_sec)
        ok = client.press_key("space")
        print(f"  key:space (ポップアップ) -> {'OK' if ok else 'NG'}")
        time.sleep(max(wait_sec, 1.0))

        # ポップアップ候補を読取
        helper_frame = capture_screen(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if helper_frame is None:
            print("  [ヘルパー単語] フレーム取得失敗 → 次の提案を試みます")
            _cancel_ime_popup_safe(client, helper_word, config=config, romaji=helper_romaji)
            continue

        _save_debug_image(helper_frame, f"ime_helper_{target_kanji}_{helper_word}")

        helper_numbered: list[tuple[int, str]] = []
        try:
            helper_numbered = _read_helper_popup_candidates(
                helper_word,
                helper_frame,
                debug_name=f"helper_{target_kanji}",
            )
            print(f"  [ヘルパー単語] ポップアップ候補: {helper_numbered}")
        except Exception as exc:
            print(f"  [ヘルパー単語] 候補読取失敗: {exc}")

        normalized_helper_numbered = _normalize_helper_popup_candidates(
            helper_word,
            helper_romaji,
            helper_numbered,
        )
        if normalized_helper_numbered != helper_numbered:
            print(f"  [ヘルパー単語] 正規化候補: {normalized_helper_numbered}")

        highlighted_match = False
        highlighted_text: Optional[str] = None
        direct_match = _find_best_candidate_match(helper_word, helper_numbered, strict=True)
        if direct_match is not None:
            display_num, highlighted_text = direct_match
            print(f"  [ヘルパー単語] 「{helper_word}」→ 表示番号 {display_num} (読取: {highlighted_text!r})")
            if display_num <= 9:
                client.send_command(f"type:{display_num}")
                time.sleep(wait_sec)
            else:
                for _ in range(display_num - 2):
                    ok = client.press_key("space")
                    print(f"  key:space (helper select) -> {'OK' if ok else 'NG'}")
                    time.sleep(wait_sec)
            highlighted_match = True
        else:
            # 候補番号や一覧には入力欄ノイズが混ざることがあるため、
            # 現在ハイライトされている候補だけを読み取りながら Space で順送りする。
            max_cycles = max(_popup_cycle_budget(helper_numbered), len(normalized_helper_numbered))
            for cycle in range(max_cycles):
                if cycle > 0:
                    ok = client.press_key("space")
                    print(f"  key:space (helper select) -> {'OK' if ok else 'NG'}")
                    time.sleep(wait_sec)
                    helper_frame = capture_screen(
                        device_index=config.capture_device_index,
                        width=config.capture_width,
                        height=config.capture_height,
                    )
                    if helper_frame is None:
                        break
                try:
                    highlighted_text = read_highlighted_popup_candidate(
                        helper_frame,
                        debug_name=f"helper_selected_{target_kanji}",
                    )
                    print(f"  [ヘルパー単語] 現在候補: {highlighted_text!r}")
                except Exception as exc:
                    print(f"  [ヘルパー単語] 現在候補読取失敗: {exc}")
                    highlighted_text = None
                if highlighted_text and highlighted_text == helper_word:
                    highlighted_match = True
                    break

        if not highlighted_match:
            if not helper_numbered:
                print(f"  [ヘルパー単語] OCR候補なし → インライン第1候補をコミット試行...")
                client.press_key("escape")  # ポップアップ → インライン変換
                time.sleep(0.3)
                client.press_key("enter")   # インライン第1候補を確定
                time.sleep(0.3)
                if backspace_count > 0:
                    print(f"  [ヘルパー単語] Backspace × {backspace_count} で余分な文字を削除...")
                    for _ in range(backspace_count):
                        client.press_key("backspace")
                        time.sleep(0.15)
                _cleanup_after_helper_backspace(
                    client,
                    config,
                    helper_word=helper_word,
                    backspace_count=backspace_count,
                )
                print(f"  [ヘルパー単語] 「{covered_prefix}」の入力完了 (インラインコミット)")
                remaining = target_kanji[len(covered_prefix):]
                if remaining:
                    print(f"  [ヘルパー単語] 残り「{remaining}」を続けて入力します...")
                    remaining_romaji = _kanji_to_romaji(remaining)
                    time.sleep(1.0)
                    type_kanji_via_ime(
                        remaining_romaji,
                        remaining,
                        wait_sec=wait_sec,
                        _current_ime_mode="japanese",
                    )
                return True
            print(f"  [ヘルパー単語] 「{helper_word}」がハイライト候補に見つかりません → 次の提案を試みます")
            _cancel_ime_popup_safe(client, helper_word, config=config, romaji=helper_romaji)
            continue

        print(f"  [ヘルパー単語] 「{helper_word}」→ 現在候補 {highlighted_text!r} を Enter で確定")
        ok = client.press_key("enter")
        print(f"  key:enter (helper confirm) -> {'OK' if ok else 'NG'}")
        time.sleep(0.3)

        # 余分な文字を削除（確定済みテキストから削除するため安全）
        if backspace_count > 0:
            print(f"  [ヘルパー単語] Backspace × {backspace_count} で余分な文字を削除...")
            for _ in range(backspace_count):
                client.press_key("backspace")
                time.sleep(0.15)
        _cleanup_after_helper_backspace(
            client,
            config,
            helper_word=helper_word,
            backspace_count=backspace_count,
        )

        print(f"  [ヘルパー単語] 「{covered_prefix}」の入力完了")

        # 残りの文字列を入力
        remaining = target_kanji[len(covered_prefix):]
        if remaining:
            print(f"  [ヘルパー単語] 残り「{remaining}」を続けて入力します...")
            remaining_romaji = _kanji_to_romaji(remaining)
            # ESP32がBackspace処理を完了するまで待機してから次のセグメントへ
            time.sleep(1.0)
            type_kanji_via_ime(
                remaining_romaji,
                remaining,
                wait_sec=wait_sec,
                _current_ime_mode="japanese",
            )

        return True

    print(f"  [ヘルパー単語] 全提案が失敗 → フォールバック失敗")
    return False



    """mlx_vlm を使って IME ポップアップ候補リスト全体を読み取る。

    ポップアップ領域を自動検出してクロップしてから Qwen3-VL に送信する。

    Returns:
        候補文字列のリスト（表示順、0-indexed）。失敗時は空リスト。
    """
    return read_popup_candidates(frame)


def _extract_ime_candidates(ocr_texts: list[str]) -> list[str]:
    """OCR テキストリストから IME ポップアップ候補を抽出する（ノイズ除去）。

    入力フィールドのコンテキストノイズ（数字・記号・アルファベット混在テキスト）を除去し、
    純粋な日本語文字のみで構成される候補のみを返す。
    """
    result = []
    for text in ocr_texts:
        text = text.strip()
        if not text:
            continue
        # 純粋な日本語文字（ひらがな U+3040-、カタカナ U+30A0-、漢字 U+4E00-）のみ 1-6 文字
        if all("\u3040" <= ch <= "\u9fff" for ch in text) and 1 <= len(text) <= 6:
            result.append(text)
    return result


def _ime_candidate_matches(target: str, combined: str) -> bool:
    """IME候補OCRにターゲット文字列が含まれているか確認する。

    OCR ノイズを 1 文字まで許容する: 先頭文字が一致し、残りの差異が 1 文字以内の
    部分文字列も一致とみなす（例: 感冒 → 感昌 のような視覚的誤読への対策）。
    """
    if target in combined:
        return True
    n = len(target)
    if n < 2:
        return False
    for i in range(len(combined) - n + 1):
        substr = combined[i:i + n]
        if substr[0] == target[0]:
            mismatches = sum(a != b for a, b in zip(substr, target))
            if mismatches <= 1:
                return True
    return False


def _has_hiragana(text: str) -> bool:
    """Return True if text contains hiragana (indicates phonetic/mixed OCR read)."""
    return any('\u3041' <= c <= '\u3096' for c in text)


def _is_pure_hiragana(text: str) -> bool:
    """Return True if text consists entirely of hiragana.

    Pure-hiragana candidates in an IME popup represent the 'unprocessed phonetic
    reading' slot (e.g., 'ちょうしゅ' for 'choushu'). Selecting them types hiragana,
    NOT kanji. These must NOT be matched when the target is a kanji string.
    """
    return bool(text) and all('\u3041' <= c <= '\u3096' for c in text)


def _is_pure_katakana(text: str) -> bool:
    """Return True if text consists entirely of katakana + prolonged sound marks.

    Pure-katakana OCR reads (e.g., 'ハイ', 'ハイゾウ') indicate the OCR
    read the phonetic pronunciation of a kanji candidate, making romaji
    comparison safe (same as hiragana phonetic reads).
    """
    return bool(text) and all('\u30A0' <= c <= '\u30FF' for c in text)


def _is_pure_kanji(text: str) -> bool:
    """Return True if text consists entirely of CJK unified ideographs (kanji).

    Used to guard the romaji-comparison pass: when the target is pure kanji,
    accepting a pure-katakana candidate would type katakana instead of kanji.
    """
    return bool(text) and all('\u4E00' <= c <= '\u9FFF' for c in text)


def _find_best_candidate_match(
    target: str, numbered: list[tuple[int, str]], *, strict: bool = False,
) -> Optional[tuple[int, str]]:
    """番号付き候補リストからターゲットに最もよく一致する候補を返す。

    完全一致を優先し、なければファジーマッチを試みる。
    strict=True の場合はファジーマッチを無効にし、完全一致のみを使用する。
    ヘルパー単語の候補照合など、組成残存による誤合致を防ぐ場合に使用する。
    例: target="痛", candidates=[(1,'通'),(2,'疼痛'),(5,'痛')] → (5, '痛')
    """
    # First pass: exact match
    for n, c in numbered:
        if c == target:
            return (n, c)
    if strict:
        return None
    # Second pass: fuzzy same-length match (OCR noise tolerance)
    for n, c in numbered:
        if len(c) == len(target) and _ime_candidate_matches(target, c):
            return (n, c)
    # Third pass: general substring/fuzzy match (e.g., target is part of candidate)
    for n, c in numbered:
        if _ime_candidate_matches(target, c):
            return (n, c)
    # Fourth pass: romaji comparison — only for targets that already contain kana.
    # For pure-kanji targets, any popup candidate that OCR/VLM read as kana-bearing text
    # is too ambiguous: selecting it can commit a same-reading but wrong candidate
    # (e.g., '兼さ' for '検査') or a literal kana entry instead of the intended kanji.
    # In those cases, fall through to helper-word / highlighted-candidate confirmation
    # rather than trusting the reading alone.
    try:
        target_romaji = _kanji_to_romaji(target)
        target_is_pure_kanji = _is_pure_kanji(target)
        if target_is_pure_kanji:
            return None
        target_has_kanji = _has_kanji(target)
        for n, c in numbered:
            # Accept only mixed hiragana+kanji or pure katakana
            if _is_pure_hiragana(c):
                continue  # hiragana-only = IME unprocessed reading slot → skip
            if _is_pure_katakana(c):
                if target_has_kanji:
                    continue  # target has kanji but candidate is katakana → skip
                # target is katakana too → allow romaji match
            elif not _has_hiragana(c):
                continue  # pure kanji = real homonym risk → skip
            try:
                if _kanji_to_romaji(c) == target_romaji:
                    print(f"  [候補照合/romaji] {c!r} → {_kanji_to_romaji(c)!r} ≈ {target!r} → 採用")
                    return (n, c)
            except Exception:
                pass
    except Exception:
        pass
    # Fifth pass: visual confusible first character.
    # OCR sometimes misreads only the first kanji (e.g., '著'→'善', '聴'→'徳').
    # If all chars except the first match exactly, the first char is a visual confusible
    # and this is almost certainly an OCR misread of the correct candidate.
    # Requires length ≥ 2 to avoid false positives on single-char targets.
    if len(target) >= 2:
        target_suffix = target[1:]
        for n, c in numbered:
            if len(c) == len(target) and c[1:] == target_suffix and c[0] != target[0]:
                print(f"  [候補照合/suffix] {c!r} suffix={target_suffix!r} ≈ {target!r} → 採用")
                return (n, c)
    return None


def _ime_fullframe_exact_match(target: str, fullframe_result: str) -> bool:
    """フルフレームVLM結果との厳格な一致判定。

    フルフレームVLMは前後の入力済み文字列まで含めて返すことがあるため、
    部分一致は使わず、完全一致のみを許容する。
    """
    if not fullframe_result or not target:
        return False
    return fullframe_result.strip() == target



def _is_ascii_only(text: str) -> bool:
    """文字列が ASCII 文字のみで構成されているか判定する。"""
    return all(ord(ch) < 128 for ch in text)


def _is_hiragana_only(text: str) -> bool:
    """文字列がひらがな（長音符含む）のみで構成されているか判定する。"""
    return bool(text) and all(
        "\u3040" <= ch <= "\u309f" or ch == "\u30fc"  # ひらがな + 長音符ー
        for ch in text
    )


def _is_katakana_only(text: str) -> bool:
    """文字列がカタカナ（長音符含む）のみで構成されているか判定する。"""
    return bool(text) and all(
        "\u30a0" <= ch <= "\u30ff" or ch == "\u30fc"  # カタカナ + 長音符ー
        for ch in text
    )


def detect_ime_mode(
    client: "BLEClient",
    config=None,
) -> Optional[str]:
    """
    'a' を1文字入力し、Qwen3-VL でスクリーンを読んで IME モードを検出する。

    英語入力モードなら 'a' が、日本語（ひらがな）入力モードなら「あ」が表示される。
    VLM で判定後に Backspace で入力した文字を削除する。

    Args:
        client: BLEClient インスタンス（キー送信に使用）
        config: キャプチャ設定（None の場合は load_config() で取得）

    Returns:
        'japanese': ひらがな入力モード
        'english':  英数字入力モード
        None:       判定不能
    """
    if config is None:
        config = load_config(skip_password=True)

    # 入力前フレームを取得（差分比較でより確実な IME モード検出に使用）
    pre_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
        flush_duration=0.5,  # 入力前なので短い flush で十分
    )

    # 'a' を1文字入力してIMEの反応を確認
    client.type_text("a")
    time.sleep(0.4)

    post_frame = capture_screen(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )

    result: Optional[str] = None
    if post_frame is not None:
        result = detect_ime_mode_from_typed_a(post_frame, pre_frame=pre_frame)
        print(f"  [IME検出/VLM] 結果: {result!r}")
    else:
        print("  [IME検出] キャプチャ失敗")

    # 入力した 'a' または 'あ' を削除する
    if result == "japanese":
        # 日本語モード: IME の未確定「あ」を Escape でキャンセル（確実）
        client.press_key("escape")
        time.sleep(0.25)
        # 念のため Backspace も送る（Escape が効かない場合への保険）
        client.press_key("backspace")
        time.sleep(0.2)
    elif result == "english":
        # 英語モード: 'a' を Backspace で削除
        client.press_key("backspace")
        time.sleep(0.2)
    else:
        # 判定不能: Escape と Backspace 両方送る
        client.press_key("escape")
        time.sleep(0.15)
        client.press_key("backspace")
        time.sleep(0.2)

    return result


def toggle_ime(client: "BLEClient") -> None:
    """半角/全角キーを送って IME モードをトグルする。"""
    print("  [IME切替] 半角/全角 を送信")
    client.press_key("zenkaku")
    time.sleep(0.5)  # IME 切替の反映を待つ


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
    # 医療用語など pykakasi が誤読みするケースの手動オーバーライド
    _ROMAJI_OVERRIDES: dict[str, str] = {
        "鼻汁": "bijuu",
        "咳嗽": "gaisou",
        "嘔吐": "outo",
        "浮腫": "fushuu",
        "倦怠": "kentai",
        "痙攣": "keiren",
        "蕁麻疹": "jinmashin",
        "喀痰": "kakutan",
        "喘鳴": "zenmei",
        "哮喘": "kozen",
        "喘息": "zensoku",
        "膿胸": "noukyo",
        "胸水": "kyousui",
        "肺炎": "haien",
        "動脈血": "doumyakuketsu",
    }
    if text in _ROMAJI_OVERRIDES:
        return _ROMAJI_OVERRIDES[text]
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


def type_japanese_sentence(text: str, clear_field: bool = False) -> None:
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
        clear_field: True の場合、入力前に Backspace を 50 回送信してフィールドをクリアする
    """
    print(f"文節分割中 ({_segmentation_runtime_label()} 優先): {text!r}")
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    if clear_field:
        # IME 未確定をキャンセルしてから末尾へ移動し、Backspace 200 回で全削除
        print("フィールドをクリア中 (Escape + Ctrl+End + Backspace×200)...")
        client.press_key("escape")
        time.sleep(0.3)
        client.press_key("ctrl+end")
        time.sleep(0.5)
        client.type_text("\x08" * 200)
    # 開始時に1回だけ IME モードを検出し、以降は内部変数でトラッキングする
    print("現在の IME モードを検出中...")
    current_mode: Optional[str] = detect_ime_mode(client, config)
    # 検出失敗(None)の場合: 前回の中断でポップアップが残っている可能性があるため
    # Escape を複数回送ってクリアし、再検出する
    if current_mode is None:
        print("  [IME回復] モード不明 → Escape×3 でクリアして再検出します")
        for _ in range(3):
            client.press_key("escape")
            time.sleep(0.2)
        time.sleep(0.5)
        current_mode = detect_ime_mode(client, config)
        print(f"  [IME回復] 再検出結果: {current_mode!r}")
    print(f"初期 IME モード: {current_mode!r}")

    segments = list(_iter_segments_for_input(text))
    for index, seg in enumerate(segments):
        seg_text = seg["text"]
        seg_romaji = seg["romaji"]
        is_last_segment = index == len(segments) - 1
        print(f"\n--- 文節: {seg_text!r} ({seg_romaji}) ---")

        if seg_text == "\n":
            current_mode = ensure_ime_mode("english", client, current_mode)
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")

        elif seg_text in ("、", "。"):
            # 句読点: ひらがなモードで IME が自動変換するキー（,/.）を送る。
            # Windows 10 では「、」「。」ともに変換候補ポップアップが表示されるため
            # Enter で確定が必要。Windows 7 では「。」のみ必要。
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  句読点入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            if seg_text == "。" or is_last_segment:
                ok = client.press_key("enter")
                print(f"key:enter -> {'OK' if ok else 'NG'}")

        elif _is_ascii_only(seg_text):
            # ASCII のみ（英単語・数字・記号）: 英数字モードで直接入力（IME 変換不要）
            current_mode = ensure_ime_mode("english", client, current_mode)
            print(f"  英数字直接入力: {seg_text!r}")
            _type_ascii_text_precisely(client, seg_text)

        elif _has_kanji(seg_text):
            # 漢字を含む文節: ひらがなモードで IME 変換候補を確認してから確定
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            type_kanji_via_ime(
                seg_romaji, seg_text,
                _current_ime_mode=current_mode,
            )
            time.sleep(0.25)

        elif _is_katakana_only(seg_text):
            # カタカナのみ: ひらがなモードでローマ字入力後 F7 でカタカナ変換して Enter で確定
            # VLM の romaji は長音符(ー)を母音重複(oo,aa)で返すことがあるため、
            # カタカナ原文から _katakana_to_romaji で IME 用 romaji を再生成する
            katakana_romaji = _katakana_to_romaji(seg_text)
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  カタカナ直接入力: {katakana_romaji!r}")
            ok = client.type_text(katakana_romaji)
            print(f"type:{katakana_romaji} -> {'OK' if ok else 'NG'}")
            ok = client.press_key("f7")  # 全角カタカナに変換
            print(f"key:f7 -> {'OK' if ok else 'NG'}")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")

        elif seg_text in ("「", "」", "『", "』"):
            # 日本語括弧: ひらがなモードで lbracket/rbracket キーを押す
            # JIS キーボードの日本語モードでは [ → 「、] → 」 になる
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            key_name = "lbracket" if seg_text in ("「", "『") else "rbracket"
            ok = client.press_key(key_name)
            print(f"key:{key_name} ({seg_text}) -> {'OK' if ok else 'NG'}")

        elif any(ch in _JP_SYMBOL_IME_READINGS or ch in _CHAR_ASCII_FALLBACK for ch in seg_text):
            # ※ などの特殊記号、または μ などの ASCII 代替文字を含む: 文字単位で処理
            # ASCII 部分は英数字モードで直接入力、特殊記号は IME で読み変換
            for ch in seg_text:
                if ch in _JP_SYMBOL_IME_READINGS:
                    reading = _JP_SYMBOL_IME_READINGS[ch]
                    current_mode = ensure_ime_mode("japanese", client, current_mode)
                    print(f"  [特殊記号] {ch!r} を IME読み {reading!r} で入力")
                    type_kanji_via_ime(
                        reading, ch,
                        _current_ime_mode=current_mode,
                    )
                elif ch in _CHAR_ASCII_FALLBACK:
                    # μ など BLE 経由で直接送れない文字は ASCII 代替文字で入力
                    fallback = _CHAR_ASCII_FALLBACK[ch]
                    if fallback:
                        current_mode = ensure_ime_mode("english", client, current_mode)
                        print(f"  [ASCII代替] {ch!r} → {fallback!r}")
                        _type_ascii_text_precisely(client, fallback)
                    else:
                        print(f"  [スキップ] {ch!r} (ASCII代替なし)")
                elif _is_ascii_only(ch):
                    current_mode = ensure_ime_mode("english", client, current_mode)
                    _type_ascii_text_precisely(client, ch)
                else:
                    current_mode = ensure_ime_mode("japanese", client, current_mode)
                    ok = client.type_text(ch)
                    if ok:
                        client.press_key("enter")

        else:
            # ひらがなのみ: ローマ字直接入力後に Enter で IME 組成を明示的に確定する
            current_mode = ensure_ime_mode("japanese", client, current_mode)
            print(f"  ひらがな直接入力: {seg_romaji!r}")
            ok = client.type_text(seg_romaji)
            print(f"type:{seg_romaji} -> {'OK' if ok else 'NG'}")
            ok = client.press_key("enter")
            print(f"key:enter -> {'OK' if ok else 'NG'}")
            # 確定後、WindowsのIMEが処理を完了するまで待機
            time.sleep(0.15)

    print("\n文章入力完了")


def _type_english_text(text: str, clear_field: bool = False) -> None:
    """
    英語テキストを英数字モードで直接入力する。

    IME を英数字モードに切替えてからテキストを送信する。
    Enter は送らない（呼び出し元がフィールド確定を制御する）。

    Args:
        text: 入力する英数字文字列
        clear_field: True の場合、入力前に Backspace を 50 回送信してフィールドをクリアする
    """
    config = load_config(skip_password=True)

    client = _wait_for_ble_connected()

    if clear_field:
        print("フィールドをクリア中 (Backspace x50)...")
        for _ in range(50):
            client.press_key("backspace")
        time.sleep(0.3)

    current_mode = detect_ime_mode(client, config)
    ensure_ime_mode("english", client, current_mode)

    print(f"英語入力: {text!r}")
    _type_ascii_text_precisely(client, _normalize_text_for_typing(text))
    print("完了")


def _print_usage() -> None:
    """使い方を表示する。"""
    print("EHR Input Automation")
    print()
    print("使い方:")
    print("  python -m automation.ehr_input [オプション] [コマンド/テキスト]")
    print()
    print("オプション:")
    print("  --clear              入力前に Backspace を 50 回送信してフィールドをクリアする")
    print("  --openrouter <model> 文節分割・IME候補読取・ヘルパー単語提案を OpenRouter のモデルで実行する（要: vision対応モデル）")
    print("  --help, -h, help     このヘルプを表示")
    print()
    print("コマンド:")
    print("  (引数なし)                         テスト患者カルテを開く")
    print("  open test                          テスト患者カルテを開く")
    print("  close record                       取り消し[F9]ボタンをクリックしてカルテを閉じる")
    print("  click history <yyyymmdd>           過去カルテの指定日付をクリック")
    print("  edit history <yyyymmdd>            過去カルテ日付クリック後に修正ボタンをクリック")
    print("  detect                             IMEモードを検出して表示")
    print()
    print("テキスト入力:")
    print("  <テキスト>                         テキストを入力（日本語はIME変換）")
    print("  <ファイル.txt>                     テキストファイルの内容を入力")
    print("  open test <テキスト/ファイル>      カルテを開いてからテキスト入力")
    print()
    print("例:")
    print('  python -m automation.ehr_input 肺炎')
    print('  python -m automation.ehr_input --clear 肺炎')
    print('  python -m automation.ehr_input --openrouter qwen/qwen3.5-9b "両肺野に"')
    print('  python -m automation.ehr_input --openrouter qwen/qwen3.5-35b-a3b "聴診"')
    print('  python -m automation.ehr_input "COVID-19の検査"')
    print('  python -m automation.ehr_input note.txt')
    print('  python -m automation.ehr_input "open test" 肺炎')
    print('  python -m automation.ehr_input "open test" "MRI所見"')
    print('  python -m automation.ehr_input "click history 20190502"')


def _run_cli_with_parsed_args(args: list[str], option_summary: dict[str, object]) -> int:
    """Run the CLI using already-normalized option state."""
    openrouter_model = option_summary["openrouter_model"]
    clear_field = bool(option_summary["clear_field"])

    _configure_runtime(openrouter_model=openrouter_model)

    if not args:
        # 引数なし: デフォルト動作（後方互換）
        open_test_patient_chart()
        return 0

    if len(args) == 1 and args[0] in ("help", "--help", "-h"):
        _print_usage()
        return 0

    if len(args) == 1 and args[0] == "open test":
        # "open test" のみ → テスト患者カルテを開く
        open_test_patient_chart()
        return 0

    if len(args) == 1 and args[0] == "close record":
        # "close record" → 「取り消し[F9]」ボタンをクリックしてカルテを閉じる
        close_record()
        return 0

    if len(args) == 1 and args[0].startswith("click history "):
        # "click history yyyymmdd" → 過去カルテ列の指定日付をクリック
        date_str = args[0][len("click history "):].strip()
        click_history(date_str)
        return 0

    if len(args) >= 2 and args[0] == "click history":
        # "click history" "yyyymmdd" (2引数) → 過去カルテ列の指定日付をクリック
        click_history(args[1])
        return 0

    if len(args) == 1 and args[0].startswith("edit history "):
        # "edit history yyyymmdd" → 過去カルテ日付クリック後に修正ボタンをクリック
        date_str = args[0][len("edit history "):].strip()
        edit_history(date_str)
        return 0

    if len(args) >= 2 and args[0] == "edit history":
        # "edit history" "yyyymmdd" (2引数) → 過去カルテ日付クリック後に修正ボタンをクリック
        edit_history(args[1])
        return 0

    if len(args) == 1:
        text = _resolve_text_argument(args[0])
        # Named key shortcut: "esc", "backspace", "enter", "space", "tab", "delete",
        # "f6", "f7", "f8", or generic "key:xxx" syntax → press the key directly.
        _NAMED_KEYS = {"esc", "escape", "backspace", "enter", "return", "space", "tab",
                       "delete", "f6", "f7", "f8", "f9", "up", "down", "left", "right"}
        if text.lower() in _NAMED_KEYS or text.lower().startswith("key:"):
            key_name = text[4:] if text.lower().startswith("key:") else text.lower()
            client = _wait_for_ble_connected()
            ok = client.press_key(key_name)
            print(f"key:{key_name} -> {'OK' if ok else 'NG'}")
            return 0
        try:
            _input_resolved_text(text, clear_field=clear_field)
        except MlxVlmSegmentationError as exc:
            print(f"mlx_vlm文節分割エラー: {exc}")
            print("omlxサーバーの動作確認: curl -s -H 'Authorization: Bearer omlxkey' http://localhost:8000/v1/models")
            return 1
        return 0

    if len(args) >= 2 and args[0] == "open test":
        # 第一引数が "open test"、第二引数がテキスト → カルテ開いてから入力
        text = _resolve_text_argument(args[1])
        print(f"テスト患者カルテを開いてから入力: {text!r}")
        open_test_patient_chart()
        try:
            _input_resolved_text(text, clear_field=clear_field)
        except MlxVlmSegmentationError as exc:
            print(f"mlx_vlm文節分割エラー: {exc}")
            print("omlxサーバーの動作確認: curl -s -H 'Authorization: Bearer omlxkey' http://localhost:8000/v1/models")
            return 1
        return 0

    _print_usage()
    return 1


def _run_cli(args: list[str]) -> int:
    """CLI entry point for manual EHR input automation."""
    positional_args, option_summary = _parse_cli_options(args)
    return _run_cli_with_parsed_args(positional_args, option_summary)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    executable_path = sys.argv[0] if sys.argv else "automation.ehr_input"
    with _capture_run_output():
        try:
            positional_args, option_summary = _parse_cli_options(args)
            print(
                _build_run_log_header(
                    executable_path,
                    args,
                    positional_args,
                    option_summary,
                )
            )
            print()
            return _run_cli_with_parsed_args(positional_args, option_summary)
        except (RuntimeError, ValueError) as exc:
            print(exc)
            return 1


if __name__ == '__main__':
    import sys

    sys.exit(main())
