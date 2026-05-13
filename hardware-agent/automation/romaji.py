"""Shared IME-friendly romaji conversion helpers."""

from __future__ import annotations

import re
from typing import Callable

import cutlet
from pykakasi import kakasi

_CUTLET = None
_KAKASI = None

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
    "生食": "seishoku",
    "静注": "seichuu",
    "日前": "nichimae",
}

_HIRAGANA_IME_OVERRIDES: dict[str, str] = {
    "は": "ha",
    "へ": "he",
    "を": "wo",
}

_ROMAJI_ARBITRATOR: Callable[[str, str, str], str] | None = None
_RUNTIME_ROMAJI_CACHE: dict[str, str] = {}


def set_romaji_arbitrator(fn: Callable[[str, str, str], str]) -> None:
    """cutlet/kakasi 不一致時に呼ばれる LLM アービトレーター関数を登録する。"""
    global _ROMAJI_ARBITRATOR
    _ROMAJI_ARBITRATOR = fn


def _get_cutlet() -> cutlet.Cutlet:
    global _CUTLET
    if _CUTLET is None:
        _CUTLET = cutlet.Cutlet(use_foreign_spelling=False, ensure_ascii=True)
    return _CUTLET


def _get_kakasi() -> kakasi:
    global _KAKASI
    if _KAKASI is None:
        _KAKASI = kakasi()
    return _KAKASI


def _normalize_cutlet_romaji(romaji: str) -> str:
    return re.sub(r"[^a-z]", "", romaji.lower())


def _normalize_kakasi_romaji(text: str) -> str:
    return re.sub(r"[^a-z]", "",
        "".join(item["hepburn"] for item in _get_kakasi().convert(text)).lower())


def _is_katakana_only(text: str) -> bool:
    return bool(text) and all(("゠" <= ch <= "ヿ") or ch in "・ー" for ch in text)


def _is_hiragana_only(text: str) -> bool:
    return bool(text) and all("぀" <= ch <= "ゟ" for ch in text)


def _romanize_hiragana_with_overrides(text: str) -> str:
    """純粋ひらがな文字列をオーバーライドを適用しながらローマ字に変換する。

    _HIRAGANA_IME_OVERRIDES の文字（は→ha 等）は個別適用。
    末尾の っ は xtsu（後続子音がないため doubled-consonant 不可）。
    それ以外は cutlet に渡す（っ+子音の二重化は cutlet が正しく処理する）。
    """
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        is_last = (i == len(text) - 1)
        if ch == "っ" and is_last:
            result.append("xtsu")
            i += 1
        elif ch in _HIRAGANA_IME_OVERRIDES:
            result.append(_HIRAGANA_IME_OVERRIDES[ch])
            i += 1
        else:
            j = i + 1
            while j < len(text):
                nc = text[j]
                if nc in _HIRAGANA_IME_OVERRIDES:
                    break
                if nc == "っ" and j == len(text) - 1:
                    break
                j += 1
            chunk = text[i:j]
            result.append(_normalize_cutlet_romaji(_get_cutlet().romaji(chunk)))
            i = j
    return "".join(result)


def katakana_to_romaji_for_ime(kana: str) -> str:
    """Convert katakana to IME-friendly ASCII romaji."""
    kana_normalized = kana.replace("ー", "-").replace("・", "/")
    return "".join(item["hepburn"] for item in _get_kakasi().convert(kana_normalized))


def romanize_for_ime(text: str) -> str:
    """Romanize Japanese text for IME input using cutlet plus IME-specific overrides."""
    override = _ROMAJI_OVERRIDES.get(text)
    if override is not None:
        return override

    kana_override = _HIRAGANA_IME_OVERRIDES.get(text)
    if kana_override is not None:
        return kana_override

    if _is_katakana_only(text):
        return katakana_to_romaji_for_ime(text)

    if _is_hiragana_only(text) and (
        any(ch in _HIRAGANA_IME_OVERRIDES for ch in text)
        or text.endswith("っ")
    ):
        return _romanize_hiragana_with_overrides(text)

    cutlet_result = _normalize_cutlet_romaji(_get_cutlet().romaji(text))
    kakasi_result = _normalize_kakasi_romaji(text)

    if cutlet_result == kakasi_result:
        return cutlet_result

    if text in _RUNTIME_ROMAJI_CACHE:
        return _RUNTIME_ROMAJI_CACHE[text]

    if _ROMAJI_ARBITRATOR is not None:
        resolved = _ROMAJI_ARBITRATOR(text, cutlet_result, kakasi_result)
        _RUNTIME_ROMAJI_CACHE[text] = resolved
        return resolved

    return cutlet_result
