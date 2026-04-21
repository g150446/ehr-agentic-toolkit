"""Shared IME-friendly romaji conversion helpers."""

from __future__ import annotations

import re

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
}

_HIRAGANA_IME_OVERRIDES: dict[str, str] = {
    "は": "ha",
    "へ": "he",
    "を": "wo",
}


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


def _is_katakana_only(text: str) -> bool:
    return bool(text) and all(("\u30A0" <= ch <= "\u30FF") or ch in "・ー" for ch in text)


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

    return _normalize_cutlet_romaji(_get_cutlet().romaji(text))
