"""Helpers for local Japanese text segmentation using sudachipy + pykakasi.

sudachipy (形態素解析) でテキストを分割し、pykakasi (ローマ字変換) でヘボン式
ローマ字を生成する。LLM ベースの実装と異なり、辞書引きによる決定論的な変換なので
長母音（ちりょう → chiryou）などを確実に正しく処理できる。
"""

from __future__ import annotations

import sudachipy
import sudachidict_core  # noqa: F401  # 辞書パッケージ（インポートで登録される）
from pykakasi import kakasi

# モジュールレベルでシングルトンを保持（初期化コストを1回に抑える）
_dic: sudachipy.Dictionary | None = None
_kks: kakasi | None = None


def _get_dic() -> sudachipy.Dictionary:
    global _dic
    if _dic is None:
        _dic = sudachipy.Dictionary(dict="core")
    return _dic


def _get_kks() -> kakasi:
    global _kks
    if _kks is None:
        _kks = kakasi()
    return _kks


# 直前のトークンに結合する品詞の組み合わせ
# (品詞, 品詞細分類1) → True なら直前に結合する
_MERGE_POS = {
    ("助詞", "接続助詞"),   # て・で・ながら 等（例: 対し+て → 対して）
    ("動詞", "非自立可能"),  # てる・ている 等の補助動詞
    ("助動詞", "*"),         # た・ない・ます 等
}

_PUNCTUATION_ROMAJI = {
    "、": ",",
    "。": ".",
    "（": "(",
    "）": ")",
    "％": "%",
    "：": ":",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
}

# Windows IME が一語として認識しにくい医療用語を手動で分割するテーブル。
# セグメントが丸ごとこのテーブルのキーに一致した場合、代わりにここで定義した
# (text, romaji) のリストに展開する。
_MANUAL_WORD_SPLITS: dict[str, list[tuple[str, str]]] = {
    "咽頭痛":  [("咽頭", "intou"), ("痛", "tsuu")],
    "関節痛":  [("関節", "kansetsu"), ("痛", "tsuu")],
    "筋肉痛":  [("筋肉", "kinniku"), ("痛", "tsuu")],
    "頭痛":    [("頭", "atama"), ("痛", "tsuu")],   # ずつう は IME に入りにくい
    "腹痛":    [("腹", "hara"), ("痛", "tsuu")],
    "胸痛":    [("胸", "mune"), ("痛", "tsuu")],
    "背部痛":  [("背部", "haibu"), ("痛", "tsuu")],
}


def _should_merge(pos: tuple[str, ...]) -> bool:
    """この品詞のトークンを直前のセグメントに結合するかどうか。"""
    return (pos[0], pos[1]) in _MERGE_POS or (pos[0], "*") in _MERGE_POS


def _katakana_to_romaji(kana: str) -> str:
    """カタカナ読みをヘボン式ローマ字に変換する。"""
    kks = _get_kks()
    return "".join(item["hepburn"] for item in kks.convert(kana))


def segment_japanese_text_locally(
    text: str,
) -> tuple[str, list[dict[str, str]]]:
    """sudachipy + pykakasi で日本語テキストを IME 入力単位に分割する。

    Returns:
        (summary, segments) のタプル。
        summary: 分割結果の要約文字列（ログ用）
        segments: [{"text": "文節", "romaji": "ローマ字"}, ...] のリスト
    """
    dic = _get_dic()
    tokenizer = dic.create()

    morphemes = tokenizer.tokenize(text, sudachipy.SplitMode.C)

    segments: list[dict[str, str]] = []
    for m in morphemes:
        pos = m.part_of_speech()  # (品詞, 品詞細分類1, ...)
        surface = m.surface()
        reading = m.reading_form()  # カタカナ読み

        # 句読点・記号は入力時に失われないよう対応する ASCII として保持
        if pos[0] == "補助記号":
            romaji = _PUNCTUATION_ROMAJI.get(surface)
            if romaji is not None:
                segments.append({"text": surface, "romaji": romaji})
            continue

        # 接続助詞などは直前セグメントに結合
        if segments and _should_merge(pos):
            prev = segments[-1]
            prev["text"] += surface
            prev["reading"] += reading
        else:
            segments.append({"text": surface, "reading": reading})

    # 各セグメントの reading → hepburn romaji に変換
    # 句読点セグメントはすでに romaji を持っているのでスキップ
    # 既知の医療用語（Windows IME が一語変換できないもの）は手動分割テーブルで展開する
    result: list[dict[str, str]] = []
    for seg in segments:
        if "romaji" in seg:
            result.append({"text": seg["text"], "romaji": seg["romaji"]})
        elif seg["text"] in _MANUAL_WORD_SPLITS:
            for sub_text, sub_romaji in _MANUAL_WORD_SPLITS[seg["text"]]:
                result.append({"text": sub_text, "romaji": sub_romaji})
        else:
            romaji = _katakana_to_romaji(seg["reading"])
            result.append({"text": seg["text"], "romaji": romaji})

    summary = " / ".join(f"{s['text']}({s['romaji']})" for s in result)
    return summary, result
