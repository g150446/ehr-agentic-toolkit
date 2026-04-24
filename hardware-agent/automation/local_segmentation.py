"""Helpers for local Japanese text segmentation using sudachipy + cutlet.

sudachipy (形態素解析) でテキストを分割し、cutlet ベースの IME 向けローマ字化
ヘルパーで ASCII ローマ字を生成する。純カタカナのみ pykakasi を併用し、
長音符や中黒を JIS IME 入力に合わせて扱う。
"""

from __future__ import annotations

import sudachipy
import sudachidict_core  # noqa: F401  # 辞書パッケージ（インポートで登録される）
from automation.romaji import katakana_to_romaji_for_ime, romanize_for_ime

# モジュールレベルでシングルトンを保持（初期化コストを1回に抑える）
_dic: sudachipy.Dictionary | None = None


def _get_dic() -> sudachipy.Dictionary:
    global _dic
    if _dic is None:
        _dic = sudachipy.Dictionary(dict="core")
    return _dic


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
    "「": "[",   # 日本語モードで lbracket キー → 「
    "」": "]",   # 日本語モードで rbracket キー → 」
    "『": "[",   # 同上
    "』": "]",
    "・": "/",   # 日本語モードで / キー → ・
}

# Windows IME が一語として認識しにくい医療用語を手動で分割するテーブル。
# セグメントが丸ごとこのテーブルのキーに一致した場合、代わりにここで定義した
# (text, romaji) のリストに展開する。
_MANUAL_WORD_SPLITS: dict[str, list[tuple[str, str]]] = {
    "咽頭痛":      [("咽頭", "intou"), ("痛", "tsuu")],
    "関節痛":      [("関節", "kansetsu"), ("痛", "tsuu")],
    "筋肉痛":      [("筋肉", "kinniku"), ("痛", "tsuu")],
    "頭痛":        [("頭", "atama"), ("痛", "tsuu")],   # ずつう は IME に入りにくい
    "腹痛":        [("腹", "hara"), ("痛", "tsuu")],
    "胸痛":        [("胸", "mune"), ("痛", "tsuu")],
    "背部痛":      [("背部", "haibu"), ("痛", "tsuu")],
    # 長い複合動詞: Windows IME が一語として変換できないため分割する
    "吸えなくなった": [("吸え", "sue"), ("なくなった", "nakunatta")],
    "使ったが":    [("使った", "tsukatta"), ("が", "ga")],
    "改善しない":  [("改善", "kaizen"), ("しない", "shinai")],
    "改善しないため": [("改善", "kaizen"), ("しない", "shinai"), ("ため", "tame")],
}


def _should_merge(pos: tuple[str, ...]) -> bool:
    """この品詞のトークンを直前のセグメントに結合するかどうか。"""
    return (pos[0], pos[1]) in _MERGE_POS or (pos[0], "*") in _MERGE_POS


def _katakana_to_romaji(kana: str) -> str:
    """カタカナ読みをヘボン式ローマ字に変換する。

    カタカナ長音符 ー (U+30FC) は日本語IMEで「-」キーで入力するため、
    pykakasi に渡す前に「-」に置換する。
    （pykakasi はー を直前の母音の繰り返し "ee" などに変換してしまうため。）

    中黒 ・ (U+30FB) は JIS キーボードの日本語モードで「/」キーで入力するため、
    pykakasi に渡す前に「/」に置換する。
    （非 ASCII 文字を BLE HID で送ると予測不能なキー入力になるため。）
    """
    kana_normalized = kana.replace("ー", "-").replace("・", "/")
    return katakana_to_romaji_for_ime(kana_normalized)


def segment_japanese_text_locally(
    text: str,
) -> tuple[str, list[dict[str, str]]]:
    """sudachipy + cutlet で日本語テキストを IME 入力単位に分割する。

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

    # 各セグメントの surface を IME 向け romaji に変換する。
    # 句読点セグメントはすでに romaji を持っているのでスキップ。
    # 既知の医療用語（Windows IME が一語変換できないもの）は手動分割テーブルで展開する。
    result: list[dict[str, str]] = []
    for seg in segments:
        if "romaji" in seg:
            result.append({"text": seg["text"], "romaji": seg["romaji"]})
        elif seg["text"] in _MANUAL_WORD_SPLITS:
            for sub_text, sub_romaji in _MANUAL_WORD_SPLITS[seg["text"]]:
                result.append({"text": sub_text, "romaji": sub_romaji})
        else:
            romaji = romanize_for_ime(seg["text"])
            result.append({"text": seg["text"], "romaji": romaji})

    summary = " / ".join(f"{s['text']}({s['romaji']})" for s in result)
    return summary, result
