"""CLI probe for testing local (sudachipy + cutlet) Japanese segmentation."""

from __future__ import annotations

import sys

from automation.local_segmentation import segment_japanese_text_locally


def main(argv: list[str] | None = None) -> int:
    """Run the probe from the command line."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("使い方:")
        print('  python -m automation.local_segment_probe "肺炎に対して抗菌薬による治療を行う"')
        return 1

    text = args[0]
    print(f"対象文: {text!r}")
    print("エンジン: sudachipy (SplitMode.C) + cutlet (+ pykakasi for katakana)")

    summary, segments = segment_japanese_text_locally(text)

    print(f"分割サマリ: {summary}")
    print("分割結果:")
    for index, segment in enumerate(segments, start=1):
        print(f"  {index}. {segment['text']!r} ({segment['romaji']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
