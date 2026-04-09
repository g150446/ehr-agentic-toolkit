"""CLI probe for testing local Ollama Japanese segmentation responses."""

from __future__ import annotations

import sys

from automation.ollama_segmentation import (
    OLLAMA_SEGMENTATION_MODEL,
    OLLAMA_SEGMENTATION_TIMEOUT,
    OLLAMA_SEGMENTATION_URL,
    OllamaSegmentationError,
    segment_japanese_text_with_ollama,
)


def main(argv: list[str] | None = None) -> int:
    """Run the probe from the command line."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("使い方:")
        print('  python -m automation.ollama_segment_probe "肺炎に対して抗菌薬による治療を行う"')
        return 1

    text = args[0]
    print(f"対象文: {text!r}")
    print(f"endpoint: {OLLAMA_SEGMENTATION_URL}")
    print(f"model: {OLLAMA_SEGMENTATION_MODEL}")
    print(f"timeout: {OLLAMA_SEGMENTATION_TIMEOUT:g}秒")

    try:
        raw_content, segments = segment_japanese_text_with_ollama(text)
    except OllamaSegmentationError as exc:
        print(f"エラー: {exc}")
        return 1

    print(f"ollama応答: {raw_content!r}")
    print("分割結果:")
    for index, segment in enumerate(segments, start=1):
        print(f"  {index}. {segment['text']!r} ({segment['romaji']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
