"""CLI probe for testing local mlx_vlm Japanese segmentation responses."""

from __future__ import annotations

import sys

from automation.mlx_vlm_segmentation import (
    MLX_VLM_SEGMENTATION_MODEL,
    MLX_VLM_SEGMENTATION_TIMEOUT,
    MLX_VLM_SEGMENTATION_URL,
    MlxVlmSegmentationError,
    segment_japanese_text_with_mlx_vlm,
)


def main(argv: list[str] | None = None) -> int:
    """Run the probe from the command line."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("使い方:")
        print('  python -m automation.mlx_vlm_segment_probe "肺炎に対して抗菌薬による治療を行う"')
        return 1

    text = args[0]
    print(f"対象文: {text!r}")
    print(f"endpoint: {MLX_VLM_SEGMENTATION_URL}")
    print(f"model: {MLX_VLM_SEGMENTATION_MODEL}")
    print(f"timeout: {MLX_VLM_SEGMENTATION_TIMEOUT:g}秒")

    try:
        raw_content, segments = segment_japanese_text_with_mlx_vlm(text)
    except MlxVlmSegmentationError as exc:
        print(f"エラー: {exc}")
        return 1

    print(f"mlx_vlm応答: {raw_content!r}")
    print("分割結果:")
    for index, segment in enumerate(segments, start=1):
        print(f"  {index}. {segment['text']!r} ({segment['romaji']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
