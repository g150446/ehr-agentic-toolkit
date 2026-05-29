"""surya-ocr 動作確認スクリプト。crop 画像に対して OCR を実行し検出結果を表示する。"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_CAPTURES_DIR = Path(__file__).resolve().parent.parent / "captures"
_CROP_DIR = _CAPTURES_DIR / "crop"


def _latest_crop() -> Path:
    crops = sorted(_CROP_DIR.glob("*past_chart_crop*.png"))
    if not crops:
        raise FileNotFoundError(f"crop 画像が見つかりません: {_CROP_DIR}")
    return crops[-1]


def _html_to_text(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


def run_surya(image_path: Path) -> None:
    from surya.recognition import RecognitionPredictor

    print(f"画像: {image_path}")
    pil_image = Image.open(image_path).convert("RGB")

    rec_predictor = RecognitionPredictor()

    # full_page=True でページ全体を OCR（layout_results 省略で自動）
    predictions = rec_predictor([pil_image])
    page = predictions[0]

    # y 座標 → x 座標の順でソート
    blocks = sorted(page.blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    print(f"\n検出ブロック数: {len(blocks)}\n")
    for block in blocks:
        x1, y1, x2, y2 = [int(v) for v in block.bbox]
        conf = block.confidence if block.confidence is not None else 0.0
        text = _html_to_text(block.html)
        label = block.label
        skipped = " [skipped]" if block.skipped else ""
        print(f"[{x1:4d},{y1:4d},{x2:4d},{y2:4d}] conf={conf:.2f} label={label}{skipped}  {text!r}")

    # デバッグ画像: バウンディングボックスをオーバーレイ
    img = cv2.imread(str(image_path))
    for block in page.blocks:
        x1, y1, x2, y2 = [int(v) for v in block.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        preview = _html_to_text(block.html)[:24]
        cv2.putText(
            img, preview, (x1, max(y1 - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _CAPTURES_DIR / f"surya_debug_{ts}.png"
    cv2.imwrite(str(out), img)
    print(f"\nデバッグ画像保存: {out}")


if __name__ == "__main__":
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_crop()
    run_surya(img)
