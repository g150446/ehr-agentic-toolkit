"""
Analyze the past-history area using OCR anchors and layout-aware OCR strategies.

This tool is for offline analysis on saved screenshots. It compares:
- PaddleOCR + full-image OCR
- PaddleOCR + UI detection OCR
- PP-StructureV3 + Paddle OCR models
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2

from automation.config import load_config
from automation.mlx_vlm_history import _date_matches_text
from automation.model_manager import ModelManager, ModelType
from automation.screen_analyzer import (
    load_paddleocr_reader,
    load_ppstructure_reader,
    run_ocr,
    run_ppstructure_ocr,
)
from automation.utils import draw_bounding_boxes, format_timestamp


DATE_LIKE_YEAR_RE = re.compile(r"(19|20)\d{2}")


@dataclass
class OcrStrategyResult:
    name: str
    ocr_backend: str
    layout_mode: str
    ocr_results: list[tuple]
    date_candidates: list[dict[str, Any]]
    roi: Optional[dict[str, int]]
    error: Optional[str] = None


def _bbox_rect(bbox: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [int(point[0]) for point in bbox]
    ys = [int(point[1]) for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_center(bbox: list[list[int]]) -> tuple[int, int]:
    x1, y1, x2, y2 = _bbox_rect(bbox)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _load_ocr_backend(backend: str, config):
    if backend == "paddleocr":
        return load_paddleocr_reader(config.ocr_languages)
    if backend == "ppstructure":
        return load_ppstructure_reader(config.ocr_languages)
    raise ValueError(f"Unsupported OCR backend: {backend}")


def _run_region_ocr(image, config, ocr_reader, model_type: ModelType, confidence: float) -> list[tuple]:
    mgr = ModelManager(config)
    mgr.switch_model(model_type)
    detections = mgr.detect(image, confidence=confidence)

    results = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.shape[1], int(x2))
        y2 = min(image.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        for bbox, text, conf in run_ocr(ocr_reader, crop):
            abs_bbox = [[int(p[0]) + x1, int(p[1]) + y1] for p in bbox]
            results.append((abs_bbox, text, float(conf)))
    return results


def _run_ppstructure_blocks(image, config) -> list[tuple]:
    reader = load_ppstructure_reader(config.ocr_languages)
    return run_ppstructure_ocr(reader, image)


def _is_date_like_text(text: str) -> bool:
    return (
        bool(DATE_LIKE_YEAR_RE.search(text))
        and "年" in text
        and "月" in text
        and "日" in text
    )


def _extract_date_candidates(
    ocr_results: list[tuple],
    *,
    image_width: int,
    target_date: str | None = None,
    min_x: int = 200,
    max_x_margin: int = 420,
) -> list[dict[str, Any]]:
    if image_width < 900:
        min_x = 0
        max_x = image_width
    else:
        max_x = max(min_x + 1, image_width - max_x_margin)
    candidates: list[dict[str, Any]] = []

    for idx, (bbox, text, conf) in enumerate(ocr_results):
        cx, cy = _bbox_center(bbox)
        x1, y1, x2, y2 = _bbox_rect(bbox)
        if cx < min_x or cx > max_x:
            continue
        if not _is_date_like_text(text):
            continue

        exact_match = False
        if target_date:
            year = int(target_date[:4])
            month = int(target_date[4:6])
            day = int(target_date[6:8])
            exact_match = _date_matches_text(text, year, month, day)

        candidates.append(
            {
                "index": idx,
                "text": text,
                "confidence": float(conf),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
                "cx": int(cx),
                "cy": int(cy),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "exact_match": exact_match,
            }
        )

    return candidates


def infer_history_panel_roi(
    candidates: list[dict[str, Any]],
    *,
    image_shape: tuple[int, int, int] | tuple[int, int],
    x_padding: int = 24,
    y_padding: int = 20,
    cluster_gap: int | None = None,
) -> Optional[dict[str, int]]:
    if not candidates:
        return None

    widths = [candidate["width"] for candidate in candidates if candidate["width"] > 0]
    median_width = int(statistics.median(widths)) if widths else 80
    gap = cluster_gap if cluster_gap is not None else max(48, min(140, int(median_width * 1.2)))

    sorted_candidates = sorted(candidates, key=lambda candidate: candidate["cx"])
    clusters: list[list[dict[str, Any]]] = []
    current_cluster: list[dict[str, Any]] = []

    for candidate in sorted_candidates:
        if not current_cluster:
            current_cluster = [candidate]
            continue

        current_mean_x = sum(item["cx"] for item in current_cluster) / len(current_cluster)
        if abs(candidate["cx"] - current_mean_x) <= gap:
            current_cluster.append(candidate)
        else:
            clusters.append(current_cluster)
            current_cluster = [candidate]

    if current_cluster:
        clusters.append(current_cluster)

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])

    def cluster_score(cluster: list[dict[str, Any]]):
        y_values = [item["cy"] for item in cluster]
        x_values = [item["cx"] for item in cluster]
        exact_matches = sum(1 for item in cluster if item["exact_match"])
        return (
            exact_matches,
            len(cluster),
            max(y_values) - min(y_values),
            -abs((sum(x_values) / len(x_values)) - image_width / 2),
        )

    best_cluster = max(clusters, key=cluster_score)
    x1 = max(0, min(item["bbox"]["x1"] for item in best_cluster) - x_padding)
    y1 = max(0, min(item["bbox"]["y1"] for item in best_cluster) - y_padding)
    x2 = min(image_width, max(item["bbox"]["x2"] for item in best_cluster) + x_padding)
    y2 = min(image_height, max(item["bbox"]["y2"] for item in best_cluster) + y_padding)

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "candidate_count": len(best_cluster),
        "cluster_count": len(clusters),
    }


def _annotate_strategy_image(image, strategy: OcrStrategyResult):
    boxes = [
        [
            candidate["bbox"]["x1"],
            candidate["bbox"]["y1"],
            candidate["bbox"]["x2"],
            candidate["bbox"]["y2"],
        ]
        for candidate in strategy.date_candidates
    ]
    labels = [
        f"{idx + 1}{'*' if candidate['exact_match'] else ''}:{candidate['text'][:16]}"
        for idx, candidate in enumerate(strategy.date_candidates)
    ]
    scores = [candidate["confidence"] for candidate in strategy.date_candidates]
    annotated = draw_bounding_boxes(image, boxes, labels=labels, scores=scores, color=(0, 255, 0))

    if strategy.roi:
        cv2.rectangle(
            annotated,
            (strategy.roi["x1"], strategy.roi["y1"]),
            (strategy.roi["x2"], strategy.roi["y2"]),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"history-roi ({strategy.layout_mode}, {strategy.ocr_backend})",
            (strategy.roi["x1"], max(20, strategy.roi["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _save_strategy_summary(run_dir: Path, strategy: OcrStrategyResult):
    summary_path = run_dir / f"{strategy.name}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Strategy: {strategy.name}\n")
        f.write(f"OCR backend: {strategy.ocr_backend}\n")
        f.write(f"Layout mode: {strategy.layout_mode}\n")
        f.write(f"OCR segments: {len(strategy.ocr_results)}\n")
        f.write(f"Date-like candidates: {len(strategy.date_candidates)}\n")
        if strategy.roi:
            f.write(
                "ROI: "
                f"({strategy.roi['x1']},{strategy.roi['y1']})-"
                f"({strategy.roi['x2']},{strategy.roi['y2']}) "
                f"cluster_candidates={strategy.roi['candidate_count']} "
                f"cluster_count={strategy.roi['cluster_count']}\n"
            )
        else:
            f.write("ROI: none\n")
        if strategy.error:
            f.write(f"Error: {strategy.error}\n")
        f.write("\n")
        for candidate in strategy.date_candidates:
            marker = "*" if candidate["exact_match"] else "-"
            f.write(
                f"{marker} ({candidate['cx']},{candidate['cy']}) "
                f"{candidate['text']!r} conf={candidate['confidence']:.3f}\n"
            )
    return summary_path


def _run_strategy(
    *,
    image,
    config,
    ocr_backend: str,
    layout_mode: str,
    target_date: str | None = None,
) -> OcrStrategyResult:
    strategy_name = f"{layout_mode}_{ocr_backend}".replace("-", "_")

    try:
        ocr_reader = _load_ocr_backend(ocr_backend, config)
        if layout_mode == "full-image":
            ocr_results = run_ocr(ocr_reader, image)
        elif layout_mode == "ui-detection":
            ocr_results = _run_region_ocr(
                image,
                config,
                ocr_reader,
                ModelType.UI_DETECTION,
                confidence=0.25,
            )
        elif layout_mode == "pp-structure":
            ocr_results = _run_ppstructure_blocks(image, config)
        else:
            raise ValueError(f"Unsupported layout mode: {layout_mode}")
    except Exception as exc:
        return OcrStrategyResult(
            name=strategy_name,
            ocr_backend=ocr_backend,
            layout_mode=layout_mode,
            ocr_results=[],
            date_candidates=[],
            roi=None,
            error=str(exc),
        )

    date_candidates = _extract_date_candidates(
        ocr_results,
        image_width=image.shape[1],
        target_date=target_date,
    )
    roi = infer_history_panel_roi(date_candidates, image_shape=image.shape)
    return OcrStrategyResult(
        name=strategy_name,
        ocr_backend=ocr_backend,
        layout_mode=layout_mode,
        ocr_results=ocr_results,
        date_candidates=date_candidates,
        roi=roi,
    )


def _choose_best_strategy(strategies: list[OcrStrategyResult]) -> Optional[OcrStrategyResult]:
    valid = [strategy for strategy in strategies if strategy.roi]
    if not valid:
        return None

    def strategy_score(strategy: OcrStrategyResult):
        exact_matches = sum(1 for candidate in strategy.date_candidates if candidate["exact_match"])
        return (
            exact_matches,
            len(strategy.date_candidates),
            len(strategy.ocr_results),
        )

    return max(valid, key=strategy_score)


def analyze_history_panel(
    image_path: str,
    *,
    target_date: str | None = None,
    output_dir: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    config = load_config(skip_password=True)
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"画像を読み込めませんでした: {image_path}")

    base_dir = Path(output_dir) if output_dir else config.output_dir / "history_panel_analysis"
    actual_run_name = run_name or f"{Path(image_path).stem}_{format_timestamp()}"
    run_dir = base_dir / actual_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    source_path = run_dir / "source.png"
    cv2.imwrite(str(source_path), image)

    layout_strategies = [
        _run_strategy(image=image, config=config, ocr_backend="paddleocr", layout_mode="full-image", target_date=target_date),
        _run_strategy(image=image, config=config, ocr_backend="paddleocr", layout_mode="ui-detection", target_date=target_date),
        _run_strategy(image=image, config=config, ocr_backend="ppstructure", layout_mode="pp-structure", target_date=target_date),
    ]

    saved_strategies = []
    for strategy in layout_strategies:
        annotated = _annotate_strategy_image(image, strategy)
        annotated_path = run_dir / f"{strategy.name}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated)
        summary_path = _save_strategy_summary(run_dir, strategy)
        saved_strategies.append(
            {
                "name": strategy.name,
                "ocr_backend": strategy.ocr_backend,
                "layout_mode": strategy.layout_mode,
                "ocr_segments": len(strategy.ocr_results),
                "date_candidates": len(strategy.date_candidates),
                "exact_matches": sum(1 for candidate in strategy.date_candidates if candidate["exact_match"]),
                "roi": strategy.roi,
                "error": strategy.error,
                "annotated_path": str(annotated_path),
                "summary_path": str(summary_path),
            }
        )

    best_strategy = _choose_best_strategy(layout_strategies)
    roi_crop_path = None
    if best_strategy and best_strategy.roi:
        roi = best_strategy.roi
        roi_crop = image[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]]
        roi_crop_path = run_dir / "history_roi.png"
        cv2.imwrite(str(roi_crop_path), roi_crop)

    recommendations = []
    if best_strategy is None:
        recommendations.append("日付候補クラスタから履歴ROIを推定できませんでした。OCRアンカー条件の緩和が必要です。")
    else:
        recommendations.append(
            f"最有力レイアウト戦略は {best_strategy.layout_mode} + {best_strategy.ocr_backend} です。"
        )
        ppstructure_result = next(
            (item for item in saved_strategies if item["layout_mode"] == "pp-structure"),
            None,
        )
        ui_result = next(
            (item for item in saved_strategies if item["layout_mode"] == "ui-detection"),
            None,
        )
        if ppstructure_result and ppstructure_result["date_candidates"] > 0:
            recommendations.append("PP-Structure は補助的なブロック分割として確認価値があります。")
        full_result = next(
            (item for item in saved_strategies if item["layout_mode"] == "full-image"),
            None,
        )
        if (
            full_result
            and all(
                item["date_candidates"] < full_result["date_candidates"]
                for item in saved_strategies
                if item["layout_mode"] != "full-image" and item["error"] is None
            )
        ):
            recommendations.append("この画像では detector-first より全画面 OCR の方が履歴日付候補を多く拾えています。")
        recommendations.append("必要なら次段階で ROI 内に対する PaddleOCR 前処理強化を検討してください。")

    manifest = {
        "source_path": str(source_path),
        "image_path": image_path,
        "target_date": target_date,
        "strategies": saved_strategies,
        "best_strategy": best_strategy.name if best_strategy else None,
        "history_roi_path": str(roi_crop_path) if roi_crop_path else None,
        "roi_engine_results": [],
        "recommendations": recommendations,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = run_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        if target_date:
            f.write(f"Target date: {target_date}\n")
        f.write("\n=== Layout strategy comparison ===\n")
        for item in saved_strategies:
            f.write(
                f"- {item['name']}: segments={item['ocr_segments']} "
                f"date_candidates={item['date_candidates']} "
                f"exact_matches={item['exact_matches']} "
                f"roi={'yes' if item['roi'] else 'no'}"
            )
            if item["error"]:
                f.write(f" error={item['error']}")
            f.write("\n")
        if recommendations:
            f.write("\n=== Recommendations ===\n")
            for recommendation in recommendations:
                f.write(f"- {recommendation}\n")

    manifest["manifest_path"] = str(manifest_path)
    manifest["summary_path"] = str(summary_path)
    manifest["run_dir"] = str(run_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the past-history panel using OCR anchors and layout comparisons."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--date", help="Optional target date in yyyymmdd format")
    parser.add_argument("--run-name", help="Optional output run directory name")
    parser.add_argument("--output-dir", help="Optional base output directory")
    args = parser.parse_args(argv)

    manifest = analyze_history_panel(
        args.image,
        target_date=args.date,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )

    print(f"保存先: {manifest['run_dir']}")
    print(f"サマリ: {manifest['summary_path']}")
    print(f"JSON: {manifest['manifest_path']}")
    if manifest["best_strategy"]:
        print(f"最有力戦略: {manifest['best_strategy']}")
    for recommendation in manifest["recommendations"]:
        print(f"- {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
