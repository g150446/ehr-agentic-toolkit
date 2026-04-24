"""
GUI Image Analyzer

Analyzes screenshots to find text coordinates and textbox positions.
Uses EasyOCR for text extraction and edge detection for empty textbox finding.

Usage:
    # Find text coordinates
    python -m automation.gui_image_analyzer screenshot.png "患者検索"

    # Find textbox right to a label
    python -m automation.gui_image_analyzer screenshot.png --find-textbox "フリガナ"
"""

import argparse
import cv2
import sys
import time
from typing import Optional, Tuple

from automation.config import AutomationConfig, load_config
from automation.screen_analyzer import load_ocr_reader, run_ocr, run_ocr_word_split


def _load_ocr(config: AutomationConfig):
    """Load the OCR reader selected by config.ocr_backend."""
    return load_ocr_reader(config.ocr_languages, config.ocr_use_gpu)


def _run_yolo_ocr(image, config: AutomationConfig, ocr_reader) -> list:
    """
    Detect individual UI elements with YOLO, then OCR each element separately.

    This avoids OCR merging adjacent menu items / tab labels into a single text segment.
    Returns results in the same (bbox, text, confidence) format as run_ocr().
    Coordinates in bbox are absolute (relative to the original image).

    Falls back to full-image OCR if YOLO detects no elements.
    """
    from automation.model_manager import ModelManager, ModelType

    mgr = ModelManager(config)
    mgr.switch_model(ModelType.UI_DETECTION)

    print("🔲 Running YOLO UI detection...")
    yolo_start = time.time()
    try:
        elements = mgr.detect(image, confidence=0.25)
    except Exception as e:
        print(f"⚠️  YOLO UI detection failed ({e}), falling back to full-image OCR")
        return run_ocr(ocr_reader, image)

    print(f"🔲 Detected {len(elements)} UI elements [{time.time() - yolo_start:.2f}s]")

    if not elements:
        print("⚠️  No UI elements detected, falling back to word-split OCR")
        return run_ocr_word_split(ocr_reader, image)

    results = []
    for elem in elements:
        x1, y1, x2, y2 = elem.bbox
        # Guard against out-of-bounds coords
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        crop_results = run_ocr(ocr_reader, crop)

        for bbox, text, conf in crop_results:
            # Translate bbox coordinates from crop-local to image-absolute
            abs_bbox = [[p[0] + x1, p[1] + y1] for p in bbox]
            results.append((abs_bbox, text, conf))

    if not results:
        print("⚠️  YOLO elements yielded no OCR text, falling back to word-split OCR")
        return run_ocr_word_split(ocr_reader, image)

    return results


def analyze_image_for_text(image_path: str, search_text: str, config: AutomationConfig) -> Optional[Tuple[int, int]]:
    """
    Analyze an image to find the coordinates of specific text.

    Args:
        image_path: Path to the image file
        search_text: Text to search for in the image
        config: AutomationConfig object

    Returns:
        Tuple of (x, y) coordinates of text center, or None if not found
    """
    start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    load_time = time.time() - start_time
    print(f"📷 Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]}) [{load_time:.2f}s]")

    print(f"🔍 Loading OCR model ({config.ocr_backend})...")
    ocr_start = time.time()
    try:
        ocr_reader = _load_ocr(config)
    except Exception as e:
        print(f"❌ Failed to load OCR: {e}")
        return None

    ocr_load_time = time.time() - ocr_start
    print(f"✅ OCR model loaded [{ocr_load_time:.2f}s]")

    print(f"🔍 Running {'YOLO + per-element OCR' if config.detection_mode == 'yolo' else 'full-image OCR'}...")
    ocr_run_start = time.time()
    if config.detection_mode == 'yolo':
        ocr_results = _run_yolo_ocr(image, config, ocr_reader)
    else:
        ocr_results = run_ocr(ocr_reader, image)
    ocr_run_time = time.time() - ocr_run_start
    print(f"📝 Extracted {len(ocr_results)} text segments [{ocr_run_time:.2f}s]")

    search_lower = search_text.lower()
    match_start = time.time()
    matches = []

    for bbox, text, conf in ocr_results:
        if search_lower in text.lower():
            matches.append((bbox, text, conf))

    if not matches:
        print(f"❌ Text \"{search_text}\" not found in image")
        print("\n📝 Available text:")
        for i, (bbox, text, conf) in enumerate(ocr_results[:30], 1):
            print(f"  {i}. \"{text}\" (conf={conf:.2f})")
        if len(ocr_results) > 30:
            print(f"  ... and {len(ocr_results) - 30} more")
        return None

    best_match = max(matches, key=lambda x: x[2])
    bbox, text, conf = best_match

    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    center_x = int((min(xs) + max(xs)) / 2)
    center_y = int((min(ys) + max(ys)) / 2)

    match_time = time.time() - match_start
    print(f"\n📍 Text \"{text}\" found at coordinates: (x={center_x}, y={center_y}) [{match_time:.2f}s]")
    print(f"   Confidence: {conf:.2f}")
    print(f"   Bounding box: [[{int(bbox[0][0])},{int(bbox[0][1])}], [{int(bbox[2][0])},{int(bbox[2][1])}]]")

    total_time = time.time() - start_time
    print(f"\n⏱️  Total processing time: {total_time:.2f}s")

    return (center_x, center_y)


def find_textbox_right_of_label(image_path: str, label_text: str, config: AutomationConfig,
                                 y_tolerance: int = 30, max_distance: int = 300) -> Optional[Tuple[int, int]]:
    """
    Find the textbox to the right of a label text.

    Args:
        image_path: Path to the image file
        label_text: Label text to find (e.g., "フリガナ")
        config: AutomationConfig object
        y_tolerance: Maximum vertical distance for alignment (default: 30px)
        max_distance: Maximum horizontal distance to search (default: 300px)

    Returns:
        Tuple of (x, y) coordinates of textbox center, or None if not found
    """
    start_time = time.time()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    load_time = time.time() - start_time
    print(f"📷 Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]}) [{load_time:.2f}s]")
    print(f"🔍 Finding textbox right of \"{label_text}\"...")

    print(f"🔍 Loading OCR model ({config.ocr_backend})...")
    ocr_start = time.time()
    try:
        ocr_reader = _load_ocr(config)
    except Exception as e:
        print(f"❌ Failed to load OCR: {e}")
        return None

    ocr_load_time = time.time() - ocr_start
    print(f"✅ OCR model loaded [{ocr_load_time:.2f}s]")

    print(f"🔍 Running {'YOLO + per-element OCR' if config.detection_mode == 'yolo' else 'full-image OCR'}...")
    ocr_run_start = time.time()
    if config.detection_mode == 'yolo':
        ocr_results = _run_yolo_ocr(image, config, ocr_reader)
    else:
        ocr_results = run_ocr(ocr_reader, image)
    ocr_run_time = time.time() - ocr_run_start
    print(f"📝 Extracted {len(ocr_results)} text segments [{ocr_run_time:.2f}s]")

    # Find the label (prefer exact match over partial)
    label_lower = label_text.lower()
    match_start = time.time()
    exact_match = None
    partial_match = None

    for bbox, text, conf in ocr_results:
        text_lower = text.lower()
        if text_lower == label_lower:
            exact_match = (bbox, text, conf)
            break
        elif label_lower in text_lower and partial_match is None:
            partial_match = (bbox, text, conf)

    label_match = exact_match or partial_match
    match_time = time.time() - match_start

    if label_match is None:
        print(f"❌ Label \"{label_text}\" not found in image")
        print("\n📝 Available text:")
        for i, (bbox, text, conf) in enumerate(ocr_results[:30], 1):
            print(f"  {i}. \"{text}\" (conf={conf:.2f})")
        if len(ocr_results) > 30:
            print(f"  ... and {len(ocr_results) - 30} more")
        return None

    label_bbox, label_text_found, label_conf = label_match

    label_x1 = int(min(p[0] for p in label_bbox))
    label_y1 = int(min(p[1] for p in label_bbox))
    label_x2 = int(max(p[0] for p in label_bbox))
    label_y2 = int(max(p[1] for p in label_bbox))
    label_cx = (label_x1 + label_x2) // 2
    label_cy = (label_y1 + label_y2) // 2

    print(f"📍 Label \"{label_text_found}\" found at ({label_cx}, {label_cy}) [{match_time:.2f}s]")
    print(f"   Label bbox: ({label_x1}, {label_y1}) -> ({label_x2}, {label_y2})")

    # Find textbox to the right
    textbox_start = time.time()
    candidates = []

    for bbox, text, conf in ocr_results:
        x1 = int(min(p[0] for p in bbox))
        y1 = int(min(p[1] for p in bbox))
        x2 = int(max(p[0] for p in bbox))
        y2 = int(max(p[1] for p in bbox))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if x1 <= label_x2:
            continue

        if abs(cy - label_cy) > y_tolerance:
            continue

        if text == label_text_found:
            continue

        distance = x1 - label_x2

        if distance > max_distance:
            continue

        candidates.append((distance, bbox, text, conf, cx, cy))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best = candidates[0]
        distance, bbox, text, conf, cx, cy = best

        textbox_time = time.time() - textbox_start
        print(f"\n📍 Textbox right of \"{label_text_found}\" found at: (x={cx}, y={cy}) [{textbox_time:.2f}s]")
        print(f"   Text in textbox: \"{text}\" (conf={conf:.2f})")
        print(f"   Distance from label: {distance}px")
        print(f"   Bounding box: ({int(min(p[0] for p in bbox))}, {int(min(p[1] for p in bbox))}) -> ({int(max(p[0] for p in bbox))}, {int(max(p[1] for p in bbox))})")

        total_time = time.time() - start_time
        print(f"\n⏱️  Total processing time: {total_time:.2f}s")

        return (cx, cy)
    else:
        # Detect textbox visually using edge detection
        edge_start = time.time()
        crop_y1 = max(0, label_y1 - 30)
        crop_y2 = min(image.shape[0], label_y2 + 50)
        # Start crop at label_x1 (not label_x2) so input fields starting just after
        # the label are fully within the crop and produce complete quadrilateral contours.
        crop_x1 = max(0, label_x1)
        crop_x2 = min(image.shape[1], label_x2 + max_distance + 100)

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_box = None
            best_score = 0

            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    img_x = x + crop_x1  # convert to image coordinates

                    # Only consider boxes whose left edge is to the right of the label
                    if img_x <= label_x2:
                        continue

                    if w > 50 and h > 15 and w > h:
                        aspect = w / h
                        if 2 < aspect < 30:
                            rect_score = area  # 面積優先（幅広の入力フィールドを選ぶ）
                            if rect_score > best_score:
                                best_score = rect_score
                                best_box = (img_x, y + crop_y1, w, h)

            if best_box:
                bx, by, bw, bh = best_box
                textbox_cx = bx + bw // 2
                textbox_cy = by + bh // 2

                edge_time = time.time() - edge_start
                total_time = time.time() - start_time
                print(f"\n📍 Textbox right of \"{label_text_found}\" detected visually at: (x={textbox_cx}, y={textbox_cy}) [{edge_time:.2f}s]")
                print(f"   Textbox area: ({bx}, {by}) -> ({bx + bw}, {by + bh})")
                print(f"   Size: {bw}x{bh}px")
                print(f"   Distance from label: {bx - label_x2}px")
                print(f"\n⏱️  Total processing time: {total_time:.2f}s")

                return (textbox_cx, textbox_cy)

        # Fallback: estimate textbox position
        textbox_width = 200
        textbox_start = label_x2 + 20
        textbox_center_x = textbox_start + textbox_width // 2
        textbox_center_y = label_cy

        total_time = time.time() - start_time
        print(f"\n📍 No textbox detected right of \"{label_text_found}\" (within {max_distance}px)")
        print(f"   Estimated textbox center: (x={textbox_center_x}, y={textbox_center_y})")
        print(f"   Estimated textbox area: ({textbox_start}, {label_y1}) -> ({textbox_start + textbox_width}, {label_y2})")
        print(f"   (Textbox may be empty or visually indistinguishable)")
        print(f"\n⏱️  Total processing time: {total_time:.2f}s")

        return (textbox_center_x, textbox_center_y)


def find_first_patient_row(image_path: str, config: AutomationConfig) -> Optional[Tuple[int, int]]:
    """
    患者一覧画面から先頭の患者行の中心座標を返す。

    手順:
    1. 全画面 OCR を実行
    2. OCR 結果を Y 座標でクラスタリングして「行」に分割
    3. 患者一覧のヘッダー行（患者番号・氏名・生年月日 などのキーワードを含む行）を特定
    4. ヘッダー行より下にある最初のデータ行の中心を返す

    ヘッダーが見つからない場合は、画面上部から2番目の行（最初の行はタブ/メニューバーが多い）を
    先頭患者行とみなしてフォールバックする。

    Args:
        image_path: スクリーンショット画像のパス
        config: AutomationConfig インスタンス

    Returns:
        先頭患者行の (x, y) 中心座標。検出失敗時は None
    """
    # ヘッダー行を識別するキーワード
    HEADER_KEYWORDS = ['患者番号', '患者id', '氏名', 'フリガナ', '生年月日', '性別', '年齢', '診察']

    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    img_h, img_w = image.shape[:2]
    print(f"📷 Loaded image: {image_path} ({img_w}x{img_h})")

    print(f"🔍 Loading OCR model ({config.ocr_backend})...")
    try:
        ocr_reader = _load_ocr(config)
    except Exception as e:
        print(f"❌ Failed to load OCR: {e}")
        return None

    print("🔍 Running full-image OCR...")
    ocr_results = run_ocr(ocr_reader, image)
    print(f"📝 Extracted {len(ocr_results)} text segments [{time.time() - start_time:.2f}s]")

    if not ocr_results:
        print("❌ OCR returned no results")
        return None

    # --- Y クラスタリング: 近い Y 座標のセグメントを同一行にまとめる ---
    ROW_TOLERANCE = 12  # px

    def _center(bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2))

    # (cy, cx, text) のリストを Y でソート
    segments = []
    for bbox, text, conf in ocr_results:
        cx, cy = _center(bbox)
        segments.append((cy, cx, text))
    segments.sort(key=lambda s: s[0])

    # 近い Y のものを同じ行グループにまとめる
    rows = []  # list of (mean_y, min_x, max_x, [text, ...])
    for cy, cx, text in segments:
        if rows and abs(cy - rows[-1][0]) <= ROW_TOLERANCE:
            prev_y, prev_x1, prev_x2, texts = rows[-1]
            new_y = (prev_y + cy) // 2
            rows[-1] = (new_y, min(prev_x1, cx), max(prev_x2, cx), texts + [text])
        else:
            rows.append((cy, cx, cx, [text]))

    print(f"📊 Clustered into {len(rows)} rows")
    for i, (ry, rx1, rx2, texts) in enumerate(rows):
        print(f"  行 {i+1} (y={ry}): {' | '.join(texts)}")

    # --- ヘッダー行を探す ---
    header_row_idx = None
    for i, (ry, rx1, rx2, texts) in enumerate(rows):
        combined = ' '.join(texts).lower()
        hits = sum(1 for kw in HEADER_KEYWORDS if kw in combined)
        if hits >= 2:  # 2つ以上のヘッダーキーワードが一致したらヘッダー行とみなす
            header_row_idx = i
            print(f"📋 ヘッダー行を検出: 行 {i + 1} (y={ry}) — キーワード {hits}個一致")
            break

    if header_row_idx is None:
        print("⚠️  ヘッダー行が見つかりません。2行目をデータ行とみなします（フォールバック）")
        # メニューバー等を避けるため Y が全体の上 1/3 より下の最初の行を選ぶ
        threshold_y = img_h // 3
        data_rows = [r for r in rows if r[0] > threshold_y]
        if not data_rows:
            data_rows = rows[1:] if len(rows) > 1 else rows
        if not data_rows:
            print("❌ データ行が見つかりませんでした")
            return None
        first_data = data_rows[0]
    else:
        # ヘッダー行の次の行をデータ行とする
        next_idx = header_row_idx + 1
        if next_idx >= len(rows):
            print("❌ ヘッダーの次の行が存在しません（患者データなし）")
            return None
        first_data = rows[next_idx]

    ry, rx1, rx2, texts = first_data
    cx = (rx1 + rx2) // 2
    cy = ry

    print(f"\n📍 先頭患者行: (x={cx}, y={cy})")
    print(f"   テキスト: {' | '.join(texts)}")
    print(f"\n⏱️  Total: {time.time() - start_time:.2f}s")

    return (cx, cy)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GUI Image Analyzer - Find text coordinates and textbox positions in screenshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find text coordinates
  python -m automation.gui_image_analyzer screenshot.png "患者検索"
  
  # Find textbox right to a label
  python -m automation.gui_image_analyzer screenshot.png --find-textbox "フリガナ"
  python -m automation.gui_image_analyzer form.png --find-textbox "氏名"
        """
    )

    parser.add_argument(
        "image_path",
        help="Path to image file for analysis"
    )

    parser.add_argument(
        "search_text",
        nargs="?",
        help="Text to search for in the image"
    )

    parser.add_argument(
        "--find-textbox",
        type=str,
        metavar="LABEL",
        help="Find textbox to the right of label text (e.g., --find-textbox \"フリガナ\")"
    )

    parser.add_argument(
        "--detection-mode",
        type=str,
        choices=["yolo", "ocr"],
        default=None,
        help="Detection mode: 'yolo' (UI element detection + per-element OCR, default) or 'ocr' (full-image OCR only)"
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.env_file, skip_password=True)
    if args.detection_mode is not None:
        config.detection_mode = args.detection_mode

    print(f"\n🔍 GUI Image Analysis Mode")
    print(f"   Image: {args.image_path}\n")

    # Find textbox mode
    if args.find_textbox:
        print(f"🔍 Finding textbox right of \"{args.find_textbox}\"")
        result = find_textbox_right_of_label(args.image_path, args.find_textbox, config)
    # Search text mode
    elif args.search_text:
        print(f"   Search: \"{args.search_text}\"")
        result = analyze_image_for_text(args.image_path, args.search_text, config)
    else:
        print("❌ Error: Either provide search_text or --find-textbox")
        print("   Usage: gui_image_analyzer.py <image> <search_text>")
        print("          gui_image_analyzer.py <image> --find-textbox <label>")
        sys.exit(1)

    if result:
        x, y = result
        print(f"\n✅ Success! Coordinates: ({x}, {y})")
        sys.exit(0)
    else:
        print("\n❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
