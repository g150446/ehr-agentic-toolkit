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
from typing import Optional, Tuple

from automation.config import AutomationConfig, load_config
from automation.screen_analyzer import load_ocr_reader


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
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    print(f"📷 Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")

    print("🔍 Loading OCR model...")
    try:
        ocr_reader = load_ocr_reader(config.ocr_languages, config.ocr_use_gpu)
    except Exception as e:
        print(f"❌ Failed to load OCR: {e}")
        return None

    print("🔍 Running OCR on full image...")
    ocr_results = ocr_reader.readtext(image)
    print(f"📝 Extracted {len(ocr_results)} text segments")

    search_lower = search_text.lower()
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

    print(f"\n📍 Text \"{text}\" found at coordinates: (x={center_x}, y={center_y})")
    print(f"   Confidence: {conf:.2f}")
    print(f"   Bounding box: [[{int(bbox[0][0])},{int(bbox[0][1])}], [{int(bbox[2][0])},{int(bbox[2][1])}]]")

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
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None

    print(f"📷 Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    print(f"🔍 Finding textbox right of \"{label_text}\"...")

    print("🔍 Loading OCR model...")
    try:
        ocr_reader = load_ocr_reader(config.ocr_languages, config.ocr_use_gpu)
    except Exception as e:
        print(f"❌ Failed to load OCR: {e}")
        return None

    print("🔍 Running OCR on full image...")
    ocr_results = ocr_reader.readtext(image)
    print(f"📝 Extracted {len(ocr_results)} text segments")

    # Find the label (prefer exact match over partial)
    label_lower = label_text.lower()
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

    print(f"📍 Label \"{label_text_found}\" found at ({label_cx}, {label_cy})")
    print(f"   Label bbox: ({label_x1}, {label_y1}) -> ({label_x2}, {label_y2})")

    # Find textbox to the right
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

        print(f"\n📍 Textbox right of \"{label_text_found}\" found at: (x={cx}, y={cy})")
        print(f"   Text in textbox: \"{text}\" (conf={conf:.2f})")
        print(f"   Distance from label: {distance}px")
        print(f"   Bounding box: ({int(min(p[0] for p in bbox))}, {int(min(p[1] for p in bbox))}) -> ({int(max(p[0] for p in bbox))}, {int(max(p[1] for p in bbox))})")

        return (cx, cy)
    else:
        # Detect textbox visually using edge detection
        crop_y1 = max(0, label_y1 - 30)
        crop_y2 = min(image.shape[0], label_y2 + 50)
        crop_x1 = label_x2 + 5
        crop_x2 = min(image.shape[1], crop_x1 + max_distance + 100)
        
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
                    
                    if w > 50 and h > 15 and w > h:
                        aspect = w / h
                        if 2 < aspect < 12:
                            rect_score = area * (1.0 / (1.0 + abs(aspect - 5)))
                            if rect_score > best_score:
                                best_score = rect_score
                                best_box = (x + crop_x1, y + crop_y1, w, h)
            
            if best_box:
                bx, by, bw, bh = best_box
                textbox_cx = bx + bw // 2
                textbox_cy = by + bh // 2
                
                print(f"\n📍 Textbox right of \"{label_text_found}\" detected visually at: (x={textbox_cx}, y={textbox_cy})")
                print(f"   Textbox area: ({bx}, {by}) -> ({bx + bw}, {by + bh})")
                print(f"   Size: {bw}x{bh}px")
                print(f"   Distance from label: {bx - label_x2}px")
                
                return (textbox_cx, textbox_cy)
        
        # Fallback: estimate textbox position
        textbox_width = 200
        textbox_start = label_x2 + 20
        textbox_center_x = textbox_start + textbox_width // 2
        textbox_center_y = label_cy

        print(f"\n📍 No textbox detected right of \"{label_text_found}\" (within {max_distance}px)")
        print(f"   Estimated textbox center: (x={textbox_center_x}, y={textbox_center_y})")
        print(f"   Estimated textbox area: ({textbox_start}, {label_y1}) -> ({textbox_start + textbox_width}, {label_y2})")
        print(f"   (Textbox may be empty or visually indistinguishable)")

        return (textbox_center_x, textbox_center_y)


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
