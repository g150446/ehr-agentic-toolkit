import argparse
import cv2
import glob
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

CAPTURE_DIR = "captures"
MATCH_DIR = Path(__file__).resolve().parent.parent / "match_templates"
TEMPLATE_FILES = ["gray_printer.jpg", "printer_button.png"]

os.makedirs(CAPTURE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")


def clear_captures():
    removed = 0
    for ext in IMAGE_EXTENSIONS:
        for filepath in glob.glob(os.path.join(CAPTURE_DIR, ext)):
            os.remove(filepath)
            removed += 1
    print(f"✓ {removed} 件の画像ファイルを削除しました ({CAPTURE_DIR}/)")
    return removed


def scan_video_devices(max_devices=10):
    print("=" * 60)
    print("ビデオデバイススキャン開始...")
    print("=" * 60)

    available_devices = []

    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()

            ret, frame = cap.read()

            available_devices.append({
                'index': i,
                'width': width,
                'height': height,
                'fps': fps,
                'backend': backend,
                'working': ret and frame is not None
            })

            status = "✓ 動作中" if ret else "⚠ フレーム読み込み失敗"
            print(f"\nデバイス {i}: {status}")
            print(f"  解像度: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  バックエンド: {backend}")

            if ret and frame is not None:
                mean_brightness = frame.mean()
                print(f"  フレーム平均輝度: {mean_brightness:.2f}")
                if mean_brightness < 1.0:
                    print(f"  警告: フレームが真っ黒です")

            cap.release()
        else:
            print(f"\nデバイス {i}: 使用不可")

    print("\n" + "=" * 60)
    print(f"検出されたデバイス数: {len(available_devices)}")
    print("=" * 60)

    return available_devices


def capture_from_device(device_index, num_warmup_frames=5):
    print(f"\nデバイス {device_index} からキャプチャを開始...")

    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(f"✗ エラー: デバイス {device_index} を開けませんでした")
        return None

    print(f"✓ デバイス {device_index} をオープンしました")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()

    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  バックエンド: {backend}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  設定後の解像度: {actual_width}x{actual_height}")

    print(f"\nウォームアップ中（{num_warmup_frames}フレームスキップ）...")
    for i in range(num_warmup_frames):
        ret, frame = cap.read()
        if ret:
            mean_brightness = frame.mean() if frame is not None else 0
            print(f"  フレーム {i+1}/{num_warmup_frames}: 輝度={mean_brightness:.2f}")
        else:
            print(f"  フレーム {i+1}/{num_warmup_frames}: 読み込み失敗")
        time.sleep(0.1)

    print("\n実際のキャプチャフレーム取得中...")
    ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        print("✗ エラー: フレームの読み込みに失敗しました")
        return None

    height, width = frame.shape[:2]
    mean_brightness = frame.mean()

    print(f"✓ フレーム取得成功")
    print(f"  解像度: {width}x{height}")
    print(f"  平均輝度: {mean_brightness:.2f}")

    if mean_brightness < 1.0:
        print(f"⚠ 警告: フレームが真っ黒です（輝度 < 1.0）")
        print(f"  デバイスが正しく接続されているか確認してください")

    return frame


def _find_gray_divider_candidates(
    frame: np.ndarray,
    *,
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
    kernel_h_div: int = 6,
) -> list[int]:
    h, w = frame.shape[:2]
    y1 = int(h * 0.05)
    y2 = int(h * 0.95)
    band = frame[y1:y2, :]
    b = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    r = band[:, :, 2].astype(np.int16)
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    value = ((b + g + r) / 3.0)
    mask = ((spread <= spread_max) & (value >= value_min) & (value <= 235)).astype(np.uint8) * 255
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, (y2 - y1) // kernel_h_div)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))

    col_strength = np.count_nonzero(mask, axis=0)
    threshold = max(int((y2 - y1) * coverage_ratio), 40)
    edge_margin = max(5, w // 100)
    candidates: list[int] = []
    start: Optional[int] = None
    for x, strength in enumerate(col_strength):
        if strength >= threshold and start is None:
            start = x
        elif strength < threshold and start is not None:
            end = x - 1
            cx = (start + end) // 2
            if 1 <= end - start + 1 <= max(18, w // 30) and edge_margin <= cx <= w - edge_margin:
                candidates.append(cx)
            start = None
    if start is not None:
        end = len(col_strength) - 1
        cx = (start + end) // 2
        if 1 <= end - start + 1 <= max(18, w // 30) and edge_margin <= cx <= w - edge_margin:
            candidates.append(cx)
    return candidates


def _measure_divider_thicknesses(
    frame: np.ndarray,
    candidates: list[int],
    *,
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
) -> list[tuple[int, int]]:
    h, w = frame.shape[:2]
    y1 = int(h * 0.05)
    y2 = int(h * 0.95)
    band = frame[y1:y2, :]
    b = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    r = band[:, :, 2].astype(np.int16)
    spread = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
    value = ((b + g + r) / 3.0)
    mask = ((spread <= spread_max) & (value >= value_min) & (value <= 235)).astype(np.uint8) * 255

    result = []
    for x in candidates:
        left = max(0, x - 15)
        right = min(w, x + 15)
        col_strength = np.count_nonzero(mask[:, left:right], axis=0)
        threshold = max(int((y2 - y1) * coverage_ratio), 40)
        active = col_strength >= threshold
        if active.any():
            indices = np.where(active)[0]
            thickness = int(indices[-1] - indices[0] + 1)
        else:
            thickness = 0
        result.append((x, thickness))
    return result


def _select_thick_gray_dividers(
    frame: np.ndarray,
    *,
    num_lines: int = 3,
    min_gap: int = 80,
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
    kernel_h_div: int = 6,
) -> list[int]:
    gray_candidates = _find_gray_divider_candidates(
        frame,
        spread_max=spread_max,
        value_min=value_min,
        coverage_ratio=coverage_ratio,
        kernel_h_div=kernel_h_div,
    )
    if not gray_candidates:
        return []

    thicknesses = _measure_divider_thicknesses(
        frame,
        gray_candidates,
        spread_max=spread_max,
        value_min=value_min,
        coverage_ratio=coverage_ratio,
    )
    thicknesses.sort(key=lambda t: t[1], reverse=True)

    selected = []
    for x, thickness in thicknesses:
        if thickness == 0:
            continue
        if all(abs(x - sx) >= min_gap for sx in selected):
            selected.append(x)
        if len(selected) >= num_lines:
            break

    return sorted(selected)


def _try_detect_dividers(
    frame: np.ndarray,
    *,
    spread_max: int = 14,
    value_min: int = 120,
    coverage_ratio: float = 0.45,
    kernel_h_div: int = 6,
    debug: bool = False,
) -> Optional[list[int]]:
    accepted = _select_thick_gray_dividers(
        frame,
        num_lines=3,
        min_gap=80,
        spread_max=spread_max,
        value_min=value_min,
        coverage_ratio=coverage_ratio,
        kernel_h_div=kernel_h_div,
    )

    if debug:
        h = frame.shape[0]
        overlay = frame.copy()
        gray_candidates = _find_gray_divider_candidates(
            frame,
            spread_max=spread_max,
            value_min=value_min,
            coverage_ratio=coverage_ratio,
            kernel_h_div=kernel_h_div,
        )
        for x in gray_candidates:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 215, 255), 1)
        if accepted:
            for x in accepted:
                cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(CAPTURE_DIR, "debug_dividers.jpg"), overlay)

    if len(accepted) >= 3:
        return accepted
    return None


def _detect_all_dividers(
    frame: np.ndarray,
    *,
    debug: bool = False,
) -> Optional[list[int]]:
    print("  区切り線検出 第1段階（通常パラメータ: spread≤14, value≥120, カバレッジ45%）...")
    accepted = _try_detect_dividers(
        frame, spread_max=14, value_min=120, coverage_ratio=0.45, debug=debug,
    )
    if accepted is not None:
        print(f"    第1段階で成功: {accepted[:3]}")
        return accepted[:3]

    print("  区切り線検出 第2段階（緩和パラメータ: spread≤25, value≥100, カバレッジ30%）...")
    accepted = _try_detect_dividers(
        frame, spread_max=25, value_min=100, coverage_ratio=0.30, debug=debug,
    )
    if accepted is not None:
        print(f"    第2段階で成功: {accepted[:3]}")
        return accepted[:3]

    print("  区切り線検出 第3段階（短カーネル: spread≤14, value≥120, カバレッジ25%, kernel÷20）...")
    accepted = _try_detect_dividers(
        frame, spread_max=14, value_min=120, coverage_ratio=0.25, kernel_h_div=20, debug=debug,
    )
    if accepted is not None:
        print(f"    第3段階で成功: {accepted[:3]}")
        return accepted[:3]

    print("  区切り線検出 第4段階（最大緩和パラメータ: spread≤50, value≥60, カバレッジ15%）...")
    accepted = _try_detect_dividers(
        frame, spread_max=50, value_min=60, coverage_ratio=0.15, debug=debug,
    )
    if accepted is not None:
        print(f"    第4段階で成功: {accepted[:3]}")
        return accepted[:3]

    print("    検出失敗")
    return None


def detect_icon_bottom_y(
    frame: np.ndarray,
    x_start: int,
    x_end: int,
    *,
    threshold: float = 0.7,
) -> int | None:
    roi = frame[:, x_start:x_end]
    if roi.size == 0:
        return None

    for template_name in TEMPLATE_FILES:
        template_path = MATCH_DIR / template_name
        if not template_path.exists():
            print(f"[WARNING] テンプレート画像が見つかりません: {template_path}")
            continue

        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            print(f"[WARNING] テンプレート画像の読み込みに失敗しました: {template_path}")
            continue

        if template.shape[0] > roi.shape[0] or template.shape[1] > roi.shape[1]:
            print(f"[WARNING] テンプレートがパネル領域より大きいためスキップ: {template_name}")
            continue

        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        th, tw = template.shape[:2]

        loc = np.where(result >= threshold)
        points = list(zip(*loc[::-1]))

        if not points:
            _, max_val, _, _ = cv2.minMaxLoc(result)
            print(f"  {template_name} 未検出 (最高スコア {max_val:.3f} < {threshold})")
            continue

        best_x, best_y = max(points, key=lambda p: p[1])
        bottom_y = best_y + th
        score = float(result[best_y, best_x])
        print(
            f"  {template_name} 検出: "
            f"({best_x + x_start}, {best_y})〜({best_x + x_start + tw}, {bottom_y}) "
            f"(score={score:.3f})"
        )
        return bottom_y

    return None


def crop_past_records(
    frame: np.ndarray,
    *,
    threshold: float = 0.7,
    debug: bool = False,
) -> np.ndarray | None:
    print("区切り線を検出中...")
    dividers = _detect_all_dividers(frame, debug=debug)
    if not dividers or len(dividers) < 2:
        print("[ERROR] 患者カルテ画面の区切り線（太いグレーの縦線）を検出できませんでした")
        return None
    print(f"区切り線検出: x={dividers}")

    x_start = dividers[0]
    x_end = dividers[1]

    print("\n過去記録フィールドの上部アンカー（プリンターアイコン）を検出中...")
    y_start = detect_icon_bottom_y(frame, x_start, x_end, threshold=threshold)
    if y_start is None:
        print("[ERROR] プリンターアイコンを検出できませんでした")
        return None
    y_start = max(0, y_start)

    cropped = frame[y_start:, x_start:x_end]
    print(
        f"過去記録フィールド切り出し: x={x_start}-{x_end}, "
        f"y={y_start}-{frame.shape[0]} (size={cropped.shape[1]}x{cropped.shape[0]})"
    )

    if debug:
        overlay = frame.copy()
        h = overlay.shape[0]
        for x in dividers:
            cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
        cv2.rectangle(overlay, (x_start, y_start), (x_end, h - 1), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(CAPTURE_DIR, "debug_past_records_region.jpg"), overlay)

    return cropped


def parse_args():
    parser = argparse.ArgumentParser(description="過去記録フィールド キャプチャツール (past records crop)")
    parser.add_argument("--name", default=None, help="出力ファイル名")
    parser.add_argument("--clear", action="store_true", help="キャプチャ前にcapturesフォルダの画像を全削除")
    parser.add_argument("--device", type=int, default=None, help="ビデオデバイス番号（省略時は自動スキャン）")
    parser.add_argument("--threshold", type=float, default=0.7, help="アイコンテンプレートマッチングの閾値 (default: 0.7)")
    parser.add_argument("--debug", action="store_true", help="区切り線・切り出し領域のデバッグ画像を保存")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("過去記録フィールド キャプチャツール (past records crop)")
    print("=" * 60)

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    if args.clear:
        clear_captures()

    if args.device is not None:
        target_device = args.device
        print(f"\n指定デバイスを使用: {target_device}")
        frame = capture_from_device(target_device, num_warmup_frames=5)
    else:
        devices = scan_video_devices(max_devices=5)

        if not devices:
            print("\n✗ エラー: ビデオデバイスが見つかりませんでした")
            print("  - HDMIキャプチャデバイス（MiraBox）が接続されているか確認してください")
            print("  - USBケーブルが正しく接続されているか確認してください")
            return

        working_devices = [d for d in devices if d['working']]

        if not working_devices:
            print("\n✗ 警告: フレームを読み込めるデバイスがありません")
            print(f"  検出されたデバイス: {len(devices)}個")
            print("  すべてのデバイスでフレーム読み込みに失敗しました")
            print("\nデバイス 0 で再試行します...")
            frame = capture_from_device(0, num_warmup_frames=10)
        else:
            target_device = working_devices[0]['index']
            print(f"\n使用するデバイス: {target_device}")
            frame = capture_from_device(target_device, num_warmup_frames=5)

    if frame is None:
        print("\n" + "=" * 60)
        print("✗ キャプチャ失敗")
        print("=" * 60)
        return

    cropped = crop_past_records(frame, threshold=args.threshold, debug=args.debug)
    if cropped is None:
        print("\n" + "=" * 60)
        print("✗ 過去記録フィールドの切り出し失敗")
        print("=" * 60)
        return

    if args.name:
        out_name = args.name
        if not out_name.lower().endswith('.jpg'):
            out_name += '.jpg'
        filename = os.path.join(CAPTURE_DIR, out_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"past_records_{timestamp}.jpg")

    cv2.imwrite(filename, cropped)
    print(f"✓ 保存完了: {filename}")

    print("\n" + "=" * 60)
    print("✓ 過去記録フィールド キャプチャ完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
