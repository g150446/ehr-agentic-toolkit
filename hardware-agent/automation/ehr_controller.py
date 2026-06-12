"""
音声操作による EHR コントローラー。

実行方法:
  python -m automation.ehr_controller --copy-prev-rx
  python -m automation.ehr_controller --open-note
  python -m automation.ehr_controller --open-test
  python -m automation.ehr_controller --care-plan
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date as _date
from pathlib import Path

import cv2
import numpy as np

from automation.config import load_config
from automation.ehr_input import open_test_patient_chart as _open_test_patient_chart
from automation.ehr_reader import (
    _wait_for_ble_connected,
    _detect_all_dividers,
    _detect_edit_button_bottom,
    _save_debug_frame,
)
from automation.screen_analyzer import (
    capture_screen as _capture_screen_hdmi,
    run_ocr_backend,
)


def _find_time_series_button(
    frame: np.ndarray,
    *,
    threshold: float = 0.7,
) -> tuple[int, int] | None:
    """画面全体から time_series_button.png を検出し、一致矩形の中心座標 (x, y) を返す。"""
    template_path = (
        Path(__file__).resolve().parent.parent
        / "match_templates"
        / "time_series_button.png"
    )
    if not template_path.exists():
        print(f"[WARNING] テンプレート画像が見つかりません: {template_path}")
        return None

    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"[WARNING] テンプレート画像の読み込みに失敗しました: {template_path}")
        return None

    if template.shape[2] == 4:
        b, g, r, a = cv2.split(template)
        alpha = a.astype(np.float32) / 255.0
        white_bg = np.full_like(b, 255, dtype=np.uint8)
        b = (b.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        g = (g.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        r = (r.astype(np.float32) * alpha + white_bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        template = cv2.merge([b, g, r])

    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]

    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        print(f"  time_series_button 未検出 (最高スコア {max_val:.3f} < {threshold})")
        return None

    cx = max_loc[0] + w // 2
    cy = max_loc[1] + h // 2
    print(f"  time_series_button 検出: ({max_loc[0]}, {max_loc[1]})〜({max_loc[0] + w}, {max_loc[1] + h}) 中心=({cx}, {cy}) (score={max_val:.3f})")
    return (cx, cy)


def _click_time_series_button(click_x: int, click_y: int) -> bool:
    """time_series_button の中心座標にマウスを移動してクリックする。"""
    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(click_x, click_y)
    print(f"moveto ({click_x}, {click_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"click (time_series_button) -> {'OK' if ok else 'NG'}")
    return ok


def _find_column2_click_pos(frame: np.ndarray) -> tuple[int, int] | None:
    """時系列テーブルの第2列中心座標を返す。

    優先: Hough縦線検出で3列の水平レイアウトを認識 (table2.jpg型)
    代替: 彩度バンドで垂直積み重ねレイアウトを認識 (drug.jpg型)
    """
    h, w = frame.shape[:2]
    print(f"  フレームサイズ: {w}x{h}")

    debug_path = Path(__file__).resolve().parent.parent / "captures" / "debug_timeseries.jpg"
    cv2.imwrite(str(debug_path), frame)
    print(f"  デバッグ画像: {debug_path}")

    # --- Method 1: Hough 縦線 → 水平3列レイアウト ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 2,
        threshold=15,
        minLineLength=30,
        maxLineGap=30,
    )
    raw_xs: list[int] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 3:
                raw_xs.append(int((x1 + x2) // 2))

    if raw_xs:
        # 15px 以内の近傍線をまとめてクラスタ重心を算出
        raw_xs = sorted(set(raw_xs))
        centroids: list[int] = []
        total, count, prev = raw_xs[0], 1, raw_xs[0]
        for x in raw_xs[1:]:
            if x - prev > 15:
                centroids.append(total // count)
                total, count = x, 1
            else:
                total += x
                count += 1
            prev = x
        centroids.append(total // count)

        # 全境界 [0, ...centroids..., w] から幅 w/10 超の区間を列とみなす
        all_b = [0] + centroids + [w]
        min_col_w = w // 10
        wide = [
            (all_b[i], all_b[i + 1])
            for i in range(len(all_b) - 1)
            if all_b[i + 1] - all_b[i] > min_col_w
        ]
        print(f"  Hough列候補: {wide}")

        if len(wide) >= 3:
            # 3列以上検出: 右から2番目の列（最新1つ前 = 最終処方が格納された列）
            col2_x0, col2_x1 = wide[-2]
            cx = int((col2_x0 + col2_x1) // 2)
            cy = int(h * 0.17)
            print(f"  列境界検出(Hough縦線): 右から2番目列 x={col2_x0}-{col2_x1}")
            print(f"  列中心: ({cx}, {cy})")
            return (int(cx), int(cy))

        if len(wide) == 2:
            # 1本しか区切り線を検出できなかった場合: 右列の幅から等幅と仮定してcol2を推定
            right_col_w = wide[-1][1] - wide[-1][0]
            col2_x1 = wide[-1][0]
            col2_x0 = max(wide[0][0], col2_x1 - right_col_w)
            cx = int((col2_x0 + col2_x1) // 2)
            cy = int(h * 0.17)
            print(f"  列境界検出(等幅推定): 推定col2 x={col2_x0}-{col2_x1}")
            print(f"  列中心: ({cx}, {cy})")
            return (int(cx), int(cy))

    # --- Method 2: 彩度バンド → 垂直積み重ねレイアウト ---
    print("  Hough未検出 → 彩度バンド法にフォールバック")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x_start, x_end = 100, min(w - 10, 1600)
    sat = hsv[:, x_start:x_end, 1]
    row_counts = (sat > 35).sum(axis=1).astype(np.float32)
    kernel = np.ones(5, dtype=np.float32) / 5
    smoothed = np.convolve(row_counts, kernel, mode="same")
    threshold = (x_end - x_start) * 0.35
    high_rows = np.where(smoothed > threshold)[0]

    if len(high_rows) == 0:
        print("[WARNING] 処方エントリーを検出できませんでした")
        return None

    groups_s: list[tuple[int, int, float]] = []
    s = prev = int(high_rows[0])
    peak = float(smoothed[s])
    for y in high_rows[1:]:
        y = int(y)
        if y - prev > 20:
            groups_s.append((s, prev, peak))
            s, peak = y, float(smoothed[y])
        else:
            peak = max(peak, float(smoothed[y]))
        prev = y
    groups_s.append((s, prev, peak))

    min_height = max(20, int(h * 0.03))
    entry_bands = [(s, e, p) for s, e, p in groups_s if s > 100 and e - s >= min_height]
    print(f"  処方エントリーバンド: {len(entry_bands)} 個")
    for i, b in enumerate(entry_bands):
        print(f"    Band {i + 1}: y={b[0]}-{b[1]}")

    if len(entry_bands) < 2:
        print("[WARNING] 2番目のエントリーを特定できませんでした")
        return None

    target = entry_bands[1]
    cy = int((target[0] + target[1]) // 2)
    band_sat = hsv[target[0]:target[1] + 1, x_start:x_end, 1]
    col_sat = (band_sat > 35).sum(axis=0)
    sat_xs = np.where(col_sat > (target[1] - target[0]) * 0.3)[0]
    cx = int((int(sat_xs[0]) + int(sat_xs[-1])) // 2) + x_start if len(sat_xs) > 0 else w // 2

    print(f"  第2エントリー中心: ({cx}, {cy})")
    return (int(cx), int(cy))


def _right_click_column2(click_x: int, click_y: int) -> bool:
    """第2列の座標にマウスを移動して右クリックする。"""
    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(click_x, click_y)
    print(f"moveto ({click_x}, {click_y}) -> {'OK' if ok else 'NG'}")

    ok = client.right_click()
    print(f"right_click (第2列) -> {'OK' if ok else 'NG'}")
    return ok


def _find_care_plan_row(
    frame: np.ndarray,
    dividers: list[int],
    *,
    keyword: str = "生活",
    min_confidence: float = 0.3,
) -> tuple[int, int] | None:
    """書状タブのテーブルから keyword を含む行を検出し、最上位行の全画面座標 (x, y) を返す。

    1. printer_button.png でコンテンツ開始 y を検出（失敗時は OCR で「表示名」を探す）
    2. 書状パネル領域を切り出して OCR
    3. keyword を含む結果を y 昇順でソートし最小 y の行を返す
    """
    x_start = dividers[0]
    x_end = dividers[1]

    y_start = _detect_edit_button_bottom(frame, x_start, x_end)
    if y_start is None:
        print("  printer_button 未検出 → OCR で「表示名」を検索してフォールバック...")
        panel = frame[:, x_start:x_end]
        ocr_results = run_ocr_backend(panel, backend="ndlocr")
        y_start = None
        for bbox, text, conf in ocr_results:
            if conf >= min_confidence and "表示名" in text:
                ys = [p[1] for p in bbox]
                y_start = int(max(ys))
                print(f"  「表示名」ヘッダー検出: y_start={y_start}")
                break
        if y_start is None:
            print("  「表示名」も未検出 → y_start=0 で全領域を対象とします")
            y_start = 0

    cropped = frame[y_start:, x_start:x_end]

    overlay = frame.copy()
    h = overlay.shape[0]
    for x in dividers:
        cv2.line(overlay, (x, 0), (x, h - 1), (0, 255, 0), 2)
    cv2.rectangle(overlay, (x_start, y_start), (x_end, h - 1), (0, 0, 255), 2)
    _save_debug_frame(overlay, "care_plan_region", subdir="crop")
    _save_debug_frame(cropped, "care_plan_crop", subdir="crop")

    from automation.mlx_vlm_ime import _get_ndlocr
    _detector, _recognizer = _get_ndlocr()
    _all_dets = _detector.detect(cropped)
    # run_ocr_ndlocr のデフォルト閾値 0.75 ではテーブル行が除外されるため、
    # ここでは 0.65 に下げて line_main を直接処理する。
    _line_dets = [
        d for d in _all_dets
        if d["class_name"] == "line_main" and d["confidence"] >= 0.65
    ]
    ocr_results = []
    for det in sorted(_line_dets, key=lambda d: d["box"][1]):
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        region = cropped[max(0, y1):y2, max(0, x1):x2]
        if region.size == 0:
            continue
        text = _recognizer.read(region).strip()
        if not text:
            continue
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        ocr_results.append((bbox, text, float(det["confidence"])))

    print(f"  OCR結果 ({len(ocr_results)}件):")
    for bbox, text, conf in ocr_results:
        print(f"    [{conf:.2f}] {text}")

    candidates: list[tuple[int, int, str, float]] = []
    for bbox, text, conf in ocr_results:
        if conf < min_confidence:
            continue
        if keyword in text:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            cx = x_start + int(sum(xs) / len(xs))
            cy = y_start + int(sum(ys) / len(ys))
            candidates.append((cy, cx, text, conf))
            print(f"  '{keyword}' 候補: text='{text}' conf={conf:.2f} 座標=({cx}, {cy})")

    if not candidates:
        print(f"  '{keyword}' を含む行が見つかりませんでした")
        return None

    candidates.sort(key=lambda c: c[0])
    cy, cx, text, conf = candidates[0]
    print(f"  最上位行を選択: text='{text}' 座標=({cx}, {cy})")
    return (cx, cy)


def _right_click_care_plan_row(click_x: int, click_y: int) -> bool:
    """指定座標へマウスを移動して左クリック → 500ms待機 → 右クリックする。"""
    client = _wait_for_ble_connected()

    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")

    ok = client.move_mouse_to_position(click_x, click_y)
    print(f"moveto ({click_x}, {click_y}) -> {'OK' if ok else 'NG'}")

    ok = client.click()
    print(f"left_click (care_plan_row) -> {'OK' if ok else 'NG'}")

    time.sleep(0.5)

    ok = client.right_click()
    print(f"right_click (care_plan_row) -> {'OK' if ok else 'NG'}")

    time.sleep(0.5)

    ok = client.switch_to_keyboard_mode()
    print(f"mode:keyboard -> {'OK' if ok else 'NG'}")

    ok = client.press_key("f2")
    print(f"press_key(f2) -> {'OK' if ok else 'NG'}")
    return ok


def _find_word_text_edges(
    frame: np.ndarray,
    keyword: str,
    *,
    min_confidence: float = 0.1,
    backend: str = "easyocr",
    scale: float = 1.0,
    y_offset: int = 0,
    x_offset: int = 0,
) -> tuple[int, int, int] | None:
    """frame 内で keyword を OCR 検索し (left_x, y_center, right_x) を返す。

    scale/y_offset/x_offset: クロップ・拡大済みフレームを渡す場合に元座標へ戻すための変換値。
    """
    ocr_results = run_ocr_backend(frame, backend=backend)
    for bbox, text, conf in ocr_results:
        if conf < min_confidence:
            continue
        matched = keyword in text
        if not matched and len(keyword) >= 2:
            matched = keyword[0] in text and keyword[-1] in text
        if matched:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            left_x = int(min(xs) / scale) + x_offset
            right_x = int(max(xs) / scale) + x_offset
            y_center = int(sum(ys) / len(ys) / scale) + y_offset
            print(f"  '{keyword}' 検出: left={left_x} right={right_x} y={y_center} conf={conf:.2f} text={repr(text)}")
            return (left_x, y_center, right_x)
    print(f"  '{keyword}' が見つかりませんでした")
    return None


def _find_age_to_delete(
    all_results: list,
) -> tuple[int, int, int] | None:
    """OCR全結果から年齢パターン [（(]\d+才[)）] の位置を返す。

    テキストブロック内のマッチ位置を線形補間して、ドラッグ開始・終了X座標を計算する。
    1件目を見つけたら即座に返す（1箇所のみの出現を想定）。
    Returns: (drag_start_x, drag_end_x, y) or None.
    """
    pattern = re.compile(r'[〔（(\[]\s*(\d+)\s*才\s*[)）〕\]]')
    for bbox, text, conf in all_results:
        if conf < 0.1:
            continue
        m = pattern.search(text)
        if m:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            bx1 = int(min(xs))
            bx2 = int(max(xs))
            y_c = int(sum(ys) / len(ys))
            # パターンの開始・終了位置を線形補間してドラッグ範囲を計算
            text_len = max(len(text), 1)
            ratio_start = m.start() / text_len
            ratio_end = m.end() / text_len
            drag_x1 = int(bx1 + ratio_start * (bx2 - bx1))
            drag_x2 = int(bx1 + ratio_end * (bx2 - bx1))
            print(f"  年齢パターン検出: '{m.group()}' in '{text}'")
            print(f"    drag range: ({drag_x1}, {y_c}) -> ({drag_x2}, {y_c})")
            return (drag_x1, drag_x2, y_c)
    print("  年齢パターンは検出されませんでした")
    return None


def _find_care_plan_header_bounds(
    frame: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Word療養計画書のヘッダー行から (記入日left_x, y, 回目right_x, y) を返す。

    「令和」を空間的アンカーとして、その左側の最近傍テキスト（記入日）の左端と
    右側テキスト（回目）の右端を取得する。OCRの誤読に依存しない位置ベースの手法。
    """
    results = run_ocr_backend(frame, backend="easyocr")

    # 「令和」を高信頼度で検索
    reiwa_bbox = None
    for bbox, text, conf in results:
        if "令和" in text and conf > 0.3:
            reiwa_bbox = bbox
            print(f"  '令和' 基準: conf={conf:.3f}")
            break
    if reiwa_bbox is None:
        print("  '令和' が検出できませんでした")
        return None

    reiwa_ys = [p[1] for p in reiwa_bbox]
    reiwa_xs = [p[0] for p in reiwa_bbox]
    ry = int(sum(reiwa_ys) / len(reiwa_ys))
    rleft = int(min(reiwa_xs))
    rright = int(max(reiwa_xs))

    left_candidates: list[tuple[int, int, int]] = []   # (x1, x2, y_center)
    right_candidates: list[tuple[int, int, int]] = []  # (x1, x2, y_center)

    for bbox, text, conf in results:
        if conf < 0.1:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        y_c = int(sum(ys) / len(ys))
        x1 = int(min(xs))
        x2 = int(max(xs))
        if abs(y_c - ry) > 10:
            continue
        if x2 <= rleft:
            left_candidates.append((x1, x2, y_c))
        elif x1 >= rright:
            right_candidates.append((x1, x2, y_c))

    if not left_candidates:
        print("  記入日 左側候補なし")
        return None
    if not right_candidates:
        print("  回目 右側候補なし")
        return None

    # 令和の直前にある（x1最大）テキストの左端を記入日位置とする
    best_left = max(left_candidates, key=lambda t: t[0])
    # 右側テキストの最大right_xを回目位置とする
    best_right = max(right_candidates, key=lambda t: t[1])

    print(f"  記入日: x={best_left[0]}-{best_left[1]}, y={best_left[2]}")
    print(f"  回目:   x={best_right[0]}-{best_right[1]}, y={best_right[2]}")
    return (best_left[0], best_left[2], best_right[1], best_right[2])


def _parse_reiwa_date_from_text(text: str) -> tuple[int, int, int] | None:
    """令和OCRテキストから (reiwa_year, month, day) を抽出する。"""
    m = re.search(r'(\d+)年.*?(\d+)月.*?(\d+)日', text)
    if not m:
        return None
    return (int(m[1]), int(m[2]), int(m[3]))


def _estimate_digit_x_before(
    text: str, landmark: str, bbox_left: int, bbox_right: int
) -> int:
    """テキスト中の landmark（年/月/日）直前の数字の推定X座標を返す（線形補間）。"""
    idx = text.find(landmark)
    if idx <= 0:
        return (bbox_left + bbox_right) // 2
    ratio = (idx - 0.5) / len(text)
    return int(bbox_left + ratio * (bbox_right - bbox_left))


def _find_care_plan_header_data(frame: np.ndarray) -> dict | None:
    """Word療養計画書ヘッダーの位置情報とOCRデータをdictで返す。

    _find_care_plan_header_bounds と同ロジック + 令和テキスト・kaime左端・全結果を追加。
    """
    results = run_ocr_backend(frame, backend="easyocr")

    reiwa_bbox = None
    reiwa_text = ""
    for bbox, text, conf in results:
        if "令和" in text and conf > 0.3:
            reiwa_bbox = bbox
            reiwa_text = text
            print(f"  '令和' 基準: conf={conf:.3f} text={repr(text)}")
            break
    if reiwa_bbox is None:
        print("  '令和' が検出できませんでした")
        return None

    reiwa_ys = [p[1] for p in reiwa_bbox]
    reiwa_xs = [p[0] for p in reiwa_bbox]
    ry = int(sum(reiwa_ys) / len(reiwa_ys))
    rleft = int(min(reiwa_xs))
    rright = int(max(reiwa_xs))

    left_candidates: list[tuple[int, int, int]] = []
    right_candidates: list[tuple[int, int, int]] = []

    for bbox, text, conf in results:
        if conf < 0.1:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        y_c = int(sum(ys) / len(ys))
        x1 = int(min(xs))
        x2 = int(max(xs))
        if abs(y_c - ry) > 10:
            continue
        if x2 <= rleft:
            left_candidates.append((x1, x2, y_c))
        elif x1 >= rright:
            right_candidates.append((x1, x2, y_c))

    if not left_candidates:
        print("  記入日 左側候補なし")
        return None
    if not right_candidates:
        print("  回目 右側候補なし")
        return None

    best_left = max(left_candidates, key=lambda t: t[0])
    best_right = max(right_candidates, key=lambda t: t[1])

    print(f"  記入日: x={best_left[0]}-{best_left[1]}, y={best_left[2]}")
    print(f"  回目:   x={best_right[0]}-{best_right[1]}, y={best_right[2]}")
    return {
        "kinen_left_x": best_left[0],
        "kinen_right_x": best_left[1],
        "kinen_y": best_left[2],
        "kaime_right_x": best_right[1],
        "kaime_y": best_right[2],
        "reiwa_text": reiwa_text,
        "reiwa_left_x": rleft,
        "reiwa_right_x": rright,
        "reiwa_y": ry,
        "kaime_left_x": best_right[0],
        "all_results": results,
    }


def _find_kaime_number(
    all_results: list,
    reiwa_right_x: int,
    ry: int,
    frame: np.ndarray,
    kaime_right_x: int = 0,
) -> tuple[int | None, int | None]:
    """OCR全結果からN回目のNとフレーム内X座標を返す。戻り値: (n, x)。"""

    def _digit_center_x(bbox, text: str, digit_str: str,
                        offset_x: int = 0, scale: int = 1) -> int:
        xs = [p[0] for p in bbox]
        bx1, bx2 = int(min(xs)), int(max(xs))
        idx = text.find(digit_str)
        ratio = (idx + len(digit_str) / 2) / max(len(text), 1) if idx >= 0 else 0.5
        return int(bx1 + ratio * (bx2 - bx1)) // scale + offset_x

    # 1. 令和ブロック右側・同行トークンから digit を探す
    for bbox, text, conf in all_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1 = int(min(xs))
        y_c = int(sum(ys) / len(ys))
        if x1 >= reiwa_right_x - 10 and abs(y_c - ry) <= 15:
            m = re.search(r'(\d+)', text)
            if m:
                n = int(m[1])
                x = _digit_center_x(bbox, text, m.group())
                print(f"  N検出(右側トークン): {n} at x={x} from {repr(text)}")
                return n, x

    # 2. 全トークンから "N回" または "（N" パターンを探す（同行のみ）
    for bbox, text, conf in all_results:
        ys = [p[1] for p in bbox]
        y_c = int(sum(ys) / len(ys))
        if abs(y_c - ry) > 15:
            continue
        for pattern in [r'(\d+)回', r'[（(](\d+)']:
            m = re.search(pattern, text)
            if m:
                n = int(m.group(1))
                x = _digit_center_x(bbox, text, m.group(1))
                print(f"  N検出(パターン): {n} at x={x} from {repr(text)}")
                return n, x

    # 3. クロップ再OCR (4x拡大)
    h, w = frame.shape[:2]
    crop_x1 = max(0, reiwa_right_x - 5)
    crop_x2 = min(w, max(kaime_right_x + 10, reiwa_right_x + 150))
    crop_y1 = max(0, ry - 25)
    crop_y2 = min(h, ry + 25)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size > 0:
        scale = 4
        upscaled = cv2.resize(
            crop,
            (crop.shape[1] * scale, crop.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        crop_results = run_ocr_backend(upscaled, backend="easyocr")
        for bbox, text, _ in crop_results:
            m = re.search(r'(\d+)', text)
            if m:
                n = int(m[1])
                x = _digit_center_x(bbox, text, m.group(), offset_x=crop_x1, scale=scale)
                print(f"  N検出(クロップ再OCR): {n} at x={x} from {repr(text)}")
                return n, x

    print("  [WARNING] N(回目の数字)が検出できませんでした")
    return None, None


def _find_date_digit_positions(
    frame: np.ndarray, rleft: int, rright: int, ry: int
) -> dict[str, tuple[int, int]]:
    """令和ブロックを4倍拡大OCRして year/month/day の (center_x, right_edge_x) を返す。

    right_edge_x はダブルクリックを避けるため、数字の右端（直後の漢字の直前）の座標。
    Returns: {'year': (center, right), 'month': (center, right), 'day': (center, right)} — 検出分のみ
    """
    h, w = frame.shape[:2]
    scale = 4
    margin = 10
    y1 = max(0, ry - 22)
    y2 = min(h, ry + 22)
    x1_c = max(0, rleft - margin)
    x2_c = min(w, rright + margin)
    crop = frame[y1:y2, x1_c:x2_c]
    if crop.size == 0:
        return {}
    upscaled = cv2.resize(
        crop,
        (crop.shape[1] * scale, crop.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    crop_results = run_ocr_backend(upscaled, backend="easyocr")
    positions: dict[str, tuple[int, int]] = {}
    for bbox, text, conf in crop_results:
        if conf < 0.2:
            continue
        xs = [p[0] for p in bbox]
        bx1, bx2 = int(min(xs)), int(max(xs))
        for landmark, key in [("年", "year"), ("月", "month"), ("日", "day")]:
            if key in positions or landmark not in text:
                continue
            idx = text.find(landmark)
            before = text[:idx]
            dm = re.search(r'(\d+)\s*$', before)
            if not dm:
                continue
            digit = dm.group(1)
            d_idx = before.rfind(digit)
            # center of digit
            center_char = d_idx + len(digit) / 2
            ratio_c = center_char / max(len(text), 1)
            cx_up = bx1 + ratio_c * (bx2 - bx1)
            center_x = int(cx_up / scale + x1_c)
            # right edge of digit (just before the following kanji)
            ratio_r = (d_idx + len(digit)) / max(len(text), 1)
            rx_up = bx1 + ratio_r * (bx2 - bx1)
            right_edge_x = int(rx_up / scale + x1_c)
            positions[key] = (center_x, right_edge_x)
            print(f"  {key}数字位置(高解像度OCR): center={center_x}, right={right_edge_x} ({repr(text)} conf={conf:.2f})")
    return positions



def _word_care_plan_interaction(click_x: int, click_y: int, config) -> bool:
    """F2押下後: 4秒待機 → 同位置左クリック → Wordフレームキャプチャ → 記入日と回目の数字を更新。"""
    print("4秒待機中 (Word 読み込み)...")
    time.sleep(4.0)

    client = _wait_for_ble_connected()
    ok = client.switch_to_mouse_mode()
    print(f"mode:mouse -> {'OK' if ok else 'NG'}")
    ok = client.move_mouse_to_position(click_x, click_y)
    print(f"moveto ({click_x}, {click_y}) -> {'OK' if ok else 'NG'}")
    ok = client.click()
    print(f"left_click (same pos) -> {'OK' if ok else 'NG'}")
    time.sleep(0.5)

    print("Wordフレームをキャプチャ中...")
    word_frame = _capture_screen_hdmi(
        device_index=config.capture_device_index,
        width=config.capture_width,
        height=config.capture_height,
    )
    if word_frame is None:
        print("[ERROR] Word フレーム取得失敗", file=sys.stderr)
        return False
    _save_debug_frame(word_frame, "care_plan_word", subdir="crop")

    print("ヘッダー情報を検出中...")
    header = _find_care_plan_header_data(word_frame)
    if header is None:
        print("[ERROR] ヘッダー情報の検出に失敗しました", file=sys.stderr)
        return False

    kinen_left_x  = header["kinen_left_x"]
    kinen_right_x = header["kinen_right_x"]
    ry            = header["kinen_y"]
    kaime_right_x = header["kaime_right_x"]

    # "記入日" トークンは3文字の漢字 → 1文字分の幅を推定
    kinen_char_w = max(10, (kinen_right_x - kinen_left_x) // 3)
    drag_start_x = max(0, kinen_left_x - kinen_char_w)
    print(f"  ドラッグ範囲: ({drag_start_x}, {ry}) → ({kaime_right_x}, {ry})")

    # ドラッグ選択: 記入日の1文字前 ～ 回目の右端
    client.switch_to_mouse_mode()
    client.move_mouse_to_position(drag_start_x, ry)
    client.mouse_down()
    time.sleep(0.15)
    client.move_mouse_to_position(kaime_right_x, ry)
    time.sleep(0.15)
    client.mouse_up()
    time.sleep(0.3)

    # 選択範囲を削除
    client.switch_to_keyboard_mode()
    client.press_key("backspace")
    time.sleep(0.3)

    # 半角/全角キーで半角モードに切り替え (ehr_input.toggle_ime と同パターン)
    print("  [IME切替] 半角/全角 を送信")
    client.press_key("zenkaku")
    time.sleep(0.5)

    # 今日の日付を YYYY/M/D 形式で入力
    today = _date.today()
    date_str = f"{today.year}/{today.month}/{today.day}"
    print(f"  入力する日付: {date_str}")
    client.type_text(date_str)
    time.sleep(0.2)

    # 年齢パターン (例: 〔93才)) の検出と削除
    print("年齢パターンを検出中...")
    age_pos = _find_age_to_delete(header["all_results"])
    if age_pos:
        drag_x1, drag_x2, age_y = age_pos
        client.switch_to_mouse_mode()
        client.move_mouse_to_position(drag_x1, age_y)
        client.mouse_down()
        time.sleep(0.15)
        client.move_mouse_to_position(drag_x2, age_y)
        time.sleep(0.15)
        client.mouse_up()
        time.sleep(0.3)
        client.switch_to_keyboard_mode()
        client.press_key("backspace")
        time.sleep(0.3)
        print("  年齢パターンを削除しました")

    # 全処理完了後、IMEを半角モードに切り替え
    print("  [IME切替] 半角/全角 を送信")
    client.switch_to_keyboard_mode()
    client.press_key("zenkaku")
    time.sleep(0.5)

    print("ヘッダー更新完了")
    return True


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    do_last_prescription = "--copy-prev-rx" in args
    do_open_note = "--open-note" in args
    do_open_test = "--open-test" in args
    do_care_plan = "--care-plan" in args

    if not any([do_last_prescription, do_open_note, do_open_test, do_care_plan]):
        print("[ERROR] --copy-prev-rx / --open-note / --open-test / --care-plan オプションが必要です", file=sys.stderr)
        print("使用例: python -m automation.ehr_controller --copy-prev-rx", file=sys.stderr)
        print("       python -m automation.ehr_controller --open-note", file=sys.stderr)
        print("       python -m automation.ehr_controller --open-test", file=sys.stderr)
        print("       python -m automation.ehr_controller --care-plan", file=sys.stderr)
        return 1

    if do_open_test:
        _open_test_patient_chart()
        return 0

    if do_care_plan:
        config = load_config(skip_password=True)
        print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
        frame = _capture_screen_hdmi(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame is None:
            print("[ERROR] HDMIキャプチャデバイスからフレームを取得できませんでした", file=sys.stderr)
            return 1

        print("区切り線を検出中...")
        dividers = _detect_all_dividers(frame)
        if dividers is None:
            print("[ERROR] 患者カルテ画面の区切り線（太いグレーの縦線3本）を検出できませんでした", file=sys.stderr)
            return 1
        print(f"区切り線検出: x={dividers}")

        print("\n書状テーブルから '生活' 行を検索中...")
        pos = _find_care_plan_row(frame, dividers)
        if pos is None:
            print("[ERROR] '生活' を含む行が見つかりませんでした", file=sys.stderr)
            return 1

        click_x, click_y = pos
        print(f"\n右クリック: ({click_x}, {click_y})")
        _right_click_care_plan_row(click_x, click_y)
        print("\nWordドキュメント操作中...")
        _word_care_plan_interaction(click_x, click_y, config)
        return 0

    if do_open_note:
        client = _wait_for_ble_connected()
        ok = client.press_key("win")
        print(f"press_key(win) -> {'OK' if ok else 'NG'}")
        time.sleep(1.0)
        ok = client.type_text("note")
        print(f"type_text(note) -> {'OK' if ok else 'NG'}")
        time.sleep(0.5)
        ok = client.press_key("enter")
        print(f"press_key(enter) -> {'OK' if ok else 'NG'}")
        time.sleep(3.0)
        print("メモ帳を開きました")

    if do_last_prescription:
        config = load_config(skip_password=True)
        print(f"HDMIデバイス (index={config.capture_device_index}) からキャプチャ中...")
        frame = _capture_screen_hdmi(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame is None:
            print("[ERROR] HDMIキャプチャデバイスからフレームを取得できませんでした", file=sys.stderr)
            return 1

        print("\ntime_series_button.png を画面全体から検索中...")
        pos = _find_time_series_button(frame)
        if pos is None:
            print("[ERROR] time_series_button.png が画面内に見つかりませんでした", file=sys.stderr)
            return 1
        click_x, click_y = pos
        print(f"time_series_button クリック: ({click_x}, {click_y})")
        _click_time_series_button(click_x, click_y)

        print("テーブル表示待機中...")
        time.sleep(2.5)

        print("\n画面を再キャプチャ中...")
        frame2 = _capture_screen_hdmi(
            device_index=config.capture_device_index,
            width=config.capture_width,
            height=config.capture_height,
        )
        if frame2 is None:
            print("[ERROR] 再キャプチャ失敗", file=sys.stderr)
            return 1

        print("\n第2列のクリック座標を算出中...")
        pos2 = _find_column2_click_pos(frame2)
        if pos2 is None:
            print("[ERROR] 列境界を検出できませんでした", file=sys.stderr)
            return 1
        click2_x, click2_y = pos2
        print(f"第2列 右クリック: ({click2_x}, {click2_y})")
        _right_click_column2(click2_x, click2_y)

        time.sleep(0.5)
        follow_x = click2_x + 10
        follow_y = click2_y + 10
        print(f"フォローアップ左クリック: ({follow_x}, {follow_y})")
        follow_client = _wait_for_ble_connected()
        follow_client.move_mouse_to_position(follow_x, follow_y)
        ok = follow_client.click()
        print(f"click (フォロー) -> {'OK' if ok else 'NG'}")

        time.sleep(0.3)
        follow_client.switch_to_keyboard_mode()
        ok = follow_client.press_key("f12")
        print(f"press_key F12 -> {'OK' if ok else 'NG'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
