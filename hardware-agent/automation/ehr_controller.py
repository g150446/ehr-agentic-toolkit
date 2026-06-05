"""
音声操作による EHR コントローラー。

実行方法:
  python -m automation.ehr_controller --last-prescription
  python -m automation.ehr_controller --open-note
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from automation.config import load_config
from automation.ehr_reader import _wait_for_ble_connected
from automation.screen_analyzer import capture_screen as _capture_screen_hdmi


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


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    do_last_prescription = "--last-prescription" in args
    do_open_note = "--open-note" in args

    if not do_last_prescription and not do_open_note:
        print("[ERROR] --last-prescription または --open-note オプションが必要です", file=sys.stderr)
        print("使用例: python -m automation.ehr_controller --last-prescription", file=sys.stderr)
        print("       python -m automation.ehr_controller --open-note", file=sys.stderr)
        return 1

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
