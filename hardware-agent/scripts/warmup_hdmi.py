#!/usr/bin/env python3
"""
HDMI キャプチャデバイス ウォームアップ

Mac Mini コールドスタート時、MiraBox USB HDMI キャプチャデバイスが
HDMI 信号をロックして有効なフレームを返すまで待機する。

AVCaptureSession を保持し続けることで QuickTime Player の
「新規ムービー収録」と同等の初期化を行う。

Exit 0: 成功 (有効なフレームを取得)
Exit 1: タイムアウト またはデバイスを開けない
"""
import sys
import time
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="HDMI デバイスウォームアップ")
    parser.add_argument("--timeout",    type=float, default=30.0,
                        help="タイムアウト秒数 (デフォルト: 30)")
    parser.add_argument("--min-warmup", type=float, default=5.0,
                        help="最小ウォームアップ秒数 (デフォルト: 5)")
    parser.add_argument("--threshold",  type=float, default=5.0,
                        help="有効フレーム判定の輝度閾値 (デフォルト: 5.0)")
    args = parser.parse_args()

    import cv2
    import numpy as np
    from automation.config import AutomationConfig

    cfg          = AutomationConfig()
    device_index = cfg.capture_device_index
    width        = cfg.capture_width
    height       = cfg.capture_height

    print(f"[HDMI warmup] デバイス {device_index} ({width}x{height}) 初期化中...", flush=True)

    start               = time.time()
    timeout_deadline    = start + args.timeout
    min_warmup_deadline = start + args.min_warmup
    valid_frame_seen    = False

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"[HDMI warmup] エラー: デバイス {device_index} を開けません。", flush=True)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("[HDMI warmup] デバイスオープン完了。HDMI 信号ロック待機中...", flush=True)

    try:
        while time.time() < timeout_deadline:
            ret, frame = cap.read()
            if ret and frame is not None:
                brightness = float(np.mean(frame))
                if brightness > args.threshold:
                    if not valid_frame_seen:
                        elapsed = time.time() - start
                        print(
                            f"[HDMI warmup] 有効フレーム検出 "
                            f"(輝度: {brightness:.1f}, {elapsed:.1f}秒経過)",
                            flush=True,
                        )
                        valid_frame_seen = True
                    if time.time() >= min_warmup_deadline:
                        elapsed = time.time() - start
                        print(f"[HDMI warmup] ウォームアップ完了 (合計: {elapsed:.1f}秒)", flush=True)
                        return 0
            time.sleep(0.03)
    finally:
        cap.release()

    elapsed = time.time() - start
    print(
        f"[HDMI warmup] タイムアウト ({elapsed:.0f}秒): 有効フレームを取得できませんでした。",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
