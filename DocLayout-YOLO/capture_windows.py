import cv2
import os
from datetime import datetime

# デバイス0 = MiraBox Video Capture (Windows PC)
WINDOWS_DEVICE = 0

# キャプチャー画像の保存先フォルダ
CAPTURE_DIR = "captures"

# フォルダが存在しない場合は作成
os.makedirs(CAPTURE_DIR, exist_ok=True)

print("Windows画面をキャプチャしています...")

cap = cv2.VideoCapture(WINDOWS_DEVICE)

if not cap.isOpened():
    print("エラー: キャプチャーデバイスを開けませんでした")
    exit(1)

# フレームを読み込む
ret, frame = cap.read()

if ret:
    # タイムスタンプ付きファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CAPTURE_DIR, f"windows_capture_{timestamp}.jpg")

    # 画像を保存
    cv2.imwrite(filename, frame)

    height, width = frame.shape[:2]
    print(f"✓ キャプチャ成功")
    print(f"  解像度: {width}x{height}")
    print(f"  保存先: {filename}")
else:
    print("エラー: フレームの読み込みに失敗しました")

cap.release()
