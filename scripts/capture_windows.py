import argparse
import cv2
import glob
import os
import time
from datetime import datetime

# キャプチャー画像の保存先フォルダ
CAPTURE_DIR = "captures"

# フォルダが存在しない場合は作成
os.makedirs(CAPTURE_DIR, exist_ok=True)

def scan_video_devices(max_devices=10):
    """利用可能なビデオデバイスをスキャン"""
    print("=" * 60)
    print("ビデオデバイススキャン開始...")
    print("=" * 60)

    available_devices = []

    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # デバイス情報を取得
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()

            # テストフレームを読み込み
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
                # フレームが真っ黒かチェック
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

def capture_from_device(device_index, num_warmup_frames=5, output_filename=None):
    """指定したデバイスからキャプチャ"""
    print(f"\nデバイス {device_index} からキャプチャを開始...")

    # デバイスを開く
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        print(f"✗ エラー: デバイス {device_index} を開けませんでした")
        return False

    print(f"✓ デバイス {device_index} をオープンしました")

    # デバイス情報を表示
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()

    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  バックエンド: {backend}")

    # 解像度を明示的に設定（MiraBox用）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 設定後の解像度を確認
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  設定後の解像度: {actual_width}x{actual_height}")

    # ウォームアップ: 最初の数フレームをスキップ
    # （デバイスによっては最初のフレームが黒い場合がある）
    print(f"\nウォームアップ中（{num_warmup_frames}フレームスキップ）...")
    for i in range(num_warmup_frames):
        ret, frame = cap.read()
        if ret:
            mean_brightness = frame.mean() if frame is not None else 0
            print(f"  フレーム {i+1}/{num_warmup_frames}: 輝度={mean_brightness:.2f}")
        else:
            print(f"  フレーム {i+1}/{num_warmup_frames}: 読み込み失敗")
        time.sleep(0.1)  # 少し待機

    # 実際のキャプチャフレームを読み込む
    print("\n実際のキャプチャフレーム取得中...")
    ret, frame = cap.read()

    if not ret or frame is None:
        print("✗ エラー: フレームの読み込みに失敗しました")
        cap.release()
        return False

    # フレームの状態をチェック
    height, width = frame.shape[:2]
    mean_brightness = frame.mean()

    print(f"✓ フレーム取得成功")
    print(f"  解像度: {width}x{height}")
    print(f"  平均輝度: {mean_brightness:.2f}")

    if mean_brightness < 1.0:
        print(f"⚠ 警告: フレームが真っ黒です（輝度 < 1.0）")
        print(f"  デバイスが正しく接続されているか確認してください")

    # ファイル名の決定
    if output_filename:
        if not output_filename.lower().endswith('.jpg'):
            output_filename += '.jpg'
        filename = os.path.join(CAPTURE_DIR, output_filename)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CAPTURE_DIR, f"windows_capture_{timestamp}.jpg")

    # 画像を保存
    cv2.imwrite(filename, frame)
    print(f"✓ 保存完了: {filename}")

    cap.release()
    return True

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")


def clear_captures():
    """capturesフォルダ内の画像ファイルをすべて削除"""
    removed = 0
    for ext in IMAGE_EXTENSIONS:
        for filepath in glob.glob(os.path.join(CAPTURE_DIR, ext)):
            os.remove(filepath)
            removed += 1
    print(f"✓ {removed} 件の画像ファイルを削除しました ({CAPTURE_DIR}/)")
    return removed


def parse_args():
    parser = argparse.ArgumentParser(description="Windows HDMI キャプチャツール (MiraBox)")
    parser.add_argument("--name", default=None, help="出力ファイル名")
    parser.add_argument("--clear", action="store_true", help="キャプチャ前にcapturesフォルダの画像を全削除")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("Windows HDMI キャプチャツール (MiraBox)")
    print("=" * 60)

    if args.clear:
        clear_captures()

    custom_name = args.name

    # ステップ1: デバイススキャン
    devices = scan_video_devices(max_devices=5)

    if not devices:
        print("\n✗ エラー: ビデオデバイスが見つかりませんでした")
        print("  - HDMIキャプチャデバイス（MiraBox）が接続されているか確認してください")
        print("  - USBケーブルが正しく接続されているか確認してください")
        return

    # 動作しているデバイスを見つける
    working_devices = [d for d in devices if d['working']]

    if not working_devices:
        print("\n✗ 警告: フレームを読み込めるデバイスがありません")
        print(f"  検出されたデバイス: {len(devices)}個")
        print("  すべてのデバイスでフレーム読み込みに失敗しました")

        # それでも最初のデバイスで試行
        print("\nデバイス 0 で再試行します...")
        capture_from_device(0, num_warmup_frames=10, output_filename=custom_name)
        return

    # 最初の動作デバイスを使用
    target_device = working_devices[0]['index']
    print(f"\n使用するデバイス: {target_device}")

    # ステップ2: キャプチャ実行
    success = capture_from_device(target_device, num_warmup_frames=5, output_filename=custom_name)

    if success:
        print("\n" + "=" * 60)
        print("✓ キャプチャ完了")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ キャプチャ失敗")
        print("=" * 60)

if __name__ == "__main__":
    main()
