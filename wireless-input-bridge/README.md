# Wireless Input Bridge

ESP32でBLE UART経由でUSBマウスとキーボードを制御するArduinoスケッチです。

## 必要なハードウェア

- ESP32-S3（Native USB対応）— M5AtomS3U で動作確認済み
- Arduino IDE または arduino-cli

## 機能

BLE UART経由で以下のコマンドを送信してUSBマウスとキーボードを制御できます。

### モード切替コマンド（互換性のため残存、動作に影響なし）

| コマンド | 説明 |
|---------|------|
| `mode:mouse` | マウスモードに切り替え（互換用） |
| `mode:keyboard` | キーボードモードに切り替え（互換用） |

### マウスコマンド

| コマンド | 説明 | 例 |
|---------|------|-----|
| `moveto:X,Y` | カーソルを絶対座標に移動（HID単位 0〜32767） | `moveto:16383,16383` |
| `move:DX,DY` | 現在位置からの相対移動（HID単位） | `move:1706,0` |
| `click` | 左クリックを実行 | `click` |
| `scroll:N` | スクロール（正の値で下、負の値で上） | `scroll:3` |

### キーボードコマンド

| コマンド | 説明 | 例 |
|---------|------|-----|
| `type:テキスト` | 指定したテキストを入力 | `type:Hello` |
| `key:enter` | Enterキーを押す | `key:enter` |
| `key:tab` | Tabキーを押す | `key:tab` |
| `key:backspace` | Backspaceキーを押す | `key:backspace` |
| `key:delete` | Deleteキーを押す | `key:delete` |
| `key:esc` | Escキーを押す | `key:esc` |

---

## moveto と move の違い

### moveto:X,Y — 絶対座標移動

スクリーン上の絶対位置にカーソルをジャンプさせます。

- X, Y は HID 座標系（0〜32767）で指定します
- Python側では画面ピクセル座標を自動的に HID 座標に変換して送信します
  - 例: 1920×1080 画面の pixel (960, 540) → `moveto:16383,16383`
- `USBHIDAbsoluteMouse` を使用するため、Windowsのマウス速度設定や
  ポインター精度設定の影響を受けません

```
moveto:16383,16383   # 1920x1080スクリーンの中央へ
moveto:0,0           # 左上隅へ
moveto:32767,32767   # 右下隅へ
```

### move:DX,DY — 相対移動

現在のカーソル位置からの差分でカーソルを移動させます。

- DX, DY は HID 単位の差分値（正負どちらも可）
- ファームウェア内部で現在位置に加算するため、現在位置を知らなくても使えます
- Python側では `move_mouse(x_px, y_px)` がピクセルを HID 単位に変換して送信します
  - 変換式: `HID_delta = pixel_delta * 32767 / screen_width`

```
move:1706,0      # 右へ約100px（1920px幅のスクリーン想定）
move:-1706,0     # 左へ約100px
move:0,1524      # 下へ約100px（1080px高のスクリーン想定）
move:1706,1524   # 右100px・下100px
```

> **注意**: 電源投入直後はファームウェアの追跡位置がスクリーン中央
> (16383, 16383) に初期化されています。最初に `moveto` で既知の位置に
> 移動してから `move` を使うと、より正確に動作します。

---

## OTAアップデート

WiFi経由でファームウェアを無線アップデートできます。USB接続は不要です。

### 初回セットアップ

1. `wifi_config.h` を編集してWiFi認証情報を入力します（gitignore済み）:

```cpp
#define WIFI_SSID     "your-ssid"
#define WIFI_PASSWORD "your-password"
```

2. USBケーブルでMacに接続し、通常通りファームウェアを書き込みます
3. シリアルモニタ（115200 baud）で以下が表示されれば準備完了：

```
WiFi connected, IP: 192.168.x.x
OTA ready — hostname: ble-hid-bridge
```

### OTAアップデート手順

以降のアップデートはUSB不要：

```bash
# ホスト名で自動検出（推奨）
./scripts/upload_firmware_ota.sh

# IPアドレスを直接指定する場合
./scripts/upload_firmware_ota.sh 192.168.1.42
```

スクリプトはコンパイルと転送を自動で実行します。転送後、デバイスは自動的に再起動します。

### Arduino IDEからのOTA

`Tools > Port` に `ble-hid-bridge at 192.168.x.x` が表示されたら、
そのポートを選択して通常通りアップロードできます。

---

## セットアップ

1. `wifi_config.h` にWiFi認証情報を設定
2. このスケッチをESP32にUSBでアップロード（初回のみ）
3. 以降は `./scripts/upload_firmware_ota.sh` でOTA更新

## BLE UART接続方法

### Mac（Pythonスクリプト）

```bash
./scripts/run_ble_test.sh
```

CLIコマンド例：

```
connect
mouse
moveto 960 540       # 画面中央へ（絶対座標）
move 100 0           # 右へ100px（相対移動）
move -100 50         # 左100px・下50px（相対移動）
click
keyboard
type hello
key:enter
```

### スマートフォンアプリ

- **Android**: Serial Bluetooth Terminal, nRF Connect
- **iOS**: LightBlue, nRF Connect

デバイス名 "BLE Mouse & Keyboard" に接続し、
UARTサービス（UUID: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E）へコマンドを送信します。

---

## 技術詳細

### アーキテクチャ

```
Mac
 └─(BLE UART)─► ESP32 M5AtomS3U ─(USB HID)─► Windows PC
                      │
                      └─(WiFi OTA)─► 無線ファームウェア更新
```

### 使用ライブラリ

- `BLEDevice.h`, `BLEServer.h`, `BLEUtils.h`, `BLE2902.h` — BLE UART通信
- `USBHIDMouse.h` (`USBHIDAbsoluteMouse`) — USB絶対座標マウス
- `USBHIDKeyboard.h` — USBキーボード
- `WiFi.h`, `ArduinoOTA.h` — OTAアップデート

### UUIDs（BLE UART / Nordic UART Service）

- Service UUID: `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`
- RX Characteristic: `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` (WRITE)
- TX Characteristic: `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` (NOTIFY)

### HID座標系と画面ピクセルの変換

`USBHIDAbsoluteMouse` は 0〜32767 の絶対座標系を使用します。

```
HID_x = pixel_x * 32767 / screen_width
HID_y = pixel_y * 32767 / screen_height
```

例（1920×1080スクリーン）:

| 画面位置 | ピクセル | HID座標 |
|---------|---------|---------|
| 左上 | (0, 0) | (0, 0) |
| 中央 | (960, 540) | (16383, 16383) |
| 右下 | (1920, 1080) | (32767, 32767) |
| フリガナ欄 | (254, 236) | (4329, 7152) |

---

## トラブルシューティング

### Windows で「コード10 / このデバイスを開始できません」が出る場合

**解決手順:**

1. Arduino IDE の設定を確認する
   - `Tools → USB CDC On Boot → Enabled`
   - `Tools → USB Mode → USB-OTG (TinyUSB)`
2. 正しい設定でファームウェアを**再ビルド・再書き込み**する
3. Windows のデバイスマネージャーで当該デバイスを**アンインストール**する
   - 右クリック → 「デバイスのアンインストール」
   - 「このデバイスのドライバーソフトウェアを削除する」にチェック
4. ESP32 を USB から**抜いて再接続**する
5. 「HID準拠マウス」「HID準拠キーボード」として表示されれば成功

### OTA接続が見つからない場合

```bash
# pingで確認
ping ble-hid-bridge.local

# mDNSで検索
dns-sd -B _arduino._tcp local

# IPを直接指定
./scripts/upload_firmware_ota.sh 192.168.x.x
```

### OTA後にBLEが切断される場合

OTAアップデート中はBLE接続が一時切断されます。アップデート完了後に自動的に再起動し、
BLEアドバタイジングを再開します。

## ライセンス

Public Domain
