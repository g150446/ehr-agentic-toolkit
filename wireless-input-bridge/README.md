# Wireless Input Bridge

ESP32でBLE UART経由でUSBマウスとキーボードを制御するArduinoスケッチです。

## 必要なハードウェア

- ESP32-S2またはESP32-S3（Native USB対応）
- Arduino IDE

## 機能

BLE UART経由で以下のコマンドを送信してUSBマウスとキーボードを制御できます：

### モード切替コマンド

| コマンド | 説明 |
|---------|------|
| `mode:mouse` | マウスモードに切り替え |
| `mode:keyboard` | キーボードモードに切り替え |

### マウスコマンド（マウスモード時）

| コマンド | 説明 | 例 |
|---------|------|-----|
| `click` | 左クリックを実行 | `click` |
| `up:N` | カーソルをN ピクセル上に移動 | `up:10` |
| `down:N` | カーソルをN ピクセル下に移動 | `down:10` |
| `left:N` | カーソルをN ピクセル左に移動 | `left:10` |
| `right:N` | カーソルをN ピクセル右に移動 | `right:10` |
| `scroll:N` | N 単位スクロール（正の値で下、負の値で上） | `scroll:5` または `scroll:-5` |

### キーボードコマンド（キーボードモード時）

| コマンド | 説明 | 例 |
|---------|------|-----|
| `type:テキスト` | 指定したテキストを入力 | `type:Hello World` |
| `key:enter` | Enterキーを押す | `key:enter` |
| `key:tab` | Tabキーを押す | `key:tab` |
| `key:backspace` | Backspaceキーを押す | `key:backspace` |
| `key:esc` | Escキーを押す | `key:esc` |

## スクロール機能について

**スクロールは可能です！**

`Mouse.move(x, y, scroll)` 関数の3番目のパラメータがスクロールホイールの値になっています。

- 正の値：下にスクロール
- 負の値：上にスクロール

例：
```
scroll:5    // 下に5単位スクロール
scroll:-5   // 上に5単位スクロール
```

## セットアップ

1. このスケッチをESP32にアップロード
2. シリアルモニタを開く（115200 baud）
3. BLEデバイス名 "BLE Mouse & Keyboard" に接続
4. UARTサービス（UUID: 6E400001-B5A3-F393-E0A9-E50E24DCCA9E）経由でコマンドを送信

## BLE UART接続方法

### スマートフォンアプリを使用する場合

以下のようなBLE UARTアプリが使えます：
- **Android**: Serial Bluetooth Terminal, nRF Connect
- **iOS**: LightBlue, nRF Connect

1. アプリで "BLE Mouse & Keyboard" を検索
2. 接続
3. UART RX Characteristic（UUID: 6E400002-B5A3-F393-E0A9-E50E24DCCA9E）にコマンドを書き込み

## 使用例

### モード切替
```
mode:mouse      // マウスモードに切り替え
mode:keyboard   // キーボードモードに切り替え
```

### マウス操作（マウスモード時）
```
click           // 左クリック
up:10           // 10ピクセル上に移動
down:20         // 20ピクセル下に移動
left:15         // 15ピクセル左に移動
right:15        // 15ピクセル右に移動
scroll:3        // 下に3単位スクロール
scroll:-3       // 上に3単位スクロール
```

### キーボード操作（キーボードモード時）
```
type:Hello World        // "Hello World"と入力
type:こんにちは         // 日本語入力（要IME設定）
type:user@example.com   // メールアドレス入力
key:enter               // Enterキーを押す
key:tab                 // Tabキーを押す
key:backspace           // Backspaceキーを押す
key:esc                 // Escキーを押す
```

## 技術詳細

### 使用しているライブラリ

- `BLEDevice.h`, `BLEServer.h`, `BLEUtils.h`, `BLE2902.h` - BLE通信
- `USB.h`, `USBHIDMouse.h`, `USBHIDKeyboard.h` - USB HIDマウス・キーボードエミュレーション

### UUIDs

- Service UUID: `6E400001-B5A3-F393-E0A9-E50E24DCCA9E`
- RX Characteristic: `6E400002-B5A3-F393-E0A9-E50E24DCCA9E` (WRITE)
- TX Characteristic: `6E400003-B5A3-F393-E0A9-E50E24DCCA9E` (NOTIFY)

## 注意事項

- このスケッチはArduino USB ModeがUSBモード（OTGモードではない）の時に動作します
- ESP32-S2またはESP32-S3のようなNative USB対応のボードが必要です
- マウス・キーボード制御が有効になると、コンピュータの入力デバイスが乗っ取られます。使用には注意してください
- キーボードモードでの日本語入力には、コンピュータ側でIMEを有効にする必要があります
- デフォルトではマウスモードで起動します

## ライセンス

Public Domain
