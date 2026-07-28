# harness-node

M5Core2 を **BLE キーボード** または **BLE UART（設定用）** として動かし、物理ボタンで役割を切り替えるファームウェア。

Aボタン長押しで、あらかじめ登録したフレーズを Bluetooth キーボードとして対象ホスト（Windows/Mac 等）に入力します。フレーズは Bボタンで切り替えた CONFIG モードで BLE UART 経由で登録でき、NVS に永続化されます（未登録時は `config.h` の `DEFAULT_PHRASE`）。

## ハードウェア

- **M5Core2**（ESP32 classic）
- Arduino-ESP32 core 3.x（NimBLE ベースの BLE スタックを使用）

## 必要ライブラリ

外部ライブラリの追加インストールは不要です。

- `M5Unified`（既に導入済み）
- BLE/HID/Preferences は esp32 core にバンドル（`BLEHIDDevice` / `BLEDevice` / `Preferences`）

## ボタン操作

| ボタン | 動作 |
|---|---|
| **A** 長押し | アクティブフレーズを BLE キーボード送信（TYPE モードのみ） |
| **B** クリック | BLE 役割を **TYPE ⇄ CONFIG** で完全切替（BLE 再初期化、約1秒） |
| **C** | 未使用 |

## 2つのモード

| モード（既定） | BLE 役割 | 接続相手 | 用途 |
|---|---|---|---|
| **TYPE** | BLE HID Keyboard（HOGP）広告 | 対象ホスト | A長押しでフレーズ入力 |
| **CONFIG** | BLE UART（Nordic UART Service）広告 | Mac | フレーズ登録・確認 |

> モード切替時、現在小中のペアリング相手は一旦切断され、戻したときに再接続します（ボンディング情報は保持）。

## CONFIG モードのコマンド（BLE UART）

Mac 側から Nordic UART Service（UUID: `6E400001-…-E50E24DCCA9E`、RX `…0002…`、TX `…0003…`）へ送信：

| コマンド | 動作 |
|---|---|
| `setphrase:Hello EHR` | フレーズをNVS保存→即アクティブ（ASCII） |
| `clearphrase` | 上書きを削除→ `DEFAULT_PHRASE` に復帰 |
| `get` | 現在のフレーズを Notify で応答 |
| `help` | コマンド一覧を応答 |

※フレーズは **ASCII（英数字・US記号・空白・タブ・改行）** のみ対応。

## ビルド / 書き込み

```bash
# コンパイル（検証）
arduino-cli compile --fqbn esp32:esp32:m5stack_core2 harness-node

# USB 書き込み（ポートは環境に合わせて）
arduino-cli upload  --fqbn esp32:esp32:m5stack_core2 \
                    --port /dev/cu.SLAB_USBtoUART harness-node
```

シリアルモニタ（115200 baud）で動作ログ・受信コマンドを確認できます。

## カスタマイズ（`config.h`）

| 項目 | 既定値 | 説明 |
|---|---|---|
| `DEFAULT_PHRASE` | `"Hello EHR"` | 未登録時のフレーズ |
| `BLE_NAME_TYPE` / `BLE_NAME_CONFIG` | `Harness Node` / `… [CFG]` | 各モードのBLEデバイス名 |
| `TYPE_CHAR_DELAY_MS` | `20` | 1文字あたりの送信間隔 |
| `HOLD_THRESHOLD_MS` | `500` | Aボタン長押し判定時間 |

## 注意点・制約

- **モード切替の堅牢性**: `BLEDevice::deinit(true)` → 再 `init` で排他制御しています。実機で切替が不安定な場合は切替後の再接続タイミングの調整余地があります。
- **HID レポート**: Report ID 1 を使用し、notify 値の先頭に Report ID バイトを含める方式（esp32 core バンドルの `Server_Gamepad` サンプル準拠）。万が一ホストでキーが反映されない場合は、レポート構成の調整が必要です。
- **セキュリティ**: Just-Works + ボンディングでペアリングします（PIN 入力不要）。
- **CONFIG 中の A**: 無効（画面にヒント表示）。

## ライセンス

Public Domain
