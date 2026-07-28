# harness-node 進捗・TODO

M5Core2 を「BLE HIDキーボード（フレーズ送信）」兼「BLE UART（フレーズ設定）」とし、
物理ボタンBで2役割を完全切替するファームウェア。

---

## ✅ 完了したこと

### ファイル構成
`hardware-agent/harness-node/`
- `harness-node.ino` — メイン（モード管理 / B切替 / A長押しHID送信 / CONFIGのUART受信 / 画面描画）
- `config.h` — `DEFAULT_PHRASE` / デバイス名 / 送信間隔 / 長押し閾値
- `README.md` — ビルド手順・コマンド・モード・注意点

### 実装仕様
| 項目 | 内容 |
|---|---|
| 対象HW | M5Core2（ESP32 classic） |
| BLEスタック | esp32 core 3.3.10 バンドルの NimBLE（`BLEHIDDevice` 使用）。**外部ライブラリ不要** |
| FQBN | `esp32:esp32:m5stack_core2` |
| モード切替 | Bクリック → `BLEDevice::deinit(true)` → 反対役割で再init（TYPE⇄CONFIG 完全排他） |
| TYPE（既定） | BLE HID Keyboard(HOGP)。A長押しでアクティブフレーズを1文字ずつ送信 |
| CONFIG | BLE UART（Nordic UART Service）。`setphrase:` / `clearphrase` / `get` / `help` |
| 永続化 | `Preferences`(NVS) にフレーズ上書き保存、未設定時は `DEFAULT_PHRASE` |
| フレーズ | ASCII（英数字・US記号）限定 |
| 画面 | タイトル / モードバッジ / BLE接続状態 / アクティブフレーズ / ボタン押下ハイライト |

### 解決した不具合（2件）
1. **起動パニック（LoadProhibited）**
   - 原因: `hid->manufacturer(String)`（setter単独）は、getter `manufacturer()` で特性を**生成した後でないと**未初期化ポインタを参照してクラッシュ。
   - 対策: バンドルの gamepad サンプルと同じ **`hid->manufacturer()->setValue(...)`**（getter→setValue）に修正。
2. **A長押しで文字が入力されない**
   - 原因: HIDレポート通知値の先頭に Report ID バイトを付加していた。HOGP仕様では Report Reference descriptor がIDを示すため**値にIDは含めない**。先頭 `0x01` が修飾子として解釈され `Ctrl+キー` になり可視文字が出なかった。
   - 対策: レポート構造体から `reportId` を除去し、8バイト値 `[modifiers, reserved, keys[6]]` を送るよう修正。

### 検証結果
- `arduino-cli compile` … EXIT=0 / 警告エラー0件（Flash 20% / RAM 1%）
- 書込 … ハッシュ検証OK・ハードリセットOK
- ブートログ … パニック解消、`TYPE advertising started` → `>> TYPE mode active` → `[BLE] connected`
- 動作 … **A長押しで `Hello EHR` が入力されることを確認**

### 環境メモ
- デバッグ用に `hardware-agent/venv` へ `pyserial` を追加インストール済み（不要なら `venv/bin/pip uninstall pyserial`）。
- シリアル読取には `esptool run` による reset-to-run を使用（`cu.*` ポートはオープンで自動リセットしないため。DTR/RTSの手動トグルはダウンロードモードに入ってしまうので注意）。

---

## 📋 次に行うべきこと

### 優先度：高
- [ ] **CONFIGモードの動作確認**
  - Bクリックで画面が橙`CONFIG`に切替されるか
  - Macで `Harness Node [CFG]` をBLE UART接続（NUS: `6E400001-…-E50E24DCCA9E`）
  - `setphrase:<新フレーズ>` 送信 → `get` で確認 → BでTYPEに戻す → A長押しで反映されているか
  - 再起動後もNVSに保持されるか
- [ ] **CONFIG送信ヘルパースクリプト作成**（`hardware-agent/scripts/`）
  - `wireless-input-bridge/scripts/run_ble_test.sh` 相当。`bleak` 等でNUSに接続し `setphrase:` 等を送るPythonラッパ。Macから1コマンドでフレーズ変更できるように。

### 優先度：中
- [ ] **モード切替の堅牢性検証**（README「注意点」に明記済み）
  - TYPE⇄CONFIGを繰り返した際の `deinit`/再init の安定性、ホスト再接続タイミングの確認。必要なら切替後に広告再開のディレイ調整。
- [ ] **実運用ホストでの検証**（Windows EHR PC 等）
  - ペアリング・文字入力の安定性、修飾子解放漏れ（キーが stuck しないか）の確認。
- [ ] **長押し閾値の体感調整**（現在 `HOLD_THRESHOLD_MS=500`、`config.h`）

### 優先度：低（必要なら）
- [ ] B/Cボタンへの追加割当（複数フレーズ切替、Enter送信 等）
- [ ] フレーズの多段登録（NCSに複数キー保存）
- [ ] バッテリ残量の実値反映（現在 `setBatteryLevel(100)` 固定）
- [ ] 画面UIのブラッシュアップ（フレーズ長時の折返し・スクロール）

---

## 🔧 ビルド／書込（参照用）
```bash
cd hardware-agent
arduino-cli compile --fqbn esp32:esp32:m5stack_core2 harness-node
arduino-cli upload  --fqbn esp32:esp32:m5stack_core2 \
                    --port /dev/cu.usbserial-54FC0189691 harness-node
```
