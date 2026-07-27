# Plan: HDMI キャプチャ CXR → MedGemma コメント CLI

**実装済み**（`automation/medgemma_cxr_comment.py` + `automation/cxr_crop.py`）。以下は最終仕様。

## 目的

HDMI キャプチャした画面に含まれる**胸部レントゲン（CXR）領域**を切り出し、MedGemma で短い所見コメントを生成し、**ターミナル上に表示**するスクリプトを `automation/` に追加する。

教育・実験用。診断用途ではない。

## 配置・実行

| 項目 | 内容 |
|------|------|
| 新規スクリプト | `automation/medgemma_cxr_comment.py` |
| 実行 | `python -m automation.medgemma_cxr_comment [options]` |
| 本プラン | `automation/MEDGEMMA_CXR_COMMENT_PLAN.md` |

既存の `gui_image_analyzer` / `monitor_stream` と同パターン（`python -m automation.xxx`）。

## 参考にする既存実装

| 既存 | 流用点 |
|------|--------|
| `cxr_crop.crop_cxr` | 胸部 X 線 ROI 切り出し + 焼き込み注記の黒塗り（実装済み。上記フローの Step 2 全体をこれに置き換え） |
| `screen_analyzer.capture_screen` | HDMI 1フレーム取得（リングバッファ flush 付き） |
| `config.AutomationConfig` | `capture_device_index` / 解像度 / 出力ディレクトリ |
| `monitor_stream` | `argparse` CLI、デバイス index、キャプチャ初期化 |
| `gui_image_analyzer` | モジュール入口・ログ・結果の stdout 表示 |
| `utils.setup_logging` | ログ設定 |
| medgemma notebook（`cxr_anatomy_localization_...`） | プロンプト + `Comment:` 抽出ロジック |

関連リポジトリ側の実績:

- モデル: `google/medgemma-1.5-4b-it`（`~/.cache/huggingface/hub` にフルダウンロード済み・HF トークン設定済みで確認済み）
- コメント形式: `Comment: ...`（1–2 文）
- 実行環境: `hardware-agent/venv` の torch 2.12 / transformers 4.57.6 / huggingface_hub 0.36.2 でそのまま動作。`accelerate` は不要（`device_map="auto"` ではなく `pipeline(..., device=<device>)` を使用）
- 実機（Apple Silicon, MPS）で実推論を確認済み。モデルロード ~13秒、1推論 ~10秒

## 処理フロー

1. **入力取得**（`--image` と `--capture`/`--device` は排他）
   - `--image PATH` … **既に切り出し済みの CXR 画像**をそのまま使う（crop 処理なし）
   - `--capture PATH` … crop 前の画面キャプチャ画像。`crop_cxr()` を通す
   - 引数なし（`--device N`） … `capture_screen()` で HDMI キャプチャ → `crop_cxr()` を通す
2. **胸部 X 線 ROI 切り出し**（`--image` 使用時はスキップ） — `automation/cxr_crop.py` の `crop_cxr()` に委譲（実装・検証済み）
   - 既定: 青いビューア選択枠 → 枠内の無彩色・高輝度ブロブを自動検出（`find_viewer_viewport` → `find_cxr_region`）。
     枠が見つからない場合は全画面に対して同じブロブ検出を行うフォールバックあり
   - 焼き込み注記（体位ラベル等）は `mask_overlay_annotations()` で常時黒塗り（解剖構造と非連結の画素を除去）
   - `--full` … 全画面をそのまま CXR として扱う（`crop_cxr(image, full=True)`）
   - `--roi x,y,w,h` … 手動 ROI（`crop_cxr(image, roi=(x,y,w,h))`）
   - 検出失敗時は `None` を返す（`--full` / `--roi` を案内）
3. **MedGemma 推論**
   - モデル: `google/medgemma-1.5-4b-it`
   - プロンプト: **肺結節（pulmonary nodule/mass）の検出を主眼**とし、有無を明示的に述べさせる（`Comment:` 行に集約）。`--prompt` で上書き可能
   - bbox は任意（主目的はコメント）
4. **ターミナル表示**

```text
=== MedGemma CXR Comment ===
source: hdmi device=0 | crop=x,y,w,h | size=WxH
Comment: A small nodule is visible in the right upper lung field.
```

5. **任意保存**
   - `--save-dir DIR` で全画面キャプチャ・crop・コメントテキストを保存

## CLI 案

```bash
# 既に切り出し済みの CXR 画像を直接読影
python -m automation.medgemma_cxr_comment --image captures/cxr.png

# crop 前の画面キャプチャを渡す → 自動 crop → コメント表示
python -m automation.medgemma_cxr_comment --capture captures/xray.jpg

# HDMI 1回キャプチャ → 自動 crop → コメント表示
python -m automation.medgemma_cxr_comment

# 全画面を CXR として扱う / 手動 ROI（--capture, --device 用）
python -m automation.medgemma_cxr_comment --capture captures/xray.jpg --full
python -m automation.medgemma_cxr_comment --device 0 --roi 200,80,900,900

# 連続（N 秒ごと、Ctrl+C で終了）。モデルは保持して再利用（--image とは併用不可）
python -m automation.medgemma_cxr_comment --capture captures/xray.jpg --watch 10

# 結果保存
python -m automation.medgemma_cxr_comment --capture captures/xray.jpg --save-dir ./automation_outputs/medgemma_cxr
```

### 主なオプション

| オプション | 説明 |
|------------|------|
| `--image PATH` | 既に切り出し済みの CXR 画像。そのまま読影（crop なし） |
| `--capture PATH` | crop 前の画面キャプチャ画像。`crop_cxr()` を通してから読影 |
| `--device N` | キャプチャデバイス index（default: config）。`--image`/`--capture` 省略時のデフォルト経路 |
| `--roi x,y,w,h` / `--full` / `--no-mask-overlay` | `--capture`/`--device` 使用時に `crop_cxr()` へ透過 |
| `--model ID` | HF モデル ID（default: `google/medgemma-1.5-4b-it`） |
| `--compute-device {auto,mps,cuda,cpu}` | 推論デバイス（default: auto = mps > cuda > cpu） |
| `--max-new-tokens N` | 生成トークン数上限（default: 300） |
| `--prompt TEXT` | 既定プロンプトの上書き |
| `--watch SEC` | 連続実行の間隔秒（`--image` とは併用不可） |
| `--save-dir DIR` | キャプチャ・crop・コメント保存先 |
| `--env-file PATH` | `.env` パス |
| `--debug` / `--debug-dir DIR` | デバッグログ + crop 中間画像保存 |

## 依存関係

既存（hardware-agent）のみで動作、追加インストール不要:

- `opencv-python`, `numpy`, `torch` (2.12), `transformers` (4.57.6), `huggingface_hub` (0.36.2), `pillow`, `python-dotenv`

`accelerate` は意図的に不要（`device_map="auto"` を避け、`pipeline(..., device=<device>)` で単一デバイスに直接配置）。
`bitsandbytes` も不要（4B モデルは bf16 のまま Apple Silicon の統合メモリに収まる）。

- MedGemma は gated → Hugging Face ログイン済み前提（このマシンでは確認済み: `~/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it` に8GB完全ダウンロード済み、`huggingface_hub.get_token()` がトークンを返す）

## 実装ファイル

1. **実装済み** `automation/medgemma_cxr_comment.py`
   - capture / crop（`cxr_crop.crop_cxr()` に委譲） / MedGemma load+infer / CLI / ターミナル出力
2. **実装済み** `automation/README.md` に Usage 追記（`## MedGemma CXR Comment` セクション）

## 実装上の注意

- 自動 crop は EHR レイアウト依存。失敗時は `--full` または `--roi` を案内
- 出力・docstring に「診断用途ではない」を明記
- クロップ失敗（非CXR画像等）はモデルロード前に検出し、無駄なロード時間を発生させない（単発実行時。`--watch` はモデルを保持しつつ毎サイクル再取得を試みる）
- 初回モデルロードは実測 ~13秒（Apple Silicon MPS、bf16、キャッシュ済みモデル）。`--watch` ではプロセス内でモデルを保持・再利用（1回のみロードすることを確認済み）
- HDMI 未接続時は `capture_screen()` からの明確なエラーメッセージ（warmup スクリプト `scripts/warmup_hdmi.py` を案内）

## 受け入れ条件

- [x] `python -m automation.medgemma_cxr_comment --image <cxr.png>` でコメントが stdout に出る
- [x] `--capture <screen.png>` で crop → コメントが出る（HDMI 接続時は `--device` で同様）
- [x] `--full` / `--roi` が動作する
- [x] コメントが `Comment:` 形式で短文である
- [x] 診断用途でない旨が出力に含まれる（`[教育・実験用。診断用途ではありません。]`）

## 関連パス

- hardware-agent: `/Users/kazami/projects/ehr/toolkit-dev/hardware-agent`
- MedGemma 実験: `/Users/kazami/projects/medgemma`
- HF モデル: `google/medgemma-1.5-4b-it`
