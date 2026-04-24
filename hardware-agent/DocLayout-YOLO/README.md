# DocLayout-YOLO: Document Layout Analysis with OCR

文書画像のレイアウト解析とOCR（文字認識）を行うツールです。DocLayout-YOLOモデルを使用して、文書内の要素（テキスト、表、図など）を検出し、EasyOCRでテキストを抽出します。

## 環境セットアップ

```bash
# 仮想環境の作成
conda create -n doclayout_yolo python=3.10
conda activate doclayout_yolo
pip install -e .

# または、推論のみが必要な場合
pip install doclayout-yolo
```

## モデルのダウンロード

HuggingFaceから事前学習済みモデルをダウンロードします：

```bash
python download_model.py
```

これにより、`./models/doclayout_docstructbench.pt` が保存されます。

## スクリプト

### 1. `test_with_ocr.py` - 基本レイアウト解析 + OCR

文書画像のレイアウト解析とOCRを実行します。

```bash
python test_with_ocr.py <画像パス> [オプション]
```

**オプション:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model` | モデルファイルのパス | `./models/doclayout_docstructbench.pt` |
| `--output-dir` | 切り出し画像の出力ディレクトリ | `outputs/extracted_regions` |
| `--conf` | 信頼度閾値 | `0.1` |
| `--imgsz` | 推論時の画像サイズ | `1024` |
| `--no-ocr` | OCRをスキップ | - |

**使用例:**

```bash
# 基本
python test_with_ocr.py ./document.png

# フルオプション
python test_with_ocr.py ./document.png --conf 0.2 --imgsz 1024

# OCRなしでレイアウト解析のみ
python test_with_ocr.py ./document.png --no-ocr
```

**出力ファイル:**

| ファイル | 内容 |
|---------|------|
| `outputs/result_<画像名>.jpg` | 検出結果を注釈付きで可視化した画像 |
| `outputs/extracted_regions/` | 切り出された領域画像 |
| `outputs/extracted_text_<画像名>.txt` | OCRで抽出されたテキスト |

---

### 2. `test_with_ocr_positions.py` - 位置情報付きレイアウト解析 + OCR

基本機能に加え、**位置情報の詳細出力**と**読み順ソート**（上→下、左→右）を提供します。

```bash
python test_with_ocr_positions.py <画像パス> [オプション]
```

**オプション:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model` | モデルファイルのパス | `./models/doclayout_docstructbench.pt` |
| `--output-dir` | 切り出し画像の出力ディレクトリ | `outputs/extracted_regions` |
| `--conf` | 信頼度閾値 | `0.1` |
| `--imgsz` | 推論時の画像サイズ | `1024` |
| `--no-ocr` | OCRをスキップ | - |

**使用例:**

```bash
# 基本
python test_with_ocr_positions.py ./document.png

# フルオプション
python test_with_ocr_positions.py ./document.png --conf 0.2 --output-dir ./my_regions

# OCRなし
python test_with_ocr_positions.py ./document.png --no-ocr
```

**出力ファイル:**

| ファイル | 内容 |
|---------|------|
| `outputs/result_with_positions_<画像名>.jpg` | 番号付きで領域を可視化した画像 |
| `outputs/result_detailed_<画像名>.jpg` | 従来の注釈付き画像 |
| `outputs/extracted_regions/` | 切り出された領域画像 |
| `outputs/extracted_text_<画像名>.txt` | 位置情報付きOCR結果 |

## 検出クラス

モデルは以下の要素を検出できます：

- `title`: タイトル
- `text`: テキスト
- `figure`: 図
- `table`: 表
- `header`: ヘッダー
- `footer`: フッター
- `reference`: 参考文献
- `equation`: 数式

## 出力ディレクトリ構成

```
DocLayout-YOLO/
├── outputs/
│   ├── result_document.jpg
│   ├── result_with_positions_document.jpg
│   ├── result_detailed_document.jpg
│   ├── extracted_text_document.txt
│   └── extracted_regions/
│       ├── 01_text.png
│       ├── 02_table.png
│       └── ...
```

## 注意事項

- 出力ファイルはすべて `outputs/` ディレクトリ内に生成されます
- `outputs/` は `.gitignore` に含まれており、リポジトリにはコミットされません
- OCR機能を使用するには `easyocr` のインストールが必要です：`pip install easyocr`
