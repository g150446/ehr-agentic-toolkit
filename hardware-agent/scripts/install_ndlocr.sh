#!/bin/bash
# ndlocr-lite をクローンし、依存パッケージをインストールするスクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NDBOCR_DIR="$PROJECT_ROOT/ndlocr-lite"

# ── 仮想環境確認 ───────────────────────────────────────────────
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    PYTHON="$PROJECT_ROOT/venv/bin/python"
    PIP="$PROJECT_ROOT/venv/bin/pip"
else
    echo "Error: Virtual environment not found. Run ./scripts/setup_automation.sh first."
    exit 1
fi

# ── クローン ────────────────────────────────────────────────────
if [ -d "$NDBOCR_DIR" ]; then
    echo "[ndlocr-lite] 既に存在します: $NDBOCR_DIR"
    echo "[ndlocr-lite] git pull で更新します..."
    cd "$NDBOCR_DIR"
    git pull
else
    echo "[ndlocr-lite] クローン中: https://github.com/ndl-lab/ndlocr-lite ..."
    cd "$PROJECT_ROOT"
    git clone https://github.com/ndl-lab/ndlocr-lite.git
    cd "$NDBOCR_DIR"
fi

# ── 依存パッケージインストール ──────────────────────────────────
echo "[ndlocr-lite] 依存パッケージをインストール中..."
if [ ! -f "$NDBOCR_DIR/requirements.txt" ]; then
    echo "Error: requirements.txt が見つかりません: $NDBOCR_DIR/requirements.txt"
    echo ""
    echo "リポジトリのクローンが不完全な可能性があります。"
    echo "以下のコマンドで再クローンしてください："
    echo "  rm -rf $NDBOCR_DIR"
    echo "  ./scripts/install_ndlocr.sh"
    echo ""
    echo "または手動で requirements.txt を確認："
    echo "  ls -la $NDBOCR_DIR/"
    exit 1
fi
$PIP install -r "$NDBOCR_DIR/requirements.txt"

# ── モデルファイル確認 ───────────────────────────────────────────
MODEL_DIR="$NDBOCR_DIR/src/model"
CONFIG_DIR="$NDBOCR_DIR/src/config"

REQUIRED_MODELS=(
    "deim-s-1024x1024.onnx"
    "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"
)
REQUIRED_CONFIGS=(
    "ndl.yaml"
    "NDLmoji.yaml"
)

MISSING=0

echo "[ndlocr-lite] モデルファイルを確認中..."
for model in "${REQUIRED_MODELS[@]}"; do
    if [ -f "$MODEL_DIR/$model" ]; then
        echo "  ✅ $model"
    else
        echo "  ❌ $model が見つかりません"
        MISSING=$((MISSING + 1))
    fi
done

for config in "${REQUIRED_CONFIGS[@]}"; do
    if [ -f "$CONFIG_DIR/$config" ]; then
        echo "  ✅ $config"
    else
        echo "  ❌ $config が見つかりません"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "Error: 必要なファイルが $MISSING 個見つかりません。"
    echo "git clone に問題がある可能性があります。"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ ndlocr-lite のインストールが完了しました"
echo "   パス: $NDBOCR_DIR"
echo "=========================================="
echo ""
echo "start_servers.sh を実行してください："
echo "  ./scripts/start_servers.sh"
