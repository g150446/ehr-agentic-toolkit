#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../venv/bin/activate"

if [[ $# -gt 0 ]]; then
    MODEL_INPUT="$1"
    shift
else
    MODEL_INPUT="${MLX_VLM_SERVER_MODEL:-qwen}"
fi

case "$MODEL_INPUT" in
    qwen)
        MODEL="mlx-community/Qwen3.5-4B-MLX-4bit"
        ;;
    gemma)
        MODEL="mlx-community/gemma-4-e2b-it-4bit"
        ;;
    *)
        MODEL="$MODEL_INPUT"
        ;;
esac

exec python -m mlx_vlm.server \
    --model "$MODEL" \
    --host "${MLX_VLM_SERVER_HOST:-127.0.0.1}" \
    --port "${MLX_VLM_SERVER_PORT:-8181}" \
    "$@"
