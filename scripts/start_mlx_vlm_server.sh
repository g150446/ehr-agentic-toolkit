#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../venv/bin/activate"
exec python -m mlx_vlm.server \
    --model mlx-community/gemma-4-e2b-it-4bit \
    --host 127.0.0.1 \
    --port 8181
