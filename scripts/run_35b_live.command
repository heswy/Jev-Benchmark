#!/bin/bash
# Live eval terminal for JEV-Benchmark (Qwen3.5-35B full 1050)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

echo "=============================================="
echo " JEV-Benchmark · Qwen/Qwen3.5-35B-A3B"
echo " 1050 samples · thinking=OFF · temp=0"
echo " incremental checkpoint ON"
echo "=============================================="
echo "start $(date '+%H:%M:%S')"

PY="${MIMO_PYTHON:-python3}"

"$PY" -u -m jevbench.runner \
  --models qwen3.5-35b \
  --datasets banking77 clinc150 sst5 boolq agnews \
  --concurrency 10 \
  --tag qwen35-35b-full-1050

echo
echo "done $(date '+%H:%M:%S')"
echo "raw: results/raw/*qwen3.5-35b*"
echo "summary: results/summary_latest.json"
echo "按回车关闭窗口..."
read -r
