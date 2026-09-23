#!/bin/bash
# Live eval terminal: Qwen/Qwen3.5-122B-A10B on full 1050 gold set
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
echo " JEV-Benchmark · Qwen/Qwen3.5-122B-A10B"
echo " 1050 samples · thinking=OFF · temp=0 · JSON"
echo " incremental checkpoint ON · concurrency 8"
echo "=============================================="
echo "start $(date '+%H:%M:%S')"

PY="${MIMO_PYTHON:-python3}"

"$PY" -u -m jevbench.runner \
  --models qwen3.5-122b \
  --datasets banking77 clinc150 sst5 boolq agnews \
  --concurrency 8 \
  --tag qwen35-122b-full-1050

echo
echo "done $(date '+%H:%M:%S')"
echo "raw: results/raw/*qwen3.5-122b*"
echo "summary: results/summary_latest.json"
echo "按回车关闭窗口..."
read -r
