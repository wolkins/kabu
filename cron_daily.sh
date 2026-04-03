#!/bin/bash
# さくらVPS用 日次バッチスクリプト
# crontab: 0 6 * * 1-5 /path/to/kabu/cron_daily.sh >> /path/to/kabu/logs/cron.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ログディレクトリ
mkdir -p "$SCRIPT_DIR/logs"

echo "========================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') バッチ開始"
echo "========================================"

# Python仮想環境を有効化
source "$SCRIPT_DIR/.venv/bin/activate"

# バッチ実行（Optuna無しで高速化、週1でOptunaを回す場合は --optuna を付ける）
python batch_predict.py

# HF Spaceにデプロイ
bash deploy_to_hf.sh

echo "========================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') バッチ完了"
echo "========================================"
