#!/bin/bash
# さくらVPS用 バッチスクリプト
#
# cron設定例:
#   # 3時間ごと: データ取得+予測のみ（学習スキップ、軽量）
#   0 9,12,15,18 * * 1-5 /home/ubuntu/work/kabu/cron_daily.sh --skip-train
#
#   # 毎朝6時: データ取得+通常学習+予測
#   0 6 * * 1-5 /home/ubuntu/work/kabu/cron_daily.sh
#
#   # 週1月曜5時: Optuna最適化+予測
#   0 5 * * 1 /home/ubuntu/work/kabu/cron_daily.sh --optuna
#
# ログ: >> /home/ubuntu/work/kabu/logs/cron.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ログディレクトリ
mkdir -p "$SCRIPT_DIR/logs"

echo "========================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') バッチ開始 (引数: $@)"
echo "========================================"

# Python仮想環境を有効化
source "$SCRIPT_DIR/.venv/bin/activate"

# バッチ実行（引数をそのまま渡す）
python batch_predict.py "$@"

# HF Spaceにデプロイ
bash deploy_to_hf.sh

echo "========================================"
echo "$(date '+%Y-%m-%d %H:%M:%S') バッチ完了"
echo "========================================"
