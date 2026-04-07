#!/bin/bash
# 予測結果をHugging Face Spaceにデプロイするスクリプト
# huggingface_hub Pythonライブラリを使用（Dockerリビルド回避）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# venvがあれば有効化
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

python deploy_to_hf.py

# デプロイ後にSpaceを起こす（スリープ回避）
echo "Spaceにアクセスしてウォームアップ中..."
curl -s -o /dev/null -w "  ウォームアップ: HTTP %{http_code}\n" \
    "https://fukuhaera-kabu-dashboard.hf.space/" \
    --max-time 30 || true
