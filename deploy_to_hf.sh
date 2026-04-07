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
# Privateなのでトークンが必要
TOKEN_FILE="$SCRIPT_DIR/.hf_token"
if [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN=$(cat "$TOKEN_FILE")
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Spaceにアクセスしてウォームアップ中..."
    curl -s -o /dev/null -w "  ウォームアップ: HTTP %{http_code}\n" \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        "https://fukuhaera-kabu-dashboard.hf.space/" \
        --max-time 30 || true
fi
