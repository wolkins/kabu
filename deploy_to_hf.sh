#!/bin/bash
# 予測結果をHugging Face Spaceにデプロイするスクリプト
# 使い方: ./deploy_to_hf.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HF_SPACE_DIR="$SCRIPT_DIR/hf_space"
PREDICTIONS_DIR="$SCRIPT_DIR/outputs/predictions"

# 予測結果が存在するか確認
if [ ! -f "$PREDICTIONS_DIR/latest.json" ]; then
    echo "エラー: 予測結果が見つかりません。先に batch_predict.py を実行してください。"
    exit 1
fi

# HF Spaceのdataディレクトリに予測結果をコピー
echo "予測結果をHF Spaceにコピー中..."
cp "$PREDICTIONS_DIR/latest.json" "$HF_SPACE_DIR/data/"
cp "$PREDICTIONS_DIR/chart_data.json" "$HF_SPACE_DIR/data/" 2>/dev/null || true

# HF Spaceリポジトリにpush
cd "$HF_SPACE_DIR"

# HFリポジトリが初期化されていない場合のガード
if [ ! -d ".git" ]; then
    echo "エラー: HF Spaceリポジトリが初期化されていません。"
    echo "以下を実行してください:"
    echo "  cd $HF_SPACE_DIR"
    echo "  git init"
    echo "  git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>"
    exit 1
fi

git add -A
git diff --cached --quiet && echo "変更なし。スキップします。" && exit 0

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "予測更新: $TIMESTAMP"
git push origin main

echo "デプロイ完了!"
