#!/bin/bash
# 予測結果をHugging Face Spaceにデプロイするスクリプト
# HF APIでファイルアップロード（Dockerリビルドを回避）
#
# 初回のみ: app.py等のコード変更時は git push が必要
# 日常運用: このスクリプトでデータのみ更新（リビルド不要）

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREDICTIONS_DIR="$SCRIPT_DIR/outputs/predictions"
HF_SPACE_DIR="$SCRIPT_DIR/hf_space"

# HFの設定
HF_REPO="fukuhaera/kabu-dashboard"
HF_REPO_TYPE="space"

# トークンファイルから読み込み（なければ環境変数）
TOKEN_FILE="$SCRIPT_DIR/.hf_token"
if [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN=$(cat "$TOKEN_FILE")
elif [ -z "$HF_TOKEN" ]; then
    echo "エラー: HFトークンが設定されていません。"
    echo "以下のいずれかで設定してください:"
    echo "  echo 'hf_xxxx' > $TOKEN_FILE"
    echo "  export HF_TOKEN=hf_xxxx"
    exit 1
fi

# 予測結果が存在するか確認
if [ ! -f "$PREDICTIONS_DIR/latest.json" ]; then
    echo "エラー: 予測結果が見つかりません。先に batch_predict.py を実行してください。"
    exit 1
fi

echo "HF SpaceにAPIでファイルアップロード中..."

# latest.json をアップロード
curl -s -X PUT \
    "https://huggingface.co/api/spaces/${HF_REPO}/blob/main/data/latest.json" \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -H "Content-Type: application/json" \
    --data-binary @"$PREDICTIONS_DIR/latest.json" \
    -o /dev/null -w "  latest.json: HTTP %{http_code}\n"

# chart_data.json をアップロード
if [ -f "$PREDICTIONS_DIR/chart_data.json" ]; then
    curl -s -X PUT \
        "https://huggingface.co/api/spaces/${HF_REPO}/blob/main/data/chart_data.json" \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        -H "Content-Type: application/json" \
        --data-binary @"$PREDICTIONS_DIR/chart_data.json" \
        -o /dev/null -w "  chart_data.json: HTTP %{http_code}\n"
fi

echo "デプロイ完了!（リビルドなし）"

# デプロイ後にSpaceを起こす（スリープ回避）
echo "Spaceにアクセスしてウォームアップ中..."
curl -s -o /dev/null -w "  ウォームアップ: HTTP %{http_code}\n" \
    "https://${HF_REPO//\//-}.hf.space/" \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    --max-time 30 || true
