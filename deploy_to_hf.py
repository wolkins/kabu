"""HF Spaceに予測結果をアップロード（Dockerリビルド回避）

huggingface_hub の upload_file API を使用。Git LFSも自動対応。
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = SCRIPT_DIR / "outputs" / "predictions"
TOKEN_FILE = SCRIPT_DIR / ".hf_token"

HF_REPO = "fukuhaera/kabu-dashboard"
HF_REPO_TYPE = "space"


def get_token() -> str:
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("エラー: HFトークンが設定されていません。")
        print(f"  echo 'hf_xxxx' > {TOKEN_FILE}")
        print("  または export HF_TOKEN=hf_xxxx")
        sys.exit(1)
    return token


def main():
    latest = PREDICTIONS_DIR / "latest.json"
    chart = PREDICTIONS_DIR / "chart_data.json"

    if not latest.exists():
        print(f"エラー: 予測結果が見つかりません: {latest}")
        sys.exit(1)

    api = HfApi(token=get_token())

    print(f"HF Spaceにファイルアップロード中: {HF_REPO}")

    # latest.json
    api.upload_file(
        path_or_fileobj=str(latest),
        path_in_repo="data/latest.json",
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        commit_message="Update predictions",
    )
    size_kb = latest.stat().st_size / 1024
    print(f"  ✓ data/latest.json ({size_kb:.1f} KB)")

    # chart_data.json
    if chart.exists():
        api.upload_file(
            path_or_fileobj=str(chart),
            path_in_repo="data/chart_data.json",
            repo_id=HF_REPO,
            repo_type=HF_REPO_TYPE,
            commit_message="Update chart data",
        )
        size_kb = chart.stat().st_size / 1024
        print(f"  ✓ data/chart_data.json ({size_kb:.1f} KB)")

    print("デプロイ完了!（リビルドなし）")


if __name__ == "__main__":
    main()
