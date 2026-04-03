# デプロイ手順（さくらVPS + HF Spaces）

## 1. Hugging Face Spaces のセットアップ

### 1-1. HFアカウント作成 & Spaceの作成
1. https://huggingface.co でアカウント作成（未登録の場合）
2. https://huggingface.co/new-space から新しいSpaceを作成
   - Space name: `kabu-dashboard`（任意）
   - Visibility: **Private**
   - SDK: Streamlit
3. 作成後、SpaceのURLをメモ（例: `https://huggingface.co/spaces/yourname/kabu-dashboard`）

### 1-2. HFアクセストークン作成
1. https://huggingface.co/settings/tokens
2. 「New token」→ Write権限で作成
3. トークンをメモ（`hf_xxxx` 形式）

## 2. さくらVPS側のセットアップ

### 2-1. リポジトリのclone
```bash
cd ~/work
git clone <your-github-repo-url> kabu
cd kabu
```

### 2-2. Python環境構築
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2-3. HF Spaceリポジトリの初期化
```bash
cd hf_space

# gitの初期化
git init
git remote add origin https://huggingface.co/spaces/<yourname>/kabu-dashboard

# HF認証設定（トークンをURLに埋め込む方法）
git remote set-url origin https://<yourname>:<hf_token>@huggingface.co/spaces/<yourname>/kabu-dashboard

# 初回push
git add -A
git commit -m "Initial commit"
git push -u origin main
```

### 2-4. 動作確認（手動実行）
```bash
cd ~/work/kabu
source .venv/bin/activate

# バッチ実行
python batch_predict.py

# HFにデプロイ
bash deploy_to_hf.sh
```

HF SpacesのURLにアクセスしてダッシュボードが表示されればOK。

### 2-5. cron登録
```bash
crontab -e
```
以下を追加（平日の朝6時に実行）:
```
0 6 * * 1-5 /home/ubuntu/work/kabu/cron_daily.sh >> /home/ubuntu/work/kabu/logs/cron.log 2>&1
```

### （オプション）週1でOptuna最適化
```
0 5 * * 1 /home/ubuntu/work/kabu/.venv/bin/python /home/ubuntu/work/kabu/batch_predict.py --optuna >> /home/ubuntu/work/kabu/logs/cron.log 2>&1
```

## 3. ファイル構成

```
kabu/
├── batch_predict.py        # バッチスクリプト（データ取得→学習→予測→JSON保存）
├── cron_daily.sh           # cron用ラッパー
├── deploy_to_hf.sh         # HF Spaceへのデプロイ
├── outputs/predictions/    # 予測結果JSON
│   ├── latest.json
│   ├── chart_data.json
│   └── 20260403.json      # 日付別アーカイブ
├── hf_space/               # HF Spaceリポジトリ（別git）
│   ├── app.py              # 表示専用ダッシュボード
│   ├── requirements.txt
│   ├── README.md           # HF Space設定
│   └── data/
│       ├── latest.json     # ← deploy_to_hf.shがコピー
│       └── chart_data.json
└── logs/
    └── cron.log            # バッチログ
```

## トラブルシューティング

- **HF Spaceが起動しない**: `hf_space/README.md` のSDKバージョンを確認
- **git pushが失敗する**: HFトークンの権限(Write)を確認
- **メモリ不足**: `batch_predict.py` でOptuna無しで実行（デフォルト）
- **yfinanceエラー**: ネットワーク一時障害の可能性、cronが次回再実行すれば復旧
