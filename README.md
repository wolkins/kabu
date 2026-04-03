# 日本株 AI 予測システム

LightGBM を用いて東証上場 25 銘柄の 5 営業日後アルファ（対日経 225 超過リターン）を予測し、Streamlit ダッシュボードで可視化するシステム。

## 特徴

- **アルファ予測**: 日経 225 をベンチマークとした超過リターンの上昇/下落を二値分類
- **豊富な特徴量**: テクニカル指標 + 動的ファンダメンタルズ + クロスマーケット + カレンダー + レジーム + クロスセクション（銘柄間相対値）
- **2 種類のモデル**: 銘柄別個別モデル / 全銘柄一括クロスセクションモデル
- **SHAP 説明性**: 予測根拠を特徴量貢献度として可視化
- **インタラクティブダッシュボード**: ローソク足・MACD・RSI・出来高チャート + AI 予測 + 全銘柄一括サマリー

## プロジェクト構成

```
kabu/
├── app.py                     # Streamlit ダッシュボード
├── run_pipeline.py            # 一括実行（データ取得→学習→予測）
├── config/
│   └── config.yaml            # 銘柄リスト・モデルパラメータ等
├── src/
│   ├── data/
│   │   └── fetch.py           # yfinance データ取得・キャッシュ
│   ├── features/
│   │   ├── technical.py       # テクニカル指標（RSI, MACD, BB, OBV, MFI 等）
│   │   ├── fundamental.py     # 動的ファンダメンタルズ（PER, PBR, 配当利回り）
│   │   └── builder.py         # 特徴量構築パイプライン
│   ├── models/
│   │   └── train.py           # LightGBM 学習・予測・SHAP 分析
│   └── utils/
│       └── config.py          # 設定ファイル読み込み
├── data/
│   ├── raw/                   # yfinance 生データ（parquet）
│   └── processed/             # 特徴量加工済みデータ
├── models/                    # 学習済みモデル（.txt）
└── requirements.txt
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使い方

### パイプライン一括実行

データ取得からモデル学習・予測まで一括で実行:

```bash
python run_pipeline.py
```

### ダッシュボード起動

```bash
streamlit run app.py
```

## 対象銘柄

トヨタ自動車、ソニーグループ、ソフトバンクグループ、三菱 UFJ フィナンシャル、キーエンス、大和ハウス工業、味の素、日本たばこ産業、信越化学工業、武田薬品工業、オリエンタルランド、日本製鉄、日立製作所、村田製作所、三菱重工業、キヤノン、任天堂、伊藤忠商事、三菱商事、オリックス、東京海上 HD、日本郵船、ANA ホールディングス、日本電信電話、ファーストリテイリング

## 特徴量一覧

| カテゴリ | 特徴量 |
|---|---|
| テクニカル | RSI, MACD（3 系列）, ボリンジャーバンド（幅・%B）, SMA 比率（5/20/60 日）, OBV（正規化）, MFI |
| ファンダメンタルズ | 動的 PER, Forward PER, PBR, 配当利回り（終値 / EPS 等で日次算出） |
| 価格変動 | 日次リターン, High-Low 比率, 20 日ボラティリティ |
| ラグ | 1/2/3/5/10 日リターン・出来高変化率 |
| クロスマーケット | 日経 225・S&P500・USD/JPY のリターン・ボラティリティ |
| カレンダー | 曜日, 月, 四半期, 月初/月末フラグ |
| レジーム | ボラティリティレジーム, トレンド強度, リターン z-score |
| クロスセクション | 上記主要指標の銘柄間ランク・z-score |

## モデル

- **アルゴリズム**: LightGBM（二値分類）
- **検証**: Walk-forward Cross-Validation（5 分割）
- **評価指標**: AUC, Accuracy
- **ターゲット**: 5 営業日後の対日経 225 アルファが正なら 1、負なら 0

## 技術スタック

- Python 3.11
- LightGBM / scikit-learn / SHAP
- yfinance / ta（テクニカル指標ライブラリ）
- Streamlit / Plotly
