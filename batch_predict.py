"""日次バッチ: データ取得 → 学習 → 予測 → JSON保存"""

import json
import traceback
from datetime import datetime
from pathlib import Path

from src.data.fetch import fetch_all
from src.features.builder import build_features, build_all_features, get_feature_columns
from src.models.train import train_cross_sectional, predict_latest, MODEL_DIR
from src.utils.config import load_config, ticker_list, ticker_name, ticker_display

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "predictions"


def run_batch(use_optuna: bool = False, n_trials: int = 50,
              skip_train: bool = False):
    """バッチ実行: データ取得 → 学習 → 全銘柄予測 → JSON保存

    skip_train: Trueなら学習をスキップし、既存モデルで予測のみ実行
    """
    config = load_config()
    tickers = ticker_list(config)
    horizon = config["model"]["target_horizon"]
    use_alpha = config["model"].get("benchmark") is not None

    # Step 1: データ取得
    print("=" * 60)
    print("Step 1: データ取得")
    print("=" * 60)
    fetch_all(config)

    # Step 2: クロスセクション学習
    cv_auc = 0.0
    if skip_train:
        cross_path = MODEL_DIR / "cross_sectional_lgbm.txt"
        if not cross_path.exists():
            print("\n  モデルが存在しないため、学習を実行します...")
            skip_train = False
        else:
            print("\n  学習スキップ（既存モデルを使用）")
    if not skip_train:
        print("\n" + "=" * 60)
        print("Step 2: クロスセクション学習")
        print("=" * 60)
        result = train_cross_sectional(config, use_optuna=use_optuna, n_trials=n_trials)
        cv_auc = result["scores"]["auc"].mean()
        print(f"  CV Mean AUC: {cv_auc:.4f}")

    # Step 3: 全銘柄予測
    print("\n" + "=" * 60)
    print("Step 3: 全銘柄予測")
    print("=" * 60)

    # 特徴量を1回だけ構築
    df_all = build_all_features(config)

    predictions = []
    errors = []
    for ticker in tickers:
        try:
            r = predict_latest(ticker, config, df_all_cache=df_all)
            conf_value = abs(r["prediction_proba"] - 0.5) * 200

            # SHAP上位10を保存
            shap_vals = r["shap_values"]
            if isinstance(shap_vals, list):
                shap_arr = shap_vals[1]
            else:
                shap_arr = shap_vals
            shap_flat = shap_arr.flatten().tolist()
            feature_names = r["feature_names"]
            shap_pairs = sorted(
                zip(feature_names, shap_flat),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:15]

            # 特徴量の値も保存
            feature_values = {
                name: float(r["feature_values"].get(name, 0))
                for name, _ in shap_pairs
            }

            predictions.append({
                "ticker": ticker,
                "name": ticker_name(config, ticker),
                "display": ticker_display(config, ticker),
                "date": str(r["date"].date()),
                "close": float(r["close"]),
                "prediction": r["prediction"],
                "prediction_proba": float(r["prediction_proba"]),
                "confidence": float(conf_value),
                "horizon": r["horizon"],
                "model_type": r.get("model_type", "-"),
                "use_alpha": r.get("use_alpha", False),
                "shap_top15": [
                    {"feature": name, "shap_value": float(val), "feature_value": feature_values[name]}
                    for name, val in shap_pairs
                ],
            })
            print(f"  {ticker_display(config, ticker)}: {r['prediction']} "
                  f"(確率: {r['prediction_proba']:.1%}, 確信度: {conf_value:.1f}%)")
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
            print(f"  {ticker}: エラー - {e}")
            traceback.print_exc()

    # Step 4: JSON保存
    print("\n" + "=" * 60)
    print("Step 4: 予測結果をJSON保存")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at": datetime.now().isoformat(),
        "horizon": horizon,
        "use_alpha": use_alpha,
        "cv_auc": float(cv_auc),
        "num_tickers": len(tickers),
        "num_predictions": len(predictions),
        "num_errors": len(errors),
        "predictions": predictions,
        "errors": errors,
    }

    # latest.json（常に最新を上書き）
    latest_path = OUTPUT_DIR / "latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  保存: {latest_path}")

    # 日付別にもアーカイブ
    date_str = datetime.now().strftime("%Y%m%d")
    archive_path = OUTPUT_DIR / f"{date_str}.json"
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  保存: {archive_path}")

    # 株価チャート用データも保存（各銘柄の直近500日分のOHLCV+テクニカル指標）
    print("\n  チャート用データを保存中...")
    chart_data = {}
    for ticker in tickers:
        try:
            df = build_features(ticker, config)
            df_recent = df.tail(500)
            cols = ["Open", "High", "Low", "Close", "Volume",
                    "rsi", "macd", "macd_signal", "macd_diff",
                    "bb_high", "bb_mid", "bb_low"]
            # SMA
            for w in config["features"]["sma_windows"]:
                cols.append(f"sma_{w}")
            available_cols = [c for c in cols if c in df_recent.columns]
            chart_df = df_recent[available_cols].copy()
            chart_df.index = chart_df.index.strftime("%Y-%m-%d")
            chart_data[ticker] = chart_df.to_dict(orient="index")
        except Exception as e:
            print(f"    {ticker} チャートデータ保存エラー: {e}")

    chart_path = OUTPUT_DIR / "chart_data.json"
    with open(chart_path, "w", encoding="utf-8") as f:
        json.dump(chart_data, f, ensure_ascii=False)
    print(f"  保存: {chart_path}")

    print("\n" + "=" * 60)
    print(f"完了! {len(predictions)}銘柄の予測を保存しました")
    print("=" * 60)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="日次バッチ予測")
    parser.add_argument("--optuna", action="store_true", help="Optuna最適化を有効化")
    parser.add_argument("--trials", type=int, default=50, help="Optuna試行回数")
    parser.add_argument("--skip-train", action="store_true",
                        help="学習をスキップし既存モデルで予測のみ実行")
    args = parser.parse_args()
    run_batch(use_optuna=args.optuna, n_trials=args.trials,
              skip_train=args.skip_train)
