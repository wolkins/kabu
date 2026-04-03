"""一括実行スクリプト: データ取得 → 特徴量構築 → モデル学習 → 予測"""

from src.data.fetch import fetch_all
from src.models.train import train_model, train_cross_sectional, predict_latest
from src.utils.config import load_config, ticker_list, ticker_display


def main():
    config = load_config()

    print("=" * 60)
    print("Step 1: データ取得")
    print("=" * 60)
    fetch_all(config)

    print("\n" + "=" * 60)
    print("Step 2: 個別モデル学習")
    print("=" * 60)
    for ticker in ticker_list(config):
        print(f"\n--- {ticker_display(config, ticker)} ---")
        train_model(ticker, config)

    print("\n" + "=" * 60)
    print("Step 3: クロスセクション学習（全銘柄一括 + Optuna最適化）")
    print("=" * 60)
    train_cross_sectional(config, use_optuna=True, n_trials=50)

    print("\n" + "=" * 60)
    print("Step 4: 最新予測（クロスセクションモデル）")
    print("=" * 60)
    horizon = config["model"]["target_horizon"]
    use_alpha = config["model"].get("benchmark") is not None
    prob_label = "市場超過確率" if use_alpha else "上昇確率"
    for ticker in ticker_list(config):
        result = predict_latest(ticker, config)
        print(f"  {ticker_display(config, ticker)}: {result['prediction']} "
              f"({prob_label}: {result['prediction_proba']:.1%}, "
              f"{horizon}日後, モデル: {result['model_type']})")

    print("\n" + "=" * 60)
    print("完了! ダッシュボードを起動するには:")
    print("  .venv/bin/streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
