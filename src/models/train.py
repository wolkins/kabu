"""LightGBMモデルの学習・予測・評価パイプライン"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features.builder import build_features, build_all_features, get_feature_columns
from src.utils.config import load_config, ticker_list

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def _train_lgbm(X_train, y_train, X_val, y_val, params: dict):
    """LightGBM単体の学習"""
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    p = dict(params)
    n_estimators = p.pop("n_estimators", 500)
    early_stopping_rounds = p.pop("early_stopping_rounds", 50)

    model = lgb.train(
        p, train_set,
        num_boost_round=n_estimators,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )
    return model


def train_model(ticker: str, config: dict | None = None) -> dict:
    """Walk-forward検証付きでLightGBMモデルを学習（単一銘柄）"""
    if config is None:
        config = load_config()

    df = build_features(ticker, config)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["target"]

    model_cfg = config["model"]
    tscv = TimeSeriesSplit(n_splits=model_cfg["n_splits"])

    scores = []
    last_model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = _train_lgbm(X_train, y_train, X_val, y_val, model_cfg["lgbm_params"])
        last_model = model

        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        scores.append({"fold": fold, "accuracy": acc, "auc": auc})
        print(f"  Fold {fold}: ACC={acc:.4f}, AUC={auc:.4f}")

    # モデル保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("^", "IDX_").replace("=", "_")
    model_path = MODEL_DIR / f"{safe_name}_lgbm.txt"
    last_model.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": last_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    scores_df = pd.DataFrame(scores)
    print(f"\n  Mean ACC: {scores_df['accuracy'].mean():.4f}")
    print(f"  Mean AUC: {scores_df['auc'].mean():.4f}")

    return {
        "model": last_model, "scores": scores_df,
        "importance": importance, "feature_cols": feature_cols, "df": df,
    }


def train_cross_sectional(config: dict | None = None) -> dict:
    """全銘柄を一括学習（クロスセクション学習）"""
    if config is None:
        config = load_config()

    df_all = build_all_features(config)
    feature_cols = get_feature_columns(df_all)
    X = df_all[feature_cols]
    y = df_all["target"]

    model_cfg = config["model"]

    # 時系列分割（日付ベース）
    unique_dates = df_all.index.unique().sort_values()
    n = len(unique_dates)
    n_splits = model_cfg["n_splits"]

    scores = []
    last_model = None

    for fold in range(n_splits):
        split_point = int(n * (fold + 1) / (n_splits + 1))
        val_start = int(n * (fold + 1) / (n_splits + 1))
        val_end = int(n * (fold + 2) / (n_splits + 1))

        train_dates = unique_dates[:split_point]
        val_dates = unique_dates[val_start:val_end]

        train_mask = df_all.index.isin(train_dates)
        val_mask = df_all.index.isin(val_dates)

        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        if len(X_val) == 0:
            continue

        model = _train_lgbm(X_train, y_train, X_val, y_val, model_cfg["lgbm_params"])
        last_model = model

        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        scores.append({"fold": fold, "accuracy": acc, "auc": auc})
        print(f"  Fold {fold}: ACC={acc:.4f}, AUC={auc:.4f} (train={len(X_train)}, val={len(X_val)})")

    # 全銘柄共通モデル保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "cross_sectional_lgbm.txt"
    last_model.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": last_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    scores_df = pd.DataFrame(scores)
    print(f"\n  Mean ACC: {scores_df['accuracy'].mean():.4f}")
    print(f"  Mean AUC: {scores_df['auc'].mean():.4f}")

    return {
        "model": last_model, "scores": scores_df,
        "importance": importance, "feature_cols": feature_cols,
    }


def predict_latest(ticker: str, config: dict | None = None, use_cross: bool = True) -> dict:
    """最新データに対する予測を実行"""
    if config is None:
        config = load_config()

    # クロスセクションモデル優先、なければ個別モデル
    safe_name = ticker.replace("^", "IDX_").replace("=", "_")
    cross_path = MODEL_DIR / "cross_sectional_lgbm.txt"
    single_path = MODEL_DIR / f"{safe_name}_lgbm.txt"

    if use_cross and cross_path.exists():
        model = lgb.Booster(model_file=str(cross_path))
        model_type = "クロスセクション"
    elif single_path.exists():
        model = lgb.Booster(model_file=str(single_path))
        model_type = "個別"
    else:
        raise FileNotFoundError(f"モデルが見つかりません: {ticker}")

    df = build_features(ticker, config)
    feature_cols = get_feature_columns(df)

    latest = df[feature_cols].iloc[[-1]]
    proba = model.predict(latest)[0]

    # SHAP値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest)

    horizon = config["model"]["target_horizon"]

    return {
        "ticker": ticker,
        "date": df.index[-1],
        "close": df["Close"].iloc[-1],
        "prediction_proba": proba,
        "prediction": "上昇" if proba > 0.5 else "下落",
        "horizon": horizon,
        "model_type": model_type,
        "shap_values": shap_values,
        "feature_names": feature_cols,
        "feature_values": latest.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    config = load_config()
    for ticker in ticker_list(config):
        print(f"\n{'='*50}")
        print(f"Training: {ticker}")
        print(f"{'='*50}")
        train_model(ticker, config)
