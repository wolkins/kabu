"""LightGBMモデルの学習・予測・評価パイプライン"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import shap
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features.builder import build_features, build_all_features, get_feature_columns
from src.utils.config import load_config, ticker_list

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# クロスセクション学習時のカテゴリ特徴量
CATEGORICAL_FEATURES = ["ticker_id", "sector_id"]


def _train_lgbm(X_train, y_train, X_val, y_val, params: dict,
                categorical_feature="auto"):
    """LightGBM単体の学習"""
    train_set = lgb.Dataset(X_train, label=y_train,
                            categorical_feature=categorical_feature)
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


def optimize_params(df_all: pd.DataFrame, feature_cols: list[str],
                    config: dict, n_trials: int = 50,
                    categorical_feature: list[str] | str = "auto") -> dict:
    """Optunaでハイパーパラメータを最適化（Walk-forward CVベース）"""
    X = df_all[feature_cols].copy()
    y = df_all["target"]

    # カテゴリ型変換
    if categorical_feature != "auto":
        for col in categorical_feature:
            if col in X.columns:
                X[col] = X[col].astype("category")

    unique_dates = df_all.index.unique().sort_values()
    n = len(unique_dates)
    n_splits = config["model"]["n_splits"]

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }

        aucs = []
        for fold in range(n_splits):
            split_point = int(n * (fold + 1) / (n_splits + 1))
            val_start = split_point
            val_end = int(n * (fold + 2) / (n_splits + 1))

            train_dates = unique_dates[:split_point]
            val_dates = unique_dates[val_start:val_end]

            train_mask = df_all.index.isin(train_dates)
            val_mask = df_all.index.isin(val_dates)

            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]

            if len(X_val) == 0 or y_val.nunique() < 2:
                continue

            model = _train_lgbm(X_train, y_train, X_val, y_val,
                                {**params, "n_estimators": 500,
                                 "early_stopping_rounds": 30},
                                categorical_feature=categorical_feature)
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            aucs.append(auc)

            # Fold単位でPruning判定
            trial.report(auc, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(aucs) if aucs else 0.5

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update({
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "n_estimators": 1000,
        "early_stopping_rounds": 80,
    })

    print(f"  Optuna最適化完了 (best AUC: {study.best_value:.4f})")
    print(f"  最適パラメータ: {study.best_params}")

    return best


def train_model(ticker: str, config: dict | None = None) -> dict:
    """Walk-forward検証付きでLightGBMモデルを学習（単一銘柄）"""
    if config is None:
        config = load_config()

    df = build_features(ticker, config)
    feature_cols = get_feature_columns(df)
    # 個別モデルではticker_id, sector_idは不要（全行同じ値）
    feature_cols = [c for c in feature_cols if c not in ("ticker_id", "sector_id")]
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


def train_cross_sectional(config: dict | None = None,
                          use_optuna: bool = False,
                          n_trials: int = 50) -> dict:
    """全銘柄を一括学習（クロスセクション学習）"""
    if config is None:
        config = load_config()

    df_all = build_all_features(config)
    feature_cols = get_feature_columns(df_all)
    X = df_all[feature_cols].copy()
    y = df_all["target"]

    # カテゴリ特徴量を変換
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in X.columns:
            X[cat_col] = X[cat_col].astype("category")

    model_cfg = config["model"]
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    # --- Optunaによるハイパーパラメータ最適化 ---
    if use_optuna:
        print(f"  Optunaでハイパーパラメータ最適化中 ({n_trials}試行)...")
        lgbm_params = optimize_params(
            df_all, feature_cols, config, n_trials=n_trials,
            categorical_feature=cat_features,
        )
    else:
        lgbm_params = model_cfg["lgbm_params"]

    # --- Walk-forward CV で評価 ---
    unique_dates = df_all.index.unique().sort_values()
    n = len(unique_dates)
    n_splits = model_cfg["n_splits"]

    scores = []
    best_model = None
    best_auc = -1

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

        model = _train_lgbm(X_train, y_train, X_val, y_val,
                            lgbm_params,
                            categorical_feature=cat_features)

        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        scores.append({"fold": fold, "accuracy": acc, "auc": auc,
                        "num_trees": model.num_trees()})
        print(f"  Fold {fold}: ACC={acc:.4f}, AUC={auc:.4f}, "
              f"trees={model.num_trees()} (train={len(X_train)}, val={len(X_val)})")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # --- 最終モデル: 全データの80%で学習、残り20%をバリデーション ---
    print(f"\n  全データで最終モデルを学習中...")
    split_idx = int(n * 0.8)
    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]
    train_mask = df_all.index.isin(train_dates)
    val_mask = df_all.index.isin(val_dates)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    final_model = _train_lgbm(X_train, y_train, X_val, y_val,
                              lgbm_params,
                              categorical_feature=cat_features)

    final_pred = final_model.predict(X_val)
    final_auc = roc_auc_score(y_val, final_pred)
    final_acc = accuracy_score(y_val, (final_pred > 0.5).astype(int))
    print(f"  最終モデル: ACC={final_acc:.4f}, AUC={final_auc:.4f}, "
          f"trees={final_model.num_trees()}")

    # 木が十分育ったモデルを保存（最終モデル or ベストCV）
    save_model = final_model if final_model.num_trees() > 5 else best_model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "cross_sectional_lgbm.txt"
    save_model.save_model(str(model_path))
    print(f"  保存モデルの木数: {save_model.num_trees()}")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": save_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    scores_df = pd.DataFrame(scores)
    print(f"\n  CV Mean ACC: {scores_df['accuracy'].mean():.4f}")
    print(f"  CV Mean AUC: {scores_df['auc'].mean():.4f}")

    print(f"\n  特徴量重要度 TOP15:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.1f}")

    return {
        "model": save_model, "scores": scores_df,
        "importance": importance, "feature_cols": feature_cols,
    }


def predict_latest(ticker: str, config: dict | None = None, use_cross: bool = True) -> dict:
    """最新データに対する予測を実行"""
    if config is None:
        config = load_config()

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

    # クロスセクションモデルの場合、全銘柄の特徴量を構築してランク特徴量を得る
    if model_type == "クロスセクション":
        df_all = build_all_features(config)
        feature_cols = get_feature_columns(df_all)
        # カテゴリ特徴量を変換（学習時と一致させる）
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in df_all.columns:
                df_all[cat_col] = df_all[cat_col].astype("category")
        # 対象銘柄の最新行を取得
        ticker_id = ticker_list(config).index(ticker)
        mask = df_all["ticker_id"] == ticker_id
        df_ticker = df_all[mask]
        latest = df_ticker[feature_cols].iloc[[-1]]
        latest_date = df_ticker.index[-1]
        latest_close = df_ticker["Close"].iloc[-1]
    else:
        df = build_features(ticker, config)
        feature_cols = get_feature_columns(df)
        feature_cols = [c for c in feature_cols if c != "ticker_id"]
        latest = df[feature_cols].iloc[[-1]]
        latest_date = df.index[-1]
        latest_close = df["Close"].iloc[-1]

    proba = model.predict(latest)[0]

    # SHAP値の計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest)

    horizon = config["model"]["target_horizon"]
    use_alpha = config["model"].get("benchmark") is not None

    if use_alpha:
        prediction = "市場超過" if proba > 0.5 else "市場未満"
    else:
        prediction = "上昇" if proba > 0.5 else "下落"

    return {
        "ticker": ticker,
        "date": latest_date,
        "close": latest_close,
        "prediction_proba": proba,
        "prediction": prediction,
        "horizon": horizon,
        "model_type": model_type,
        "use_alpha": use_alpha,
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
