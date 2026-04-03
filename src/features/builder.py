"""特徴量構築パイプライン"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.fetch import load_raw
from src.features.technical import add_technical_indicators
from src.utils.config import load_config, ticker_list

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def add_lag_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """ラグ特徴量（過去N日のリターン等）を追加"""
    lag_days = config["features"]["lag_days"]
    close = df["Close"]

    for lag in lag_days:
        df[f"return_lag_{lag}"] = close.pct_change(lag)
        df[f"volume_lag_{lag}"] = df["Volume"].pct_change(lag)

    return df


def add_market_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """クロスマーケット特徴量（日経225, S&P500, USD/JPY）を追加"""
    for idx_ticker in config["market_indices"]:
        try:
            idx_df = load_raw(idx_ticker)
        except FileNotFoundError:
            continue

        safe = idx_ticker.replace("^", "").replace("=", "").replace(".", "_")
        idx_close = idx_df["Close"].rename(f"{safe}_close")
        idx_return = idx_close.pct_change().rename(f"{safe}_return")

        df = df.join(idx_close, how="left")
        df = df.join(idx_return, how="left")

    # 前方参照を防ぐため、市場指標は1日シフト
    market_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["N225", "GSPC", "JPY"]
    )]
    for col in market_cols:
        df[col] = df[col].shift(1)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """カレンダー特徴量（曜日・月末効果等）"""
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_month_end"] = df.index.is_month_end.astype(int)
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["quarter"] = df.index.quarter
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """市場レジーム特徴量（ボラティリティレジーム）"""
    ret = df["Close"].pct_change()

    # 短期・長期ボラティリティ比率
    vol_short = ret.rolling(5).std()
    vol_long = ret.rolling(60).std()
    df["vol_regime"] = vol_short / vol_long

    # トレンド強度（ADXの簡易版: 方向性の一貫性）
    df["trend_strength"] = ret.rolling(20).mean() / ret.rolling(20).std()

    # 直近リターンの分布位置（z-score）
    df["return_zscore"] = (ret - ret.rolling(60).mean()) / ret.rolling(60).std()

    return df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """予測ターゲット: N日後リターンが正なら1、負なら0"""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["future_return"] = future_return
    df["target"] = (future_return > 0).astype(int)
    return df


def build_features(ticker: str, config: dict | None = None) -> pd.DataFrame:
    """単一銘柄の全特徴量を構築"""
    if config is None:
        config = load_config()

    df = load_raw(ticker)
    df = add_technical_indicators(df, config)
    df = add_lag_features(df, config)
    df = add_market_features(df, config)
    df = add_calendar_features(df)
    df = add_regime_features(df)
    df = add_target(df, config["model"]["target_horizon"])

    # 銘柄識別用カラム（一括学習用）
    df["ticker"] = ticker

    # NaN行を除去
    df = df.dropna()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("^", "IDX_").replace("=", "_")
    df.to_parquet(PROCESSED_DIR / f"{safe_name}_features.parquet")

    return df


def build_all_features(config: dict | None = None) -> pd.DataFrame:
    """全銘柄の特徴量を結合（クロスセクション学習用）"""
    if config is None:
        config = load_config()

    dfs = []
    for ticker in ticker_list(config):
        try:
            df = build_features(ticker, config)
            dfs.append(df)
        except FileNotFoundError:
            print(f"  {ticker}: データなし、スキップ")
    return pd.concat(dfs, axis=0).sort_index()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """学習に使う特徴量カラム名のリストを返す"""
    exclude = {"Open", "High", "Low", "Close", "Volume", "target", "future_return", "ticker"}
    return [c for c in df.columns if c not in exclude]
