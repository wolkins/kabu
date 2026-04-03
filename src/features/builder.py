"""特徴量構築パイプライン"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data.fetch import load_raw
from src.features.technical import add_technical_indicators
from src.features.fundamental import add_fundamental_features
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
        idx_close = idx_df["Close"]

        # リターン（絶対値ではなく変化率を使う）
        idx_return = idx_close.pct_change().rename(f"{safe}_return")
        # 5日リターン
        idx_return5 = idx_close.pct_change(5).rename(f"{safe}_return5")
        # 20日ボラティリティ
        idx_vol = idx_close.pct_change().rolling(20).std().rename(f"{safe}_vol20")

        df = df.join(idx_return, how="left")
        df = df.join(idx_return5, how="left")
        df = df.join(idx_vol, how="left")

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


def add_multiframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """マルチタイムフレーム特徴量（週足・月足トレンド）"""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --- 週足レベル（5日） ---
    # 週足レンジ内ポジション（ストキャスティクス的）
    weekly_high = high.rolling(5).max()
    weekly_low = low.rolling(5).min()
    weekly_range = weekly_high - weekly_low
    df["weekly_range_pos"] = np.where(
        weekly_range > 0, (close - weekly_low) / weekly_range, 0.5
    )
    # 週足トレンド安定度（5日リターンの Sharpe 風指標）
    weekly_ret = close.pct_change(5)
    weekly_std = weekly_ret.rolling(4).std()
    df["weekly_trend_stability"] = np.where(
        weekly_std > 0, weekly_ret.rolling(4).mean() / weekly_std, 0.0
    )
    # 週足モメンタム加速度（5日リターンの変化率）
    df["weekly_momentum_accel"] = weekly_ret.diff(5)

    # --- 月足レベル（20日） ---
    monthly_high = high.rolling(20).max()
    monthly_low = low.rolling(20).min()
    monthly_range = monthly_high - monthly_low
    df["monthly_range_pos"] = np.where(
        monthly_range > 0, (close - monthly_low) / monthly_range, 0.5
    )
    # 月足トレンド安定度
    monthly_ret = close.pct_change(20)
    monthly_std = monthly_ret.rolling(4).std()
    df["monthly_trend_stability"] = np.where(
        monthly_std > 0, monthly_ret.rolling(4).mean() / monthly_std, 0.0
    )
    # 月足モメンタム加速度
    df["monthly_momentum_accel"] = monthly_ret.diff(20)

    # --- MAクロス（ゴールデンクロス/デッドクロス）---
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    df["ma_cross_5_20"] = (sma5 / sma20) - 1  # 比率で表現（連続値）
    df["ma_cross_20_60"] = (sma20 / sma60) - 1

    # --- 長期トレンド（60日）---
    quarterly_high = high.rolling(60).max()
    quarterly_low = low.rolling(60).min()
    quarterly_range = quarterly_high - quarterly_low
    df["quarterly_range_pos"] = np.where(
        quarterly_range > 0, (close - quarterly_low) / quarterly_range, 0.5
    )

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


def add_target(df: pd.DataFrame, horizon: int = 5,
               benchmark: str | None = None) -> pd.DataFrame:
    """予測ターゲット: N日後アルファ（対ベンチマーク超過リターン）が正なら1、負なら0

    benchmark が指定されている場合、銘柄リターンからベンチマークリターンを差し引いた
    アルファを予測対象とする。未指定の場合は従来の絶対リターンを使用。
    """
    stock_return = df["Close"].shift(-horizon) / df["Close"] - 1

    if benchmark:
        bench_df = load_raw(benchmark)
        bench_return = bench_df["Close"].shift(-horizon) / bench_df["Close"] - 1
        bench_return = bench_return.reindex(df.index)
        alpha = stock_return - bench_return
        df["future_return"] = alpha
        df["benchmark_return"] = bench_return.reindex(df.index)
    else:
        df["future_return"] = stock_return

    df["target"] = (df["future_return"] > 0).astype(int)
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
    df = add_multiframe_features(df)
    df = add_regime_features(df)
    df = add_fundamental_features(df, ticker, config)
    df = add_target(df, config["model"]["target_horizon"],
                    benchmark=config["model"].get("benchmark"))

    # 銘柄識別用カラム（一括学習用: カテゴリ型）
    tickers = ticker_list(config)
    df["ticker_id"] = tickers.index(ticker) if ticker in tickers else -1

    # NaN行を除去
    df = df.dropna()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = ticker.replace("^", "IDX_").replace("=", "_")
    df.to_parquet(PROCESSED_DIR / f"{safe_name}_features.parquet")

    return df


def build_all_features(config: dict | None = None) -> pd.DataFrame:
    """全銘柄の特徴量を結合し、クロスセクション特徴量を追加"""
    if config is None:
        config = load_config()

    dfs = []
    for ticker in ticker_list(config):
        try:
            df = build_features(ticker, config)
            dfs.append(df)
        except FileNotFoundError:
            print(f"  {ticker}: データなし、スキップ")

    df_all = pd.concat(dfs, axis=0).sort_index()

    # --- クロスセクション特徴量: 同一日付内での銘柄間相対値 ---
    rank_cols = ["rsi", "daily_return", "volume_ratio", "volatility_20",
                 "macd_diff", "bb_pct", "trend_strength", "return_zscore",
                 "obv_norm", "mfi",
                 "dynamic_per", "dynamic_pbr", "dynamic_div_yield",
                 "weekly_range_pos", "monthly_range_pos", "quarterly_range_pos",
                 "ma_cross_5_20", "ma_cross_20_60",
                 "weekly_trend_stability", "monthly_trend_stability",
                 "weekly_momentum_accel", "monthly_momentum_accel"]

    for col in rank_cols:
        if col in df_all.columns:
            # 同一日のランク（0~1に正規化）
            df_all[f"{col}_rank"] = df_all.groupby(df_all.index)[col].rank(pct=True)
            # 同一日のz-score
            grp = df_all.groupby(df_all.index)[col]
            df_all[f"{col}_cs_zscore"] = (df_all[col] - grp.transform("mean")) / grp.transform("std")

    # NaN除去（クロスセクション特徴量で発生しうる）
    df_all = df_all.dropna()

    return df_all


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """学習に使う特徴量カラム名のリストを返す"""
    # 絶対値系の価格カラムを除外（銘柄間でスケールが異なり支配的になる）
    exclude = {
        "Open", "High", "Low", "Close", "Volume", "target", "future_return",
        "benchmark_return",
        "bb_high", "bb_low", "bb_mid",  # 比率版(bb_width, bb_pct)を使う
    }
    # SMA絶対値も除外（比率版 sma_X_ratio を使う）
    sma_abs = {c for c in df.columns if c.startswith("sma_") and not c.endswith("_ratio")}
    exclude |= sma_abs
    return [c for c in df.columns if c not in exclude]
