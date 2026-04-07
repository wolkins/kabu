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

    # --- 市場レジーム特徴量（日経225ベース）---
    try:
        n225_df = load_raw("^N225")
        n225_ret = n225_df["Close"].pct_change()
        n225_sma20 = n225_df["Close"].rolling(20).mean()
        n225_sma60 = n225_df["Close"].rolling(60).mean()

        # 市場トレンド方向
        market_trend = (n225_sma20.pct_change(5) * 100).rename("market_regime_trend")
        df = df.join(market_trend, how="left")

        # 市場トレンド強度
        market_strength = (np.where(
            n225_sma60 > 0, (n225_sma20 / n225_sma60) - 1, 0.0
        ))
        df["market_regime_strength"] = pd.Series(
            market_strength, index=n225_df.index
        ).reindex(df.index)

        # 市場ボラティリティ状態
        market_vol = n225_ret.rolling(20).std()
        market_vol_pctl = market_vol.rolling(252).rank(pct=True).rename("market_vol_state")
        df = df.join(market_vol_pctl, how="left")
    except FileNotFoundError:
        pass

    # --- ボラティリティ指数（VIX, 日経VI）---
    for vix_ticker in config.get("volatility_indices", []):
        try:
            vix_df = load_raw(vix_ticker)
        except FileNotFoundError:
            continue

        safe = vix_ticker.replace("^", "").replace("=", "").replace(".", "_")
        vix_close = vix_df["Close"]

        # レベル（生値）
        df[f"{safe}_level"] = vix_close.reindex(df.index)
        # 5日変化率
        df[f"{safe}_change5"] = vix_close.pct_change(5).reindex(df.index)
        # 1日変化率
        df[f"{safe}_change1"] = vix_close.pct_change().reindex(df.index)
        # 252日パーセンタイル順位（高ボラ/低ボラレジーム）
        vix_pctl = vix_close.rolling(252).rank(pct=True)
        df[f"{safe}_regime"] = vix_pctl.reindex(df.index)
        # 20日平均からの乖離
        vix_sma20 = vix_close.rolling(20).mean()
        df[f"{safe}_dev20"] = ((vix_close / vix_sma20) - 1).reindex(df.index)

    # 前方参照を防ぐため、市場指標は1日シフト
    market_prefixes = ["N225", "GSPC", "JPY", "EURJPY", "ESF", "NQF",
                       "TNX", "IRX", "market_", "VIX", "N225VI"]
    market_cols = [c for c in df.columns if any(
        c.startswith(p) for p in market_prefixes
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
    """市場レジーム特徴量（ボラティリティレジーム + 市場レジームスイッチング）"""
    ret = df["Close"].pct_change()

    # 短期・長期ボラティリティ比率
    vol_short = ret.rolling(5).std()
    vol_long = ret.rolling(60).std()
    df["vol_regime"] = np.where(vol_long > 0, vol_short / vol_long, 1.0)

    # トレンド強度（ADXの簡易版: 方向性の一貫性）
    ret_std_20 = ret.rolling(20).std()
    df["trend_strength"] = np.where(
        ret_std_20 > 0, ret.rolling(20).mean() / ret_std_20, 0.0
    )

    # 直近リターンの分布位置（z-score）
    ret_std_60 = ret.rolling(60).std()
    df["return_zscore"] = np.where(
        ret_std_60 > 0, (ret - ret.rolling(60).mean()) / ret_std_60, 0.0
    )

    # --- 市場レジームスイッチング ---
    # レジーム判定: 20日SMAトレンド + ボラティリティで分類
    sma20 = df["Close"].rolling(20).mean()
    sma60 = df["Close"].rolling(60).mean()
    sma20_slope = sma20.pct_change(5)  # SMA20の5日変化率

    # 上昇/下降/レンジ を連続値で表現（モデルが自然に学習可能）
    # regime_trend: 正=上昇トレンド、負=下降トレンド、0近辺=レンジ
    df["regime_trend"] = sma20_slope * 100  # スケーリング

    # regime_strength: トレンドの強さ（SMA20とSMA60の乖離度）
    df["regime_strength"] = np.where(
        sma60 > 0, (sma20 / sma60) - 1, 0.0
    )

    # regime_vol_state: 高ボラ/低ボラ状態（パーセンタイル）
    vol_20 = ret.rolling(20).std()
    vol_percentile = vol_20.rolling(252).rank(pct=True)  # 1年間での順位
    df["regime_vol_state"] = vol_percentile

    # regime_momentum_consistency: リターンの方向一貫性
    # 直近20日中、正リターンの割合
    df["regime_up_ratio"] = ret.rolling(20).apply(lambda x: (x > 0).mean(), raw=True)

    # regime_drawdown: 直近高値からの下落率（ドローダウン）
    rolling_max = df["Close"].rolling(60).max()
    df["regime_drawdown"] = np.where(
        rolling_max > 0, (df["Close"] / rolling_max) - 1, 0.0
    )

    return df


def add_sector_commodity_features(df: pd.DataFrame, ticker: str,
                                  config: dict) -> pd.DataFrame:
    """セクター・コモディティ相関の動的特徴量を追加"""
    sectors = config.get("sectors", {})
    commodities = config.get("commodities", {})

    # --- セクターID（カテゴリ特徴量）---
    sector_name = sectors.get(ticker, "unknown")
    unique_sectors = sorted(set(sectors.values()))
    df["sector_id"] = unique_sectors.index(sector_name) if sector_name in unique_sectors else -1

    # --- セクター内相対リターン ---
    # 同一セクター銘柄のリターン平均を計算
    same_sector_tickers = [t for t, s in sectors.items() if s == sector_name and t != ticker]
    sector_returns = []
    if same_sector_tickers:
        for st in same_sector_tickers:
            try:
                st_df = load_raw(st)
                sector_returns.append(st_df["Close"].pct_change().rename(st))
            except FileNotFoundError:
                continue

    # 同セクターに他銘柄がいない場合は市場平均（N225）で代用
    if not sector_returns:
        try:
            n225_df = load_raw("^N225")
            sector_returns.append(n225_df["Close"].pct_change().rename("N225_fallback"))
        except FileNotFoundError:
            pass

    if sector_returns:
        sector_avg = pd.concat(sector_returns, axis=1).mean(axis=1)
        ticker_ret = df["Close"].pct_change()
        df["sector_avg_return"] = sector_avg.reindex(df.index)
        df["sector_relative_return"] = ticker_ret - df["sector_avg_return"]
        sector_avg_5 = sector_avg.rolling(5).mean()
        df["sector_momentum_5d"] = sector_avg_5.reindex(df.index)
    else:
        df["sector_avg_return"] = 0.0
        df["sector_relative_return"] = 0.0
        df["sector_momentum_5d"] = 0.0

    # --- コモディティ特徴量 ---
    for cmd_ticker, cmd_name in commodities.items():
        try:
            cmd_df = load_raw(cmd_ticker)
        except FileNotFoundError:
            continue
        safe = cmd_ticker.replace("=", "").replace(".", "_")
        cmd_close = cmd_df["Close"]
        # リターン
        cmd_ret = cmd_close.pct_change().rename(f"{safe}_return")
        df = df.join(cmd_ret, how="left")
        # 5日リターン
        cmd_ret5 = cmd_close.pct_change(5).rename(f"{safe}_return5")
        df = df.join(cmd_ret5, how="left")
        # 20日ボラティリティ
        cmd_vol = cmd_close.pct_change().rolling(20).std().rename(f"{safe}_vol20")
        df = df.join(cmd_vol, how="left")

    # コモディティは前方参照防止で1日シフト
    cmd_cols = [c for c in df.columns if any(
        c.startswith(cmd_ticker.replace("=", "").replace(".", "_"))
        for cmd_ticker in commodities.keys()
    )]
    for col in cmd_cols:
        df[col] = df[col].shift(1)

    # セクター平均も1日シフト（前方参照防止）
    for col in ["sector_avg_return", "sector_relative_return", "sector_momentum_5d"]:
        if col in df.columns:
            df[col] = df[col].shift(1)

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
    df = add_sector_commodity_features(df, ticker, config)
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
                 "dynamic_psr", "eps_growth",
                 "weekly_range_pos", "monthly_range_pos", "quarterly_range_pos",
                 "ma_cross_5_20", "ma_cross_20_60",
                 "weekly_trend_stability", "monthly_trend_stability",
                 "weekly_momentum_accel", "monthly_momentum_accel",
                 "regime_trend", "regime_strength", "regime_vol_state",
                 "regime_up_ratio", "regime_drawdown",
                 "sector_relative_return",
                 # Phase 1 ミクロ構造特徴量
                 "overnight_gap", "intraday_return", "intraday_reversal",
                 "parkinson_vol_20", "garman_klass_vol_20",
                 "amihud_illiq_20_log", "volume_shock_5d",
                 "volume_shock_persist"]

    # クロスセクション特徴量を一括計算（fragmentation回避）
    cs_frames = []
    for col in rank_cols:
        if col in df_all.columns:
            grp = df_all.groupby(df_all.index)[col]
            rank_s = grp.rank(pct=True).rename(f"{col}_rank")
            zscore_s = ((df_all[col] - grp.transform("mean"))
                        / grp.transform("std")).rename(f"{col}_cs_zscore")
            cs_frames.extend([rank_s, zscore_s])

    if cs_frames:
        df_all = pd.concat([df_all] + cs_frames, axis=1)

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
