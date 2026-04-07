"""テクニカル指標の計算"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator


def add_technical_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """DataFrameにテクニカル指標カラムを追加"""
    feat = config["features"]
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # RSI
    df["rsi"] = RSIIndicator(close=close, window=feat["rsi_window"]).rsi()

    # MACD
    macd = MACD(
        close=close,
        window_slow=feat["macd_slow"],
        window_fast=feat["macd_fast"],
        window_sign=feat["macd_signal"],
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # ボリンジャーバンド
    bb = BollingerBands(close=close, window=feat["bb_window"], window_dev=feat["bb_std"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    df["bb_pct"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    # 移動平均線
    for w in feat["sma_windows"]:
        df[f"sma_{w}"] = SMAIndicator(close=close, window=w).sma_indicator()
        df[f"sma_{w}_ratio"] = close / df[f"sma_{w}"]

    # 出来高変化率
    df["volume_ratio"] = volume / volume.rolling(20).mean()

    # OBV（On Balance Volume）— 期間変化量を総出来高で正規化
    obv = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["obv_norm"] = (obv - obv.shift(20)) / volume.rolling(20).sum()

    # MFI（Money Flow Index）— 出来高加重RSI
    df["mfi"] = MFIIndicator(
        high=high, low=low, close=close, volume=volume,
        window=feat.get("mfi_window", 14),
    ).money_flow_index()

    # 価格変動
    df["daily_return"] = close.pct_change()
    df["high_low_ratio"] = (high - low) / close
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    # === ミクロ構造特徴量（Phase 1: Codex/Gemini提案）===
    open_ = df["Open"]
    prev_close = close.shift(1)

    # Overnight gap: 前日終値→当日始値の変化率
    df["overnight_gap"] = (open_ / prev_close) - 1

    # Intraday return: 当日始値→当日終値の変化率
    df["intraday_return"] = (close / open_) - 1

    # Intraday reversal: オーバーナイトと日中の符号が逆ならリバーサル
    # 連続値: -1 ~ 1（負ならリバーサル傾向）
    df["intraday_reversal"] = -np.sign(df["overnight_gap"]) * df["intraday_return"]

    # Parkinson volatility（高値・安値ベース、20日窓）
    # σ_P = sqrt( (1/(4*ln(2))) * mean( (ln(H/L))^2 ) )
    log_hl_sq = (np.log(high / low) ** 2)
    df["parkinson_vol_20"] = np.sqrt(
        log_hl_sq.rolling(20).mean() / (4 * np.log(2))
    )

    # Garman-Klass volatility（OHLCベース、20日窓）
    # σ_GK = sqrt( mean( 0.5*(ln(H/L))^2 - (2ln2 - 1)*(ln(C/O))^2 ) )
    log_co_sq = (np.log(close / open_) ** 2)
    gk_term = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
    df["garman_klass_vol_20"] = np.sqrt(gk_term.rolling(20).mean().clip(lower=0))

    # Amihud illiquidity: |return| / (volume * close)（出来高×価格 = 取引代金）
    # 日次値を20日平均で平滑化
    dollar_volume = volume * close
    amihud_daily = np.abs(df["daily_return"]) / dollar_volume.replace(0, np.nan)
    df["amihud_illiq_20"] = amihud_daily.rolling(20).mean()
    # スケールが極端なのでlog化
    df["amihud_illiq_20_log"] = np.log1p(df["amihud_illiq_20"].fillna(0) * 1e9)

    # Volume shock persistence: 直近5日の出来高比率の安定度
    # 出来高ショック（volume_ratio）が継続しているかを表す
    df["volume_shock_5d"] = df["volume_ratio"].rolling(5).mean()
    df["volume_shock_persist"] = (
        (df["volume_ratio"] > 1.5).rolling(5).sum() / 5
    )

    return df
