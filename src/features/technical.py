"""テクニカル指標の計算"""

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

    return df
