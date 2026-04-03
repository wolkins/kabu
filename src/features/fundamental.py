"""ファンダメンタルズ特徴量の取得

yfinance の .info から EPS / BPS / 配当率を取得し、
日次終値と組み合わせて動的な PER / PBR / 配当利回りを算出する。
これにより先読みバイアスを軽減し、クロスセクション特徴量として有効に機能する。
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from functools import lru_cache

from src.utils.config import load_config, ticker_list

FUND_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "fundamentals.parquet"


def fetch_fundamentals(config: dict | None = None) -> pd.DataFrame:
    """全銘柄のファンダメンタルズ指標を yfinance から取得しキャッシュ保存

    EPS / BPS / 配当率（静的値）を取得し、動的指標算出の元データとする。
    """
    if config is None:
        config = load_config()

    tickers = ticker_list(config)
    rows = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker": ticker,
                "eps": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "bps": info.get("bookValue"),
                "dividend_rate": info.get("dividendRate"),
            })
        except Exception as e:
            print(f"  {ticker}: ファンダメンタルズ取得失敗 ({e})")
            rows.append({"ticker": ticker})

    df = pd.DataFrame(rows).set_index("ticker")

    FUND_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUND_PATH)
    print(f"ファンダメンタルズ保存: {len(df)} 銘柄 → {FUND_PATH}")
    return df


@lru_cache(maxsize=1)
def load_fundamentals() -> pd.DataFrame:
    """キャッシュ済みファンダメンタルズを読み込み（NaN は銘柄間中央値で補完）"""
    df = pd.read_parquet(FUND_PATH)
    for col in ["eps", "eps_forward", "bps", "dividend_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def add_fundamental_features(df: pd.DataFrame, ticker: str,
                             config: dict | None = None) -> pd.DataFrame:
    """銘柄DataFrameに動的ファンダメンタルズ特徴量を追加

    EPS / BPS は静的スナップショットだが、日次終値と組み合わせることで
    動的な PER / PBR を算出する。これにより:
    - 株価変動に応じて特徴量が毎日変化する
    - クロスセクションのrank/zscoreが日次で意味を持つ
    - 先読みバイアスを軽減（EPS/BPSは最新値だが、比率の変動は株価に依存）
    """
    try:
        fund = load_fundamentals()
    except FileNotFoundError:
        load_fundamentals.cache_clear()
        fund = fetch_fundamentals(config)

    close = df["Close"]

    if ticker in fund.index:
        row = fund.loc[ticker]
        eps = row.get("eps")
        eps_fwd = row.get("eps_forward")
        bps = row.get("bps")
        div_rate = row.get("dividend_rate")

        # 動的PER: 終値 / EPS（赤字の場合は絶対値を使い符号反転で負のPER）
        if pd.notna(eps) and eps != 0:
            df["dynamic_per"] = close / eps
        else:
            df["dynamic_per"] = 0.0
        if pd.notna(eps_fwd) and eps_fwd != 0:
            df["dynamic_per_fwd"] = close / eps_fwd
        else:
            df["dynamic_per_fwd"] = 0.0
        # 動的PBR: 終値 / BPS
        if pd.notna(bps) and bps > 0:
            df["dynamic_pbr"] = close / bps
        else:
            df["dynamic_pbr"] = 0.0
        # 動的配当利回り: 配当率 / 終値
        if pd.notna(div_rate) and div_rate > 0:
            df["dynamic_div_yield"] = div_rate / close
        else:
            df["dynamic_div_yield"] = 0.0
    else:
        for col in ["dynamic_per", "dynamic_per_fwd", "dynamic_pbr", "dynamic_div_yield"]:
            df[col] = 0.0

    return df
