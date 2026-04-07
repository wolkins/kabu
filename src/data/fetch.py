"""日本株・市場指標データの取得モジュール"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

from src.utils.config import load_config, ticker_list

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_ticker(ticker: str, start_date: str, interval: str = "1d") -> pd.DataFrame:
    """単一ティッカーのデータを取得"""
    df = yf.download(
        ticker,
        start=start_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def fetch_all(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """設定ファイルの全ティッカー+市場指標を取得し、raw/に保存"""
    if config is None:
        config = load_config()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    commodity_tickers = list(config.get("commodities", {}).keys())
    vol_tickers = config.get("volatility_indices", [])
    all_tickers = (ticker_list(config) + config["market_indices"]
                   + commodity_tickers + vol_tickers)
    results = {}

    for ticker in tqdm(all_tickers, desc="Fetching data"):
        df = fetch_ticker(ticker, config["data"]["start_date"], config["data"]["interval"])
        if df.empty:
            print(f"  WARNING: {ticker} returned no data")
            continue

        safe_name = ticker.replace("^", "IDX_").replace("=", "_")
        df.to_parquet(RAW_DIR / f"{safe_name}.parquet")
        results[ticker] = df
        print(f"  {ticker}: {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")

    # ファンダメンタルズデータも取得・キャッシュ（Point-in-Time四半期データ含む）
    from src.features.fundamental import fetch_quarterly_fundamentals
    print("Fetching quarterly fundamentals (Point-in-Time)...")
    fetch_quarterly_fundamentals(config)

    return results


def load_raw(ticker: str) -> pd.DataFrame:
    """保存済みのrawデータを読み込み"""
    safe_name = ticker.replace("^", "IDX_").replace("=", "_")
    path = RAW_DIR / f"{safe_name}.parquet"
    return pd.read_parquet(path)


if __name__ == "__main__":
    fetch_all()
