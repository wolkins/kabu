"""ファンダメンタルズ特徴量の取得（Point-in-Time方式）

四半期決算データからEPS/BPSを時系列化し、各日付で「その時点で利用可能だった
最新の決算情報」のみを使う。これにより先読みバイアスを排除する。
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from functools import lru_cache

from src.utils.config import load_config, ticker_list

FUND_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fundamentals"
FUND_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "fundamentals.parquet"

# 決算発表は四半期末から約45日後と仮定（先読み防止のバッファ）
REPORTING_LAG_DAYS = 45


def fetch_quarterly_fundamentals(config: dict | None = None) -> dict[str, pd.DataFrame]:
    """全銘柄の四半期決算データを取得しキャッシュ保存"""
    if config is None:
        config = load_config()

    FUND_DIR.mkdir(parents=True, exist_ok=True)
    tickers = ticker_list(config)
    results = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            qi = t.quarterly_income_stmt
            qb = t.quarterly_balance_sheet

            if qi is None or qi.empty or qb is None or qb.empty:
                print(f"  {ticker}: 四半期決算データなし")
                continue

            records = []
            for date in qi.columns:
                row = {"fiscal_date": date}

                # EPS = Net Income / Ordinary Shares Number
                net_income = qi.loc["Net Income", date] if "Net Income" in qi.index else np.nan
                shares = qb.loc["Ordinary Shares Number", date] if (
                    "Ordinary Shares Number" in qb.index and date in qb.columns
                ) else np.nan

                if pd.notna(net_income) and pd.notna(shares) and shares > 0:
                    row["quarterly_eps"] = net_income / shares
                else:
                    row["quarterly_eps"] = np.nan

                # BPS = Stockholders Equity / Ordinary Shares Number
                equity = qb.loc["Stockholders Equity", date] if (
                    "Stockholders Equity" in qb.index and date in qb.columns
                ) else np.nan

                if pd.notna(equity) and pd.notna(shares) and shares > 0:
                    row["quarterly_bps"] = equity / shares
                else:
                    row["quarterly_bps"] = np.nan

                # 売上高（Revenue Per Share）
                revenue = qi.loc["Total Revenue", date] if "Total Revenue" in qi.index else np.nan
                if pd.notna(revenue) and pd.notna(shares) and shares > 0:
                    row["quarterly_rps"] = revenue / shares
                else:
                    row["quarterly_rps"] = np.nan

                records.append(row)

            df_q = pd.DataFrame(records)
            if df_q.empty:
                continue

            df_q["fiscal_date"] = pd.to_datetime(df_q["fiscal_date"])
            # Point-in-Time: 決算発表は四半期末+ラグ日数後に利用可能
            df_q["available_date"] = df_q["fiscal_date"] + pd.Timedelta(days=REPORTING_LAG_DAYS)
            df_q = df_q.sort_values("fiscal_date").reset_index(drop=True)

            # TTM（直近4四半期合計）EPSを計算（4四半期揃うまではNaN）
            df_q["ttm_eps"] = df_q["quarterly_eps"].rolling(4, min_periods=4).sum()
            df_q["ttm_rps"] = df_q["quarterly_rps"].rolling(4, min_periods=4).sum()

            safe = ticker.replace(".", "_")
            df_q.to_parquet(FUND_DIR / f"{safe}_quarterly.parquet")
            results[ticker] = df_q
            print(f"  {ticker}: {len(df_q)}四半期分")

        except Exception as e:
            print(f"  {ticker}: 四半期決算取得失敗 ({e})")

    # 既存互換の静的ファンダメンタルズも更新
    _update_static_fundamentals(config)
    load_fundamentals.cache_clear()

    return results


def _update_static_fundamentals(config: dict):
    """既存互換の静的ファンダメンタルズ（フォールバック用）"""
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
        except Exception:
            rows.append({"ticker": ticker})

    df = pd.DataFrame(rows).set_index("ticker")
    FUND_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUND_PATH)


@lru_cache(maxsize=1)
def load_fundamentals() -> pd.DataFrame:
    """キャッシュ済みファンダメンタルズを読み込み（NaN は銘柄間中央値で補完）"""
    df = pd.read_parquet(FUND_PATH)
    for col in ["eps", "eps_forward", "bps", "dividend_rate"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def _load_quarterly(ticker: str) -> pd.DataFrame | None:
    """四半期データの読み込み"""
    safe = ticker.replace(".", "_")
    path = FUND_DIR / f"{safe}_quarterly.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def add_fundamental_features(df: pd.DataFrame, ticker: str,
                             config: dict | None = None) -> pd.DataFrame:
    """銘柄DataFrameにPoint-in-Timeファンダメンタルズ特徴量を追加

    四半期決算データがある場合: 各日付でその時点で利用可能な最新の決算値を使用
    四半期決算データがない場合: 静的スナップショット値にフォールバック
    """
    close = df["Close"]

    # --- Point-in-Time方式（四半期データあり）---
    df_q = _load_quarterly(ticker)
    if df_q is not None and len(df_q) > 0:
        # 各取引日に対して、available_date以前の最新四半期データをマッピング
        pit_eps = pd.Series(dtype=float, index=df.index)
        pit_bps = pd.Series(dtype=float, index=df.index)
        pit_rps = pd.Series(dtype=float, index=df.index)

        df_q_sorted = df_q.sort_values("available_date")

        for _, qrow in df_q_sorted.iterrows():
            avail = qrow["available_date"]
            mask = df.index > avail
            if pd.notna(qrow.get("ttm_eps")):
                pit_eps[mask] = qrow["ttm_eps"]
            if pd.notna(qrow.get("quarterly_bps")):
                pit_bps[mask] = qrow["quarterly_bps"]
            if pd.notna(qrow.get("ttm_rps")):
                pit_rps[mask] = qrow["ttm_rps"]

        # 動的PER (TTM EPS)
        df["dynamic_per"] = np.where(
            (pit_eps.notna()) & (pit_eps != 0), close / pit_eps, 0.0
        )
        # 動的PBR
        df["dynamic_pbr"] = np.where(
            (pit_bps.notna()) & (pit_bps > 0), close / pit_bps, 0.0
        )
        # 動的PSR (Price-to-Sales)
        df["dynamic_psr"] = np.where(
            (pit_rps.notna()) & (pit_rps > 0), close / pit_rps, 0.0
        )

        # EPS成長率（前四半期比）
        eps_growth = df_q_sorted["ttm_eps"].pct_change()
        pit_eps_growth = pd.Series(dtype=float, index=df.index)
        for i, (_, qrow) in enumerate(df_q_sorted.iterrows()):
            avail = qrow["available_date"]
            mask = df.index > avail
            if i < len(eps_growth) and pd.notna(eps_growth.iloc[i]):
                pit_eps_growth[mask] = eps_growth.iloc[i]
        df["eps_growth"] = (pit_eps_growth
                           .replace([np.inf, -np.inf], np.nan)
                           .fillna(0.0)
                           .clip(-2.0, 2.0))

        # 配当利回りは静的データにフォールバック
        try:
            fund = load_fundamentals()
            if ticker in fund.index:
                div_rate = fund.loc[ticker].get("dividend_rate")
                if pd.notna(div_rate) and div_rate > 0:
                    df["dynamic_div_yield"] = div_rate / close
                else:
                    df["dynamic_div_yield"] = 0.0
            else:
                df["dynamic_div_yield"] = 0.0
        except FileNotFoundError:
            df["dynamic_div_yield"] = 0.0

        # Forward PER（静的フォールバック）
        try:
            fund = load_fundamentals()
            if ticker in fund.index:
                eps_fwd = fund.loc[ticker].get("eps_forward")
                if pd.notna(eps_fwd) and eps_fwd != 0:
                    df["dynamic_per_fwd"] = close / eps_fwd
                else:
                    df["dynamic_per_fwd"] = 0.0
            else:
                df["dynamic_per_fwd"] = 0.0
        except FileNotFoundError:
            df["dynamic_per_fwd"] = 0.0

        return df

    # --- フォールバック: 静的スナップショット ---
    try:
        fund = load_fundamentals()
    except FileNotFoundError:
        load_fundamentals.cache_clear()
        fund = fetch_quarterly_fundamentals(config)
        fund = load_fundamentals()

    if ticker in fund.index:
        row = fund.loc[ticker]
        eps = row.get("eps")
        eps_fwd = row.get("eps_forward")
        bps = row.get("bps")
        div_rate = row.get("dividend_rate")

        df["dynamic_per"] = close / eps if pd.notna(eps) and eps != 0 else 0.0
        df["dynamic_per_fwd"] = close / eps_fwd if pd.notna(eps_fwd) and eps_fwd != 0 else 0.0
        df["dynamic_pbr"] = close / bps if pd.notna(bps) and bps > 0 else 0.0
        df["dynamic_psr"] = 0.0
        df["dynamic_div_yield"] = div_rate / close if pd.notna(div_rate) and div_rate > 0 else 0.0
        df["eps_growth"] = 0.0
    else:
        for col in ["dynamic_per", "dynamic_per_fwd", "dynamic_pbr",
                     "dynamic_psr", "dynamic_div_yield", "eps_growth"]:
            df[col] = 0.0

    return df
