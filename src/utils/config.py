import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else _CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def ticker_list(config: dict) -> list[str]:
    """ティッカーコードのリストを返す"""
    return list(config["tickers"].keys())


def ticker_name(config: dict, ticker: str) -> str:
    """ティッカーコードから会社名を返す"""
    return config["tickers"].get(ticker, ticker)


def ticker_display(config: dict, ticker: str) -> str:
    """'7203.T トヨタ自動車' 形式の表示名を返す"""
    name = ticker_name(config, ticker)
    return f"{ticker} {name}"
