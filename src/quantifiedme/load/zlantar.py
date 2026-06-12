"""
Loads personal finance data from a Zlantar export.

Zlantar (https://zlantar.se/) is a Swedish personal-finance aggregator that
pulls transactions from connected bank accounts and categorizes them. Because
it aggregates every connected bank, a single loader covers all accounts — there
is no need for per-bank parsers.

The export is a ``.zip`` containing:
  - ``transactions.json`` — every transaction with English field names
  - ``transaktioner.csv``  — the same data with Swedish column names
  - ``data.json``          — accounts, budgets, agreements (not loaded here)

This loader reads ``transactions.json`` (cleanest schema). The ``path`` may
point at the ``.zip``, at an extracted ``transactions.json``, or at a directory
containing it.

Amounts are in SEK and signed from the account's perspective:
  - income:   positive (money in)
  - expense:  negative (money out; a few positive rows are refunds)
  - transfer: mixed (moves between own accounts — internal)
  - savings:  negative (moved into a savings account — internal)

Transfers and savings are internal movements, so the daily summary excludes
them from income/expense/net to avoid double-counting real cash flow.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

from ..config import load_config

_TRANSACTIONS_FILE = "transactions.json"

# Transaction types that represent real cash flow (vs. internal account moves).
_CASHFLOW_TYPES = {"income", "expense"}


def _read_transactions(path: Path) -> list[dict]:
    """Read the raw transaction list from a zip, json file, or directory."""
    if path.is_dir():
        path = path / _TRANSACTIONS_FILE

    if not path.exists():
        raise FileNotFoundError(f"Zlantar export not found at {path}")

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            if _TRANSACTIONS_FILE not in zf.namelist():
                raise ValueError(
                    f"{path} does not contain {_TRANSACTIONS_FILE}. "
                    f"Found: {zf.namelist()}"
                )
            with zf.open(_TRANSACTIONS_FILE) as f:
                return json.load(f)

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_transactions_df(path: Path | None = None) -> pd.DataFrame:
    """
    Load all Zlantar transactions.

    Returns a DataFrame indexed by UTC timestamp with columns:
    - amount: signed amount in SEK
    - transaction_type: income | expense | transfer | savings
    - category: main category (e.g. food, shopping, transport)
    - subcategory: finer category (may be empty)
    - description: free-text description from the bank
    - bank_name, account_name, account_number: account identifiers
    - tags, notes: user annotations (often empty)
    """
    if path is None:
        config = load_config()
        path = Path(config["data"]["zlantar"]).expanduser()
    else:
        path = Path(path).expanduser()

    records = _read_transactions(path)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No transactions found in {path}")

    df["amount"] = pd.to_numeric(df["amount"])
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop(columns=["date", "index"], errors="ignore")
    df = df.set_index("timestamp").sort_index()

    return df


def load_daily_df(path: Path | None = None) -> pd.DataFrame:
    """
    Load Zlantar data aggregated to daily cash-flow stats.

    Returns a DataFrame indexed by UTC date with columns:
    - income: total money in (SEK)
    - expense: total money out as a positive number (refunds reduce it)
    - net: income - expense
    - savings: amount moved into savings accounts (positive)
    - n_transactions: number of income/expense transactions that day

    Internal transfers and savings movements are excluded from income/expense/net.
    """
    df = load_transactions_df(path=path)
    df = df.assign(day=pd.DatetimeIndex(df.index).floor("D"))

    def _sum_by_day(mask: pd.Series, sign: int) -> pd.Series:
        sub = df[mask]
        return sign * sub.groupby("day")["amount"].sum()

    income = _sum_by_day(df["transaction_type"] == "income", 1)
    expense = _sum_by_day(df["transaction_type"] == "expense", -1)
    savings = _sum_by_day(df["transaction_type"] == "savings", -1)
    cashflow = df[df["transaction_type"].isin(_CASHFLOW_TYPES)]
    n_transactions = cashflow.groupby("day")["amount"].size()

    daily = pd.DataFrame(
        {
            "income": income,
            "expense": expense,
            "savings": savings,
            "n_transactions": n_transactions,
        }
    )
    daily[["income", "expense", "savings"]] = daily[
        ["income", "expense", "savings"]
    ].fillna(0.0)
    daily["n_transactions"] = daily["n_transactions"].fillna(0).astype(int)
    daily["net"] = daily["income"] - daily["expense"]

    daily = daily[["income", "expense", "net", "savings", "n_transactions"]]
    daily.index.name = "date"
    return daily.sort_index()


def load_category_spending_df(
    path: Path | None = None, freq: str = "MS"
) -> pd.DataFrame:
    """
    Load expense spending broken down by main category over time.

    Returns a DataFrame indexed by period start (UTC), one column per main
    category, with positive SEK spending magnitude per period. Only ``expense``
    transactions are included.

    Parameters
    ----------
    path:
        Path to the Zlantar export. Falls back to config if None.
    freq:
        Pandas resample frequency for the period buckets (default "MS",
        month-start). Any frequency accepted by ``pd.Grouper`` works.
    """
    df = load_transactions_df(path=path)
    expenses = df[df["transaction_type"] == "expense"].copy()
    expenses["spend"] = -expenses["amount"]

    pivot = (
        expenses.groupby([pd.Grouper(freq=freq), "category"])["spend"]
        .sum()
        .unstack("category")
        .fillna(0.0)
    )
    pivot.index.name = "period"
    return pivot.sort_index()
