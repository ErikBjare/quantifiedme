"""Tests for the Zlantar personal-finance data loader."""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.zlantar import (
    load_category_spending_df,
    load_daily_df,
    load_transactions_df,
)

SAMPLE_TRANSACTIONS = [
    {
        "index": 0,
        "description": "Lön",
        "date": "2024-01-02T00:00:00.000Z",
        "amount": 30000,
        "transaction_type": "income",
        "category": "salary",
        "subcategory": "",
        "tags": [],
        "notes": "",
        "bank_name": "swedbank",
        "account_name": "Lönekonto",
        "account_number": "1234",
    },
    {
        "index": 1,
        "description": "ICA",
        "date": "2024-01-02T00:00:00.000Z",
        "amount": -350,
        "transaction_type": "expense",
        "category": "food",
        "subcategory": "groceries",
        "tags": [],
        "notes": "",
        "bank_name": "swedbank",
        "account_name": "Lönekonto",
        "account_number": "1234",
    },
    {
        "index": 2,
        "description": "Återbäring",
        "date": "2024-01-03T00:00:00.000Z",
        "amount": 50,  # positive expense row = refund
        "transaction_type": "expense",
        "category": "food",
        "subcategory": "groceries",
        "tags": [],
        "notes": "",
        "bank_name": "swedbank",
        "account_name": "Lönekonto",
        "account_number": "1234",
    },
    {
        "index": 3,
        "description": "H&M",
        "date": "2024-01-03T00:00:00.000Z",
        "amount": -800,
        "transaction_type": "expense",
        "category": "shopping",
        "subcategory": "clothes",
        "tags": [],
        "notes": "",
        "bank_name": "swedbank",
        "account_name": "Lönekonto",
        "account_number": "1234",
    },
    {
        "index": 4,
        "description": "Till sparkonto",
        "date": "2024-01-03T00:00:00.000Z",
        "amount": -5000,
        "transaction_type": "savings",
        "category": "savings",
        "subcategory": "",
        "tags": [],
        "notes": "",
        "bank_name": "swedbank",
        "account_name": "Sparkonto",
        "account_number": "5678",
    },
    {
        "index": 5,
        "description": "Överföring",
        "date": "2024-01-03T00:00:00.000Z",
        "amount": -1000,
        "transaction_type": "transfer",
        "category": "other",
        "subcategory": "",
        "tags": [],
        "notes": "",
        "bank_name": "swedbank",
        "account_name": "Lönekonto",
        "account_number": "1234",
    },
]


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    p = tmp_path / "transactions.json"
    p.write_text(json.dumps(SAMPLE_TRANSACTIONS))
    return p


@pytest.fixture
def sample_zip(tmp_path: Path) -> Path:
    p = tmp_path / "export.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("transactions.json", json.dumps(SAMPLE_TRANSACTIONS))
    return p


def test_load_transactions_df(sample_json: Path) -> None:
    df = load_transactions_df(path=sample_json)

    assert isinstance(df, pd.DataFrame)
    assert "amount" in df.columns
    assert "transaction_type" in df.columns
    assert "category" in df.columns
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"  # explicitly UTC, not just tz-aware
    assert len(df) == len(SAMPLE_TRANSACTIONS)
    assert df.index.is_monotonic_increasing
    # amount coerced to numeric
    assert pd.api.types.is_numeric_dtype(df["amount"])


def test_load_transactions_df_from_zip(sample_zip: Path) -> None:
    df = load_transactions_df(path=sample_zip)
    assert len(df) == len(SAMPLE_TRANSACTIONS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_load_transactions_df_from_dir(sample_json: Path) -> None:
    df = load_transactions_df(path=sample_json.parent)
    assert len(df) == len(SAMPLE_TRANSACTIONS)


def test_load_daily_df(sample_json: Path) -> None:
    df = load_daily_df(path=sample_json)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "income",
        "expense",
        "net",
        "savings",
        "n_transactions",
    ]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "date"
    # Two distinct days in the sample
    assert len(df) == 2

    jan2 = pd.Timestamp("2024-01-02", tz="UTC")
    jan3 = pd.Timestamp("2024-01-03", tz="UTC")

    # Jan 2: 30000 income, 350 expense
    assert df.loc[jan2, "income"] == 30000
    assert df.loc[jan2, "expense"] == 350
    assert df.loc[jan2, "net"] == 30000 - 350
    assert df.loc[jan2, "n_transactions"] == 2

    # Jan 3: expense = 800 (H&M) - 50 (refund) = 750; refund reduces spend
    assert df.loc[jan3, "expense"] == 750
    assert df.loc[jan3, "income"] == 0
    # savings movement counted in savings, excluded from expense/net
    assert df.loc[jan3, "savings"] == 5000
    # transfer + savings excluded from n_transactions (only income/expense)
    assert df.loc[jan3, "n_transactions"] == 2


def test_load_daily_df_excludes_internal_from_net(sample_json: Path) -> None:
    df = load_daily_df(path=sample_json)
    # Total net = total income - total expense, unaffected by savings/transfer
    assert df["net"].sum() == 30000 - (350 + 750)


def test_load_category_spending_df(sample_json: Path) -> None:
    df = load_category_spending_df(path=sample_json, freq="MS")

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.name == "period"
    # One month in the sample
    assert len(df) == 1
    # food spend = 350 - 50 refund = 300; shopping = 800
    assert df["food"].iloc[0] == 300
    assert df["shopping"].iloc[0] == 800
    # savings/transfer are not expenses → not present as spend
    assert "salary" not in df.columns


def test_load_transactions_df_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="nonexistent.json"):
        load_transactions_df(path=missing)


def test_load_zip_without_transactions(tmp_path: Path) -> None:
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("something_else.json", "{}")
    with pytest.raises(ValueError, match="does not contain transactions.json"):
        load_transactions_df(path=p)
