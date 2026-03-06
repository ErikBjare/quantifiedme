"""Tests for the Cronometer nutrition data loader."""

import io
from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.cronometer import create_fake_nutrition_df, load_nutrition_df


SAMPLE_CSV = """\
Date,Energy (kcal),Protein (g),Net Carbs (g),Carbs (g),Fat (g),Sugar (g),Fiber (g),Caffeine (mg),Calcium (mg)
2024-01-01,2100,115,210,235,88,45,25,120,800
2024-01-02,1950,130,190,218,72,38,28,200,750
2024-01-03,2300,140,240,275,90,55,35,80,820
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "dailysummary.csv"
    p.write_text(SAMPLE_CSV)
    return p


def test_load_nutrition_df(sample_csv: Path) -> None:
    df = load_nutrition_df(path=sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.index.name == "timestamp"
    assert df.index.tz is not None  # timezone-aware
    # Check key columns are present (normalized names)
    assert "energy_kcal" in df.columns
    assert "protein_g" in df.columns
    assert "carbs_g" in df.columns
    assert "fat_g" in df.columns


def test_load_nutrition_df_sorted(sample_csv: Path) -> None:
    df = load_nutrition_df(path=sample_csv)
    assert df.index.is_monotonic_increasing


def test_create_fake_nutrition_df() -> None:
    df = create_fake_nutrition_df(start="2024-01-01", end="2024-03-31")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 91  # Jan + Feb + Mar days
    assert df.index.name == "timestamp"
    assert df.index.tz is not None

    # Values in plausible ranges
    assert (df["energy_kcal"] > 1000).all()
    assert (df["energy_kcal"] < 4000).all()
    assert (df["protein_g"] > 0).all()
    assert (df["fiber_g"] > 0).all()


def test_fake_nutrition_df_reproducible() -> None:
    df1 = create_fake_nutrition_df(seed=42)
    df2 = create_fake_nutrition_df(seed=42)
    pd.testing.assert_frame_equal(df1, df2)
