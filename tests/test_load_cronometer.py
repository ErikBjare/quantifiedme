"""Tests for the Cronometer nutrition data loader."""

from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.cronometer import (
    create_fake_nutrition_df,
    load_nutrition_df,
    load_servings_df,
)


SAMPLE_CSV = """\
Date,Energy (kcal),Protein (g),Net Carbs (g),Carbs (g),Fat (g),Sugar (g),Fiber (g),Caffeine (mg),Calcium (mg)
2024-01-01,2100,115,210,235,88,45,25,120,800
2024-01-02,1950,130,190,218,72,38,28,200,750
2024-01-03,2300,140,240,275,90,55,35,80,820
"""

SAMPLE_SERVINGS_CSV = """\
Day,Food Name,Amount,Unit,Energy (kcal),Protein (g),Carbs (g),Fat (g)
2024-01-01,Oat Milk,240,ml,120,3,16,5
2024-01-01,Banana,1,medium,89,1,23,0
2024-01-02,Greek Yogurt,200,g,130,17,9,0
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "dailysummary.csv"
    p.write_text(SAMPLE_CSV)
    return p


@pytest.fixture
def sample_servings_csv(tmp_path: Path) -> Path:
    p = tmp_path / "servings.csv"
    p.write_text(SAMPLE_SERVINGS_CSV)
    return p


def test_load_nutrition_df(sample_csv: Path) -> None:
    df = load_nutrition_df(path=sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    # Check key columns are present (normalized names)
    assert "energy_kcal" in df.columns
    assert "protein_g" in df.columns
    assert "carbs_g" in df.columns
    assert "fat_g" in df.columns


def test_load_nutrition_df_sorted(sample_csv: Path) -> None:
    df = load_nutrition_df(path=sample_csv)
    assert df.index.is_monotonic_increasing


def test_load_nutrition_df_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cronometer export not found"):
        load_nutrition_df(path=tmp_path / "nonexistent.csv")


def test_load_servings_df(sample_servings_csv: Path) -> None:
    df = load_servings_df(path=sample_servings_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    assert "Food Name" in df.columns
    assert "Energy (kcal)" in df.columns


def test_load_servings_df_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Cronometer servings export not found"):
        load_servings_df(path=tmp_path / "nonexistent.csv")


def test_create_fake_nutrition_df() -> None:
    df = create_fake_nutrition_df(start="2024-01-01", end="2024-03-31")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 91  # Jan + Feb + Mar days
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"

    # Values in plausible ranges
    assert (df["energy_kcal"] > 1000).all()
    assert (df["energy_kcal"] < 4000).all()
    assert (df["protein_g"] > 0).all()
    assert (df["fiber_g"] > 0).all()


def test_fake_nutrition_df_reproducible() -> None:
    df1 = create_fake_nutrition_df(seed=42)
    df2 = create_fake_nutrition_df(seed=42)
    pd.testing.assert_frame_equal(df1, df2)
