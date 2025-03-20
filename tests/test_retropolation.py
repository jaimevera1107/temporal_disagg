import pytest
import pandas as pd
import numpy as np
from temporal_disagg.retropolation import Retropolation

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "year": range(2000, 2010),
        "new_col": [100, 110, 120, np.nan, np.nan, 160, 170, np.nan, 190, 200],
        "old_col": [90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    })

def test_initialization(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    assert isinstance(obj.df, pd.DataFrame)
    assert "new_col" in obj.df.columns
    assert "old_col" in obj.df.columns

def test_proportion(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    result = obj.retropolate(method="proportion")
    assert not result.isnull().any()
    assert len(result) == len(sample_df)

def test_linear_regression(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    result = obj.retropolate(method="linear_regression")
    assert not result.isnull().any()
    assert len(result) == len(sample_df)

def test_polynomial_regression(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    result = obj.retropolate(method="polynomial_regression")
    assert not result.isnull().any()
    assert len(result) == len(sample_df)

def test_exponential_smoothing(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    result = obj.retropolate(method="exponential_smoothing")
    assert not result.isnull().any()
    assert len(result) == len(sample_df)

def test_mlp_regression(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    result = obj.retropolate(method="mlp_regression")
    assert not result.isnull().any()
    assert len(result) == len(sample_df)

def test_invalid_method(sample_df):
    obj = Retropolation(sample_df, "new_col", "old_col")
    with pytest.raises(ValueError, match="Invalid method 'invalid_method'. Choose from: .*"):
        obj.retropolate(method="invalid_method")
