import pandas as pd
import pytest
from temporal_disagg.preprocessing import TimeSeriesPreprocessor

def test_preprocessing_valid_data():
    df = pd.DataFrame({
        "Index": pd.date_range(start="2020-01-01", periods=5, freq="ME"),
        "Grain": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50]
    })

    preprocessor = TimeSeriesPreprocessor(df, index_col="Index", grain_col="Grain", value_col="y")
    processed_df = preprocessor.preprocess()

    assert not processed_df["y"].isnull().any()

def test_preprocessing_missing_values():
    df = pd.DataFrame({
        "Index": pd.date_range(start="2020-01-01", periods=5, freq="ME"),
        "Grain": [1, 2, 3, 4, 5],
        "y": [10, None, 30, None, 50]
    })

    preprocessor = TimeSeriesPreprocessor(df, index_col="Index", grain_col="Grain", value_col="y")
    processed_df = preprocessor.preprocess()

    assert not processed_df["y"].isnull().any()  # Debe llenar los valores vacíos
