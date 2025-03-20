import pandas as pd
import pytest
from temporal_disagg.aggregation import TemporalAggregation

def test_aggregation_sum():
    df = pd.DataFrame({
        "time": pd.date_range(start="2020-01-01", periods=6, freq="D"),
        "value": [10, 20, 30, 40, 50, 60]
    })

    aggregator = TemporalAggregation(conversion="sum")
    aggregated_df = aggregator.aggregate(df, time_col="time", value_col="value", freq="ME")

    assert aggregated_df["value"].sum() == sum(df["value"])  # La suma debe coincidir

def test_aggregation_invalid_method():
    with pytest.raises(ValueError):
        TemporalAggregation(conversion="invalid")
