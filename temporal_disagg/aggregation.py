import pandas as pd
import numpy as np

class TemporalAggregation:
    """
    Class for temporal aggregation of high-frequency time series into low-frequency series.
    """

    def __init__(self, conversion="sum"):
        """
        Initializes the TemporalAggregation class.

        Parameters:
            conversion (str): Specifies the aggregation method:
                - "sum": Sum of the high-frequency values.
                - "average": Mean of the high-frequency values.
                - "first": First observed value.
                - "last": Last observed value.
        """
        if conversion not in ["sum", "average", "first", "last"]:
            raise ValueError("Invalid conversion method. Choose from 'sum', 'average', 'first', 'last'.")
        
        self.conversion = conversion

    def aggregate(self, df, time_col, value_col, freq):
        """
        Aggregates high-frequency data into low-frequency data.

        Parameters:
            df (pd.DataFrame): DataFrame containing time series data.
            time_col (str): Column representing the time index.
            value_col (str): Column with values to aggregate.
            freq (str): Target frequency ('M' for monthly, 'Q' for quarterly, 'A' for annual).

        Returns:
            pd.DataFrame: Aggregated time series.
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        
        if self.conversion == "sum":
            aggregated = df[value_col].resample(freq).sum()
        elif self.conversion == "average":
            aggregated = df[value_col].resample(freq).mean()
        elif self.conversion == "first":
            aggregated = df[value_col].resample(freq).first()
        elif self.conversion == "last":
            aggregated = df[value_col].resample(freq).last()
        
        return aggregated.reset_index()