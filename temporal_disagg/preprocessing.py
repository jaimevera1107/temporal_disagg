import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed


class TimeSeriesPreprocessor:
    """
    Preprocessor class for preparing time series data for temporal disaggregation.

    This class ensures consistency, fills missing values, handles interpolation,
    and generates a complete grid of (Index, Grain) combinations for proper alignment.
    """

    def __init__(
        self,
        df=None,
        index_col=None,
        grain_col=None,
        value_col=None,
        indicator_col=None,
        interp_method='nearest'
    ):
        """
        Initializes the preprocessor with required parameters.

        Parameters:
            df (pd.DataFrame): The input time series DataFrame.
            index_col (str): Column indicating the low-frequency index.
            grain_col (str): Column indicating the high-frequency grain/position.
            value_col (str): Column containing the target variable (low-freq).
            indicator_col (str): Column containing the high-frequency indicator.
            interp_method (str): Interpolation method for missing values (default: 'nearest').
        """
        self.df = df.copy() if df is not None else None
        self.index_col = index_col
        self.grain_col = grain_col
        self.value_col = value_col
        self.indicator_col = indicator_col
        self.interp_method = interp_method

        if self.df is not None:
            self.validate_data()

    def validate_data(self):
        """
        Validates the input DataFrame and required columns.
        """
        if self.df is None:
            raise ValueError("Input DataFrame cannot be None.")

        required_cols = [self.index_col, self.grain_col, self.value_col]
        if self.indicator_col:
            required_cols.append(self.indicator_col)

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required column(s): {missing_cols}")

        if self.df[self.value_col].isnull().all():
            raise ValueError(f"All values in '{self.value_col}' are missing.")

        for col in [self.value_col, self.indicator_col]:
            if col and col in self.df.columns and not np.issubdtype(self.df[col].dtype, np.number):
                raise TypeError(f"Column '{col}' must be numeric.")

    def normalize_dates(self):
        """
        Ensures numeric format for index and encodes grain labels numerically if needed.
        """
        if not np.issubdtype(self.df[self.index_col].dtype, np.number):
            self.df[self.index_col] = pd.to_numeric(self.df[self.index_col], errors='coerce')

        if self.grain_col in self.df.columns:
            if pd.api.types.is_object_dtype(self.df[self.grain_col].dtype):
                self.df[self.grain_col] = pd.factorize(self.df[self.grain_col])[0]

    def handle_missing_values(self):
        """
        Interpolates and fills missing values in the value column.
        """
        self.df[self.value_col] = (
            self.df[self.value_col]
            .interpolate(method=self.interp_method)
            .ffill()
            .bfill()
        )

    def create_complete_series(self):
        """
        Expands the DataFrame to contain all combinations of (Index, Grain).
        """
        all_indices = self.df[self.index_col].unique()
        all_grains = self.df[self.grain_col].unique()
        full_index = pd.MultiIndex.from_product(
            [all_indices, all_grains],
            names=[self.index_col, self.grain_col]
        )
        self.df = (
            self.df.set_index([self.index_col, self.grain_col])
            .reindex(full_index)
            .reset_index()
        )

    def preprocess(self):
        """
        Runs the full preprocessing pipeline.

        Returns:
            pd.DataFrame: The cleaned and completed time series DataFrame.
        """
        self.validate_data()
        self.normalize_dates()
        self.create_complete_series()
        self.handle_missing_values()
        return self.df
