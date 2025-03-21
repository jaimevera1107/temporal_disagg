import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from .preprocessing import TimeSeriesPreprocessor
from .base import TempDisBase
from .models.chow_lin import ChowLin, ChowLinFixed, ChowLinOpt, ChowLinEcotrim, ChowLinQuilis
from .models.denton import Denton
from .models.denton_cholette import DentonCholette
from .models.dynamic_models import DynamicChowLin, DynamicLitterman
from .models.fast import Fast
from .models.fernandez import Fernandez
from .models.litterman import Litterman, LittermanOpt
from .models.ols import OLS
from .models.uniform import Uniform


class TempDisModel(TempDisBase):
    """
    Main class for temporal disaggregation using various econometric methods.

    This class allows estimation of high-frequency series based on a low-frequency
    target and a high-frequency indicator, supporting multiple disaggregation methods,
    ensemble predictions, and negative value adjustment.
    """

    def __init__(self,
                 df,
                 index_col="Index",
                 grain_col="Grain",
                 value_col="y",
                 indicator_col="X",
                 conversion="sum",
                 method="chow-lin",
                 ensemble=False,
                 adjust_negative_values=False,
                 y_hat_name="y_hat",
                 interp_method="nearest",
                 **kwargs):
        """
        Initializes the disaggregation model.

        Parameters:
            df (pd.DataFrame): Input DataFrame with required columns.
            index_col (str): Column name identifying low-frequency period.
            grain_col (str): Column name identifying high-frequency partition.
            value_col (str): Name of the low-frequency series.
            indicator_col (str): Name of the high-frequency indicator.
            conversion (str): Aggregation method ('sum', 'average', 'first', 'last').
            method (str): Disaggregation method to use.
            ensemble (bool): Whether to use ensemble of methods.
            adjust_negative_values (bool): Whether to adjust negative predictions.
            y_hat_name (str): Name of the prediction column.
            interp_method (str): Method used for interpolating missing values.
            **kwargs: Additional arguments for the estimator.
        """
        self.index_col = index_col
        self.grain_col = grain_col
        self.value_col = value_col
        self.indicator_col = indicator_col
        self.conversion = conversion
        self.method = method
        self.ensemble = ensemble
        self.adjust_negative_values = adjust_negative_values
        self.y_hat_name = y_hat_name
        self.method_kwargs = kwargs
        self.interp_method = interp_method

        self.valid_methods = {
            "ols", "denton", "denton-cholette", "chow-lin", "chow-lin-opt",
            "chow-lin-ecotrim", "chow-lin-quilis", "chow-lin-fixed", "litterman",
            "litterman-opt", "dynamic-litterman", "dynamic-chowlin", "fernandez",
            "fast", "uniform"
        }

        if self.method not in self.valid_methods:
            raise ValueError(f"Method '{self.method}' is not supported. Valid methods: {self.valid_methods}")

        preprocessor = TimeSeriesPreprocessor(
            df,
            index_col=self.index_col,
            grain_col=self.grain_col,
            value_col=self.value_col,
            indicator_col=self.indicator_col,
            interp_method=self.interp_method
        )
        self.df = preprocessor.preprocess()

    def preprocess_data(self):
        """
        Preprocesses the input data, builds the conversion matrix,
        and extracts the aligned low-frequency and indicator vectors.

        Returns:
            Tuple: (processed DataFrame, y_l, X, C)
        """
        df_processed = self.df.copy()
        C = self.build_conversion_matrix(df_processed, self.conversion)
        y_l = df_processed.groupby(df_processed.columns[0])["y"].first().values.reshape(-1, 1)
        X = df_processed["X"].values.reshape(-1, 1)
        return df_processed, y_l, X, C

    def predict(self):
        """
        Runs the disaggregation prediction using the specified method or ensemble.

        Returns:
            pd.DataFrame: DataFrame with predicted high-frequency series.
        """
        all_methods = {
            "ols": OLS(),
            "denton": Denton(),
            "denton-cholette": DentonCholette(),
            "chow-lin": ChowLin(),
            "chow-lin-opt": ChowLinOpt(),
            "chow-lin-ecotrim": ChowLinEcotrim(),
            "chow-lin-quilis": ChowLinQuilis(),
            "chow-lin-fixed": ChowLinFixed(),
            "litterman": Litterman(),
            "litterman-opt": LittermanOpt(),
            "dynamic-litterman": DynamicLitterman(),
            "dynamic-chowlin": DynamicChowLin(),
            "fernandez": Fernandez(),
            "fast": Fast(),
            "uniform": Uniform(),
        }

        df_predicted, y_l, X, C = self.preprocess_data()

        if self.ensemble:
            df_predicted = self.ensemble_prediction(
                df_predicted,
                self.conversion,
                {k: v for k, v in all_methods.items() if k != self.method}
            )
            if self.adjust_negative_values:
                df_predicted = self.adjust_negative_values_method(df_predicted)
            return df_predicted

        if self.method not in all_methods:
            raise ValueError(f"Method '{self.method}' is not supported.")

        try:
            y_hat = all_methods[self.method].estimate(y_l, X, C, **self.method_kwargs)
        except Exception:
            for fallback in ["fast", "uniform"]:
                try:
                    y_hat = all_methods[fallback].estimate(y_l, X, C, **self.method_kwargs)
                    break
                except Exception:
                    raise
            else:
                raise RuntimeError(f"All fallback methods failed when '{self.method}' was estimated.")

        df_predicted[self.y_hat_name] = y_hat.flatten()

        if self.adjust_negative_values:
            df_predicted = self.adjust_negative_values_method(df_predicted)

        return df_predicted

    def adjust_negative_values_method(self, df):
        """
        Adjusts negative predicted values within each low-frequency group.

        Parameters:
            df (pd.DataFrame): DataFrame with predicted 'y_hat' column.

        Returns:
            pd.DataFrame: Adjusted DataFrame with non-negative 'y_hat'.
        """
        df_adjusted = df.copy()
        negative_indexes = df_adjusted[df_adjusted[self.y_hat_name] < 0]["Index"].unique()

        for index in negative_indexes:
            group = df_adjusted[df_adjusted["Index"] == index].reset_index(drop=True)
            y_hat = group[self.y_hat_name].values

            if (y_hat >= 0).all():
                continue

            if self.conversion == "sum":
                negative_sum = np.abs(y_hat[y_hat < 0].sum())
                positive_values = y_hat[y_hat > 0]
                positive_sum = positive_values.sum()

                if positive_sum > 0:
                    weights = positive_values / positive_sum
                    y_hat[y_hat > 0] -= negative_sum * weights
                    y_hat[y_hat < 0] = 0
                else:
                    y_hat[:] = negative_sum / len(y_hat)

            elif self.conversion == "average":
                negative_sum = np.abs(y_hat[y_hat < 0].sum())
                positive_values = y_hat[y_hat > 0]
                positive_sum = positive_values.sum()

                if positive_sum > 0:
                    weights = positive_values / positive_sum
                    y_hat[y_hat > 0] -= negative_sum * weights
                else:
                    y_hat[:] = 0

                avg_before = y_hat.mean()
                y_hat[y_hat < 0] = 0
                avg_after = y_hat.mean()

                if avg_after > 0:
                    y_hat *= avg_before / avg_after

            elif self.conversion == "first":
                first_value = y_hat[0]
                remaining_values = y_hat[1:]

                if remaining_values.sum() < 0:
                    remaining_values[:] = 0
                else:
                    negative_sum = np.abs(remaining_values[remaining_values < 0].sum())
                    positive_values = remaining_values[remaining_values > 0]
                    positive_sum = positive_values.sum()

                    if positive_sum > 0:
                        weights = positive_values / positive_sum
                        remaining_values[remaining_values > 0] -= negative_sum * weights

                    remaining_values[remaining_values < 0] = 0

                y_hat[1:] = remaining_values
                y_hat[0] = first_value

            elif self.conversion == "last":
                last_value = y_hat[-1]
                remaining_values = y_hat[:-1]

                if remaining_values.sum() < 0:
                    remaining_values[:] = 0
                else:
                    negative_sum = np.abs(remaining_values[remaining_values < 0].sum())
                    positive_values = remaining_values[remaining_values > 0]
                    positive_sum = positive_values.sum()

                    if positive_sum > 0:
                        weights = positive_values / positive_sum
                        remaining_values[remaining_values > 0] -= negative_sum * weights

                    remaining_values[remaining_values < 0] = 0

                y_hat[:-1] = remaining_values
                y_hat[-1] = last_value

            df_adjusted.loc[df_adjusted["Index"] == index, self.y_hat_name] = y_hat

        return df_adjusted
