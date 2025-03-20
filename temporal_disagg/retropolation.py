import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


class Retropolation:
    """
    Class for retropolarizing a new data series based on an old data series,
    using different statistical and Machine Learning methods.
    """

    def __init__(self, df, new_col, old_col, interp_method="linear"):
        """
        Initializes the class with the DataFrame and relevant columns.

        Parameters:
        - df: pd.DataFrame -> DataFrame containing the data series.
        - new_col: str -> Name of the column with the new methodology.
        - old_col: str -> Name of the column with the old methodology.
        - interp_method: str -> Interpolation method to fill missing values (default: "linear").
        """
        try:
            self.df = df.copy()
            self.new_col = new_col
            self.old_col = old_col
            self.interp_method = interp_method

            # Fill missing values using interpolation
            self.df[self.new_col] = self.df[self.new_col].interpolate(method=self.interp_method).ffill().bfill()
            self.df[self.old_col] = self.df[self.old_col].interpolate(method=self.interp_method).ffill().bfill()

            # 🔹 Si después de la interpolación hay menos de 3 muestras, deshabilitar estimación
            valid_data = self.df.dropna(subset=[self.new_col, self.old_col])
            self.disable_estimation = valid_data.shape[0] < 3

        except Exception as e:
            print(f"Error initializing Retropolation: {e}")

    def _convert_input(self, X, y=None):
        """Converts inputs to NumPy arrays or Pandas Series depending on their structure."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy().reshape(-1, 1)
        if y is not None and isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy().ravel()
        return X, y

    def _linear_regression(self, mask_retropolar):
        """Applies linear regression for retropolarization."""
        try:
            valid_data = self.df.dropna(subset=[self.new_col, self.old_col])
            if valid_data.shape[0] < 2:
                return

            X, y = self._convert_input(valid_data[self.old_col], valid_data[self.new_col])
            model = LinearRegression().fit(X, y)
            X_pred, _ = self._convert_input(self.df.loc[mask_retropolar, self.old_col])
            self.df.loc[mask_retropolar, self.new_col] = model.predict(X_pred)

        except Exception as e:
            print(f"Error in _linear_regression: {e}")

    def _polynomial_regression(self, mask_retropolar):
        """Applies polynomial regression for retropolarization."""
        try:
            valid_data = self.df.dropna(subset=[self.new_col, self.old_col])
            if valid_data.shape[0] < 3:
                return

            X, y = self._convert_input(valid_data[self.old_col], valid_data[self.new_col])
            poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            poly_model.fit(X, y)
            X_pred, _ = self._convert_input(self.df.loc[mask_retropolar, self.old_col])
            self.df.loc[mask_retropolar, self.new_col] = poly_model.predict(X_pred)

        except Exception as e:
            print(f"Error in _polynomial_regression: {e}")

    def _exponential_smoothing(self, mask_retropolar, alpha=0.5):
        """Applies exponential smoothing to retropolarize values."""
        try:
            smoothed_values = self.df[self.new_col].ewm(alpha=alpha, adjust=False).mean()
            if smoothed_values.dropna().shape[0] < 3:
                return

            self.df.loc[mask_retropolar, self.new_col] = smoothed_values.iloc[-1]

        except Exception as e:
            print(f"Error in _exponential_smoothing: {e}")

    def _mlp_regression(self, mask_retropolar):
        """Applies regression using a neural network (MLP Regressor)."""
        try:
            valid_data = self.df.dropna(subset=[self.new_col, self.old_col])
            if valid_data.shape[0] < 5:
                return

            X, y = self._convert_input(valid_data[self.old_col], valid_data[self.new_col])
            model = MLPRegressor(hidden_layer_sizes=(1000,), max_iter=10000, activation="tanh", alpha=0.001, random_state=0)
            model.fit(X, y)
            X_pred, _ = self._convert_input(self.df.loc[mask_retropolar, self.old_col])
            self.df.loc[mask_retropolar, self.new_col] = model.predict(X_pred)

        except Exception as e:
            print(f"Error in _mlp_regression: {e}")

    def retropolate(self, method="proportion"):
        """
        Executes retropolarization using the specified method.

        Parameter:
        - method: str -> Method to use ('proportion', 'linear_regression', 'polynomial_regression',
                          'exponential_smoothing', 'mlp_regression').

        Returns:
        - pd.Series -> Series with retropolarized values.
        """
        try:
            methods = {
                "proportion": self._linear_regression,
                "linear_regression": self._linear_regression,
                "polynomial_regression": self._polynomial_regression,
                "exponential_smoothing": self._exponential_smoothing,
                "mlp_regression": self._mlp_regression,
            }

            if method not in methods:
                print(f"🚨 Método inválido detectado: {method}")
                raise ValueError(f"Invalid method '{method}'. Choose from: {list(methods.keys())}")
            
            if self.disable_estimation:
                warnings.warn("Not enough data for estimation. Returning the interpolated series.")
                return self.df[self.new_col]

            mask_retropolar = self.df[self.new_col].isna() & self.df[self.old_col].notnull()

            methods[method](mask_retropolar)

            return self.df[self.new_col]

        except Exception as e:
            print(f"Error in retropolate: {e}")
            raise
