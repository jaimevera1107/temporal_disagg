import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed

class TimeSeriesPreprocessor:
    def __init__(self, df=None, index_col=None, grain_col=None, value_col=None, indicator_col=None, freq_mapping=None, interp_method='nearest'):
        self.df = df.copy() if df is not None else None
        self.index_col = index_col
        self.grain_col = grain_col
        self.value_col = value_col
        self.indicator_col = indicator_col
        self.freq_mapping = freq_mapping if freq_mapping else {}
        self.interp_method = interp_method

        if self.df is not None:
            self.validate_data()

    def validate_data(self):
        if self.df is None:
            raise ValueError("El DataFrame no puede ser None.")

        required_cols = [self.index_col, self.grain_col, self.value_col]
        if self.indicator_col:
            required_cols.append(self.indicator_col)

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas obligatorias: {missing_cols}")

        if self.df[self.value_col].isnull().all():
            raise ValueError(f"Todos los valores de la columna '{self.value_col}' están vacíos.")

        for col in [self.value_col, self.indicator_col]:
            if col and col in self.df.columns and not np.issubdtype(self.df[col].dtype, np.number):
                raise TypeError(f"La columna '{col}' debe ser numérica.")

    def normalize_dates(self):
        if not np.issubdtype(self.df[self.index_col].dtype, np.number):
            self.df[self.index_col] = pd.to_numeric(self.df[self.index_col], errors='coerce')
        if self.grain_col in self.df.columns:
            if self.freq_mapping and self.df[self.grain_col].dtype == 'O':
                self.df[self.grain_col] = self.df[self.grain_col].map(self.freq_mapping)
        if self.df[self.grain_col].dtype == 'O':
            self.df[self.grain_col] = pd.factorize(self.df[self.grain_col])[0]

    def handle_missing_values(self):
        self.df[self.value_col] = self.df[self.value_col].interpolate(method=self.interp_method).ffill().bfill()

    def create_complete_series(self):
        all_indices = self.df[self.index_col].unique()
        all_grains = self.df[self.grain_col].unique()
        full_index = pd.MultiIndex.from_product([all_indices, all_grains], names=[self.index_col, self.grain_col])
        self.df = self.df.set_index([self.index_col, self.grain_col]).reindex(full_index).reset_index()

    def preprocess(self):
        self.validate_data()
        self.normalize_dates()
        self.create_complete_series()
        self.handle_missing_values()
        return self.df