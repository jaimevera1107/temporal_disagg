import pandas as pd
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

class TempDisBase:
    def preprocess_inputs(self, y_l, X, C):
        y_l = np.atleast_2d(y_l).reshape(-1, 1)
        X = np.atleast_2d(X).reshape(-1, 1)

        if not isinstance(C, np.ndarray):
            raise TypeError("C must a numpy array.")
        
        if C.shape[0] != y_l.shape[0]:
            raise ValueError(f"Shape mismatch: C.shape[0] ({C.shape[0]}) != y_l.shape[0] ({y_l.shape[0]})")
        if C.shape[1] != X.shape[0]:
            raise ValueError(f"Shape mismatch: C.shape[1] ({C.shape[1]}) != X.shape[0] ({X.shape[0]})")

        return y_l, X, C

    def build_conversion_matrix(self, df, conversion):
        unique_combinations = df[["Index", "Grain"]].drop_duplicates().sort_values(["Index", "Grain"])
        unique_indexes = unique_combinations["Index"].unique()
        n_l = len(unique_indexes)
        C = np.zeros((n_l, len(df)))

        def get_conversion_vector(size, conversion):
            if conversion == "sum":
                return np.ones(size)
            elif conversion == "average":
                return np.ones(size) / size
            elif conversion == "first":
                vec = np.zeros(size)
                vec[0] = 1
                return vec
            elif conversion == "last":
                vec = np.zeros(size)
                vec[-1] = 1
                return vec
            raise ValueError("Invalid method in conversion.")

        for i, idx in enumerate(unique_indexes):
            mask = (df["Index"] == idx).values
            num_valid = np.sum(mask)
            C[i, mask] = get_conversion_vector(num_valid, conversion)

        return C
    
    def ensemble_prediction(self, df, conversion, methods):
        df_predicted = df.copy()
        C = self.build_conversion_matrix(df_predicted, conversion)
        y_l = df_predicted.groupby("Index")["y"].first().values.reshape(-1, 1)
        X = df_predicted["X"].values.reshape(-1, 1)

        def predict_method(method_name, method_class):
            try:
                pred = method_class.estimate(y_l, X, C)
                if pred is not None and not np.isnan(pred).all() and pred.shape[0] == X.shape[0]:
                    return method_name, pred.flatten()
            except Exception as e:
                print(f"Error en el método {method_name}: {e}")
            return method_name, np.full_like(X.flatten(), np.nan)

        results = Parallel(n_jobs=-1)(delayed(predict_method)(name, cls) for name, cls in methods.items())

        predictions = {name: pred for name, pred in results if not np.isnan(pred).all()}

        if not predictions:
            raise ValueError("No prediction could be estimated.")

        y_hats = np.column_stack([pred for pred in predictions.values() if not np.isnan(pred).all()])

        if y_hats.shape[1] == 0:
            raise ValueError("No prediction was valid.")

        residuals = y_l - C @ y_hats
        rmse = np.sqrt(np.nanmean(residuals**2, axis=0))
        volatility = np.nanstd(y_hats, axis=0)

        if rmse.max() == rmse.min():
            rmse_norm = np.ones_like(rmse)
        else:
            rmse_norm = (rmse - rmse.min()) / (rmse.max() - rmse.min())

        if volatility.max() == volatility.min():
            volatility_norm = np.ones_like(volatility)
        else:
            volatility_norm = (volatility - volatility.min()) / (volatility.max() - volatility.min())

        correlation = np.array([
            np.corrcoef(y_hats[:, i], y_l.flatten())[0, 1] if y_hats.shape[0] == y_l.shape[0] else 0
            for i in range(y_hats.shape[1])
        ])

        scores = (1 / (1 + rmse_norm)) * (correlation ** 2) / (1 + volatility_norm)

        def loss(w):
            return -np.dot(w, scores)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in scores]

        result = minimize(loss, x0=np.ones(len(scores)) / len(scores), bounds=bounds, constraints=constraints)
        weights = result.x if result.success else np.ones(len(scores)) / len(scores)

        df_predicted["y_hat"] = np.dot(y_hats, weights)

        return df_predicted
