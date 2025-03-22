import pandas as pd
import numpy as np
import pytest
from temporal_disagg.estimation import TempDisModel

def test_temp_dis_model():
    df = pd.DataFrame({
        "Index": np.repeat(np.arange(2000, 2010), 4),
        "Grain": np.tile(np.arange(1, 5), 10),
        "X": np.random.rand(40) * 100,
        "y": np.repeat(np.random.rand(10) * 1000, 4)
    })

    model = TempDisModel(
            df=df,
            method="ols"
        )

    result = model.predict()

    assert "y_hat" in result.columns
    assert not result["y_hat"].isnull().all()


def test_temp_dis_model_custom_cols():
    df = pd.DataFrame({
        "anio": np.repeat(np.arange(2000, 2010), 4),
        "mes": np.tile(np.arange(1, 5), 10),
        "meta": np.repeat(np.random.rand(10) * 1000, 4),
        "indicador": np.random.rand(40) * 100
    })

    model = TempDisModel(
        df=df,
        index_col="anio",
        grain_col="mes",
        value_col="meta",
        indicator_col="indicador",
        method="ols"
    )

    result = model.predict()


    assert "y_hat" in result.columns
    assert not result["y_hat"].isnull().all()