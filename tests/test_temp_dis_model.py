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
