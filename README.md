# Temporal Disaggregation

A **Python library** for **temporal disaggregation** of time series data using various statistical and econometric methods.

This library enables **high-frequency estimation** from low-frequency data, ensuring consistency with the original aggregated series. It includes **multiple disaggregation methods**, such as:

- **Chow-Lin** (and variations: maxlog, minrss, fixed, ecotrim, quilis)
- **Denton** and **Denton-Cholette**
- **Fernandez**
- **Litterman**
- **OLS-based methods**
- **Machine Learning-based retropolation**
- **Exponential Smoothing**
- **Ensemble Predictions**

## 🚀 **Features**
✅ Disaggregate **low-frequency time series** into **higher frequencies** (e.g., annual → quarterly).  
✅ Supports **multiple statistical and econometric methods** for temporal disaggregation.  
✅ Includes **machine learning models** for time series retropolation.  
✅ Provides **interpolation and missing value handling**.  
✅ Compatible with **pandas**, **numpy**, and **scikit-learn**.  

---

## 📦 **Installation**
You can install the package using **pip**:

```bash
pip install git+https://github.com/jaimevera1107/temp_disagg.git
```

## ⚙️ **Example of usage**:

```python
import pandas as pd
import numpy as np
from temporal_disagg import TempDisModel

# Create sample data
df = pd.DataFrame({
    "Index": np.repeat(np.arange(2000, 2010), 4),  # Years (low frequency)
    "Grain": np.tile(np.arange(1, 5), 10),  # Quarterly data
    "X": np.random.rand(40) * 100,  # High-frequency indicator
    "y": np.repeat(np.random.rand(10) * 1000, 4)  # Low-frequency series
})

# Instantiate the model
model = TempDisModel(
    df, 
    conversion="average", 
    method="chow-lin-opt", 
    ensemble=True, 
    adjust_negative_values=True
)

# Run prediction
df_pred = model.predict()

# Display results
print(df_pred.head())

```