# 🧠 `temporal-disagg`: Temporal Disaggregation Toolkit

📚 **Intro**  
Many official statistics and business indicators are reported at low frequencies (e.g., annually or quarterly), but decision-making often demands high-frequency data. Temporal disaggregation bridges this gap by estimating high-frequency series that remain consistent with aggregated values.  
**TempDis** provides a flexible and modular interface for performing temporal disaggregation using multiple statistical, econometric, and machine learning techniques — all in Python.

---

🎊 **Features**

**Classic disaggregation methods:**
- `Chow-Lin`: Regression-based approach using auxiliary indicators.
- `Denton` & `Denton-Cholette`: Smooth adjustment minimizing volatility.
- `Litterman` & `Fernandez`: Econometric methods rooted in time series modeling.
- `OLS`, `Fast`, `Uniform`: Simpler or fallback estimators.

**Optimized and specialized variants:**
- `Chow-Lin-opt`: Log-likelihood optimization.
- `Chow-Lin-ecotrim`, `Chow-Lin-quilis`, `Chow-Lin-fixed`: Inspired by official implementations (Eurostat, INE).

**Dynamic models:**
- `Dynamic-ChowLin`, `Dynamic-Litterman`: Incorporate time-dependent structure.

**Ensemble forecasts:**
- Combine multiple methods for more robust predictions.

**Value adjustments:**
- Optional post-processing to avoid negative values while maintaining coherence.

**Temporal aggregation:**
- Convert high-frequency data into low-frequency (`sum`, `average`, `first`, `last`).

**Retropolation tools:**
- Fill gaps or reconstruct historical data using:
  - Proportional relationships
  - Linear & Polynomial Regression
  - Exponential Smoothing
  - Neural Networks (MLP)

---

📖 **Why?**

**Short:**  
This library aims to provide a solid, extensible baseline for temporal disaggregation tasks in official statistics, forecasting, and applied data science.

**Verbose:**  
`temporal-disagg` integrates reliable methods inspired by both econometric literature and official statistics agencies. It supports reproducible workflows, custom preprocessing, and multiple disaggregation paradigms. Inspired by the R package `tempdisagg`, our goal is to create a Python-native solution that can evolve with the needs of statistical offices, researchers, and machine learning practitioners. The toolkit also includes interpolation, conversion schemes, negative-value handling, and dynamic extensions — all designed for flexibility, performance, and clarity.

---

💻 **Installation**

```bash
pip install temporal-disagg
```

---


## ⚙️ Main Class: `TempDisModel`

The `TempDisModel` is the main interface for performing disaggregation. It allows users to easily switch between different methods, customize input column names, and fine-tune prediction behavior.

### 💡 Example Usage

```python
import pandas as pd
import numpy as np
from temporal_disagg import TempDisModel

# Sample dataset
df = pd.DataFrame({
    "Index": np.repeat(np.arange(2000, 2010), 4),     # Low-frequency ID
    "Grain": np.tile(np.arange(1, 5), 10),            # High-frequency position
    "X": np.random.rand(40) * 100,                    # High-frequency indicator
    "y": np.repeat(np.random.rand(10) * 1000, 4)      # Low-frequency target
})

# Initialize the model
model = TempDisModel(
    df,
    index_col="Index",
    grain_col="Grain",
    value_col="y",
    indicator_col="X",
    method="chow-lin-opt",
    conversion="average",
    ensemble=True,
    adjust_negative_values=True
)

# Predict high-frequency series
df_pred = model.predict()
print(df_pred.head())
```

🧩 **Related Projects**

**In R:**
- [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) – Reference package for temporal disaggregation.

---

📚 **References and Acknowledgements**

This library draws inspiration from the R ecosystem and academic literature on temporal disaggregation.

Their research laid the foundation for many techniques implemented here.  
For a deeper review, we encourage exploring the reference section in the [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) R package.

---

📃 **License**  
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for more details.
