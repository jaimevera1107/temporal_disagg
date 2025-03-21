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
<br><br>

💻 **Installation**

```bash
pip install temporal-disagg
```
<br><br>
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
    ensemble=False,
    adjust_negative_values=False
)

# Predict high-frequency series
df_pred = model.predict()
print(df_pred.head())
```
<br><br><br>

## 🧠 Optional Parameters Explained:

You can control additional behavior by setting the following parameters when initializing the model:

✅ `ensemble=True`: Combines predictions from multiple methods into a single ensemble forecast.

```python
model = TempDisModel(
    df,
    method="chow-lin-opt",
    ensemble=True  # ← Enables ensemble mechanism
)
```
Useful when you want to increase robustness and mitigate the risk of poor performance from a single method.

---
### 🤖 How does the Ensemble Prediction work?:

The ensemble option in TempDisModel allows combining multiple disaggregation methods into a single, optimized forecast. Instead of relying on a single model (e.g., only Chow-Lin), it computes predictions using multiple alternative methods in parallel and then integrates them into one final estimate.

Here's how it works:

- Parallel Estimation: All candidate methods (excluding the one selected as method) are run independently to generate predictions.

- Quality Scoring: For each method, the following metrics are computed:

  - RMSE between the aggregated prediction and the original low-frequency target.

  - Volatility of the high-frequency prediction.

  - Correlation between the predicted and true aggregated series.

- Weighted Aggregation: A scoring function based on the metrics above assigns higher weights to more accurate and stable models. These weights are optimized using a constrained minimization problem (with weights summing to 1).

- Final Output: The ensemble forecast is a weighted average of all valid predictions.


**This ensemble mechanism is particularly useful when no single method performs consistently well across all periods or series. It improves robustness by exploiting the strengths of different disaggregation algorithms**.

---
<br><br>

✅ `adjust_negative_values=True`: Automatically adjusts negative predicted values after disaggregation.

```python
model = TempDisModel(
    df,
    adjust_negative_values=True  # ← Enables correction of negative values
)
```
Important when disaggregating non-negative series like production, sales, or population.

---

### 🚫 How does the Negative Value Adjustment work?:

Some disaggregation methods (especially regression-based ones) may yield negative values in the high-frequency output, which are often not meaningful (e.g., negative GDP or population).

When adjust_negative_values=True, TempDisModel automatically rebalances the forecast within each low-frequency group to eliminate negatives without violating the original aggregate.

Depending on the aggregation method used (sum, average, first, last), a different adjustment rule is applied:

- Sum / Average:

  - Negative values are zeroed.

  - The "missing mass" is proportionally redistributed among positive values.

- First / Last:

  - The first or last value is preserved.

  - All remaining values are corrected to keep the total (or mean) consistent while eliminating negatives.

**This adjustment is done after prediction and helps maintain the interpretability of the results, especially for non-negative variables like production, employment, or financial indicators**.


---
<br><br><br>

## 🗂️ Input Time Series Format

To use `TempDisModel`, your time series data must be organized in a **long-format DataFrame** with one row per high-frequency observation. The model requires the following columns:

| Column          | Description |
|-----------------|-------------|
| `Index`         | Identifier for the low-frequency group (e.g., year, quarter). This groups the target values. |
| `Grain`         | Identifier for the high-frequency breakdown within each `Index` (e.g., month, quarter number). |
| `y`             | The **low-frequency target variable** (repeated across the group). This is the variable to disaggregate. |
| `X`             | The **high-frequency indicator** variable (available at the granular level). Used to guide the disaggregation. |

---

#### 🔢 Example Structure

| Index | Grain | y       | X         |
|-------|-------|---------|-----------|
| 2000  | 1     | 1000.00 | 80.21     |
| 2000  | 2     | 1000.00 | 91.13     |
| 2000  | 3     | 1000.00 | 85.44     |
| 2000  | 4     | 1000.00 | 92.32     |
| 2001  | 1     | 1200.00 | 88.71     |
| 2001  | 2     | 1200.00 | 93.55     |
| ...   | ...   | ...     | ...       |

---

#### 📌 Notes

- The `y` column (low-frequency target) should have the **same value repeated** within each `Index` group.
- The number of `Grain` values per `Index` must be **consistent** (e.g., always 4 quarters, 12 months).
- The `X` column (indicator) should vary across rows and be on the **high-frequency level**.
- You may use any column names, but you must pass them via:
  ```python
  TempDisModel(..., index_col="your_index", grain_col="your_grain", ...)
  ```
---

### 📊 Summary of Temporal Disaggregation Methods
| Method             | Estimation Approach                                                    | Requires Rho | Optimizes Rho | Covariance / Penalization Structure              | Notes                                                              |
|--------------------|------------------------------------------------------------------------|--------------|----------------|--------------------------------------------------|--------------------------------------------------------------------|
| `OLS`              | Ordinary Least Squares: ŷ = Xβ                                        | ❌           | ❌             | None                                             | Simple linear regression, no temporal smoothing                    |
| `ChowLin`          | GLS with Toeplitz(ρ) covariance matrix                                | ✅           | ❌             | Σ = (1 / (1 - ρ²)) · Toeplitz(ρ)                 | Classical Chow-Lin with fixed autocorrelation                      |
| `ChowLinFixed`     | Same as `ChowLin` but with ρ = 0.9 fixed                              | ✅           | ❌             | Toeplitz                                         | Hard-coded autocorrelation                                         |
| `ChowLinOpt`       | Same as `ChowLin`, but optimizes ρ via log-likelihood                 | ✅           | ✔️             | Toeplitz                                         | Finds ρ that maximizes likelihood                                  |
| `ChowLinEcotrim`   | Chow-Lin variant with ρ ≈ 0.75 fixed                                  | ✅           | ❌             | Toeplitz                                         | Mimics Eurostat's Ecotrim                                          |
| `ChowLinQuilis`    | Chow-Lin with numerical correction for ρ                              | ✅           | ❌             | Toeplitz with epsilon adjustment                 | Quilis method (INE Spain)                                          |
| `Litterman`        | GLS with autoregressive structure Σ = (I - ρL)⁻¹ (I - ρL)⁻ᵀ           | ✅           | ❌             | AR smoothing                                    | Smooths differences with lagged dependence                         |
| `LittermanOpt`     | Same as `Litterman`, but optimizes ρ by minimizing residuals          | ✅           | ✔️             | AR smoothing                                    | Optimal smoothing parameter                                        |
| `DynamicChowLin`   | Optimized ρ + Chow-Lin estimation                                      | ✅           | ✔️             | Toeplitz                                         | Same as ChowLinOpt                                                 |
| `DynamicLitterman` | Optimized ρ + Litterman estimation                                     | ✅           | ✔️             | AR smoothing                                    | Same as LittermanOpt                                               |
| `Denton`           | Minimizes squared differences: ∑(Δʰu)²                                | ❌           | ❌             | Penalization via difference matrix              | Smooths changes across high-freq series                            |
| `DentonCholette`   | Same as `Denton`, but adjusts relative to an indicator X              | ❌           | ❌             | Penalizes difference from indicator             | Cholette's refinement                                              |
| `Fernandez`        | Covariance from inverse of ΔᵀΔ                                        | ❌           | ❌             | Inverse difference penalty                       | Based on AR(1)-like assumptions                                   |
| `Fast`             | Litterman with ρ = 0.9 fixed                                           | ✅           | ❌             | AR smoothing                                    | Fast fallback without optimization                                |
| `Uniform`          | No smoothing; assumes identity matrix (Σ = I)                         | ❌           | ❌             | None                                             | Spreads residuals uniformly                                        |

---
<br><br><br>

## 🧩 **Related Projects**

**In R:**
- [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) – Reference package for temporal disaggregation.

---

## 📚 **References and Acknowledgements**

This library draws inspiration from the R ecosystem and academic literature on temporal disaggregation.

Their research laid the foundation for many techniques implemented here.  
For a deeper review, we encourage exploring the reference section in the [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) R package.

---

## 📃 **License**  
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for more details.
