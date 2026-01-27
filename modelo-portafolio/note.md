# Portfolio Optimization Pipeline — Notes

## Overview
This document describes the **end-to-end pipeline** used to optimize an **existing equity portfolio** using:
- **Yahoo Finance** data
- **Monthly SARIMA forecasts** per asset
- **Robust expected returns (μ)**
- **Mean–Variance Optimization (MV)** with real-world constraints

The goal is to provide **a stable, defensible portfolio recommendation** for a **~1-year horizon**, not frequent trading.

---

## 1. Portfolio Input
The pipeline starts from a **user-defined portfolio** expressed as invested amounts:

```python
portfolio = {
    "AAPL": 1500.0,
    "MSFT": 2000.0,
    "GOOGL": 1800.0,
    "AMZN": 3000.0,
    "INTC": 3000.0
}
```

From this:
- Tickers are extracted automatically
- Current portfolio weights are computed as:

\[
w_i = \frac{investment_i}{\sum investment}
\]

These weights represent the **baseline portfolio** used for comparison.

---

## 2. Data Extraction (Yahoo Finance)
- Historical prices are downloaded with `yfinance`
- `auto_adjust=True` ensures dividends and splits are included
- A long historical window is used (5–10 years recommended)

Data quality checks:
- Align trading calendars
- Drop assets with insufficient data coverage

---

## 3. Monthly Returns
- Daily prices are resampled to **end-of-month**
- Monthly returns are computed:

\[
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1
\]

Monthly frequency is chosen to:
- Reduce noise
- Match the investment horizon
- Stabilize forecasting and covariance estimation

---

## 4. Forecasting with SARIMA (Per Asset)
- A **separate SARIMA model** is fit for each asset
- Frequency: **monthly**
- Horizon: **12 months (~1 year)**

Key principles:
- Forecasting is done **per asset**, never on combinations
- SARIMA is used for stability and seasonality handling
- If a model fails, a **historical-mean fallback** is applied

The output is a 12-step monthly return forecast per ticker.

---

## 5. Expected Return from Forecast (μ_ARIMA)
The 12 monthly forecasts are compounded into a **1-year expected return**:

\[
\mu^{ARIMA}_i = \prod_{k=1}^{12}(1 + \hat{r}_{i,k}) - 1
\]

This produces a **vector of expected annual returns** derived from the forecast.

---

## 6. Historical Expected Return (μ_hist)
To avoid over-reliance on forecasts:
- Historical monthly mean returns are computed
- They are annualized to obtain a long-term expectation

This represents a **structural, stable view** of each asset’s performance.

---

## 7. Robust Expected Return (μ_final)
Because Mean–Variance optimization is highly sensitive to μ, we blend:

\[
\mu_{final} = (1 - \alpha)\,\mu_{hist} + \alpha\,\mu_{ARIMA}
\]

Where:
- \( \alpha \in [0.1, 0.3] \)
- Forecasts influence the result, but do not dominate it

This improves robustness and interpretability.

---

## 8. Risk Estimation (Covariance Matrix Σ)
- Covariance is estimated from **monthly returns**
- Only portfolio assets are used
- The matrix is annualized:

\[
\Sigma_{annual} = 12 \cdot \Sigma_{monthly}
\]

This matrix captures **diversification and co-movement** between assets.

---

## 9. Mean–Variance Optimization
The portfolio is optimized by **maximizing the Sharpe ratio**:

\[
\max_w \frac{w^T \mu_{final} - r_f}{\sqrt{w^T \Sigma w}}
\]

### Constraints
- Long-only: \( w_i \ge 0 \)
- Full investment: \( \sum w_i = 1 \)
- Minimum weight per asset (e.g. 1%)
- Maximum weight per asset (e.g. 25%)
- Maximum annual volatility (e.g. 22.5%)
- Risk-free rate included (e.g. 4%)

These constraints make the solution **realistic and defensible**.

---

## 10. Portfolio Recommendation
The optimizer returns:
- Recommended weights per ticker
- Difference vs current weights (increase / decrease)

Portfolio-level metrics:
- Expected 1-year return
- Expected annual volatility
- Sharpe ratio (real, using rf)

This is a **one-time structural recommendation**, not a trading signal.

---

## 11. Performance Comparison (Visualization)
To communicate results:
- Daily prices are used
- Original and recommended portfolios are evaluated with **static weights**
- Growth of **$1 invested** is plotted over time
- Optional benchmark (e.g. SPY) can be included

This helps validate whether the recommendation:
- Improves risk-adjusted performance
- Reduces concentration
- Behaves reasonably over time

---

## 12. What This Pipeline Is (and Is Not)

### This pipeline **IS**:
- Strategic portfolio optimization
- Medium-term (≈1 year) recommendation
- Stable, explainable, business-friendly

### This pipeline **IS NOT**:
- High-frequency trading
- Monthly rebalancing engine
- Pure forecasting system

---

## 13. Possible Next Enhancements
- Covariance shrinkage (Ledoit–Wolf)
- Penalty for deviation vs current weights (turnover control)
- Transaction costs and taxes
- Rolling backtests for robustness
- Scenario / stress testing

---

## Final Note
This pipeline balances **statistical modeling** and **investment realism**.
Forecasting informs decisions, but **robust optimization and constraints drive the final result**.
