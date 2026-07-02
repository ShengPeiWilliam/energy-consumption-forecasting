# Tetouan City Energy Consumption Forecasting

10-minute interval power consumption forecasting across three distribution zones
in Tetouan, Morocco (2017), using XGBoost with temporal feature engineering to
reduce MAPE from 21--27% (ARIMA baseline) to **under 1%** in two of three zones.
Zone 3 remains harder at 2.51%. At a one-week horizon, Zone 3 error reaches 16.23%
while Zones 1 and 2 stay below 7%. Dropping collection frequency from 10 to
60 minutes costs 5 percentage points in Zone 3 but only 2 in Zone 1.

Beyond accuracy, the analysis asks whether three geographically distinct zones
are driven by the same factors, how performance degrades across forecasting
horizons, and what coarser sampling frequencies actually cost in precision.
These are the questions that determine whether a model is usable in deployment.

[See business recommendations →](#business-recommendations)

## Motivation

An earlier project ([NYC Citibike Demand Analysis](https://github.com/ShengPeiWilliam/citibike-aws-pipeline))
used Bayesian Structural Time Series to understand *why* ridership differs
across rider types: inference, not prediction. This project takes the
complementary approach. Given a full year of 10-minute power consumption data
across three zones of Tetouan, Morocco, how accurately can we predict what
comes next?

The problem generalizes to any utility: electricity, gas, water. Grid operators
need accurate short-term forecasts to anticipate demand, allocate resources, and
respond to unexpected spikes before they become outages. At the user level,
zone-level forecasts can surface when and why consumption peaks, and SHAP
importance answers which external factors actually drive it.

## Design Decisions

**Why XGBoost over a neural network?**

On moderately sized tabular time series data, XGBoost with engineered lag and
calendar features matches or outperforms neural networks without the tuning
overhead. More importantly, XGBoost supports SHAP directly, which lets us
decompose each prediction into feature contributions and explain to operators
exactly what is driving a forecast.

**Why use ARIMA as the baseline?**

ARIMA is the natural starting point: no external features, no engineering, just
the time series predicting itself. Its MAPE of 21--27% is not a failure. It is the floor you get for free. The roughly 20x gap between ARIMA and XGBoost
shows how much the engineered features are actually contributing.

**Why the sampling frequency analysis?**

In practice, not every facility can afford 10-minute sensor data. We wanted to
know the actual tradeoff: if you collect less frequently, how much accuracy do
you lose? The results give a concrete answer for each zone, which helps allocate
sensor budget where it matters most.

## Key Results

**Headline finding**: XGBoost with temporal feature engineering reduces MAPE
from 21--27% to under 1% for Zones 1 and 2. Zone 3 remains harder at 2.51%,
with higher anomaly frequency.
Forecast accuracy degrades predictably with horizon and sampling frequency,
with Zone 3 deteriorating roughly twice as fast as the other two zones.

| Model | Zone 1 MAPE | Zone 2 MAPE | Zone 3 MAPE |
|-------|-------------|-------------|-------------|
| ARIMA Baseline | 21.88% | 21.11% | 26.91% |
| **XGBoost** | **0.98%** | **0.86%** | **2.51%** |

**Forecasting horizon** (XGBoost, predicting N steps ahead):

| Horizon | Zone 1 | Zone 2 | Zone 3 |
|---------|--------|--------|--------|
| 10 min (1 step) | 0.98% | 0.86% | 2.51% |
| 1 hour (6 steps) | 2.63% | 2.79% | 6.18% |
| 6 hours (36 steps) | 3.07% | 3.88% | 8.03% |
| 1 day (144 steps) | 2.55% | 3.51% | 7.96% |
| 1 week (1008 steps) | 6.74% | 4.74% | 16.23% |

**Sampling frequency** (full pipeline retrained at each resolution):

| Frequency | Zone 1 | Zone 2 | Zone 3 | Training Rows |
|-----------|--------|--------|--------|---------------|
| 10 min | 0.98% | 0.86% | 2.51% | 52,416 |
| 20 min | 1.40% | 1.35% | 3.82% | 26,208 |
| 30 min | 1.94% | 1.75% | 4.58% | 17,472 |
| 60 min | 2.88% | 2.90% | 7.41% | 8,736 |

## Business Recommendations

**1. Match collection frequency to the operational goal.** Real-time grid balancing requires 10-minute data. The model's strongest feature is the previous-step lag. For capacity planning and load projection, 20 to 30 minute intervals sacrifice only 1 to 2 percentage points of MAPE while cutting sensor and storage costs by 50 to 67%.

**2. Calibrate collection density by zone.** Zone 3 reaches 7.41% MAPE at hourly collection; Zones 1 and 2 stay below 3%. High-variance zones cannot be downsampled as aggressively. A sampling frequency analysis should precede any deployment to determine the minimum acceptable interval per zone.

> *Two neighborhoods, same city, same sensors. One holds 2% error at 30-minute data. The other needs 10-minute data just to stay under 5%. The zone decides the infrastructure, not the other way around.*

**3. Accuracy loss from downsampling is nonlinear.** The jump from 10 to 20 minutes costs roughly 0.5 percentage points. The jump from 30 to 60 minutes costs 1 to 3 percentage points. The cost-effective range is 20 to 30 minutes. Beyond that, each step trades disproportionately more accuracy for diminishing storage savings.

**4. Use residual-based alerting as a safety net.** When prediction error exceeds the 99th percentile threshold, flag the interval for review. Zone 3 anomalies are not attributable to a single weather feature, making rule-based detection unreliable. Cross-zone correlation during flagged periods may surface grid-level events invisible to single-zone monitoring.

> *November 3, 2017: Zone 1 spikes at 10:00, Zone 2 follows at 11:20. A single-zone alert flags an anomaly. Cross-zone correlation reveals a grid-level event.*

---

## Reflections & Next Steps

The main lesson: a single accuracy number is not enough to characterize a
forecasting model's value. The same model architecture applied to three zones
produces very different operational profiles. Treating them identically would
systematically underestimate risk in the most volatile zone and over-invest
sensors in the most stable ones.

Next steps:
- **Richer weather features for longer horizons**: Short-term accuracy is
  dominated by lag features, leaving little room for weather to contribute.
  At longer horizons where lag features lose relevance, richer weather inputs
  such as cloud cover or irradiance forecasts may help reduce the steep
  accuracy drop-off observed beyond one hour.
- **Cross-zone correlation**: Residuals across the three zones may be correlated
  during anomaly periods. A multi-output model that exploits this structure
  could improve Zone 3 accuracy by borrowing information from the better-behaved zones.

## Repository

```
energy-consumption-forecasting/
├── code/
│   ├── energy_consumption_analysis.ipynb  # Full analysis pipeline
│   ├── energy_consumption_analysis.html   # Rendered notebook for viewing
│   └── config.py                          # Paths and modeling constants
└── requirements.txt                       # Python dependencies
```

## Tools

**Statistical methods**: XGBoost, ARIMA, SHAP, ACF/PACF, Anomaly Detection (residual thresholding)  
**Language**: Python  
**Libraries**: xgboost, shap, statsmodels, scikit-learn, pandas, matplotlib, seaborn  
**Data**: UCI Machine Learning Repository, Tetouan City Power Consumption

## References

Salam, A., & El Hibaoui, A. (2018). Comparison of Machine Learning Algorithms
for the Power Consumption Prediction. *2018 6th International Renewable and
Sustainable Energy Conference (IRSEC)*, 1--5.

UCI Machine Learning Repository. (2020).
[Tetouan City Power Consumption Dataset](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city).
