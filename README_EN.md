# ☢️ Nuclear Power Plant Accident Classifier

A machine learning project that predicts the type of accident occurring at a
Pressurized Water Reactor (PWR) nuclear power plant, based on sensor readings
(pressure, temperature, flow rate, etc.).

**Author:** Doston — Tashkent State Technical University, Faculty of Power
Engineering, Nuclear Power Plants specialization, 4th year

**Live demo:** [https://npp-accident-classifier-kvvxnnyg2cb4erv6cc9eot.streamlit.app/](https://npp-accident-classifier-kvvxnnyg2cb4erv6cc9eot.streamlit.app/)

## 📊 Dataset

[NPPAD (Nuclear Power Plant Accident Data)](https://github.com/thu-inet/NuclearPowerPlantAccidentData) —
a dataset generated using the **PCTRAN** simulator by Tsinghua University,
published in *Scientific Data* (Nature). It covers 18 operating conditions
(17 accident types + Normal operation) for a PWR reactor, with time-series
readings of 97 physical parameters per scenario.

## 🔧 Pipeline

1. **Data loading** — 1200+ raw CSV files (one per accident instance),
   converted into a single tabular dataset by extracting statistical
   features (mean, std, min, max) per event. Rare single-file classes
   were augmented via time-window splitting to partially address severe
   class imbalance.
2. **EDA** — analyzed class distribution, missing values (98% missing in
   24 columns, dropped), and feature correlations (dropped 288 redundant
   quantile columns highly correlated with the mean).
3. **Preprocessing** — reduced from 793 to 331 features by removing
   leaky, constant, and redundant columns; label-encoded the target.
4. **Modeling** — compared Logistic Regression, Decision Tree, and Random
   Forest, validated with 5-fold Stratified Cross-Validation to rule out
   a lucky train/test split.
5. **Evaluation** — beyond accuracy, examined per-class Precision/Recall/
   F1 and the Confusion Matrix, since class imbalance can hide poor
   performance on rare classes.

## 🏆 Results

| Model | Test Accuracy | 5-Fold CV Mean |
|---|---|---|
| Baseline (most frequent class) | 8.63% | — |
| Logistic Regression | 96.08% | 94.40% |
| Decision Tree | 97.25% | 96.76% |
| **Random Forest (final)** | **97.25%** | **97.05%** |

- **Macro F1:** 0.86 — pulled down by classes with very few test samples
- **Weighted F1:** 0.97

## ⚠️ Limitations

- Classes ATWS, LOF, Normal, SP, and TT originally had only a single raw
  simulation file each; time-windowing was used to generate more samples,
  but true physical diversity remains limited (as reflected in their lower
  F1 scores: 0.40–0.67).
- `class_weight='balanced'` did not meaningfully improve rare-class
  performance, since the bottleneck is sample scarcity rather than model
  weighting.
- This is an academic/research project, not intended for real nuclear
  plant monitoring or safety-critical use.

## 🚀 Try it yourself

A Streamlit web app is included with two modes:
- **Manual input** — enter the 20 most important sensor features (by
  Random Forest feature importance); remaining features default to
  dataset-wide means.
- **Batch CSV upload** — upload a CSV matching the model's expected
  columns for predictions on multiple events at once.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository structure

- `app.py` — Streamlit application
- `model.pkl` — trained Random Forest classifier
- `scaler.pkl` — fitted StandardScaler
- `feature_names.pkl` — ordered list of the 331 model input features
- `class_names.pkl` — accident type labels (index → name mapping)
- `feature_importance.pkl` — feature importance ranking
- `feature_means.pkl` — dataset-wide feature means (used as defaults)
- `requirements.txt` — Python dependencies
