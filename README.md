# Predicting IPC Phase 3+ Food Insecurity
**MSc Data Science Project | In Collaboration with [Zero Hunger Lab](https://www.zerohungerlab.nl/) & Tilburg University**

### Project Overview
This repository contains the end-to-end predictive analytics pipeline developed to forecast **IPC Phase 3+ (Crisis or worse) food insecurity conditions** across 5 sub-Saharan African nations (Central African Republic, Madagascar, Mozambique, Somalia, and South Sudan). Leveraging multi-domain open-source spatial-temporal datasets provided in collaboration with the **[Zero Hunger Lab](https://www.zerohungerlab.nl/)**, this project successfully applies cost-sensitive machine learning to transform volatile environmental and conflict data into high-recall, policy-relevant humanitarian insights.

### Tech Stack & Methodology
- **Language & Environment:** Python, Jupyter Notebook
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost
- **Validation Framework:** Chronological TimeSeriesSplit Cross-Validation (K=5) with strict time-based Train/Test split.
- **Imbalance Mitigation:** Cost-sensitive risk minimization (inverse frequency class weighting & `scale_pos_weight` alignment).

### Pipeline Architecture
The project is structured as a sequential, reproducible workflow:
- `01_Data Cleaning.ipynb`: Handles raw data ingestion, time-series formatting, and primary missing value imputations.
- `02_EDA.ipynb`: Conducts exploratory data analysis, mapping data missingness, and calculating early correlation heuristics.
- `03_Check Data Pipeline.ipynb`: Validates lag feature engineering (t-1 to t-3), checks for chronological integrity, and ensures zero data leakage.
- `04_Model Training.ipynb`: Deploys automated grid searches for Logistic Regression, Random Forest, and XGBoost, optimizing for minority-class recall across short-term ($h=1$) and medium-term ($h=3$) horizons.

### Key Project Insights
- **Conflict Domination:** Feature importance metrics identify recent conflict escalations (`conflict_events_lag1`) as the single most critical catalyst for sudden acute food insecurity transitions.
- **Macroeconomic & Climate Chocks:** Staple food price spikes (`maize_lag1`) and mid-season cumulative precipitation deficits (`rain_mm_lag2`) act as major lagging dependencies that prolong crisis states.
- **Added Modeling Value:** While food security is highly persistent, our predictive models successfully outperform rigid persistence baselines during crucial inflection and state-recovery phases.
