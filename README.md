# Predicting IPC Phase 3+ Food Insecurity
**MSc Data Science Project | In Collaboration with [Zero Hunger Lab](https://www.tilburguniversity.edu/research/institutes-and-research-groups/zero-hunger-lab) & Tilburg University**

### Project Overview
This repository contains the end-to-end predictive analytics pipeline developed to forecast **IPC Phase 3+ (Crisis or worse) food insecurity conditions** across 5 sub-Saharan African nations (Central African Republic, Madagascar, Mozambique, Somalia, and South Sudan) using open-source, public indicators. Leveraging multi-domain spatial-temporal datasets provided in collaboration with Tilburg University's **[Zero Hunger Lab](https://www.tilburguniversity.edu/research/institutes-and-research-groups/zero-hunger-lab)**, this project successfully applies cost-sensitive machine learning to transform volatile environmental and conflict data into high-recall, policy-relevant humanitarian insights, contributing directly to the UN Sustainable Development Goal 2 (SDG2).

### Tech Stack & Methodology
- **Language & Environment:** Python, Jupyter Notebook
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, SHAP
- **Validation Framework:** Chronological TimeSeriesSplit Cross-Validation (K=5) with strict time-based Train/Test split to prevent temporal data leakage.
- **Imbalance Mitigation:** Cost-sensitive risk minimization (inverse frequency class weighting & `scale_pos_weight` alignment) to address severe historical class skewness (90.1% Crisis vs. 9.9% Safe observations).

### Pipeline Architecture
The project is structured as a sequential, reproducible workflow:
- `01_Data Cleaning.ipynb`: Handles raw data ingestion, time-series formatting, primary missing value imputations, and sub-national population capacity estimations (`admin1_pop_est`).
- `02_EDA.ipynb`: Conducts exploratory data analysis, mapping data missingness, and calculating early correlation heuristics.
- `03_Check Data Pipeline.ipynb`: Validates lag feature engineering (t-1 to t-3), checks for chronological integrity, and ensures zero data leakage.
- `04_Model Training.ipynb`: Deploys automated grid searches for Logistic Regression, Random Forest, and XGBoost, optimizing for minority-class recall across short-term ($h=1$) and medium-term ($h=3$) horizons.

### Key Project Insights
- **IPC History Dominance:** Both baseline metrics and SHAP global feature attribution identify historical IPC records (`affected_pct_3p`) as the single most influential signal predicting future targets, reflecting the high temporal persistence of food crises.
- **Conflict & Macroeconomic Catalysts:** Feature importance and SHAP analysis highlight that recent local conflict escalations (`conflict_events_lag1`) act as primary catalysts for crisis entry, while staple food price spikes (`maize_lag1`) and mid-season cumulative precipitation deficits (`rain_mm_lag2`) serve as lagging dependencies that prolong crisis states.
- **Added Modeling Value:** While food security is highly persistent, our predictive models successfully outperform rigid persistence baselines during crucial inflection and state-recovery phases, providing critical early-warning capability in data-scarce environments.
