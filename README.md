# Predicting IPC Phase 3+ Food Insecurity
**MSc Data Science Project | In Collaboration with [Zero Hunger Lab](https://www.tilburguniversity.edu/research/institutes-and-research-groups/zero-hunger-lab) & Tilburg University**

### Project Overview
This repository contains the end-to-end predictive analytics pipeline developed to forecast **IPC Phase 3+ (Crisis or worse) food insecurity conditions** across 5 sub-Saharan African nations (Central African Republic, Madagascar, Mozambique, Somalia, and South Sudan) using open-source, public indicators[cite: 1, 2]. Leveraging multi-domain spatial-temporal datasets provided in collaboration with Tilburg University's **[Zero Hunger Lab](https://www.tilburguniversity.edu/research/institutes-and-research-groups/zero-hunger-lab)**, this project successfully applies cost-sensitive machine learning to transform volatile environmental and conflict data into high-recall, policy-relevant humanitarian insights, contributing directly to the UN Sustainable Development Goal 2 (SDG2)[cite: 1].

### Tech Stack & Methodology
- **Language & Environment:** Python, Jupyter Notebook[cite: 2]
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, SHAP[cite: 2]
- **Validation Framework:** Chronological TimeSeriesSplit Cross-Validation (K=5) with strict time-based Train/Test split to prevent temporal data leakage[cite: 4].
- **Imbalance Mitigation:** Cost-sensitive risk minimization (inverse frequency class weighting & `scale_pos_weight` alignment) to address severe historical class skewness (90.1% Crisis vs. 9.9% Safe observations)[cite: 4].

### Pipeline Architecture
The project is structured as a sequential, reproducible workflow:
- `01_Data Cleaning.ipynb`: Handles raw data ingestion, time-series formatting, primary missing value imputations, and sub-national population capacity estimations (`admin1_pop_est`)[cite: 2, 4].
- `02_EDA.ipynb`: Conducts exploratory data analysis, mapping data missingness, and calculating early correlation heuristics[cite: 1].
- `03_Check Data Pipeline.ipynb`: Validates lag feature engineering (t-1 to t-3), checks for chronological integrity, and ensures zero data leakage[cite: 1].
- `04_Model Training.ipynb`: Deploys automated grid searches for Logistic Regression, Random Forest, and XGBoost, optimizing for minority-class recall across short-term ($h=1$) and medium-term ($h=3$) horizons[cite: 1, 4].

### Key Project Insights
- **IPC History Dominance:** Both baseline metrics and SHAP global feature attribution identify historical IPC records (`affected_pct_3p`) as the single most influential signal predicting future targets, reflecting the high temporal persistence of food crises[cite: 1, 4].
- **Conflict & Macroeconomic Catalysts:** Feature importance and SHAP analysis highlight that recent local conflict escalations (`conflict_events_lag1`) act as primary catalysts for crisis entry[cite: 4], while staple food price spikes (`maize_lag1`) and mid-season cumulative precipitation deficits (`rain_mm_lag2`) serve as lagging dependencies that prolong crisis states[cite: 4].
- **Added Modeling Value:** While food security is highly persistent, our predictive models successfully outperform rigid persistence baselines during crucial inflection and state-recovery phases, providing critical early-warning capability in data-scarce environments[cite: 1, 4, 5].
