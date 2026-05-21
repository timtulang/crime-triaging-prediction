# TECHNICAL TURNOVER DOCUMENT
## PSO-Optimised XGBoost Crime Prediction System
### Prepared for: ML Engineering & MLOps Teams

---

| Field              | Detail                                            |
|--------------------|---------------------------------------------------|
| **Project Code**   | CRIME-PRED-PSO-v1.0                              |
| **Model**          | XGBoost Classifier (PSO-Tuned)                   |
| **Dataset**        | Chicago PD Crime Dataset (2015–2023)             |
| **Target**         | Binary — Violent vs. Non-Violent Crime           |
| **Prepared By**    | Lead AI Research & ML Engineering                |
| **Doc Version**    | 1.0                                              |
| **Classification** | INTERNAL — RESTRICTED                            |

---

## 1. PROJECT EXECUTIVE SUMMARY

### 1.1 Mission Objective
This system delivers a **spatial-temporal crime prediction classifier** designed to assist Chicago PD resource allocation teams in identifying patrol zones with elevated probability of violent crime occurrence. The model ingests historical incident records and outputs a binary label (**violent / non-violent**) per geographic cell per time window.

### 1.2 Chosen Architecture

| Component | Choice | Rationale |
|---|---|---|
| **Base Model** | `XGBClassifier` (XGBoost v2.x) | Handles mixed-type tabular features natively; built-in missing-value handling; state-of-the-art on structured data benchmarks |
| **Tuning Strategy** | Particle Swarm Optimization (PSO) | Population-based; escapes local optima; no gradient information required; parallelisable fitness evaluation |
| **Preprocessing** | `sklearn.Pipeline` (Imputer → Scaler → XGB) | Prevents data leakage across CV folds; deployment-safe serialisation |
| **Evaluation** | Weighted F1 + Per-class F1 | Robust to class imbalance; penalises both FP and FN asymmetrically |

### 1.3 Why PSO Over Grid / Random Search
- **Grid Search** is computationally infeasible at scale: 10 hyperparameters × 5 values each = 5^10 ≈ **10 million** configurations.
- **Random Search** improves coverage but has no memory — it cannot exploit promising regions of the search landscape.
- **PSO** maintains a **swarm of candidate solutions** that collectively converge toward optima through velocity-based movement, combining cognitive (personal best) and social (global best) signals. This escapes the flat regions and saddle points that trap greedy search methods.

---

## 2. DATA ENGINEERING PIPELINE

### 2.1 Source Dataset

| Property | Detail |
|---|---|
| **Name** | City of Chicago — Crimes 2001 to Present |
| **URL** | `https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2` |
| **Format** | CSV / Socrata API (JSON) |
| **Volume** | ~8 million records (2001–2024); recommended training window: 2015–2023 |
| **Refresh Rate** | Daily |
| **License** | City of Chicago Open Data License |

### 2.2 Raw Schema — Key Columns

| Column | Type | Usage |
|---|---|---|
| `date` | Timestamp | Temporal feature source |
| `primary_type` | String | **Target label construction** |
| `location_description` | String | Encoded categorical feature |
| `arrest` | Boolean | Binary contextual feature |
| `domestic` | Boolean | Binary contextual feature |
| `beat` | Int | Spatial granularity (finest level) |
| `district` | Int | Spatial mid-level |
| `ward` | Int | Political/administrative zone |
| `community_area` | Int | Neighbourhood-level hotspot flag source |
| `latitude` | Float | Geospatial coordinate |
| `longitude` | Float | Geospatial coordinate |
| `x_coordinate` | Int | Illinois State Plane coordinate (East) |
| `y_coordinate` | Int | Illinois State Plane coordinate (North) |
| `year` | Int | Temporal drift signal |

### 2.3 Preprocessing Steps (Ordered)

```
Raw CSV
  │
  ▼
1. LOAD & STANDARDISE
   - Strip/lowercase column names
   - Parse `date` → pd.Timestamp
   - Drop rows missing {date, latitude, longitude}

  │
  ▼
2. TARGET CONSTRUCTION
   - Map primary_type ∈ {HOMICIDE, ROBBERY, ASSAULT,
     BATTERY, SEXUAL ASSAULT} → is_violent = 1
   - All other types → is_violent = 0

  │
  ▼
3. TEMPORAL DECOMPOSITION
   - Extract: hour, day_of_week, month, quarter, year
   - Derive: is_weekend, is_night
   - Cyclical encode (sin/cos): hour, month, day_of_week
     [CRITICAL: prevents discontinuity artefacts at midnight/Monday/January]

  │
  ▼
4. SPATIAL FEATURE ENGINEERING
   - Euclidean distance to Chicago Loop centroid (41.8827, -87.6233)
   - Binary hotspot flag: community_area ∈ HOTSPOT_AREAS
   - Retain: beat, district, ward, community_area, x/y coords

  │
  ▼
5. CATEGORICAL ENCODING
   - location_description → LabelEncoder → location_enc
   - arrest, domestic → int (0/1)

  │
  ▼
6. IMPUTATION (median) — sklearn SimpleImputer
   [fit on train split ONLY; transform applied to test]

  │
  ▼
7. SCALING — StandardScaler
   [fit on train split ONLY; inside sklearn.Pipeline]

  │
  ▼
Final Feature Matrix: 23 columns
```

### 2.4 Data Leakage Prevention
- All `fit()` operations (imputer, scaler, label encoder) are performed **exclusively on the training fold**.
- Temporal train/test split is performed **before** any feature engineering to simulate real deployment conditions.
- The `primary_type` column is **dropped** from the feature matrix after target construction.

---

## 3. OPTIMIZATION FRAMEWORK

### 3.1 PSO Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `n_particles` | 30 (prod) / 10 (demo) | Population diversity vs. compute budget |
| `n_iterations` | 50 (prod) / 10 (demo) | Convergence depth |
| `w_max` | 0.9 | High initial inertia → broad exploration |
| `w_min` | 0.4 | Low final inertia → fine exploitation |
| `c1` (cognitive) | 2.0 | Attraction to personal best |
| `c2` (social) | 2.0 | Attraction to global best |
| `cv_folds` | 3 | StratifiedKFold — preserves class ratio |
| `fitness_metric` | `f1_weighted` | Class-imbalance robust |
| `velocity_clamp` | ±0.5 (unit space) | Prevents divergence |

### 3.2 Hyperparameter Search Space

| Hyperparameter | Min | Max | Type | Notes |
|---|---|---|---|---|
| `n_estimators` | 100 | 800 | int | Number of boosting rounds |
| `max_depth` | 3 | 10 | int | Tree depth — controls variance |
| `learning_rate` | 0.005 | 0.30 | float | Shrinkage factor |
| `subsample` | 0.50 | 1.00 | float | Row subsampling ratio |
| `colsample_bytree` | 0.40 | 1.00 | float | Column subsampling |
| `min_child_weight` | 1 | 10 | int | Minimum leaf node weight |
| `gamma` | 0.0 | 5.0 | float | Minimum loss reduction to split |
| `reg_alpha` | 0.0 | 2.0 | float | L1 regularisation |
| `reg_lambda` | 0.5 | 5.0 | float | L2 regularisation |
| `scale_pos_weight` | 1.0 | 10.0 | float | Class imbalance correction |

### 3.3 Fitness Function Definition

```
fitness(particle_i) = mean(StratifiedKFold CV F1_weighted)
                      where model hyperparameters = decode(position_i)
```

PSO **maximises** this function. A particle's position in [0,1]^10 is
decoded to the above domain before each XGBoost instantiation.

### 3.4 Computational Budget Estimate

| Config | Evaluations | Est. Time (8-core) |
|---|---|---|
| Demo (10×10) | 110 | ~5 min |
| Standard (20×30) | 620 | ~45 min |
| Production (30×50) | 1,530 | ~2 hrs |

---

## 4. EVALUATION METRICS & DEPLOYMENT

### 4.1 Success Criteria (Production Gate)

| Metric | Minimum Threshold | Target |
|---|---|---|
| F1 (weighted) | ≥ 0.82 | ≥ 0.87 |
| F1 (violent class) | ≥ 0.78 | ≥ 0.84 |
| Precision (violent) | ≥ 0.75 | ≥ 0.82 |
| Recall (violent) | ≥ 0.80 | ≥ 0.86 |
| False Negative Rate | ≤ 0.20 | ≤ 0.14 |

> **Critical Note:** In the law enforcement domain, **False Negatives** (missed violent crimes) carry higher operational cost than False Positives. The Recall threshold for the violent class is therefore set higher than Precision.

### 4.2 Deployment Architecture

```
[Chicago Data Portal API]
         │  (daily pull via Socrata client)
         ▼
[Raw Ingestion Layer — Apache Airflow DAG]
         │  (schema validation, deduplication)
         ▼
[Feature Engineering Service — FastAPI]
         │  (same transforms as training pipeline)
         ▼
[Model Serving — FastAPI + joblib-serialised Pipeline]
         │  predict() → probability score + binary label
         ▼
[Output Store — PostgreSQL / PostGIS]
         │  (beat-level predictions stored geospatially)
         ▼
[Dashboard — Grafana / ArcGIS]
```

### 4.3 Model Serialisation

```python
import joblib

# Save
joblib.dump(final_pipeline, "models/crime_pred_pso_v1.pkl")

# Load (inference service)
pipeline = joblib.load("models/crime_pred_pso_v1.pkl")
predictions = pipeline.predict(X_new)
```

### 4.4 Model Drift Monitoring

#### Data Drift Detection
- Monitor **PSI (Population Stability Index)** on top 5 features monthly.
  - PSI < 0.10 → Stable
  - PSI 0.10–0.25 → Warning
  - PSI > 0.25 → **Retrain triggered**
- Tool recommendation: `evidently` library

#### Concept Drift Detection
- Maintain a **rolling 30-day F1** computed on newly labelled incidents (arrest data provides ground truth with ~2-week lag).
- Trigger full PSO retraining if rolling F1 drops > 5% below production baseline.

#### Scheduled Retraining
- **Monthly**: Incremental retraining with new incident data appended.
- **Quarterly**: Full PSO hyperparameter re-optimisation.
- **Annually**: Full feature set review and spatial hotspot re-calibration.

### 4.5 Ethical & Compliance Considerations

| Risk | Mitigation |
|---|---|
| Racial/geographic bias amplification | Audit feature importance — flag if district/community_area dominates |
| Feedback loop (predictions → more arrests → more data in those areas) | Stratified evaluation across all 77 community areas quarterly |
| PII exposure | Dataset contains no individual identifiers — only incident type and location |
| Model transparency | SHAP values must be generated and available for audit |

---

## 5. ENVIRONMENT & DEPENDENCIES

### 5.1 Runtime Requirements

```
Python          >= 3.10
xgboost         >= 2.0.0
scikit-learn    >= 1.4.0
pandas          >= 2.0.0
numpy           >= 1.26.0
joblib          >= 1.3.0
sodapy          >= 2.2.0    # Chicago Data Portal API client
evidently       >= 0.4.0    # Drift monitoring
```

### 5.2 Infrastructure Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU cores | 8 | 32 |
| RAM | 16 GB | 64 GB |
| Storage | 50 GB | 200 GB |
| GPU | Not required | Optional (tree_method="gpu_hist") |

### 5.3 Key File Manifest

| File | Purpose |
|---|---|
| `crime_prediction_pso.py` | Full training + PSO pipeline |
| `pso_results.json` | PSO run artefacts (params, history, metrics) |
| `models/crime_pred_pso_v1.pkl` | Serialised production pipeline |
| `TECHNICAL_TURNOVER.md` | This document |

---

## 6. HANDOFF CHECKLIST

- [ ] Real Chicago PD CSV downloaded and path set in `load_chicago_data(filepath=...)`
- [ ] PSO parameters scaled to production (`n_particles=30`, `n_iterations=50`)
- [ ] Temporal train/test split updated to use date column (not random)
- [ ] Hotspot community areas validated against latest CPD district maps
- [ ] SHAP explainability module integrated before stakeholder demos
- [ ] Airflow DAG for daily data ingestion deployed and tested
- [ ] Drift monitoring dashboards live in Grafana
- [ ] Model card completed and signed off by ethics review board
- [ ] All thresholds in Section 4.1 met on real dataset before go-live

---

*End of Technical Turnover Document — v1.0*
