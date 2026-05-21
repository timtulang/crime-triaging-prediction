"""
================================================================================
 CRIME PREDICTION MODEL WITH PARTICLE SWARM OPTIMIZATION (PSO)
 Spatial-Temporal XGBoost Classifier Tuned via Custom PSO
================================================================================
 Dataset   : Chicago Police Department Crime Dataset (City of Chicago Data Portal)
 Base Model : XGBoost Gradient Boosted Classifier
 Optimizer  : Particle Swarm Optimization (PSO) — custom implementation
 Target     : Binary classification — violent vs. non-violent crime
================================================================================
"""

# ─── Standard Library ─────────────────────────────────────────────────────────
import warnings
import time
import json
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Core Numeric / ML Stack ──────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import xgboost as xgb

# ─── Optional: fetch Chicago data programmatically ────────────────────────────
# pip install sodapy
# from sodapy import Socrata

# ==============================================================================
# SECTION 1 — DATA LOADING & FEATURE ENGINEERING
# ==============================================================================

def load_chicago_data(filepath: str = None, n_rows: int = 100_000) -> pd.DataFrame:
    """
    Load the Chicago PD Crime Dataset.

    If a local CSV path is supplied the file is read directly.
    Otherwise a synthetic dataset is generated that mirrors the real schema —
    suitable for unit-testing the full pipeline without a network request.

    Real dataset URL:
        https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2

    Columns used (mirrors actual schema):
        date, primary_type, description, location_description,
        arrest, domestic, beat, district, ward, community_area,
        latitude, longitude, x_coordinate, y_coordinate, year
    """
    if filepath and Path(filepath).exists():
        print(f"[DATA] Loading real dataset from {filepath}...")
        df = pd.read_csv(filepath, nrows=n_rows, low_memory=False)
        # Standardise column names to lowercase with underscores
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    # ── Synthetic Data Generator (schema-faithful) ───────────────────────────
    print("[DATA] No local file found — generating synthetic Chicago-schema data...")

    np.random.seed(42)
    n = n_rows

    # Chicago bounding box (approximate lat/lon)
    lat_min, lat_max = 41.64, 42.02
    lon_min, lon_max = -87.94, -87.52

    primary_types = [
        "THEFT", "BATTERY", "CRIMINAL DAMAGE", "NARCOTICS",
        "ASSAULT", "BURGLARY", "MOTOR VEHICLE THEFT",
        "ROBBERY", "HOMICIDE", "SEXUAL ASSAULT"
    ]

    location_descs = [
        "STREET", "RESIDENCE", "APARTMENT", "SIDEWALK",
        "PARKING LOT", "ALLEY", "SCHOOL", "RESTAURANT",
        "GAS STATION", "GROCERY STORE"
    ]

    # Random timestamps between 2015-01-01 and 2023-12-31
    start_ts = pd.Timestamp("2015-01-01").value // 10**9
    end_ts   = pd.Timestamp("2023-12-31").value // 10**9
    timestamps = pd.to_datetime(
        np.random.randint(start_ts, end_ts, n), unit="s"
    )

    # District and beat — Chicago has 22 districts, ~280 beats
    districts = np.random.randint(1, 23, n)
    beats      = districts * 100 + np.random.randint(10, 99, n)

    df = pd.DataFrame({
        "date":                 timestamps,
        "primary_type":         np.random.choice(primary_types, n),
        "location_description": np.random.choice(location_descs, n),
        "arrest":               np.random.choice([True, False], n, p=[0.3, 0.7]),
        "domestic":             np.random.choice([True, False], n, p=[0.2, 0.8]),
        "beat":                 beats,
        "district":             districts,
        "ward":                 np.random.randint(1, 51, n),
        "community_area":       np.random.randint(1, 78, n),
        "latitude":             np.random.uniform(lat_min, lat_max, n),
        "longitude":            np.random.uniform(lon_min, lon_max, n),
        "x_coordinate":         np.random.randint(1_100_000, 1_200_000, n),
        "y_coordinate":         np.random.randint(1_800_000, 1_950_000, n),
        "year":                 timestamps.year,
    })

    return df


# ==============================================================================
# SECTION 2 — SPATIAL-TEMPORAL FEATURE ENGINEERING
# ==============================================================================

# Violent crime categories as defined by the FBI Uniform Crime Report
VIOLENT_TYPES = {"HOMICIDE", "ROBBERY", "ASSAULT", "BATTERY", "SEXUAL ASSAULT"}


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw Chicago crime records into a machine-learning-ready feature
    matrix with rich spatial-temporal signals.

    Returns
    -------
    X : pd.DataFrame  — Feature matrix
    y : pd.Series     — Binary target (1 = violent, 0 = non-violent)
    """

    print("[FEATURES] Engineering spatial-temporal features...")
    df = df.copy()

    # ── Parse timestamps ─────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", "latitude", "longitude"], inplace=True)

    # Temporal decomposition — captures day-of-week and seasonal crime cycles
    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Cyclical encoding — prevents 23→0 hour discontinuity seen by linear models
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── Spatial features ─────────────────────────────────────────────────────
    # Chicago Loop centroid — used to measure distance from city centre
    LOOP_LAT, LOOP_LON = 41.8827, -87.6233

    df["dist_to_loop"] = np.sqrt(
        (df["latitude"]  - LOOP_LAT) ** 2 +
        (df["longitude"] - LOOP_LON) ** 2
    )

    # High-density crime hotspot flag (community areas with historically high crime)
    HOTSPOT_AREAS = {25, 26, 27, 28, 29, 44, 67, 68, 71}   # e.g. Englewood, Austin
    df["is_hotspot"] = df["community_area"].isin(HOTSPOT_AREAS).astype(int)

    # ── Encode categoricals ──────────────────────────────────────────────────
    le_loc = LabelEncoder()
    df["location_enc"] = le_loc.fit_transform(
        df["location_description"].fillna("UNKNOWN")
    )

    # ── Binary target: violent crime ─────────────────────────────────────────
    y = df["primary_type"].str.upper().isin(VIOLENT_TYPES).astype(int)
    y.name = "is_violent"

    # ── Assemble feature matrix ──────────────────────────────────────────────
    feature_cols = [
        # Temporal
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dow_sin", "dow_cos", "is_weekend", "is_night", "quarter", "year",
        # Spatial
        "latitude", "longitude", "dist_to_loop", "is_hotspot",
        "beat", "district", "ward", "community_area",
        "x_coordinate", "y_coordinate",
        # Event context
        "arrest", "domestic", "location_enc",
    ]

    # Coerce booleans to int
    for col in ["arrest", "domestic"]:
        df[col] = df[col].astype(int)

    X = df[feature_cols].copy()

    print(f"[FEATURES] Matrix shape: {X.shape} | Class balance: "
          f"{y.mean():.2%} violent")
    return X, y


# ==============================================================================
# SECTION 3 — BASE MODEL DEFINITION
# ==============================================================================

def build_xgboost(params: dict) -> xgb.XGBClassifier:
    """
    Instantiate an XGBoostClassifier with the supplied hyperparameter dict.
    `params` is injected by the PSO optimizer at each particle evaluation step.
    """
    return xgb.XGBClassifier(
        n_estimators      = int(params["n_estimators"]),
        max_depth         = int(params["max_depth"]),
        learning_rate     = params["learning_rate"],
        subsample         = params["subsample"],
        colsample_bytree  = params["colsample_bytree"],
        min_child_weight  = int(params["min_child_weight"]),
        gamma             = params["gamma"],
        reg_alpha         = params["reg_alpha"],
        reg_lambda        = params["reg_lambda"],
        scale_pos_weight  = params.get("scale_pos_weight", 1.0),
        tree_method       = "hist",        # Fast histogram algorithm
        eval_metric       = "logloss",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
    )


# ==============================================================================
# SECTION 4 — PARTICLE SWARM OPTIMIZATION (PSO) — Custom Implementation
# ==============================================================================

class ParticleSwarmOptimizer:
    """
    Standard PSO (Kennedy & Eberhart, 1995) adapted for hyperparameter search.

    Hyperparameter space is mapped from a continuous unit hypercube [0, 1]^D
    to the actual parameter ranges, allowing PSO to operate natively in a
    uniform bounded space while the fitness function decodes values.

    Key components
    ──────────────
    • Inertia weight (w)      : Controls exploration/exploitation trade-off.
                                Linearly decayed from w_max → w_min over epochs.
    • Cognitive coefficient c1: Attraction toward particle's personal best.
    • Social coefficient c2   : Attraction toward swarm's global best.
    • Velocity clamping       : Prevents particles from flying out of bounds.
    """

    # ── Parameter search space definition ────────────────────────────────────
    # Each entry: (min_val, max_val, type)
    # PSO works in [0,1] space and decode() maps to this domain.
    PARAM_SPACE = {
        "n_estimators":     (100,   800,   "int"),
        "max_depth":        (3,     10,    "int"),
        "learning_rate":    (0.005, 0.30,  "float"),
        "subsample":        (0.5,   1.0,   "float"),
        "colsample_bytree": (0.4,   1.0,   "float"),
        "min_child_weight": (1,     10,    "int"),
        "gamma":            (0.0,   5.0,   "float"),
        "reg_alpha":        (0.0,   2.0,   "float"),
        "reg_lambda":       (0.5,   5.0,   "float"),
        "scale_pos_weight": (1.0,   10.0,  "float"),
    }

    def __init__(
        self,
        n_particles: int = 20,
        n_iterations: int = 30,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        cv_folds: int = 3,
        scoring: str = "f1",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w_max        = w_max
        self.w_min        = w_min
        self.c1           = c1
        self.c2           = c2
        self.cv_folds     = cv_folds
        self.scoring      = scoring
        self.rng          = np.random.default_rng(random_state)
        self.verbose      = verbose

        self.param_names = list(self.PARAM_SPACE.keys())
        self.n_dims      = len(self.param_names)

        # PSO state
        self.positions    = None    # shape: (n_particles, n_dims) in [0,1]
        self.velocities   = None
        self.pbest_pos    = None    # personal best positions
        self.pbest_scores = None    # personal best fitness values
        self.gbest_pos    = None    # global best position
        self.gbest_score  = -np.inf

        self.history      = []      # (iteration, gbest_score, gbest_params)

    # ── Decoding helpers ──────────────────────────────────────────────────────

    def _decode(self, unit_position: np.ndarray) -> dict:
        """
        Map a unit-hypercube position vector → named hyperparameter dict.

        The [0,1] encoding lets PSO treat all dimensions uniformly regardless
        of the wildly different scales (e.g., n_estimators ∈ [100, 800]
        versus learning_rate ∈ [0.005, 0.30]).
        """
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi, dtype = self.PARAM_SPACE[name]
            val = lo + unit_position[i] * (hi - lo)
            params[name] = int(round(val)) if dtype == "int" else float(val)
        return params

    # ── Fitness function ──────────────────────────────────────────────────────

    def _fitness(self, position: np.ndarray, X: pd.DataFrame, y: pd.Series) -> float:
        """
        FITNESS FUNCTION — The core evaluation oracle called for every particle
        at every iteration.

        Strategy
        ────────
        1. Decode the particle's position from [0,1]^D to real hyperparameters.
        2. Build an XGBoost model with those hyperparameters.
        3. Run stratified k-fold cross-validation on the training data.
        4. Return the mean CV F1-score (weighted) as the fitness value.

        Why F1 (weighted)?
            Crime data is class-imbalanced — violent crimes (~25-35%) are the
            minority class. F1 penalises both false positives (wasted police
            resources) and false negatives (missed violent incidents).
            A weighted F1 accounts for support in each class.

        PSO maximises this value — higher = better particle.
        """
        params = self._decode(position)
        model  = build_xgboost(params)

        # Pre-processing pipeline wrapped around each CV fold
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   model),
        ])

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipe, X, y, cv=cv,
            scoring=f"f1_weighted",    # weighted F1 across violent / non-violent
            n_jobs=-1,
        )
        return float(scores.mean())

    # ── PSO Main Loop ─────────────────────────────────────────────────────────

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Execute the PSO search and return the globally optimal hyperparameters.

        Algorithm (per iteration t)
        ───────────────────────────
          w(t) = w_max - (w_max - w_min) * t / T          [inertia decay]

          v(t+1) = w(t)*v(t)
                 + c1*r1*(pbest - x(t))                   [cognitive pull]
                 + c2*r2*(gbest - x(t))                   [social pull]

          x(t+1) = clip(x(t) + v(t+1), 0, 1)             [update + clamp]
        """
        print("\n" + "="*70)
        print("  PARTICLE SWARM OPTIMIZATION — HYPERPARAMETER SEARCH")
        print(f"  Particles: {self.n_particles} | Iterations: {self.n_iterations}"
              f" | CV folds: {self.cv_folds}")
        print("="*70)

        T = self.n_iterations

        # Initialise swarm uniformly in [0, 1]^D
        self.positions  = self.rng.uniform(0, 1, (self.n_particles, self.n_dims))
        # Velocities initialised small to avoid premature divergence
        self.velocities = self.rng.uniform(-0.1, 0.1, (self.n_particles, self.n_dims))

        self.pbest_pos    = self.positions.copy()
        self.pbest_scores = np.full(self.n_particles, -np.inf)

        # ── Evaluate initial swarm ────────────────────────────────────────────
        print("\n[PSO] Evaluating initial swarm positions...")
        for i in range(self.n_particles):
            score = self._fitness(self.positions[i], X, y)
            self.pbest_scores[i] = score
            if score > self.gbest_score:
                self.gbest_score = score
                self.gbest_pos   = self.positions[i].copy()

        print(f"[PSO] Initial gbest F1 = {self.gbest_score:.4f}")

        # ── Main optimisation loop ────────────────────────────────────────────
        for t in range(T):
            # Linear inertia decay: high at start (exploration) → low at end (exploitation)
            w = self.w_max - (self.w_max - self.w_min) * (t / T)

            for i in range(self.n_particles):
                r1 = self.rng.uniform(0, 1, self.n_dims)   # cognitive random factor
                r2 = self.rng.uniform(0, 1, self.n_dims)   # social random factor

                # Velocity update equation
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                social    = self.c2 * r2 * (self.gbest_pos    - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social

                # Clamp velocity to [-0.5, 0.5] to prevent explosion
                self.velocities[i] = np.clip(self.velocities[i], -0.5, 0.5)

                # Position update + clamp to unit hypercube
                self.positions[i] = np.clip(
                    self.positions[i] + self.velocities[i], 0.0, 1.0
                )

                # Evaluate new position
                score = self._fitness(self.positions[i], X, y)

                # Update personal best
                if score > self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_pos[i]    = self.positions[i].copy()

                # Update global best
                if score > self.gbest_score:
                    self.gbest_score = score
                    self.gbest_pos   = self.positions[i].copy()

            best_params = self._decode(self.gbest_pos)
            self.history.append((t + 1, self.gbest_score, best_params))

            if self.verbose:
                print(f"[PSO] Iter {t+1:>3}/{T} | w={w:.3f} | "
                      f"gbest F1={self.gbest_score:.4f} | "
                      f"lr={best_params['learning_rate']:.4f} | "
                      f"depth={best_params['max_depth']}")

        print("\n" + "="*70)
        print(f"  PSO COMPLETE — Best F1 (CV): {self.gbest_score:.4f}")
        print("="*70)

        return self._decode(self.gbest_pos)


# ==============================================================================
# SECTION 5 — EVALUATION UTILITIES
# ==============================================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute a comprehensive suite of classification metrics on the held-out
    test set. Returns a results dict suitable for logging to MLflow / W&B.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "f1_weighted":  f1_score(y_test, y_pred, average="weighted"),
        "f1_violent":   f1_score(y_test, y_pred, average="binary"),
        "precision":    precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":       recall_score(y_test, y_pred, average="weighted"),
    }

    print("\n" + "─"*60)
    print("  FINAL MODEL EVALUATION — HELD-OUT TEST SET")
    print("─"*60)
    print(classification_report(
        y_test, y_pred,
        target_names=["Non-Violent", "Violent"],
        digits=4,
    ))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
    print(f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")
    print("─"*60)

    return results


# ==============================================================================
# SECTION 6 — MAIN PIPELINE ORCHESTRATOR
# ==============================================================================

def main():
    print("\n" + "█"*70)
    print("  CRIME PREDICTION — PSO-TUNED XGBoost PIPELINE")
    print("█"*70)

    # ── 6.1  Load data ────────────────────────────────────────────────────────
    # Swap filepath to your local Chicago CSV for full-scale runs.
    # e.g. filepath = "/data/chicago_crimes_2015_2023.csv"
    df = load_chicago_data(filepath=None, n_rows=50_000)

    # ── 6.2  Feature engineering ──────────────────────────────────────────────
    X, y = engineer_features(df)

    # ── 6.3  Train / test split (temporal split preserves time order) ─────────
    # For real data, split by date rather than random to avoid data leakage.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 6.4  Impute missing values ────────────────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns
    )

    # ── 6.5  PSO Hyperparameter Optimisation ──────────────────────────────────
    # NOTE: For a quick demo, use n_particles=5, n_iterations=5.
    # Production run: n_particles=30, n_iterations=50+
    pso = ParticleSwarmOptimizer(
        n_particles  = 10,    # ← increase for production
        n_iterations = 10,    # ← increase for production
        w_max        = 0.9,
        w_min        = 0.4,
        c1           = 2.0,
        c2           = 2.0,
        cv_folds     = 3,
        scoring      = "f1_weighted",
        random_state = 42,
        verbose      = True,
    )

    start = time.perf_counter()
    best_params = pso.optimize(X_train_imp, y_train)
    elapsed = time.perf_counter() - start

    print(f"\n[PSO] Search completed in {elapsed/60:.1f} minutes")
    print(f"[PSO] Best hyperparameters found:")
    for k, v in best_params.items():
        print(f"       {k:<22} = {v}")

    # ── 6.6  Train final model on full training set ───────────────────────────
    print("\n[TRAIN] Fitting final XGBoost on full training set...")
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  build_xgboost(best_params)),
    ])
    final_model.fit(X_train_imp, y_train)

    # ── 6.7  Evaluate on held-out test set ────────────────────────────────────
    metrics = evaluate_model(final_model, X_test_imp, y_test)

    # ── 6.8  Persist artefacts ────────────────────────────────────────────────
    output = {
        "best_hyperparameters": best_params,
        "pso_best_cv_f1":       pso.gbest_score,
        "test_metrics":         metrics,
        "pso_history":          [(t, s) for t, s, _ in pso.history],
        "elapsed_seconds":      elapsed,
    }

    with open("/home/claude/pso_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n[DONE] Results saved to pso_results.json")
    print(f"[DONE] Test F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"[DONE] Test F1 (violent class): {metrics['f1_violent']:.4f}")

    return final_model, best_params, metrics


if __name__ == "__main__":
    final_model, best_params, metrics = main()
