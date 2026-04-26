import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

INSTRUMENTS = {
    "DSX":        {"cutoff": 0.4,  "unit": "IU/L", "range": (0.08, 30.0)},
    "X8":         {"cutoff": 1.5,  "unit": "IU/L", "range": (0.1,  50.0)},
    "Alinity":    {"cutoff": 3.10, "unit": "IU/L", "range": (0.27, 25.0)},
}

LATENT_CUTOFF = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_trab_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for col, cfg in INSTRUMENTS.items():
        df[f"{col}_orig_status"] = (df[col] >= cfg["cutoff"]).astype(int)

    status_cols = [f"{c}_orig_status" for c in INSTRUMENTS]
    df["true_status"] = df[status_cols].max(axis=1)

    df["sample_id"] = [f"S{i:04d}" for i in range(len(df))]

    print(f"Loaded {len(df)} samples")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TRAbNormalizer:

    def __init__(self, n_estimators: int = 300, random_state: int = 42):
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.scalers = {}
        self.pca = None
        self.rf_models = {}
        self.gb_models = {}
        self.instruments = []
        self.cv_scores_ = {}
        self.latent_train_ = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, instrument_cols: list):
        self.instruments = instrument_cols
        X_raw = df[instrument_cols].values.copy()

        # Log transform
        X_log = np.log1p(X_raw)

        # Scaling
        X_scaled = np.zeros_like(X_log)
        for i, col in enumerate(instrument_cols):
            scaler = RobustScaler()
            X_scaled[:, i] = scaler.fit_transform(
                X_log[:, i].reshape(-1, 1)).ravel()
            self.scalers[col] = scaler

        # PCA
        self.pca = PCA(n_components=len(instrument_cols))
        pca_scores = self.pca.fit_transform(X_scaled)
        latent = pca_scores[:, 0]

        # Orientation
        mean_scaled = X_scaled.mean(axis=1)
        if np.corrcoef(latent, mean_scaled)[0, 1] < 0:
            latent = -latent
            self.pca.components_[0] = -self.pca.components_[0]

        self.latent_train_ = latent

        # Regression models
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for col in instrument_cols:
            x_i = np.log1p(df[col].values).reshape(-1, 1)

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(x_i, latent)
            self.rf_models[col] = rf

            gb = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state
            )
            gb.fit(x_i, latent)
            self.gb_models[col] = gb

            cv_r2 = cross_val_score(rf, x_i, latent, cv=kf, scoring="r2")
            self.cv_scores_[col] = cv_r2

        self.is_fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def transform(self, value: float, instrument: str, model: str = "rf") -> float:
        x = np.log1p([[value]])
        if model == "rf":
            return float(self.rf_models[instrument].predict(x)[0])
        return float(self.gb_models[instrument].predict(x)[0])

    def classify(self, value: float, instrument: str,
                 latent_cutoff: float = LATENT_CUTOFF):

        ls = self.transform(value, instrument)

        x = np.log1p([[value]])
        tree_preds = np.array([t.predict(x)[0]
                               for t in self.rf_models[instrument].estimators_])
        uncertainty = float(tree_preds.std())

        orig_cutoff = INSTRUMENTS[instrument]["cutoff"]

        return {
            "raw_value": value,
            "instrument": instrument,
            "latent_score": round(ls, 4),
            "uncertainty": round(uncertainty, 4),
            "status": "POSITIVE" if ls > latent_cutoff else "NEGATIVE",
            "confidence": _confidence_from_distance(ls, latent_cutoff, uncertainty),
            "original_status": "POSITIVE" if value >= orig_cutoff else "NEGATIVE",
        }

    # ── Metrics ───────────────────────────────────────────────────────────────

    def metrics_report(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in self.instruments:
            x_i = np.log1p(df[col].values).reshape(-1, 1)
            y_pred = self.rf_models[col].predict(x_i)
            y_true = self.latent_train_

            rows.append({
                "Instrument": col,
                "R2 (train)": r2_score(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "CV R2 mean": self.cv_scores_[col].mean(),
            })

        return pd.DataFrame(rows)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_from_distance(latent, cutoff, uncertainty):
    dist = abs(latent - cutoff)
    if dist > 2 * uncertainty:
        return "High"
    elif dist > uncertainty:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    df = load_trab_dataset("/mnt/user-data/uploads/Trab.csv")

    instrument_cols = list(INSTRUMENTS.keys())

    normalizer = TRAbNormalizer()
    normalizer.fit(df, instrument_cols)

    print("\nMetrics:")
    print(normalizer.metrics_report(df))

    # Example
    result = normalizer.classify(2.5, "DSX")
    print("\nExample classification:")
    print(result)

    normalizer.save("/mnt/user-data/outputs/trab_normalizer.pkl")