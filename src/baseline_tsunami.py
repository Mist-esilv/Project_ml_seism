from pathlib import Path
import numpy as np
import pandas as pd
from global_land_mask import globe
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ALLOWED_FEATURES = ["Latitude", "Longitude", "Focal Depth (km)", "Mag"]
SPLIT_YEAR = 2012
RANDOM_STATE = 42


def load_filtered_data(csv_path: Path) -> pd.DataFrame:
    """Load the filtered earthquakes dataset."""
    return pd.read_csv(csv_path, sep=";")


def add_geophysical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create leak-free geophysical features from immediate seismic parameters.

    Replaces raw Latitude/Longitude with 3D spherical coordinates (eliminating
    the +/-180 deg Pacific discontinuity and collinearity), adds an ocean/land
    indicator via global_land_mask, and adds log-depth + magnitude-depth ratio.
    """
    lat = X["Latitude"].to_numpy(dtype=float)
    lon = X["Longitude"].to_numpy(dtype=float)
    depth = X["Focal Depth (km)"].to_numpy(dtype=float)
    mag = X["Mag"].to_numpy(dtype=float)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    log_depth = np.log1p(np.clip(depth, 0.0, None))

    # global_land_mask expects latitudes in [-90, 90] and longitudes in [-180, 180]
    is_ocean = (~globe.is_land(lat, lon)).astype(float)

    return pd.DataFrame(
        {
            "Mag": mag,
            "Focal Depth (km)": depth,
            "log_depth": log_depth,
            "mag_depth_ratio": mag / (1.0 + log_depth),
            "geo_x": np.cos(lat_rad) * np.cos(lon_rad),
            "geo_y": np.cos(lat_rad) * np.sin(lon_rad),
            "geo_z": np.sin(lat_rad),
            "is_ocean": is_ocean,
        },
        index=X.index,
    )


def temporal_train_test_split(
    df: pd.DataFrame, split_year: int = SPLIT_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically to prevent temporal data leakage."""
    y = df["Tsu"].notna().astype(int)
    X = df[ALLOWED_FEATURES]

    train_mask = df["Year"] < split_year
    test_mask = df["Year"] >= split_year

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


def build_pipeline(estimator) -> Pipeline:
    """Build a leak-free preprocessing + estimator pipeline."""
    return Pipeline(
        steps=[
            ("feat_eng", FunctionTransformer(add_geophysical_features, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def optimize_threshold_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    beta: float = 2.0,
) -> tuple[float, float]:
    """Find the probability threshold maximizing F-beta using out-of-fold CV predictions only."""
    oof_probs = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        fbeta_score(y_train, (oof_probs >= t).astype(int), beta=beta, zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def evaluate_baseline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimize_threshold: bool = True,
) -> dict[str, float]:
    """Run 5-fold StratifiedKFold CV on train, tune threshold on CV, then evaluate on temporal test."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "pr_auc": "average_precision",
        "recall": "recall",
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    }
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    if optimize_threshold:
        best_threshold, cv_f2_opt = optimize_threshold_cv(
            pipeline, X_train, y_train, cv=cv, beta=2.0
        )
    else:
        best_threshold, cv_f2_opt = 0.5, float(cv_results["test_f2"].mean())

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "cv_pr_auc_mean": float(cv_results["test_pr_auc"].mean()),
        "cv_pr_auc_std": float(cv_results["test_pr_auc"].std()),
        "cv_f2_opt": cv_f2_opt,
        "threshold": best_threshold,
        "test_pr_auc": float(average_precision_score(y_test, y_prob)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f2": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
        "test_brier": float(brier_score_loss(y_test, y_prob)),
    }

    print(f"\n=== {name} ===")
    print(
        f"CV Train (5-fold) -> "
        f"PR-AUC: {metrics['cv_pr_auc_mean']:.3f} ± {metrics['cv_pr_auc_std']:.3f} | "
        f"F2 (seuil opt CV={best_threshold:.2f}): {cv_f2_opt:.3f}"
    )
    print(
        f"Test Temporel (>={SPLIT_YEAR}, seuil={best_threshold:.2f}) -> "
        f"PR-AUC: {metrics['test_pr_auc']:.3f} | "
        f"Precision: {metrics['test_precision']:.3f} | "
        f"Recall: {metrics['test_recall']:.3f} | "
        f"F2: {metrics['test_f2']:.3f} | "
        f"Brier: {metrics['test_brier']:.3f}"
    )
    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    df = load_filtered_data(project_root / "Data" / "earthquakes_filtered.csv")

    X_train, X_test, y_train, y_test = temporal_train_test_split(df, SPLIT_YEAR)
    print(
        f"Split temporel (< {SPLIT_YEAR} vs >= {SPLIT_YEAR}) : "
        f"Train={len(X_train)} ({y_train.mean():.1%} tsunamis) | "
        f"Test={len(X_test)} ({y_test.mean():.1%} tsunamis)"
    )

    baselines = [
        (
            "Dummy (most_frequent)",
            build_pipeline(DummyClassifier(strategy="most_frequent")),
            False,
        ),
        (
            "LogisticRegression (balanced)",
            build_pipeline(
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                )
            ),
            True,
        ),
    ]

    for name, pipe, opt_thresh in baselines:
        evaluate_baseline(
            name, pipe, X_train, y_train, X_test, y_test, optimize_threshold=opt_thresh
        )
