from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ALLOWED_FEATURES = ["Latitude", "Longitude", "Focal Depth (km)", "Mag"]
SPLIT_YEAR = 2012
RANDOM_STATE = 42


def load_filtered_data(csv_path: Path) -> pd.DataFrame:
    """Load the filtered earthquakes dataset."""
    return pd.read_csv(csv_path, sep=";")


def add_geophysical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create 3D spherical coordinates and seismic energy features without leakage."""
    X_feat = X.copy()
    lat_rad = np.deg2rad(X_feat["Latitude"])
    lon_rad = np.deg2rad(X_feat["Longitude"])

    X_feat["geo_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    X_feat["geo_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    X_feat["geo_z"] = np.sin(lat_rad)
    X_feat["log_energy_proxy"] = 1.5 * X_feat["Mag"]
    X_feat["mag_depth_ratio"] = X_feat["Mag"] / (1.0 + np.log1p(X_feat["Focal Depth (km)"]))
    return X_feat


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


def evaluate_baseline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Run StratifiedKFold CV on train, then fit on train and evaluate on temporal test."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "pr_auc": "average_precision",
        "recall": "recall",
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    }
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(
        f"CV Train (5-fold) -> "
        f"PR-AUC: {cv_results['test_pr_auc'].mean():.3f} ± {cv_results['test_pr_auc'].std():.3f} | "
        f"Recall: {cv_results['test_recall'].mean():.3f} | "
        f"F2: {cv_results['test_f2'].mean():.3f}"
    )
    print(
        f"Test Temporel (>={SPLIT_YEAR}) -> "
        f"PR-AUC: {average_precision_score(y_test, y_prob):.3f} | "
        f"Recall: {recall_score(y_test, y_pred):.3f} | "
        f"F2: {fbeta_score(y_test, y_pred, beta=2, zero_division=0):.3f} | "
        f"Brier: {brier_score_loss(y_test, y_prob):.3f}"
    )


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
        ("Dummy (most_frequent)", build_pipeline(DummyClassifier(strategy="most_frequent"))),
        (
            "LogisticRegression (balanced)",
            build_pipeline(
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
                )
            ),
        ),
    ]

    for name, pipe in baselines:
        evaluate_baseline(name, pipe, X_train, y_train, X_test, y_test)
