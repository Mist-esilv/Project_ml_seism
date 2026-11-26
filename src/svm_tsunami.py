from pathlib import Path
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from baseline_tsunami import (
    RANDOM_STATE,
    SPLIT_YEAR,
    build_pipeline,
    evaluate_baseline,
    load_filtered_data,
    optimize_threshold_cv,
    temporal_train_test_split,
)


def make_calibrated_svm(C: float = 1.0, gamma: str | float = "scale") -> CalibratedClassifierCV:
    """Wrap RBF-kernel SVC with sigmoid calibration (replaces deprecated probability=True)."""
    base_svc = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5, ensemble=False)


def tune_svm(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune calibrated RBF-kernel SVC on the training set only using PR-AUC."""
    pipe = build_pipeline(make_calibrated_svm())
    param_grid = {
        "model__estimator__C": [0.1, 1.0, 5.0, 10.0],
        "model__estimator__gamma": ["scale", 0.05, 0.1, 0.3],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    df = load_filtered_data(project_root / "Data" / "earthquakes_filtered.csv")
    X_train, X_test, y_train, y_test = temporal_train_test_split(df, SPLIT_YEAR)

    # 1. Default calibrated RBF SVM
    svm_default = build_pipeline(make_calibrated_svm())
    evaluate_baseline(
        "SVM RBF (default, calibrated, balanced)",
        svm_default,
        X_train,
        y_train,
        X_test,
        y_test,
        optimize_threshold=True,
    )

    # 2. Tuned calibrated RBF SVM (GridSearchCV + CV threshold optimization on train only)
    search = tune_svm(X_train, y_train)
    best_pipe = search.best_estimator_
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_threshold, cv_f2_opt = optimize_threshold_cv(
        best_pipe, X_train, y_train, cv=cv, beta=2.0
    )

    best_pipe.fit(X_train, y_train)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    print("\n=== SVM RBF (tuned via GridSearchCV) ===")
    print(f"Meilleurs hyperparamètres : {search.best_params_}")
    print(
        f"CV Train -> PR-AUC (best): {search.best_score_:.3f} | "
        f"F2 (seuil opt CV={best_threshold:.2f}): {cv_f2_opt:.3f}"
    )
    print(
        f"Test Temporel (>={SPLIT_YEAR}, seuil={best_threshold:.2f}) -> "
        f"PR-AUC: {average_precision_score(y_test, y_prob):.3f} | "
        f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f} | "
        f"Recall: {recall_score(y_test, y_pred, zero_division=0):.3f} | "
        f"F2: {fbeta_score(y_test, y_pred, beta=2, zero_division=0):.3f} | "
        f"Brier: {brier_score_loss(y_test, y_prob):.3f}"
    )

    cm = confusion_matrix(y_test, y_pred)
    print(f"Matrice de confusion (Test, seuil={best_threshold:.2f}) :\n{cm}")
