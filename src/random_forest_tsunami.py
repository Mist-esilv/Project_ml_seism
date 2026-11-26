from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from baseline_tsunami import (
    RANDOM_STATE,
    SPLIT_YEAR,
    build_pipeline,
    evaluate_baseline,
    load_filtered_data,
    optimize_threshold_cv,
    temporal_train_test_split,
)


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> GridSearchCV:
    """Tune RandomForestClassifier on the training set only using PR-AUC."""
    pipe = build_pipeline(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [6, 10, None],
        "model__min_samples_leaf": [1, 3, 5],
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

    # 1. Default Random Forest
    rf_default = build_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    evaluate_baseline(
        "RandomForest (default, balanced)",
        rf_default,
        X_train,
        y_train,
        X_test,
        y_test,
        optimize_threshold=True,
    )

    # 2. Tuned Random Forest (GridSearchCV + CV threshold optimization on train only)
    search = tune_random_forest(X_train, y_train)
    best_pipe = search.best_estimator_
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_threshold, cv_f2_opt = optimize_threshold_cv(
        best_pipe, X_train, y_train, cv=cv, beta=2.0
    )

    best_pipe.fit(X_train, y_train)
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    print("\n=== RandomForest (tuned via GridSearchCV) ===")
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

    # Feature importances from the fitted RandomForest inside the pipeline
    feat_names = best_pipe.named_steps["feat_eng"].transform(X_train.iloc[:1]).columns
    importances = pd.Series(
        best_pipe.named_steps["model"].feature_importances_, index=feat_names
    ).sort_values(ascending=False)
    print("\nImportance des variables :")
    print(importances.round(4).to_string())
