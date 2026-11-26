from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report



def load_earthquakes(filepath: str | Path) -> pd.DataFrame:
    """Load the earthquakes CSV file (semicolon-separated) and drop completely empty rows."""
    df = pd.read_csv(filepath, sep=";")
    df = df.dropna(how="all").reset_index(drop=True)
    return df



def classifier(df: pd.DataFrame) -> XGBClassifier:
    """Train an XGBoost classifier to predict tsunami (1) vs no tsunami (0)."""
    df = df.copy()
    df["Tsu_flag"] = df["Tsu"].notna().astype(int)

    # Features disponibles AVANT/au moment du séisme uniquement
    features = ["Latitude", "Longitude", "Focal Depth (km)", "Mag"]
    X = df[features]
    y = df["Tsu_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compense le déséquilibre de classes
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))
    return model


df=load_earthquakes("data/earthquakes_filtered.csv")
print(df.groupby('Death Description')['Deaths'].agg(['min', 'max', 'count']))
print(df["Tsu"].dropna().head())
print(df["Tsu"].dropna().nunique(), df["Tsu"].notna().sum())
print(df["Tsu"].dropna().sort_values().head(10).tolist())
df["Tsu_flag"] = df["Tsu"].notna().astype(int)
print(df["Tsu_flag"].head())
model=classifier(df)
nouveau = pd.DataFrame(
    [[38.3, 142.4, 29.0, 9.0]],
    columns=["Latitude", "Longitude", "Focal Depth (km)", "Mag"],
)
nouveaubis = pd.DataFrame(
    [[38.1, 142.4, 29.0, 7.0]],
    columns=["Latitude", "Longitude", "Focal Depth (km)", "Mag"],
)

print(model.predict(nouveau))        # [1] = tsunami, [0] = pas de tsunami
print(model.predict_proba(nouveau))
proba = model.predict_proba(nouveau)[:, 1]
print(proba)  # probabilité d'avoir un tsunami

print("hello")
