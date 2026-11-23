from pathlib import Path
import pandas as pd


def load_earthquakes(
    filepath: str | Path | None = None,
    filtered: bool = True,
) -> pd.DataFrame:
    """Load the earthquakes CSV file (semicolon-separated) and drop completely empty rows.

    If filepath is not provided, loads Data/earthquakes_filtered.csv when filtered=True,
    or Data/earthquakes.csv when filtered=False.
    """
    if filepath is None:
        filename = "earthquakes_filtered.csv" if filtered else "earthquakes.csv"
        filepath = Path(__file__).resolve().parent / "Data" / filename

    path = Path(filepath)
    df = pd.read_csv(path, sep=";").dropna(how="all").reset_index(drop=True)
    print(f"Fichier chargé : {path.name} ({df.shape[0]} lignes x {df.shape[1]} colonnes)")
    return df


if __name__ == "__main__":
    df = load_earthquakes(filtered=True)
    print(df.head())
