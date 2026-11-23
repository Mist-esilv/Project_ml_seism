from pathlib import Path
import pandas as pd


def load_earthquakes(filepath: str | Path) -> pd.DataFrame:
    """Load the earthquakes CSV file (semicolon-separated) and drop completely empty rows."""
    df = pd.read_csv(filepath, sep=";")
    df = df.dropna(how="all").reset_index(drop=True)
    return df
