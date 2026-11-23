from pathlib import Path
import pandas as pd
from read_earthquakes import load_earthquakes


CORE_GEO_COLS = ["Year", "Latitude", "Longitude", "Focal Depth (km)", "Mag"]


def filter_earthquakes(
    df: pd.DataFrame,
    min_year: int | None = 1950,
    max_year: int | None = None,
    min_mag: float | None = None,
    require_core_geo: bool = True,
    drop_sparse_cols_threshold: float | None = 0.85,
) -> pd.DataFrame:
    """Filter the earthquakes dataset for data quality and ML readiness.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from earthquakes.csv.
    min_year : int | None
        Minimum year to keep (e.g., 1950 for modern instrumental records).
    max_year : int | None
        Maximum year to keep, if any.
    min_mag : float | None
        Minimum magnitude threshold, if any.
    require_core_geo : bool
        If True, drop rows missing Latitude, Longitude, Focal Depth (km), or Mag,
        and enforce physical bounds on coordinates and depth.
    drop_sparse_cols_threshold : float | None
        If set (0 to 1), drop columns whose missing-value ratio exceeds this threshold
        after row filtering.
    """
    filtered = df.copy()

    if min_year is not None:
        filtered = filtered[filtered["Year"] >= min_year]
    if max_year is not None:
        filtered = filtered[filtered["Year"] <= max_year]
    if min_mag is not None:
        filtered = filtered[filtered["Mag"] >= min_mag]

    if require_core_geo:
        filtered = filtered.dropna(subset=CORE_GEO_COLS)
        filtered = filtered[
            filtered["Latitude"].between(-90, 90)
            & filtered["Longitude"].between(-180, 180)
            & (filtered["Focal Depth (km)"] >= 0)
            & (filtered["Mag"] > 0)
        ]

    if drop_sparse_cols_threshold is not None:
        missing_ratio = filtered.isna().mean()
        cols_to_keep = missing_ratio[missing_ratio <= drop_sparse_cols_threshold].index
        filtered = filtered[cols_to_keep]

    return filtered.reset_index(drop=True)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"
    raw_df = load_earthquakes(data_dir / "earthquakes.csv")

    filtered_df = filter_earthquakes(
        raw_df,
        min_year=1950,
        require_core_geo=True,
        drop_sparse_cols_threshold=0.85,
    )

    output_path = data_dir / "earthquakes_filtered.csv"
    filtered_df.to_csv(output_path, sep=";", index=False)

    print(f"Avant filtrage : {raw_df.shape[0]} lignes x {raw_df.shape[1]} colonnes")
    print(f"Après filtrage : {filtered_df.shape[0]} lignes x {filtered_df.shape[1]} colonnes")
    print(f"Taux de conservation des lignes : {len(filtered_df) / len(raw_df):.1%}")
    print(f"\nColonnes conservées ({len(filtered_df.columns)}) :")
    for col in filtered_df.columns:
        miss_pct = filtered_df[col].isna().mean() * 100
        print(f"  - {col} ({miss_pct:.1f}% manquants)")
    print(f"\nFichier sauvegardé : {output_path.name}")
