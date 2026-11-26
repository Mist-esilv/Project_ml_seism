from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from global_land_mask import globe
from scipy import stats

from baseline_tsunami import ALLOWED_FEATURES, SPLIT_YEAR, load_filtered_data


def run_statistical_tests(df: pd.DataFrame) -> None:
    """Check normality (Shapiro-Wilk) and run appropriate group tests (Mann-Whitney / Chi2)."""
    y = df["Tsu"].notna().astype(int)
    is_ocean = (~globe.is_land(df["Latitude"].to_numpy(), df["Longitude"].to_numpy())).astype(int)

    print("=== 1. APERÇU DU DATASET & DÉSÉQUILIBRE ===")
    print(f"Lignes : {len(df)} | Colonnes : {df.shape[1]}")
    print(f"Période : {int(df['Year'].min())} -> {int(df['Year'].max())}")
    print(f"Tsunami = 1 : {y.sum()} ({y.mean():.1%}) | Tsunami = 0 : {(1 - y).sum()} ({(1 - y).mean():.1%})")
    print(f"MMI Int renseignés : {df['MMI Int'].notna().sum()} ({df['MMI Int'].notna().mean():.1%})")

    train_mask = df["Year"] < SPLIT_YEAR
    print(
        f"Split temporel (<{SPLIT_YEAR} vs >={SPLIT_YEAR}) : "
        f"Train={train_mask.sum()} ({y[train_mask].mean():.1%} Tsu) | "
        f"Test={(~train_mask).sum()} ({y[~train_mask].mean():.1%} Tsu)"
    )

    print("\n=== 2. STATISTIQUES DESCRIPTIVES (FEATURES IMMÉDIATES) ===")
    print(df[ALLOWED_FEATURES].describe().round(3).to_string())

    print("\n=== 3. TESTS DE NORMALITÉ (SHAPIRO-WILK) & FALLBACK MANN-WHITNEY U ===")
    for col in ["Mag", "Focal Depth (km)"]:
        g0 = df.loc[y == 0, col].dropna()
        g1 = df.loc[y == 1, col].dropna()

        _, p_norm_0 = stats.shapiro(g0)
        _, p_norm_1 = stats.shapiro(g1)
        u_stat, p_mwu = stats.mannwhitneyu(g0, g1, alternative="two-sided")

        print(f"\nVariable : {col}")
        print(f"  Shapiro-Wilk p-value -> Tsu=0: {p_norm_0:.2e} | Tsu=1: {p_norm_1:.2e} (non-normal -> Mann-Whitney U)")
        print(f"  Médiane -> Tsu=0: {g0.median():.2f} (IQR={g0.quantile(0.75)-g0.quantile(0.25):.2f}) | Tsu=1: {g1.median():.2f} (IQR={g1.quantile(0.75)-g1.quantile(0.25):.2f})")
        print(f"  Mann-Whitney U = {u_stat:.1f}, p-value = {p_mwu:.2e}")

    print("\n=== 4. TEST DU CHI-2 : ÉPICENTRE EN MER (is_ocean) vs TSUNAMI ===")
    ct = pd.crosstab(is_ocean, y, rownames=["is_ocean"], colnames=["Tsu"])
    chi2, p_chi2, _, _ = stats.chi2_contingency(ct)
    print(ct)
    print(f"Taux de tsunami sur terre : {y[is_ocean == 0].mean():.1%} | en mer : {y[is_ocean == 1].mean():.1%}")
    print(f"Chi2 = {chi2:.2f}, p-value = {p_chi2:.2e}")


def generate_figures(df: pd.DataFrame, out_dir: Path) -> None:
    """Save publication-ready EDA figures to report/figures/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y = df["Tsu"].notna().astype(int)
    is_ocean = (~globe.is_land(df["Latitude"].to_numpy(), df["Longitude"].to_numpy())).astype(int)

    # Figure 1: Spatial map of earthquakes vs tsunamis
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(
        df.loc[y == 0, "Longitude"],
        df.loc[y == 0, "Latitude"],
        s=10,
        alpha=0.45,
        c="#4C78A8",
        label=f"No Tsunami (n={(y == 0).sum()})",
    )
    ax.scatter(
        df.loc[y == 1, "Longitude"],
        df.loc[y == 1, "Latitude"],
        s=18,
        alpha=0.75,
        c="#E45756",
        label=f"Tsunami Triggered (n={(y == 1).sum()})",
    )
    ax.set_title("Global Distribution of Significant Earthquakes (1950–2020)")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_xlim(-185, 185)
    ax.set_ylim(-75, 80)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "tsunami_world_map.png", dpi=180)
    plt.close(fig)

    # Figure 2: Feature distributions (Mag, Depth, Ocean/Land)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].boxplot(
        [df.loc[y == 0, "Mag"], df.loc[y == 1, "Mag"]],
        tick_labels=["No Tsunami", "Tsunami"],
        patch_artist=True,
    )
    axes[0].set_title("Magnitude by Tsunami Class")
    axes[0].set_ylabel("Magnitude (Mag)")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[1].boxplot(
        [
            np.log1p(df.loc[y == 0, "Focal Depth (km)"]),
            np.log1p(df.loc[y == 1, "Focal Depth (km)"]),
        ],
        tick_labels=["No Tsunami", "Tsunami"],
        patch_artist=True,
    )
    axes[1].set_title("Log Focal Depth by Tsunami Class")
    axes[1].set_ylabel("log(1 + Focal Depth [km])")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    rates = [y[is_ocean == 0].mean() * 100, y[is_ocean == 1].mean() * 100]
    bars = axes[2].bar(["Land", "Ocean"], rates, color=["#72B7B2", "#E45756"])
    axes[2].set_title("Tsunami Rate: Land vs Ocean")
    axes[2].set_ylabel("Tsunami Rate (%)")
    axes[2].set_ylim(0, max(rates) * 1.2)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.3)
    for bar, rate in zip(bars, rates):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            rate + 1.2,
            f"{rate:.1f}%",
            ha="center",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "tsunami_feature_distributions.png", dpi=180)
    plt.close(fig)

    # Figure 3: Temporal distribution & split year
    yearly = df.assign(TsuBin=y).groupby("Year")["TsuBin"].agg(["count", "sum"])
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.bar(yearly.index, yearly["count"] - yearly["sum"], color="#4C78A8", label="No Tsunami")
    ax.bar(
        yearly.index,
        yearly["sum"],
        bottom=yearly["count"] - yearly["sum"],
        color="#E45756",
        label="Tsunami",
    )
    ax.axvline(SPLIT_YEAR - 0.5, color="black", linestyle="--", linewidth=1.8, label=f"Temporal Split ({SPLIT_YEAR})")
    ax.set_title("Annual Earthquake Counts & Temporal Train/Test Split")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Earthquakes")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tsunami_temporal_split.png", dpi=180)
    plt.close(fig)

    print(f"\nFigures sauvegardées dans : {out_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    df = load_filtered_data(project_root / "Data" / "earthquakes_filtered.csv")
    run_statistical_tests(df)
    generate_figures(df, project_root / "report" / "figures")
