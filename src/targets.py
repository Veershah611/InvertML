"""Stage 7 Target Construction & Sensitivity Framework (src/targets.py).

Owned by B1 (Pipeline Lead for Stage 7: Target Formulation & Sensitivity Analysis).

Core Responsibilities:
1. Mathematical Target Definition:
   - Continuous Target: Health Score (0.0 to 100.0) based on physical PV derating.
   - Classification Target: 3-Class Operational Status (Healthy, Warning, Faulty).
2. Data Leakage Prevention:
   - Option A: Forecasting Health(t + h) using features up to time t.
   - Option B: Diagnostic normal-behavior proxy (purging ac_power_w from X).
3. Threshold Sensitivity Study:
   - Parameter grid sweep over Faulty thresholds [55, 85] and Warning thresholds [85, 95].
   - Visualizing class distribution trade-offs and saving empirical sensitivity tables.

Usage:
    python -m src.targets
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "config.yaml"

if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {"paths": {"processed": "data/processed"}}

PROCESSED_DIR = ROOT / CONFIG.get("paths", {}).get("processed", "data/processed")
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Standard PV Engineering Constants
DEFAULT_PV_DERATING = 0.78      # Standard IEC 61724 overall Performance Ratio derating
MIN_IRRADIANCE_DAYLIGHT = 100.0  # Daylight threshold (W/m²)
MIN_POWER_FLOOR_WATTS = 100.0    # Protect against division-by-zero during low light

# Canonical Status Thresholds
DEFAULT_HEALTHY_THRESH = 90.0
DEFAULT_FAULTY_THRESH = 70.0

# Publication Aesthetic Colors
COLOR_HEALTHY = "#2ca02c"
COLOR_WARNING = "#e67e22"
COLOR_FAULTY = "#d9534f"
COLOR_NAVY = "#0d366b"
COLOR_BLUE = "#2a78d6"
COLOR_GRID = "#e1e0d9"


def set_plot_style():
    """Apply unified, professional matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.6,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "font.size": 9.5,
        "legend.frameon": False,
    })


# ===========================================================================
# 1. Physical Target Formulation
# ===========================================================================

def compute_expected_power(
    irradiance_wm2: pd.Series | np.ndarray,
    capacity_kwp: pd.Series | np.ndarray,
    derating: float = DEFAULT_PV_DERATING,
    power_floor: float = MIN_POWER_FLOOR_WATTS
) -> pd.Series | np.ndarray:
    """Calculate the theoretical expected AC power output under standard efficiency.

    Formula:
        Expected Power (W) = (irradiance_wm2 / 1000.0) * (capacity_kwp * 1000.0) * derating

    Parameters:
        irradiance_wm2: Plane-of-array or global horizontal solar irradiance (W/m²).
        capacity_kwp: Rated peak DC capacity of the photovoltaic array (kWp).
        derating: System derating factor (default 0.78 covers thermal, wiring, and inverter losses).
        power_floor: Minimum wattage floor to prevent singularity or erratic scores during twilight.

    Returns:
        Expected AC power output in Watts.
    """
    irr_ratio = np.maximum(0.0, np.asarray(irradiance_wm2, dtype=np.float32)) / 1000.0
    cap_watts = np.maximum(1.0, np.asarray(capacity_kwp, dtype=np.float32) * 1000.0)
    expected_w = irr_ratio * cap_watts * derating
    return np.maximum(expected_w, power_floor)


def compute_health_score(
    df: pd.DataFrame,
    derating: float = DEFAULT_PV_DERATING,
    min_irradiance: float = MIN_IRRADIANCE_DAYLIGHT,
    filter_daylight: bool = False
) -> pd.DataFrame:
    """Calculate the continuous Health Score (0.0 to 100.0) for solar inverters.

    Formula:
        PR = Actual AC Power / Expected AC Power
        Health Score = min(100.0, max(0.0, PR * 100.0))

    Parameters:
        df: Input dataframe containing 'ac_power_w', 'irradiance_wm2', and 'capacity_kwp'.
        derating: Standard PR efficiency derating factor (0.78).
        min_irradiance: Minimum sunlight required for valid health assessment.
        filter_daylight: If True, returns only rows with irradiance >= min_irradiance.

    Missing AC power (telemetry gap or offline bin) gives an undefined score
    (NaN), not 0: a communication outage is not evidence of a fault. Outages
    are captured by A1's is_offline / minutes_offline_24h features instead.

    Returns:
        DataFrame enriched with 'expected_power_w' and 'health_score'.
    """
    df = df.copy()

    # Safety fallbacks
    irr = df["irradiance_wm2"].fillna(0.0) if "irradiance_wm2" in df.columns else pd.Series(500.0, index=df.index)
    cap = df["capacity_kwp"].fillna(25.0) if "capacity_kwp" in df.columns else pd.Series(25.0, index=df.index)
    ac = df["ac_power_w"] if "ac_power_w" in df.columns else pd.Series(np.nan, index=df.index)
    if "n_valid_5min" in df.columns:
        ac = ac.where(df["n_valid_5min"].gt(0))

    expected = compute_expected_power(irr, cap, derating=derating)
    df["expected_power_w"] = expected

    # Instantaneous Performance Ratio
    pr = ac / expected
    health = np.clip(pr * 100.0, 0.0, 100.0)

    # During night or heavy twilight (irradiance < min_irradiance), health score is undefined (NaN)
    is_daylight = irr >= min_irradiance
    df["health_score"] = np.where(is_daylight, health, np.nan).astype("float32")

    if filter_daylight:
        df = df[is_daylight].copy()

    return df


def assign_operational_status(
    health_scores: pd.Series | np.ndarray,
    healthy_thresh: float = DEFAULT_HEALTHY_THRESH,
    faulty_thresh: float = DEFAULT_FAULTY_THRESH
) -> np.ndarray:
    """Map continuous Health Score into 3-class Operational Status.

    Threshold Logic:
        Health Score >= healthy_thresh           -> 'Healthy' (0)
        faulty_thresh <= Health Score < healthy  -> 'Warning' (1)
        Health Score < faulty_thresh            -> 'Faulty'  (2)

    Returns:
        np.ndarray of string categorical labels ('Healthy', 'Warning', 'Faulty').
    """
    hs = np.asarray(health_scores, dtype=np.float32)
    conditions = [
        hs >= healthy_thresh,
        (hs >= faulty_thresh) & (hs < healthy_thresh),
        hs < faulty_thresh
    ]
    choices = ["Healthy", "Warning", "Faulty"]
    return np.select(conditions, choices, default="Healthy")


# ===========================================================================
# 2. Data Leakage Prevention Engine (§7.2 Compliance)
# ===========================================================================

def create_forecasting_target(
    df: pd.DataFrame,
    horizon_steps: int = 4,
    target_col: str = "health_score",
    inverter_col: str = "inverter_id"
) -> pd.DataFrame:
    """Construct future forecasting target Health(t + h) to prevent circular leakage.

    Option A (§7.2 of Project Specification):
        Instead of predicting Health(t) from features at time t (which trivially learns
        the formula), the model uses features up to time t to forecast Health(t + h).
        For 15-minute data, horizon_steps=4 corresponds to 1 hour ahead.

    Parameters:
        df: Input dataframe sorted chronologically per inverter.
        horizon_steps: Number of forward time steps to lead (default 4 = 1 hour).
        target_col: Source target column.
        inverter_col: Grouping column to prevent shifting across different rooftop inverters.

    Returns:
        DataFrame with new column f'{target_col}_lead{horizon_steps}'.
    """
    df = df.copy()
    lead_col = f"{target_col}_lead{horizon_steps}"
    df[lead_col] = df.groupby(inverter_col)[target_col].shift(-horizon_steps)
    return df


def prepare_leakage_free_matrices(
    df: pd.DataFrame,
    feature_cols: list[str],
    mode: str = "diagnostic",
    horizon_steps: int = 4
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract feature matrix X and targets y with strict data leakage guards.

    Modes:
        1. 'diagnostic' (Option B):
           Predicts current Health(t). Drops 'ac_power_w' and instantaneous power
           features from X so the model infers health purely from weather, cyclical
           time, tilt, orientation, and peer deviation.
        2. 'forecasting' (Option A):
           Predicts future Health(t + horizon_steps). Current ac_power_w at time t
           CAN be safely included because it precedes the target in time.

    Returns:
        (X, y_regression, y_classification)
    """
    df_clean = df.dropna(subset=["health_score"]).copy()

    if mode == "forecasting":
        lead_target = f"health_score_lead{horizon_steps}"
        if lead_target not in df_clean.columns:
            df_clean = create_forecasting_target(df_clean, horizon_steps=horizon_steps)
        df_clean = df_clean.dropna(subset=[lead_target])
        y_reg = df_clean[lead_target].values
        y_cls = assign_operational_status(y_reg)
        usable_features = [c for c in feature_cols if c in df_clean.columns]
    elif mode == "diagnostic":
        y_reg = df_clean["health_score"].values
        y_cls = assign_operational_status(y_reg)
        # Purge direct formula components from X to prevent circular leakage
        leakage_cols = {"ac_power_w", "specific_yield", "expected_power_w"}
        usable_features = [c for c in feature_cols if c in df_clean.columns and c not in leakage_cols]
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'diagnostic' or 'forecasting'.")

    X = df_clean[usable_features].copy()
    X = X.fillna(X.mean(numeric_only=True))
    return X, y_reg, y_cls


# ===========================================================================
# 3. Threshold Sensitivity Study (Stage 7 Research)
# ===========================================================================

def run_threshold_sensitivity_study(
    health_scores: pd.Series | np.ndarray,
    faulty_range: tuple[int, int, int] = (55, 86, 5),
    warning_thresholds: list[float] = [85.0, 90.0, 95.0]
) -> pd.DataFrame:
    """Execute grid sweep across threshold configurations to evaluate class balance shifts.

    Parameters:
        health_scores: Array of valid daylight Health Scores.
        faulty_range: (start, stop, step) for Faulty threshold sweep.
        warning_thresholds: List of Warning thresholds to test.

    Returns:
        pd.DataFrame containing sensitivity metrics and marginal shift rates.
    """
    hs = np.asarray(health_scores, dtype=np.float32)
    hs_valid = hs[~np.isnan(hs)]
    total_count = len(hs_valid)

    results = []
    faulty_vals = range(faulty_range[0], faulty_range[1], faulty_range[2])

    for f_th in faulty_vals:
        for w_th in warning_thresholds:
            if f_th >= w_th:
                continue

            n_faulty = np.sum(hs_valid < f_th)
            n_warning = np.sum((hs_valid >= f_th) & (hs_valid < w_th))
            n_healthy = np.sum(hs_valid >= w_th)

            results.append({
                "faulty_threshold": f_th,
                "warning_threshold": w_th,
                "healthy_pct": round(float(n_healthy / total_count * 100.0), 2),
                "warning_pct": round(float(n_warning / total_count * 100.0), 2),
                "faulty_pct": round(float(n_faulty / total_count * 100.0), 2),
                "healthy_count": int(n_healthy),
                "warning_count": int(n_warning),
                "faulty_count": int(n_faulty),
                "total_samples": int(total_count)
            })

    sens_df = pd.DataFrame(results)

    # Save to reports/tables/
    table_path = TABLES_DIR / "b1_target_sensitivity.csv"
    sens_df.to_csv(table_path, index=False)
    print(f"Saved sensitivity study table: {table_path.relative_to(ROOT)}")

    return sens_df


# ===========================================================================
# 4. Sensitivity Visualizations
# ===========================================================================

def plot_target_distribution(health_scores: pd.Series | np.ndarray) -> Path:
    """Plot continuous Health Score distribution and cumulative percentile curve."""
    set_plot_style()
    hs = np.asarray(health_scores, dtype=np.float32)
    hs_valid = hs[~np.isnan(hs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=120)

    # Panel 1: Histogram & KDE
    n, bins, patches = ax1.hist(
        hs_valid, bins=35, density=True, color=COLOR_BLUE, edgecolor="white", alpha=0.85
    )

    # Threshold markers
    ax1.axvline(DEFAULT_FAULTY_THRESH, color=COLOR_FAULTY, linestyle="--", linewidth=2.0,
                label=f"Faulty Threshold (< {DEFAULT_FAULTY_THRESH:.0f})")
    ax1.axvline(DEFAULT_HEALTHY_THRESH, color=COLOR_HEALTHY, linestyle="--", linewidth=2.0,
                label=f"Healthy Threshold (>= {DEFAULT_HEALTHY_THRESH:.0f})")

    ax1.set_title("A. Continuous Health Score Distribution (Daylight G >= 100 W/m²)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Health Score (0.0 to 100.0)", fontsize=9.5)
    ax1.set_ylabel("Probability Density", fontsize=9.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    # Panel 2: Cumulative Distribution Function (CDF)
    sorted_hs = np.sort(hs_valid)
    cdf = np.arange(1, len(sorted_hs) + 1) / len(sorted_hs) * 100.0

    ax2.plot(sorted_hs, cdf, color=COLOR_NAVY, linewidth=2.5, label="Empirical CDF")
    ax2.axvline(DEFAULT_FAULTY_THRESH, color=COLOR_FAULTY, linestyle="--", linewidth=1.8)
    ax2.axvline(DEFAULT_HEALTHY_THRESH, color=COLOR_HEALTHY, linestyle="--", linewidth=1.8)

    faulty_pct = np.mean(hs_valid < DEFAULT_FAULTY_THRESH) * 100.0
    healthy_pct = np.mean(hs_valid >= DEFAULT_HEALTHY_THRESH) * 100.0

    ax2.annotate(
        f"Faulty Population: {faulty_pct:.1f}%\n"
        f"Warning Population: {100.0 - faulty_pct - healthy_pct:.1f}%\n"
        f"Healthy Population: {healthy_pct:.1f}%",
        xy=(0.05, 0.65), xycoords="axes fraction",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#c3c2b7", alpha=0.9)
    )

    ax2.set_title("B. Cumulative Empirical Health Percentile", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Health Score Threshold", fontsize=9.5)
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.6)

    save_path = FIGURES_DIR / "b1_target_distribution.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


def plot_threshold_sensitivity(sens_df: pd.DataFrame) -> Path:
    """Plot multi-panel sensitivity curves showing class balance shifts across threshold sweeps."""
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=120)

    # Panel 1: Class Distribution vs Faulty Threshold (at Warning=90)
    sub90 = sens_df[sens_df["warning_threshold"] == 90.0].sort_values("faulty_threshold")

    ax1.plot(sub90["faulty_threshold"], sub90["faulty_pct"], marker="o", color=COLOR_FAULTY,
             linewidth=2.2, label="% Faulty Class")
    ax1.plot(sub90["faulty_threshold"], sub90["warning_pct"], marker="s", color=COLOR_WARNING,
             linewidth=2.2, label="% Warning Class")
    ax1.plot(sub90["faulty_threshold"], sub90["healthy_pct"], marker="^", color=COLOR_HEALTHY,
             linewidth=2.2, label="% Healthy Class (Fixed at >= 90)")

    ax1.axvline(DEFAULT_FAULTY_THRESH, color=COLOR_NAVY, linestyle=":", linewidth=1.8,
                label=f"Baseline Choice (Faulty < {DEFAULT_FAULTY_THRESH:.0f})")

    ax1.set_title("A. Class Distribution Shifts vs. Faulty Threshold (Warning = 90)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Faulty Threshold Boundary", fontsize=9.5)
    ax1.set_ylabel("Class Percentage in Telemetry (%)", fontsize=9.5)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    # Panel 2: Marginal Rate of Change (Derivative: d(Faulty%)/d(Threshold))
    x_vals = sub90["faulty_threshold"].values
    y_vals = sub90["faulty_pct"].values
    d_faulty = np.gradient(y_vals, x_vals)

    ax2.bar(x_vals, d_faulty, color="#3987e5", width=3.2, edgecolor=COLOR_NAVY, alpha=0.85)
    ax2.axhline(np.mean(d_faulty), color=COLOR_FAULTY, linestyle="--", linewidth=1.5,
                label=f"Mean Marginal Slope: {np.mean(d_faulty):.2f}% / threshold unit")

    ax2.set_title("B. Marginal Sensitivity: d(% Faulty) / d(Threshold)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Faulty Threshold Boundary", fontsize=9.5)
    ax2.set_ylabel("Marginal % Shift per Threshold Unit", fontsize=9.5)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#c3c2b7")

    save_path = FIGURES_DIR / "b1_threshold_sensitivity_study.png"
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path.relative_to(ROOT)}")
    return save_path


# ===========================================================================
# 5. Master Runner & Standalone Test
# ===========================================================================

def run_all():
    """Execute complete Task 2 target formulation and sensitivity analysis."""
    print("=" * 65)
    print("Executing Stage 7 Target Formulation & Sensitivity Analysis (B1 Lead)...")
    print("=" * 65)

    parquet_path = PROCESSED_DIR / "hkust_15min_v1.parquet"
    if parquet_path.exists():
        print(f"Loading real telemetry from {parquet_path.relative_to(ROOT)}...")
        df = pd.read_parquet(parquet_path, columns=["time", "inverter_id", "ac_power_w", "irradiance_wm2", "capacity_kwp"])
    else:
        print("Notice: Parquet file not found. Generating simulated sample.")
        times = pd.date_range("2022-01-01 06:00", "2022-12-31 18:00", freq="15min")
        df = pd.DataFrame({
            "time": times,
            "inverter_id": "test_inverter",
            "capacity_kwp": 25.0,
            "irradiance_wm2": np.random.uniform(50, 1000, len(times)),
            "ac_power_w": np.random.uniform(1000, 18000, len(times))
        })

    # 1. Compute continuous Health Score
    print("\n1. Computing continuous Health Score across daylight intervals (G >= 100 W/m²)...")
    df_daylight = compute_health_score(df, filter_daylight=True)
    print(f"Retained {len(df_daylight):,} valid daylight intervals.")

    hs = df_daylight["health_score"].dropna()
    print(f"Health Score Statistics: Mean={hs.mean():.2f}, Median={hs.median():.2f}, Std={hs.std():.2f}")

    # 2. Baseline status distribution
    print("\n2. Baseline Operational Status Distribution (Healthy >= 90, Faulty < 70):")
    df_daylight["status"] = assign_operational_status(df_daylight["health_score"])
    status_counts = df_daylight["status"].value_counts(normalize=True) * 100.0
    for cls, pct in status_counts.items():
        print(f"  * {cls:10s}: {pct:.2f}%")

    # 3. Sensitivity study grid sweep
    print("\n3. Executing Threshold Sensitivity Grid Sweep...")
    sens_df = run_threshold_sensitivity_study(hs)
    print("Top 6 rows of sensitivity table:")
    print(sens_df.head(6).to_string(index=False))

    # 4. Generate Visualizations
    print("\n4. Generating publication-grade visualizations...")
    plot_target_distribution(hs)
    plot_threshold_sensitivity(sens_df)

    print("\n" + "=" * 65)
    print("B1 Task 2 Target Construction & Sensitivity Framework Complete!")
    print("=" * 65)


if __name__ == "__main__":
    run_all()
