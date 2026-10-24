"""Shared settings used by every script: config, folders, plot helper."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import yaml

matplotlib.use("Agg")   # draw to files, no window needed

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))

RAW = ROOT / CONFIG["paths"]["raw"]
PROCESSED = ROOT / CONFIG["paths"]["processed"]
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "results"
for folder in (PROCESSED, TABLES, FIGURES, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

SEED = CONFIG["random_state"]
CLASSES = ["Healthy", "Warning", "Faulty"]

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def save_figure(fig, name):
    """Save a figure to reports/figures/<name>.png and close it."""
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=130)
    plt.close(fig)
    print(f"  saved figure {name}.png")
