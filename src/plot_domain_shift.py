"""Generate a domain shift visualization for the report.

Bar chart of KS statistics per feature, colored by domain-dependent vs invariant,
with a threshold line. Visually justifies the masking decision.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
PLOTS_DIR = OUTPUT_DIR / "plots"


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "domain_shift_analysis.csv"
    if not csv_path.exists():
        logger.info("Running domain analysis first...")
        from src.domain_analysis import main as run_analysis
        run_analysis()

    df = pd.read_csv(csv_path)
    df = df.sort_values("ks_stat", ascending=True).reset_index(drop=True)

    colors = ["#d62728" if dep else "#2ca02c" for dep in df["domain_dependent"]]

    fig, ax = plt.subplots(figsize=(10, 12))
    bars = ax.barh(range(len(df)), df["ks_stat"], color=colors, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"], fontsize=7)
    ax.set_xlabel("KS Statistic (synthetic vs real true mutations)")
    ax.set_title("Feature Domain Shift: Synthetic vs Real Tumors")
    ax.axvline(x=0.1, color="black", linestyle="--", linewidth=1, label="Masking threshold (KS=0.1)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label=f"Domain-dependent ({df['domain_dependent'].sum()} features, masked on synthetic)"),
        Patch(facecolor="#2ca02c", label=f"Domain-invariant ({(~df['domain_dependent']).sum()} features, kept everywhere)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "domain_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved domain_shift.png")


if __name__ == "__main__":
    main()
