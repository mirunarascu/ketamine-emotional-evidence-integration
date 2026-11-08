"""Detection curves: proportion of change points detected by each time after the change point.

    python analysis/figures/02_detection_curves.py --cohort in_person    # Fig 2A, S3E
    python analysis/figures/02_detection_curves.py --cohort online       # Fig S1A (+ extra)

Plots the group curves made by analysis/behaviour/01_make_behaviour_tables.py
(detection_curves in behaviour_measures.py: 1 - Kaplan-Meier, mean +/- 95% CI over participants).
Reads output/tables/<cohort>/detection_curve_by_dimension.csv and
detection_curve_by_volatility.csv. Writes the panels (one curve per dimension; stable vs
volatile per dimension) and the plotted curves in source_data/, in the figure folders or in
output/figures/extra/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import plotting as pl                       # noqa: E402
from utils.config import COHORTS, load_config, out_dir  # noqa: E402

DIMS = ["valence", "arousal"]
# table -> (folder, file name); panels not in the paper go to extra/
PANELS = {"in_person": {"by_dimension": ("Fig2", "Fig2A_detection_curve"),
                        "by_volatility": ("FigS3", "FigS3E_detection_curve_by_volatility")},
          "online": {"by_dimension": ("FigS1", "FigS1A_detection_curve"),
                     "by_volatility": ("extra", "online_detection_curve_by_volatility")}}


def plot_panel(grp, by, x_max):
    """One panel with a curve per dimension, or one panel per dimension with stable vs volatile."""
    pl.set_style()
    if by == "by_dimension":                   # one panel, a curve per dimension
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        axes = [ax, ax]
    else:                                      # a panel per dimension, stable vs volatile
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for k, (ax, dim) in enumerate(zip(axes, DIMS)):
        if by == "by_dimension":
            lines = [(None, dim.capitalize(), pl.DIMENSION_PALETTE[dim])]
        else:
            lines = [(lev, lev.capitalize(), col) for lev, col in pl.VOLATILITY_PALETTE.items()]
        for lev, label, col in lines:
            g = grp[grp["dimension"] == dim]
            if lev is not None:
                g = g[g["volatility"] == lev]
            ax.plot(g["time_s"], g["mean"], color=col, lw=2.4, label=label)
            ax.fill_between(g["time_s"], g["mean"] - g["ci95"], g["mean"] + g["ci95"], color=col, alpha=0.25,
                            linewidth=0)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1)
        pl.style_axis(ax, title="" if by == "by_dimension" else dim.capitalize(), xlabel="Time from CP onset (s)",
                      ylabel="Proportion detected" if (k == 0 or by == "by_dimension") else "")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def main(cohort):
    cfg = load_config(cohort)
    for by, (fig_folder, name) in PANELS[cohort].items():
        grp = pd.read_csv(out_dir("tables", cohort) / f"detection_curve_{by}.csv")
        folder = out_dir("figures", fig_folder)
        (folder / "source_data").mkdir(parents=True, exist_ok=True)
        grp.to_csv(folder / "source_data" / f"{name}.csv", index=False)
        pl.save_figure(plot_panel(grp, by, cfg["detection_curve"]["max_time_s"]), folder / name,
                       cfg["figures"]["formats"])
        print(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=COHORTS)
    main(ap.parse_args().cohort)
