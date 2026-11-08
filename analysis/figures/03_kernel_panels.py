"""Integration kernel panels: kernel shape and IKS, by volatility and (in person) by drug.

    python analysis/figures/03_kernel_panels.py --cohort in_person    # Fig 4B-C, 5C-D, S4
    python analysis/figures/03_kernel_panels.py --cohort online       # Fig S1J-K (+ extra)

Needs output/tables/<cohort>/<kernel>_condition.csv (analysis/kernels).
Each panel shows valence (top) and arousal (bottom) with the same y axis.
    kernel shape: mean beta per regressor (-5 .. -1, -1 = most recent), stable vs volatile
                  (dashed placebo, solid ketamine), band = within-participant SEM
    IKS:          one dot per participant and cell, grey lines join a participant's cells,
                  black dot and bar = group mean +/- within-participant SEM, joined by a line
Without drug, the two drug sessions are averaged.

"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import plotting as pl                       # noqa: E402
from utils.config import COHORTS, load_config, out_dir  # noqa: E402

DIMS, VOLS, DRUGS = ["valence", "arousal"], ["stable", "volatile"], ["placebo", "ketamine"]
DRUG_LINE = {"placebo": "--", "ketamine": "-", "pooled": "-"}
X_LABEL = {"facewise": "Face regressor (−1 = most recent)", "timebinned": "Time bin (−1 = most recent)"}
Y_LABEL = {"facewise": "Beta", "timebinned": "Beta (log-odds)"}
# y-axis limits of the IKS panels, (cohort, kernel, by drug) -> (low, high); valence and
# arousal share them. Not listed = automatic. Widened (with a note) if a value falls outside.
IKS_YLIMS = {("in_person", "facewise", False): (-0.25, 0.25), ("in_person", "facewise", True): (-0.25, 0.25),
             ("in_person", "timebinned", False): (-1.5, 1.15), ("in_person", "timebinned", True): (-2.55, 1.7),
             ("online", "facewise", False): (-0.25, 0.45)}
# (kernel, by drug) -> (shape panel, IKS panel) for each cohort; None = not drawn
PANELS = {"in_person": {("facewise", False): ("Fig4B", "Fig4C"), ("facewise", True): ("Fig5C", "Fig5D"),
                        ("timebinned", False): ("FigS4A", "FigS4B"), ("timebinned", True): ("FigS4C", "FigS4D")},
          "online": {("facewise", False): ("FigS1J", "FigS1K"), ("timebinned", False): (None, None)}}


def subject_cells(cond, by_drug):
    """One row per participant x dimension x volatility (x drug): betas b1..b5 and IKS."""
    keys = ["subID", "dimension", "volatility"] + (["drug_label"] if by_drug else [])
    cols = [c for c in cond.columns if c[0] == "b" and c[1:].isdigit()] + ["iks"]
    subj = cond.groupby(keys, as_index=False)[cols].mean()
    if not by_drug:
        subj["drug_label"] = "pooled"
    return subj


def plot_shape(ax, d, by_drug):
    """Mean kernel per volatility (and drug), band = within-participant SEM."""
    bcols = sorted([c for c in d.columns if c[0] == "b" and c[1:].isdigit()], key=lambda c: int(c[1:]))
    x = np.arange(-len(bcols), 0)
    cells = ["volatility", "drug_label"]
    for vol in VOLS:
        for drug in (DRUGS if by_drug else ["pooled"]):
            mean = [d.loc[(d.volatility == vol) & (d.drug_label == drug), c].mean() for c in bcols]
            sem = [pl.cousineau_morey_sem(d, c, "subID", cells)[(vol, drug)] for c in bcols]
            mean, sem = np.array(mean), np.array(sem)
            col = pl.VOLATILITY_PALETTE[vol]
            ax.plot(x, mean, DRUG_LINE[drug], marker="o", ms=5, lw=1.6, color=col,
                    label=vol.capitalize() if not by_drug else f"{vol.capitalize()}, {drug}")
            ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.18, lw=0)
    ax.axhline(0, ls="--", color="0.5", lw=0.8)
    ax.set_xticks(x)


def plot_iks(ax, d, by_drug):
    """IKS per participant; without drug: stable -> volatile paired; with drug: placebo ->
    ketamine paired within stable and within volatile."""
    groups = DRUGS if by_drug else ["pooled"]
    shift = pl.GROUP_PAIR_DOT_SHIFT if by_drug else 0.0
    sem = pl.cousineau_morey_sem(d, "iks", "subID", ["volatility", "drug_label"])
    pts, means = {}, {}
    for i, vol in enumerate(VOLS):
        for k, drug in enumerate(groups):
            cell = d[(d.volatility == vol) & (d.drug_label == drug)]
            x0 = i + (k - 0.5) * 2 * shift if by_drug else i
            xs = x0 + pl.jitter(cell["subID"])
            colour = pl.DRUG_PALETTE[drug] if by_drug else pl.VOLATILITY_PALETTE[vol]
            pl.scatter_subjects(ax, xs, cell["iks"], colour)
            pts[(vol, drug)] = dict(zip(cell["subID"], zip(xs, cell["iks"])))
            means[(vol, drug)] = (x0, cell["iks"].mean())
            pl.draw_mean_sem(ax, x0, cell["iks"].mean(), sem[(vol, drug)])
    # pairs of cells whose participants are joined
    pairs = [((v, "placebo"), (v, "ketamine")) for v in VOLS] if by_drug else [(("stable", "pooled"),
                                                                               ("volatile", "pooled"))]
    for a, b in pairs:
        for s in set(pts[a]) & set(pts[b]):
            ax.plot([pts[a][s][0], pts[b][s][0]], [pts[a][s][1], pts[b][s][1]], color=pl.LINE_COLOR,
                    lw=pl.LINE_LW, alpha=pl.LINE_ALPHA, zorder=5)
        ax.plot([means[a][0], means[b][0]], [means[a][1], means[b][1]], color="black", lw=1.8, zorder=25)
    if by_drug:
        for drug in DRUGS:                               # legend entries: dots coloured by drug
            ax.scatter([], [], s=pl.DOT_SIZE, color=pl.DRUG_PALETTE[drug], edgecolors="0.3", label=drug.capitalize())
    ax.axhline(0, ls="--", color="0.5", lw=0.8, zorder=1)
    ax.set_xticks(range(len(VOLS)))
    ax.set_xticklabels([v.capitalize() for v in VOLS])
    ax.set_xlim(-0.5, len(VOLS) - 0.5)


def iks_ylims(subj, cohort, kernel, by_drug):
    """IKS_YLIMS for this panel, widened if any participant value falls outside (None = automatic)."""
    lims = IKS_YLIMS.get((cohort, kernel, by_drug))
    if lims is None:
        return None
    lo, hi = lims
    pad = 0.04 * (hi - lo)
    lo2, hi2 = min(lo, subj["iks"].min() - pad), max(hi, subj["iks"].max() + pad)
    if (lo2, hi2) != (lo, hi):
        print(f"WARNING: {cohort} {kernel} IKS: values outside {lims}, limits widened to ({lo2:.3f}, {hi2:.3f})")
    return lo2, hi2


def panel(subj, kind, kernel, by_drug, ylims=None):
    """Kernel shape or IKS panel: valence (top) and arousal (bottom), same y axis."""
    pl.set_style()
    width = 3.2 if (kind == "IKS" and not by_drug) else 4.2      # IKS without drug: two columns only
    fig, axes = plt.subplots(2, 1, figsize=(width, 7.4), sharey=True, gridspec_kw={"hspace": 0.42})
    for r, (ax, dim) in enumerate(zip(axes, DIMS)):
        d = subj[subj.dimension == dim]
        if kind == "shape":
            plot_shape(ax, d, by_drug)
            pl.style_axis(ax, title=dim.capitalize(), ylabel=Y_LABEL[kernel], xlabel=X_LABEL[kernel])
        else:
            plot_iks(ax, d, by_drug)
            pl.style_axis(ax, title=dim.capitalize(), ylabel="IKS (β−1 minus β−2)")
            if ylims:
                ax.set_ylim(*ylims)
        if r == 0 and kind == "shape":
            ax.legend(frameon=False, fontsize=pl.TICK_FS - 2)
        if r == 0 and kind == "IKS" and by_drug:          # outside, so it does not cover the dots
            ax.legend(frameon=False, fontsize=pl.TICK_FS - 2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fig


def main(cohort):
    formats = load_config(cohort)["figures"]["formats"]
    for (kernel, by_drug), ids in PANELS[cohort].items():
        cond = pd.read_csv(out_dir("tables", cohort) / f"{kernel}_condition.csv")
        subj = subject_cells(cond, by_drug)
        for kind, fig_id in zip(("shape", "IKS"), ids):
            name = f"{kernel}_{'kernel' if kind == 'shape' else 'IKS'}_by_{'drug_and_' if by_drug else ''}volatility"
            if fig_id:                                       # in the paper: figure folder
                folder, file = out_dir("figures", fig_id[:-1]), f"{fig_id}_{name}"
            else:                                            # not in the paper: extra/
                folder, file = out_dir("figures", "extra"), f"{cohort}_{name}"
            (folder / "source_data").mkdir(parents=True, exist_ok=True)
            subj.to_csv(folder / "source_data" / f"{file}.csv", index=False)
            ylims = iks_ylims(subj, cohort, kernel, by_drug) if kind == "IKS" else None
            pl.save_figure(panel(subj, kind, kernel, by_drug, ylims), folder / file, formats)
            print(file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=COHORTS)
    main(ap.parse_args().cohort)
