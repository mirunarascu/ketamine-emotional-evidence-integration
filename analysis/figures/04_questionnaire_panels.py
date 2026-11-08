"""Questionnaire panels (Fig S6A-D): BPRS, CADSS-6, PANAS negative and positive affect.

    python analysis/figures/04_questionnaire_panels.py

Needs data/questionnaires_in_person.csv and, for the significance marks,
output/stats/in_person/questionnaires_wilcoxon.csv (analysis/stats/04_questionnaire_stats.R).
At each timepoint placebo (left) and ketamine (right): one dot per participant, grey lines
join a participant's two visits, black dot and bar = mean +/- within-participant SEM
(Cousineau-Morey over the drug x timepoint cells of the questionnaire).
Missing data: one participant's after-infusion CADSS-6 at the placebo visit is missing.
In Fig S6B that participant's ketamine dot is drawn without a connecting line, and the
means use all available scores (CADSS after: ketamine 1.00, n = 29; placebo 0.18, n = 28;
the test and the text use the 28 complete pairs, ketamine 1.04).
Marks (BPRS, CADSS only): Holm-corrected paired Wilcoxon tests, ketamine vs placebo,
*** p < .001, ** p < .01, * p < .05, † p < .10.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import plotting as pl                       # noqa: E402
from utils.config import data_dir, load_config, out_dir  # noqa: E402

SHIFT = {"placebo": -0.18, "ketamine": 0.18}           # position of each drug around the timepoint
PANELS = [  # (figure ID, measure, y-axis label, y limits, tick step)
    ("FigS6A", "BPRS", "BPRS total score", (3, 8.6), 1),
    ("FigS6B", "CADSS", "CADSS total score", (-2, 22), 5),     # below 0 so that scores of 0 are visible
    ("FigS6C", "PANAS_negative", "PANAS negative affect", (0, 40), 5),
    ("FigS6D", "PANAS_positive", "PANAS positive affect", (0, 80), 10),
]
TIMEPOINTS = ["before", "during", "after"]


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "†" if p < .10 else ""


def main():
    formats = load_config("in_person")["figures"]["formats"]
    data = pd.read_csv(data_dir() / "questionnaires_in_person.csv")
    tests = pd.read_csv(out_dir("stats", "in_person") / "questionnaires_wilcoxon.csv")
    marks = {(r.measure, r.timepoint): stars(r.p_holm) for r in tests.itertuples()}
    folder = out_dir("figures", "FigS6")
    (folder / "source_data").mkdir(parents=True, exist_ok=True)
    for fig_id, measure, ylabel, ylim, step in PANELS:
        d = data[data["measure"] == measure]
        sem = pl.cousineau_morey_sem(d, "score", "subID", ["drug", "timepoint"])
        timepoints = [t for t in TIMEPOINTS if t in set(d["timepoint"])]
        pl.set_style()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for i, tp in enumerate(timepoints):
            wide = d[d["timepoint"] == tp].pivot(index="subID", columns="drug", values="score")
            x = {drug: i + SHIFT[drug] + pl.jitter(wide.index) for drug in SHIFT}
            both = wide.notna().all(axis=1).to_numpy()          # participants with both visits
            for xp, xk, yp, yk in zip(x["placebo"][both], x["ketamine"][both],
                                      wide["placebo"][both], wide["ketamine"][both]):
                ax.plot([xp, xk], [yp, yk], color=pl.LINE_COLOR, alpha=pl.LINE_ALPHA, lw=pl.LINE_LW, zorder=5)
            for drug in SHIFT:
                ok = wide[drug].notna().to_numpy()
                pl.scatter_subjects(ax, x[drug][ok], wide[drug][ok], pl.DRUG_PALETTE[drug])
                pl.draw_mean_sem(ax, i + SHIFT[drug], wide[drug].mean(), sem[(drug, tp)])
            mark = marks.get((measure, tp), "")
            if mark:
                y = d.loc[d["timepoint"] == tp, "score"].max() + 0.04 * (ylim[1] - ylim[0])
                ax.plot([i + SHIFT["placebo"], i + SHIFT["ketamine"]], [y, y], color="black", lw=1.2)
                ax.text(i, y, mark, ha="center", va="bottom", fontsize=14)
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels([t.capitalize() for t in timepoints])
        ax.set_xlim(-0.6, len(timepoints) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(max(0, np.ceil(ylim[0])), ylim[1] + 0.01, step))
        pl.style_axis(ax, ylabel=ylabel)
        if measure == "BPRS":                                   # legend in the first panel only
            for drug in SHIFT:
                ax.scatter([], [], s=pl.DOT_SIZE, color=pl.DRUG_PALETTE[drug], edgecolors="0.3", label=drug.capitalize())
            ax.legend(frameon=False, loc="upper left", fontsize=pl.TICK_FS - 1)
        fig.tight_layout()
        name = f"{fig_id}_{measure}"
        d.assign(sem=[sem[(a, b)] for a, b in zip(d["drug"], d["timepoint"])]).to_csv(
            folder / "source_data" / f"{name}.csv", index=False)
        pl.save_figure(fig, folder / name, formats)
        print(name)


if __name__ == "__main__":
    main()
