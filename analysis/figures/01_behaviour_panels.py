"""Behaviour figure panels: CP accuracy, correct RT, rating error and rating changes.

    python analysis/figures/01_behaviour_panels.py --cohort in_person    # Fig 2B-F, 3, 4A, 5B, S2, S3A-D, S5
    python analysis/figures/01_behaviour_panels.py --cohort online       # Fig S1B-I (+ extra)

Needs the tables from analysis/behaviour/01_make_behaviour_tables.py.
Every panel that can be drawn for the cohort is drawn (drug panels: in person only).
Panels in the paper go to a folder per figure, the others to extra/:
    output/figures/<Fig>/<FigID>_<name>.png        e.g. output/figures/Fig3/Fig3C_....png
    output/figures/extra/<cohort>_<name>.png
with the plotted values in source_data/<same name>.csv next to them.
File formats: config/analysis_config.yaml, figures: formats.

What every panel shows
    1. rows (change points or blocks) -> mean per participant x drug session x cell;
    2. -> mean over the two drug sessions, unless the panel shows drug;
    3. one dot per participant and cell; box; half-density (most panels); black dot and
       bar = group mean +/- within-participant SEM (Cousineau-Morey). If some participants
       miss a cell, the ordinary SEM is used instead (column sem_method in source_data).
    Change points: analysable ones only; RT: correctly detected change points only.
    Grey lines join the two values of a participant for within-participant comparisons.
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import plotting as pl                      # noqa: E402
from utils.config import COHORTS, load_config, out_dir  # noqa: E402

DIMS = ["valence", "arousal"]
EMOTIONS = {"valence": ["negative", "neutral", "positive"], "arousal": ["low", "medium", "high"]}
EXTREMES_THEN_MIDDLE = {"valence": ["negative", "positive", "neutral"], "arousal": ["low", "high", "medium"]}
MIDDLE = {"valence": "neutral", "arousal": "medium"}
EXTREME = {("valence", "down"): "negative", ("valence", "up"): "positive",
           ("arousal", "down"): "low", ("arousal", "up"): "high"}
DRUGS, VOLS = ["placebo", "ketamine"], ["stable", "volatile"]
GREY_MIDDLE = {"down": "#9E9E9E", "up": "#BCBCBC"}
DRUG_LEGEND = [("Placebo", pl.COL_PLACEBO), ("Ketamine", pl.COL_KETAMINE)]

YLABEL = {"detected_correctly": "CP accuracy", "rt_first_correct_s": "Correct RT (s)",
          "RE_flipped": "RE flipped", "RE_abs": "Absolute RE",
          "n_rating_changes": "Rating changes / block", "rating_changes_per_cp": "Rating changes / true CP",
          "occupancy_ratio": "Time at rating level /\ntime true mean at level"}
# y-axis limits, widened only if a cohort's values fall outside them
YLIMS = {"detected_correctly": (0.2, 1.05), "rt_first_correct_s": (1.0, 5.0),
         "RE_flipped": (-0.6, 2.5), "RE_abs": (0.5, 1.8),
         "n_rating_changes": (10.0, 85.0), "rating_changes_per_cp": (0.0, 4.0), "occupancy_ratio": (0.0, 3.0)}
# emotion level of each rating level (for the colours of the rating-level panel)
LEVEL_EMOTION = {"valence": {2: "negative", 3: "negative", 4: "neutral", 5: "positive", 6: "positive"},
                 "arousal": {2: "low", 4: "medium", 6: "high"}}
# figure sizes (inches) of the panels with two boxes per panel: two panels side by side
# (valence | arousal) or one panel. Smaller widths bring the two boxes closer together.
PAIR_FIGSIZE, PAIR_SINGLE_FIGSIZE = (5.2, 4.0), (3.0, 4.0)
# smaller group-mean dot where the SEM is smaller than the default dot (CP magnitude panels)
SMALL_MEAN = 35
# narrow boxes for the magnitude x emotion panels
NESTED_BOX_W, NESTED_BOX_LEFT, NESTED_DOT_RIGHT, HUE_SPREAD, NESTED_JITTER = 0.14, 0.05, 0.05, 0.30, 0.016


def cap(s):
    return s[:1].upper() + s[1:]


# ------------------------------------------------------------------ input tables

def load_tables(cohort):
    """The behaviour tables of one cohort, keyed by the names used in panels()."""
    d = out_dir("tables", cohort)
    cp = pd.read_csv(d / "cp_table.csv")
    cp = cp[cp["admissible"]].copy()
    # level reached by the change point: the middle level, or the extreme in the direction
    # of the change (e.g. down -> negative); other change points get no target
    is_middle = cp["emotion_level"].eq(cp["dimension"].map(MIDDLE))
    is_extreme = cp["emotion_level"].eq([EXTREME.get(k) for k in zip(cp["dimension"], cp["cp_direction"])])
    cp["cp_target"] = np.select([is_middle, is_extreme], ["neutral", "extreme"], None)
    return {"cp": cp, "cp_correct": cp[cp["detected_correctly"] == 1].copy(),
            "re_block": pd.read_csv(d / "rating_error_block.csv"),
            "re_emotion": pd.read_csv(d / "rating_error_block_emotion.csv"),
            "changes": pd.read_csv(d / "rating_changes_block.csv"),
            "occupancy": pd.read_csv(d / "occupancy_block.csv"),
            # real vs shuffled-null participant values (in person only, from 04_null_tests.py)
            "null": (pd.read_csv(d / "null_subject_values.csv")
                     .melt(id_vars=["subID", "dimension", "measure"], value_vars=["real", "null"],
                           var_name="kind", value_name="value") if cohort == "in_person" else None)}


def subject_cells(df, dv, cols, keep_drug):
    """Steps 1-2 above: one value per participant x dimension x cols (x drug)."""
    per_drug = df.dropna(subset=[dv]).groupby(["subID", "drug_label", "dimension", *cols], as_index=False)[dv].mean()
    if not keep_drug:
        per_drug = per_drug.groupby(["subID", "dimension", *cols], as_index=False)[dv].mean()
    return per_drug.rename(columns={dv: "value"})


# ------------------------------------------------------------------ layouts
# A layout lists the cells of one panel (which rows, where the box and dots go, colour),
# the pairs of cells whose dots are joined, dividers, tick labels and an optional legend.

@dataclass
class Cell:
    where: dict
    colour: str
    box_x: float
    dot_x: float
    box_w: float = pl.BOX_WIDTH
    jitter: float = pl.DOT_JITTER
    density_side: str = "left"


@dataclass
class Layout:
    cells: list
    ticks: list
    pairs: list = field(default_factory=list)
    dividers: list = field(default_factory=list)   # (x, linestyle)
    legend: list = field(default_factory=list)     # (label, colour)
    legend_title: str = None
    legend_outside: bool = True
    xlabel: str = ""
    xpad: float = 0.55


def pair_on_two_ticks(factor, levels, colours, labels=None, t0=0.0):
    """Two levels on neighbouring ticks: boxes outside, dots inside (joined)."""
    cells = []
    for k, lev in enumerate(levels):
        sign = -1.0 if k == 0 else 1.0
        cells.append(Cell({factor: lev}, colours[lev], t0 + k + sign * pl.BOX_OUTER_SHIFT,
                          t0 + k - sign * pl.DOT_INNER_SHIFT, density_side="left" if k == 0 else "right"))
    ticks = [(t0 + k, (labels or {}).get(lev, cap(str(lev)))) for k, lev in enumerate(levels)]
    return cells, ticks


def paired_two(factor, levels, colours, labels=None):
    cells, ticks = pair_on_two_ticks(factor, levels, colours, labels)
    return Layout(cells, ticks, pairs=[(0, 1)])


def same_tick_pairs(major, majors, minor, minors, colour, labels=None, legend=None, outside=True):
    """Two minor levels side by side on each major tick, joined (e.g. drug within volatility)."""
    cells, pairs = [], []
    for t, maj in enumerate(majors):
        for k, mi in enumerate(minors):
            sign = -1.0 if k == 0 else 1.0
            cells.append(Cell({major: maj, minor: mi}, colour(maj, mi), t + sign * pl.GROUP_PAIR_BOX_SHIFT,
                              t + sign * pl.GROUP_PAIR_DOT_SHIFT, density_side="left" if k == 0 else "right"))
        pairs.append((len(cells) - 2, len(cells) - 1))
    ticks = [(t, (labels or {}).get(m, str(m))) for t, m in enumerate(majors)]
    return Layout(cells, ticks, pairs, legend=legend or [], legend_outside=outside)


def magnitudes(table, dim):
    return sorted(table.loc[table.dimension == dim, "cp_magnitude"].dropna().unique())


def lay_volatility(dim):
    return paired_two("volatility", VOLS, pl.VOLATILITY_PALETTE, labels={"stable": "stable", "volatile": "volatile"})


def lay_direction(dim):
    return paired_two("cp_direction", ["down", "up"], pl.CP_DIRECTION_PALETTE)


def lay_target(dim):
    """Down->middle | Down->extreme : Up->middle | Up->extreme, joined within direction."""
    cells, ticks, pairs = [], [], []
    for d, base in (("down", 0.0), ("up", 2.0)):
        ext = EXTREME[(dim, d)]
        col = {"neutral": GREY_MIDDLE[d], "extreme": pl.EMOTION_PALETTES[dim][ext]}
        lab = {"neutral": f"{cap(d)}→\n{MIDDLE[dim]}", "extreme": f"{cap(d)}→\n{ext}"}
        c, t = pair_on_two_ticks("cp_target", ["neutral", "extreme"], col, lab, t0=base)
        for cell in c:
            cell.where["cp_direction"] = d
        pairs.append((len(cells), len(cells) + 1))
        cells += c
        ticks += t
    return Layout(cells, ticks, pairs, dividers=[(1.5, "--")])


def lay_re_emotion(dim):
    """The two extremes joined, then a divider and the middle level."""
    lo, hi, mid = EXTREMES_THEN_MIDDLE[dim]
    pal = pl.EMOTION_PALETTES[dim]
    cells, ticks = pair_on_two_ticks("emotion_level", [lo, hi], pal)
    cells.append(Cell({"emotion_level": mid}, pal[mid], 2 - 0.14, 2 + 0.12))
    return Layout(cells, ticks + [(2.0, cap(mid))], pairs=[(0, 1)], dividers=[(1.5, "-")])


def lay_re_emotion_drug(dim):
    order = EXTREMES_THEN_MIDDLE[dim]
    lay = same_tick_pairs("emotion_level", order, "drug_label", DRUGS, lambda a, b: pl.DRUG_PALETTE[b],
                          labels={e: cap(e) for e in order}, legend=DRUG_LEGEND, outside=False)
    lay.dividers = [(1.5, "-")]
    return lay


def lay_drug_within_volatility(dim):
    return same_tick_pairs("volatility", VOLS, "drug_label", DRUGS, lambda a, b: pl.DRUG_PALETTE[b],
                           legend=DRUG_LEGEND)


def lay_rating_level(dim):
    """One box per rating level, coloured by its emotion level, no lines."""
    levels = sorted(LEVEL_EMOTION[dim])
    cells = [Cell({"level": lev}, pl.EMOTION_PALETTES[dim][LEVEL_EMOTION[dim][lev]], i - 0.14, i + 0.12)
             for i, lev in enumerate(levels)]
    return Layout(cells, [(i, str(lev)) for i, lev in enumerate(levels)], xlabel="Rating level")


def lay_real_vs_null():
    """Real | Null for valence, then for arousal, in one panel."""
    cells, ticks = [], []
    for k, dim in enumerate(DIMS):
        c, t = pair_on_two_ticks("kind", ["real", "null"], {"real": pl.DIMENSION_PALETTE[dim], "null": "#BDBDBD"},
                                 t0=3.0 * k)
        for cell in c:
            cell.where["dimension"] = dim
        cells += c
        ticks += t
    lay = Layout(cells, ticks, dividers=[(2.0, "-")], xpad=0.6)     # no connecting lines (as in Fig 2B-C)
    return lay, [(-0.45, 1.45, "Valence"), (2.55, 4.45, "Arousal")]


def lay_drug(dim):
    lay = paired_two("drug_label", DRUGS, pl.DRUG_PALETTE)
    lay.legend = DRUG_LEGEND
    return lay


def lay_magnitude_volatility(table):
    return lambda dim: same_tick_pairs("cp_magnitude", magnitudes(table, dim), "volatility", VOLS,
                                       lambda m, v: pl.VOLATILITY_PALETTE[v],
                                       labels={m: str(int(m)) for m in magnitudes(table, dim)},
                                       legend=[("Stable", pl.COL_STABLE), ("Volatile", pl.COL_VOLATILE)])


def lay_magnitude_emotion(table):
    """Emotion levels spread around each magnitude tick, narrow boxes, no lines."""
    def f(dim):
        sub = table[table.dimension == dim]
        cells, ticks, used = [], [], []
        for t, m in enumerate(magnitudes(table, dim)):
            present = [e for e in EMOTIONS[dim] if ((sub.cp_magnitude == m) & (sub.emotion_level == e)).any()]
            used += present
            for k, e in enumerate(present):
                off = 0.0 if len(present) == 1 else (k / (len(present) - 1) - 0.5) * 2 * HUE_SPREAD
                cells.append(Cell({"cp_magnitude": m, "emotion_level": e}, pl.EMOTION_PALETTES[dim][e],
                                  t + off - NESTED_BOX_LEFT, t + off + NESTED_DOT_RIGHT, NESTED_BOX_W, NESTED_JITTER))
            ticks.append((t, str(int(m))))
        legend = [(cap(e), pl.EMOTION_PALETTES[dim][e]) for e in EMOTIONS[dim] if e in used]
        return Layout(cells, ticks, legend=legend, legend_title="Emotion", xpad=0.55 + HUE_SPREAD)
    return f


def lay_magnitude_single(table):
    """Valence and arousal magnitudes in one panel, unpaired, with header boxes."""
    cells, ticks, groups = [], [], []
    for dim in DIMS:
        first = len(cells)
        for m in magnitudes(table, dim):
            i = len(cells)
            cells.append(Cell({"dimension": dim, "cp_magnitude": m}, pl.MAGNITUDE_PALETTE.get(int(m), "#D9D9D9"),
                              i - 0.14, i + 0.12))
            ticks.append((float(i), str(int(m))))
        groups.append((first - 0.45, len(cells) - 1 + 0.45, cap(dim)))
    return Layout(cells, ticks, dividers=[(groups[0][1] + 0.05, "-")], xlabel="CP magnitude"), groups


def lay_dimension_single():
    lay = paired_two("dimension", DIMS, pl.DIMENSION_PALETTE)
    lay.pairs = []          # no connecting lines (as in the published Fig 2E)
    return lay, None


# ------------------------------------------------------------------ panels
# name, table, dv, cell columns, layout; drug=True: drug is shown (in person only);
# single=(layout, header groups): one panel instead of valence | arousal;
# stacked: valence above arousal; half=False: no half-densities.

def panels(t, cohort):
    """Specification of every panel (see the comment above); drug and null panels in person only."""
    acc, rt = "detected_correctly", "rt_first_correct_s"
    P = dict(
        acc_by_magnitude=dict(table="cp", dv=acc, cols=["cp_magnitude"], single=lay_magnitude_single(t["cp"]),
                              mean_size=SMALL_MEAN),
        rt_by_dimension=dict(table="cp_correct", dv=rt, cols=[], single=lay_dimension_single(), figsize=PAIR_SINGLE_FIGSIZE),
        rt_by_magnitude=dict(table="cp_correct", dv=rt, cols=["cp_magnitude"],
                             single=lay_magnitude_single(t["cp_correct"]), mean_size=SMALL_MEAN),
        RE_by_emotion=dict(table="re_emotion", dv="RE_flipped", cols=["emotion_level"], layout=lay_re_emotion,
                           stacked=True, figsize=(4.6, 7.2), zero_line=True),
        rt_by_cp_direction=dict(table="cp_correct", dv=rt, cols=["cp_direction"], layout=lay_direction),
        # CP magnitude < 3, so that changes reaching the middle and the extreme levels are
        # matched in magnitude (the middle level cannot be reached by the largest changes)
        rt_by_cp_direction_and_level_reached=dict(table="cp_correct", dv=rt, cols=["cp_direction", "cp_target"],
                                                  layout=lay_target, query="cp_target.notna() and cp_magnitude < 3",
                                                  figsize=(10.0, 3.9)),
        rt_by_volatility=dict(table="cp_correct", dv=rt, cols=["volatility"], layout=lay_volatility),
        RE_by_emotion_and_drug=dict(table="re_emotion", dv="RE_flipped", drug=True, cols=["emotion_level"],
                                    layout=lay_re_emotion_drug, stacked=True, figsize=(4.8, 7.4), zero_line=True),
        acc_by_magnitude_and_emotion=dict(table="cp", dv=acc, cols=["cp_magnitude", "emotion_level"],
                                          layout=lay_magnitude_emotion(t["cp"]), figsize=(10.2, 4.0),
                                          xlabel="CP magnitude", half=False),
        rt_by_magnitude_and_emotion=dict(table="cp_correct", dv=rt, cols=["cp_magnitude", "emotion_level"],
                                         layout=lay_magnitude_emotion(t["cp_correct"]), figsize=(10.2, 4.0),
                                         xlabel="CP magnitude", half=False),
        # control: only change points reaching an extreme level (down towards negative/low, up towards positive/high) are included
        rt_by_cp_direction_extremes=dict(table="cp_correct", dv=rt, cols=["cp_direction"], layout=lay_direction,
                                         query="cp_target == 'extreme'"),
        rating_changes_by_volatility=dict(table="changes", dv="n_rating_changes", cols=["volatility"],
                                          layout=lay_volatility),
        rating_changes_per_cp_by_volatility=dict(table="changes", dv="rating_changes_per_cp", cols=["volatility"],
                                                 layout=lay_volatility),
        acc_by_volatility=dict(table="cp", dv=acc, cols=["volatility"], layout=lay_volatility),
        rt_by_magnitude_and_volatility=dict(table="cp_correct", dv=rt, cols=["cp_magnitude", "volatility"],
                                            layout=lay_magnitude_volatility(t["cp_correct"]), figsize=(9.6, 4.0),
                                            xlabel="CP magnitude"),
        # real data vs the mean over shuffled-rating datasets (in person only)
        real_vs_null_absolute_RE=dict(table="null", dv="RE_abs", subject_level=True, in_person_only=True,
                                      single=lay_real_vs_null(), half=False, figsize=(4.8, 4.0),
                                      ylims=(0.5, 2.0), ylabel="Absolute rating error"),
        real_vs_null_acc=dict(table="null", dv="detected_correctly", subject_level=True, in_person_only=True,
                              single=lay_real_vs_null(), half=False, figsize=(4.8, 4.0)),
        # ratio of time spent at each rating level to time the true mean was at that level
        time_at_rating_level=dict(table="occupancy", dv="occupancy_ratio", cols=["level"], layout=lay_rating_level,
                                  one_line=True, figsize=(7.4, 4.0)),
        absolute_RE_by_drug=dict(table="re_block", dv="RE_abs", drug=True, cols=[], layout=lay_drug,
                                 figsize=(6.6, 4.0)),            # wider than PAIR_FIGSIZE: legend outside
        rating_changes_by_drug_and_volatility=dict(table="changes", dv="n_rating_changes", drug=True,
                                                   cols=["volatility"], layout=lay_drug_within_volatility,
                                                   figsize=(9.6, 4.0)),
        acc_by_drug_and_volatility=dict(table="cp", dv=acc, drug=True, cols=["volatility"],
                                        layout=lay_drug_within_volatility, figsize=(9.6, 4.0)),
        rt_by_drug_and_volatility=dict(table="cp_correct", dv=rt, drug=True, cols=["volatility"],
                                       layout=lay_drug_within_volatility, figsize=(9.6, 4.0)),
    )
    if cohort == "online":                     # no drug condition online; null panels only in person (Fig 2B-C)
        P = {k: v for k, v in P.items() if not (v.get("drug") or v.get("in_person_only"))}
    return P


# Figure ID of each panel in the paper. Panels not listed are saved as extra plots.
FIGURE_IDS = {"in_person": {"Fig2B": "real_vs_null_absolute_RE", "Fig2C": "real_vs_null_acc", "Fig2D": "acc_by_magnitude", "Fig2E": "rt_by_dimension", "Fig2F": "rt_by_magnitude",
                         "Fig3A": "RE_by_emotion", "Fig3B": "rt_by_cp_direction",
                         "Fig3C": "rt_by_cp_direction_and_level_reached", "Fig4A": "rt_by_volatility",
                         "Fig5B": "RE_by_emotion_and_drug",
                         "FigS2A": "acc_by_magnitude_and_emotion", "FigS2B": "rt_by_magnitude_and_emotion",
                         "FigS2C": "rt_by_cp_direction_extremes", "FigS2D": "time_at_rating_level",
                         "FigS3A": "rating_changes_by_volatility", "FigS3B": "rating_changes_per_cp_by_volatility",
                         "FigS3C": "acc_by_volatility", "FigS3D": "rt_by_magnitude_and_volatility",
                         "FigS5A": "absolute_RE_by_drug", "FigS5B": "rating_changes_by_drug_and_volatility",
                         "FigS5C": "acc_by_drug_and_volatility", "FigS5D": "rt_by_drug_and_volatility"},
           "online": {"FigS1B": "acc_by_magnitude", "FigS1C": "rt_by_magnitude", "FigS1D": "RE_by_emotion",
                      "FigS1E": "acc_by_magnitude_and_emotion", "FigS1F": "rt_by_cp_direction",
                      "FigS1G": "rt_by_cp_direction_and_level_reached", "FigS1H": "rating_changes_by_volatility",
                      "FigS1I": "rt_by_volatility"}}


# ------------------------------------------------------------------ drawing

def draw_layout(ax, subj, lay, half_density, mean_size=pl.MEAN_SIZE):
    """Draw every cell of a layout from the participant table; return the plotted values."""
    parts, xs = [], {}
    for i, c in enumerate(lay.cells):
        m = np.ones(len(subj), bool)
        for k, v in c.where.items():
            m &= subj[k].eq(v).to_numpy()
        part = subj[m].assign(cell=i)
        parts.append(part)
        xs[i] = dict(zip(part["subID"], c.dot_x + pl.jitter(part["subID"], scale=c.jitter)))
    plotted = pd.concat(parts, ignore_index=True)
    present = [i for i, p in enumerate(parts) if len(p)]
    complete = plotted.groupby("subID")["cell"].nunique().eq(len(present)).all()
    if complete and len(present) >= 2:
        sems = pl.cousineau_morey_sem(plotted, "value", "subID", ["cell"])
        plotted["sem_method"] = "cousineau-morey"
    else:
        sems = plotted.groupby("cell")["value"].agg(lambda v: v.std(ddof=1) / np.sqrt(len(v)))
        plotted["sem_method"] = "ordinary (incomplete cells)"
    plotted["sem"] = plotted["cell"].map(sems)
    for i in present:
        c, vals = lay.cells[i], parts[i]["value"].to_numpy(float)
        if half_density:
            pl.half_density_at(ax, vals, c.box_x, c.colour, side=c.density_side)
        pl.box_at(ax, vals, c.box_x, c.colour, c.box_w)
        pl.scatter_subjects(ax, [xs[i][s] for s in parts[i]["subID"]], vals, c.colour)
        pl.draw_mean_sem(ax, c.dot_x, float(np.mean(vals)), float(sems.get(i, np.nan)), mean_size)
    for a, b in lay.pairs:
        ya, yb = dict(zip(parts[a]["subID"], parts[a]["value"])), dict(zip(parts[b]["subID"], parts[b]["value"]))
        for s in set(ya) & set(yb):
            ax.plot([xs[a][s], xs[b][s]], [ya[s], yb[s]], color=pl.LINE_COLOR, lw=pl.LINE_LW,
                    alpha=pl.LINE_ALPHA, zorder=5)
    for x, ls in lay.dividers:
        ax.axvline(x, color="0.45" if ls == "--" else "0.55", lw=1.0, ls=ls, zorder=0)
    ax.set_xticks([t for t, _ in lay.ticks])
    ax.set_xticklabels([lab for _, lab in lay.ticks])
    pad = pl.DENSITY_WIDTH if half_density else 0
    ax.set_xlim(min(t for t, _ in lay.ticks) - lay.xpad - pad, max(t for t, _ in lay.ticks) + lay.xpad + pad)
    if lay.xlabel:
        ax.set_xlabel(lay.xlabel, fontsize=pl.LABEL_FS)
    if lay.legend:
        kw = dict(loc="lower left", bbox_to_anchor=(1.01, 0.0)) if lay.legend_outside else dict(loc="upper right")
        ax.legend([Patch(facecolor=col, edgecolor="k") for _, col in lay.legend], [lab for lab, _ in lay.legend],
                  title=lay.legend_title, frameon=False, fontsize=pl.TICK_FS - 1, title_fontsize=pl.TICK_FS - 1, **kw)
    return plotted


def header_boxes(ax, groups):
    """'Valence' | 'Arousal' boxes above a single panel."""
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, label in groups:
        ax.add_patch(Rectangle((x0, 1.03), x1 - x0, 0.11, transform=trans, facecolor="white",
                               edgecolor="black", lw=1.0, clip_on=False))
        ax.text((x0 + x1) / 2, 1.085, label, ha="center", va="center", transform=trans, fontsize=pl.TICK_FS)


def make_panel(name, spec, tables, folder, formats):
    """Draw one panel from its specification and save it with its source data."""
    df = tables[spec["table"]]
    if spec.get("query"):
        df = df.query(spec["query"], engine="python")
    if spec.get("subject_level"):              # table already has one value per participant and cell
        subj = df[df["measure"] == spec["dv"]]
    else:
        subj = subject_cells(df, spec["dv"], spec["cols"], spec.get("drug", False))
    lo, hi = spec.get("ylims", YLIMS[spec["dv"]])
    pad = 0.04 * (hi - lo)
    ylims = (min(lo, subj["value"].min() - pad), max(hi, subj["value"].max() + pad))
    half = spec.get("half", True)
    pl.set_style()
    if "single" in spec:
        lay, groups = spec["single"]
        fig, ax = plt.subplots(figsize=spec.get("figsize", (5.6, 4.0)))
        plotted = draw_layout(ax, subj, lay, half, spec.get("mean_size", pl.MEAN_SIZE))
        ax.set_ylim(*ylims)
        pl.style_axis(ax, ylabel=spec.get("ylabel", YLABEL[spec["dv"]]))
        if groups:
            header_boxes(ax, groups)
        axes = [ax]
    else:
        stacked = spec.get("stacked", False)
        fig, axes = plt.subplots(2 if stacked else 1, 1 if stacked else 2, figsize=spec.get("figsize", PAIR_FIGSIZE))
        axes, plotted = list(np.ravel(axes)), []
        for ax, dim in zip(axes, DIMS):
            lay = spec["layout"](dim)
            lay.xlabel = spec.get("xlabel", lay.xlabel)
            plotted.append(draw_layout(ax, subj[subj.dimension == dim], lay, half, spec.get("mean_size", pl.MEAN_SIZE)))
            ax.set_ylim(*ylims)
            pl.style_axis(ax, title=cap(dim), ylabel=YLABEL[spec["dv"]] if (dim == "valence" or stacked) else "")
            ax.set_xlabel(lay.xlabel, fontsize=pl.LABEL_FS)
        plotted = pd.concat(plotted, ignore_index=True)
    for key, y in (("zero_line", 0), ("one_line", 1)):     # reference line at 0 or 1
        if spec.get(key):
            for ax in axes:
                ax.axhline(y, color="black", lw=1.0, ls="--", zorder=1)
    fig.tight_layout()
    (folder / "source_data").mkdir(parents=True, exist_ok=True)
    plotted.to_csv(folder / "source_data" / f"{name}.csv", index=False)
    pl.save_figure(fig, folder / name, formats)


def main(cohort):
    formats = load_config(cohort)["figures"]["formats"]
    tables = load_tables(cohort)
    figure_of = {name: fig for fig, name in FIGURE_IDS[cohort].items()}
    for name, spec in panels(tables, cohort).items():
        if name in figure_of:                  # in the paper: output/figures/Fig3/Fig3C_<name>
            fig = figure_of[name]
            folder, file = out_dir("figures", re.match(r"Fig(S?\d+)", fig).group(0)), f"{fig}_{name}"
        else:                                  # not in the paper: output/figures/extra/<cohort>_<name>
            folder, file = out_dir("figures", "extra"), f"{cohort}_{name}"
        make_panel(file, spec, tables, folder, formats)
        print(file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=COHORTS)
    main(ap.parse_args().cohort)
