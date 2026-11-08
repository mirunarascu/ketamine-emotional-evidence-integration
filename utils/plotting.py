"""Colours, sizes and small drawing helpers shared by all figure scripts.

"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors   # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

# ------------------------------------------------------------------ colours
COL_STABLE, COL_VOLATILE = "#396897", "#973968"
COL_PLACEBO, COL_KETAMINE = "#F0DCC6", "#6B668D"
COL_VALENCE, COL_AROUSAL = "#2A9D8F", "#B07AA1"
VOLATILITY_PALETTE = {"stable": COL_STABLE, "volatile": COL_VOLATILE}
DRUG_PALETTE = {"placebo": COL_PLACEBO, "ketamine": COL_KETAMINE}
DIMENSION_PALETTE = {"valence": COL_VALENCE, "arousal": COL_AROUSAL}
EMOTION_PALETTES = {"valence": {"negative": "#1B9E77", "neutral": "#9E9E9E", "positive": "#D95F02"},
                    "arousal": {"low": "#1B9E77", "medium": "#9E9E9E", "high": "#D95F02"}}
CP_DIRECTION_PALETTE = {"down": "#1B9E77", "up": "#D95F02"}
MAGNITUDE_PALETTE = {1: "#BC7A1C", 2: "#CCA978", 3: "#EDD4B1", 4: "#F5E9D7"}

# ------------------------------------------------------------------ sizes
TITLE_FS, LABEL_FS, TICK_FS = 15, 14, 12
BOX_WIDTH = 0.22
BOX_OUTER_SHIFT, DOT_INNER_SHIFT = 0.26, 0.10            # two levels on two ticks
GROUP_PAIR_BOX_SHIFT, GROUP_PAIR_DOT_SHIFT = 0.26, 0.16  # two levels on one tick
DOT_SIZE, DOT_ALPHA, DOT_EDGE_ALPHA, DOT_EDGE_LW, DOT_JITTER = 40, 0.65, 0.90, 0.65, 0.040
MEAN_SIZE, SEM_LW, SEM_CAP = 60, 2.2, 0.065             # mean marker, SEM line, cap half-width
DENSITY_WIDTH, DENSITY_ALPHA, DENSITY_N = 0.16, 0.40, 120
LINE_COLOR, LINE_ALPHA, LINE_LW = "0.45", 0.40, 1
DPI = 300


def set_style():
    plt.rcParams.update({"font.family": "Arial", "axes.linewidth": 1.0,
                         "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
                         "axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS})


def jitter(sub_ids, scale=DOT_JITTER):
    """Horizontal jitter that is the same for a participant in every figure."""
    return np.array([((int(v) * 9301 + 49297) % 233280 / 233280.0 - 0.5) for v in sub_ids]) * (scale * 2)


def _rgba(color, alpha):
    r, g, b, _ = mcolors.to_rgba(color)
    return r, g, b, alpha


def scatter_subjects(ax, xs, ys, color):
    r, g, b, _ = mcolors.to_rgba(color)
    ax.scatter(xs, ys, s=DOT_SIZE, facecolors=_rgba(color, DOT_ALPHA),
               edgecolors=(r * 0.45, g * 0.45, b * 0.45, DOT_EDGE_ALPHA), linewidths=DOT_EDGE_LW, zorder=6)


def box_at(ax, vals, x, color, width=BOX_WIDTH):
    """Box filled with the cell colour, black outline, no outlier markers."""
    vals = np.asarray(vals, float)
    bp = ax.boxplot(vals[np.isfinite(vals)], positions=[x], widths=width, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"color": "black", "linewidth": 1.0},
                    capprops={"color": "black", "linewidth": 1.0}, boxprops={"linewidth": 1.0}, zorder=3)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("black")


def half_density_at(ax, vals, x, color, side="left"):
    """Kernel density of the participant values, drawn as a half-violin next to the box."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3 or np.ptp(vals) == 0:
        return
    y = np.linspace(vals.min(), vals.max(), DENSITY_N)
    d = gaussian_kde(vals)(y)
    d = d / d.max() * DENSITY_WIDTH
    lo, hi = (x - d, x) if side == "left" else (x, x + d)
    ax.fill_betweenx(y, lo, hi, color=color, alpha=DENSITY_ALPHA, linewidth=0, zorder=2)


def draw_mean_sem(ax, x, mean, sem, size=MEAN_SIZE):
    """Group mean (black dot, area `size`) with its within-participant SEM."""
    if np.isfinite(sem) and sem > 0:
        ax.plot([x, x], [mean - sem, mean + sem], color="black", lw=SEM_LW, zorder=25)
        for y in (mean - sem, mean + sem):
            ax.plot([x - SEM_CAP, x + SEM_CAP], [y, y], color="black", lw=SEM_LW, zorder=25)
    ax.scatter([x], [mean], s=size, color="black", edgecolors="white", linewidths=1.0, zorder=26)


def cousineau_morey_sem(df, value, subject, cells):
    """Within-participant SEM per cell (Cousineau 2005, with the Morey 2008 correction).

    Each participant's values are centred on their own mean (plus the grand mean), the
    ordinary SEM is taken per cell and multiplied by sqrt(k / (k - 1)), k = number of cells.
    """
    x = df.copy()
    x["_norm"] = x[value] - x.groupby(subject)[value].transform("mean") + x[value].mean()
    k = x[cells].drop_duplicates().shape[0]
    sem = x.groupby(cells)["_norm"].agg(lambda v: v.std(ddof=1) / np.sqrt(v.count()))
    return sem * np.sqrt(k / (k - 1))


def style_axis(ax, title="", ylabel="", xlabel=""):
    ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path, formats=("png",)):
    """Save the figure once per format (e.g. png, svg), same name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
