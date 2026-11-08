"""Example blocks for Fig 1: participant rating, true mean and face values over time.

    python analysis/figures/05_example_blocks.py

Draws the first SECONDS of each block listed in EXAMPLES (in-person data):
    thin coloured line   trueVal, the emotion value of each face shown
    thick coloured line  trueMeanVal, the true mean of the generative distribution
    thick black line     the participant's rating
Colours: stable / volatile as in all other figures. Values are on the stored 2-6 scale.
Writes output/figures/Fig1/<name>.png, with the plotted values in source_data/<name>.csv.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils import plotting as pl                                 # noqa: E402
from utils.config import data_dir, load_config, out_dir          # noqa: E402

# ------------------------------------------------------------------ choose the blocks
SUBJECT, VISIT = 10, 1
EXAMPLES = {"example_stableValence": 3,       # file name: block number
            "example_volatileArousal": 2}
SECONDS = 60


def main():
    cfg = load_config("in_person")
    fps = cfg["frame_rate_hz"]
    frames = pd.read_csv(data_dir() / "frames_in_person.csv.gz")
    folder = out_dir("figures", "Fig1")
    (folder / "source_data").mkdir(parents=True, exist_ok=True)
    for name, block in EXAMPLES.items():
        b = frames[(frames.subID == SUBJECT) & (frames.visit == VISIT) & (frames.block == block)]
        assert len(b), f"no data for subject {SUBJECT}, visit {VISIT}, block {block}"
        b = b.sort_values("currentFrame").assign(time_s=lambda x: (x.currentFrame - 1) / fps)
        b = b[b.time_s <= SECONDS]
        dim, vol = b.dimension.iloc[0], b.volatility.iloc[0]
        col = pl.VOLATILITY_PALETTE[vol]

        pl.set_style()
        fig, ax = plt.subplots(figsize=(7.0, 2.8))
        ax.plot(b.time_s, b.trueVal, color=col, lw=0.8, alpha=0.6, label="Face value")
        ax.plot(b.time_s, b.trueMeanVal, color=col, lw=3.0, label="True mean")
        ax.plot(b.time_s, b.rating, color="black", lw=2.2, label="Rating")
        ax.set_xlim(0, SECONDS)
        ax.set_ylim(0.75, 8.0)               # face values span 1.0-7.8 in the data; ratings 2-6
        ax.set_yticks(range(1, 8))
        pl.style_axis(ax, title=f"{dim.capitalize()}, {vol}", xlabel="Time (s)", ylabel=dim.capitalize())
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=pl.TICK_FS - 1)
        fig.tight_layout()

        b[["subID", "visit", "block", "dimension", "volatility", "currentFrame", "time_s",
           "trueVal", "trueMeanVal", "rating"]].to_csv(folder / "source_data" / f"{name}.csv", index=False)
        pl.save_figure(fig, folder / name, cfg["figures"]["formats"])
        print(name, f"(block {block}: {dim}, {vol})")


if __name__ == "__main__":
    main()
