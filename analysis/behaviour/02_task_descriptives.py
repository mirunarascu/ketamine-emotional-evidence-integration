"""Task numbers reported in the Methods: change points per block, intervals between change
points, and face durations.

    python analysis/behaviour/02_task_descriptives.py --cohort in_person

Needs output/tables/<cohort>/cp_table.csv (01_make_behaviour_tables.py).
Writes output/tables/<cohort>/task_descriptives.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config import COHORTS, load_config, out_dir  # noqa: E402
from utils.data import load_frames                     # noqa: E402

BLOCK = ["subID", "visit", "block"]


def main(cohort):
    cfg = load_config(cohort)
    fps = cfg["frame_rate_hz"]
    frames = load_frames(cfg)
    cp = pd.read_csv(out_dir("tables", cohort) / "cp_table.csv")
    rows = []

    # change points per block (all change points)
    per_block = cp.groupby(BLOCK + ["volatility"]).size().rename("n").reset_index()
    for vol, s in per_block.groupby("volatility"):
        rows.append(("change points per block", vol, s["n"].mean(), s["n"].min(), s["n"].median(), s["n"].max()))

    # time from one change point to the next, within a block
    cp = cp.sort_values(BLOCK + ["cp_frame"])
    cp["interval_s"] = cp.groupby(BLOCK)["cp_frame"].diff() / fps
    for vol, s in cp.dropna(subset=["interval_s"]).groupby("volatility"):
        x = s["interval_s"]
        rows.append(("change-point interval (s)", vol, x.mean(), x.min(), x.median(), x.max()))

    # face durations as presented (frames per face). The face on screen when the stream paused
    # for the mid-block confidence check continues after the pause, so it is left out here.
    faces = frames.groupby(BLOCK + ["currentFace"]).agg(n_frames=("currentFrame", "size"),
                                                         probe=("probe", "max")).reset_index()
    print(f"{int(faces['probe'].sum())} faces spanning the mid-block confidence check left out of the durations")
    faces = faces[faces["probe"] == 0]
    d = faces["n_frames"] / fps
    rows.append(("face duration (s)", "all", d.mean(), d.min(), d.median(), d.max()))
    short = (faces["n_frames"] <= 12).mean() * 100
    rows.append(("faces shown for 12 frames (200 ms) or less (%)", "all", short, None, None, None))
    print(f"{len(faces):,} faces; {short:.2f}% shown for 12 frames or less")

    out = pd.DataFrame(rows, columns=["measure", "condition", "mean", "min", "median", "max"])
    out.to_csv(out_dir("tables", cohort) / "task_descriptives.csv", index=False)
    print("\n" + out.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    main(ap.parse_args().cohort)
