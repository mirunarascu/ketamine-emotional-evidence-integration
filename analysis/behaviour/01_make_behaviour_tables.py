"""Behavioural measure tables from the frame data.

    python analysis/behaviour/01_make_behaviour_tables.py --cohort in_person

Writes to output/tables/<cohort>/:
    cp_table.csv                        one row per change point (accuracy, RT)
    rating_changes_block.csv            one row per block
    occupancy_block.csv                 one row per block x rating level
    rating_error_block.csv              one row per block
    rating_error_block_emotion.csv      one row per block x emotion level
    rating_error_condition_emotion.csv  one row per subject x drug x dimension x volatility x emotion level
    diff_re_condition.csv               one row per subject x drug x dimension x volatility
    detection_curve_by_dimension.csv    detection curve per dimension (Fig 2A, S1A)
    detection_curve_by_volatility.csv   detection curve per dimension x volatility (Fig S3E)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.behaviour import behaviour_measures      # noqa: E402
from utils.config import COHORTS, load_config, out_dir  # noqa: E402
from utils.data import load_frames, probe_table        # noqa: E402


def check_tables(frames, cp, rc, re_be):
    """Basic checks on the finished tables; any failure stops the script."""
    n_blocks = frames.groupby(["subID", "visit", "block"]).ngroups
    problems = []
    if cp.duplicated(["subID", "visit", "block", "cp_frame"]).any():
        problems.append("duplicated change points")
    if (cp.loc[cp["admissible"], "window_duration_s"] <= 0).any():
        problems.append("an analysed change point has an empty response window")
    if len(rc) != n_blocks:
        problems.append(f"rating_changes_block has {len(rc)} rows for {n_blocks} blocks")
    if len(re_be) != 3 * n_blocks:
        problems.append(f"rating_error_block_emotion has {len(re_be)} rows, expected {3 * n_blocks}")
    if problems:
        raise ValueError("checks failed: " + "; ".join(problems))
    print(f"checks passed: {n_blocks} blocks")


def main(cohort):
    cfg = load_config(cohort)
    frames = load_frames(cfg)
    probes = probe_table(frames)

    cp = behaviour_measures.change_point_table(frames, probes, cfg)
    n_excl = cp.loc[~cp["admissible"], "exclusion_reason"].value_counts().to_dict()
    print(f"{len(cp):,} change points, {cp['admissible'].sum():,} analysed, excluded: {n_excl}")

    rc = behaviour_measures.rating_change_counts(frames, cfg)
    occ = behaviour_measures.occupancy_table(frames, cfg)
    re_be = behaviour_measures.rating_error_block_emotion(frames, cfg)
    re_b = behaviour_measures.rating_error_block(frames, cfg)
    re_ce = behaviour_measures.rating_error_condition_emotion(re_be)
    dre = behaviour_measures.diff_re_condition(re_ce)
    dc_dim, dc_vol = behaviour_measures.detection_curves(frames, cp, cfg)
    check_tables(frames, cp, rc, re_be)

    out = out_dir("tables", cohort)
    tables = {"cp_table": cp, "rating_changes_block": rc, "occupancy_block": occ,
              "rating_error_block": re_b, "rating_error_block_emotion": re_be,
              "rating_error_condition_emotion": re_ce, "diff_re_condition": dre,
              "detection_curve_by_dimension": dc_dim, "detection_curve_by_volatility": dc_vol}
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
    print(f"wrote {len(tables)} tables to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    main(ap.parse_args().cohort)
