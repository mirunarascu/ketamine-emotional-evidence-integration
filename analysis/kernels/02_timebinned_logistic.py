"""Time-binned logistic integration kernel (preregistered kernel; Fig S4, Table S3b).

    python analysis/kernels/02_timebinned_logistic.py --cohort in_person

After each face onset: does a rating change start soon after it, given the mean prediction
error (PE = face value - current rating) in 5 time bins before the face onset?

Steps, per block (split in two at the mid-block confidence check):
  1  rating changes: a step less than merge_frames after the previous one is merged into it,
     and the first step of the run gets the net direction of the whole run.
  2  rows: every face onset with a full look-back (5 bins x bin_width frames) inside the
     part, except the last face onset of the part. Bin 1 is the oldest; the newest bin ends
     on the frame before the face onset.
  3  outcome (+1 up, -1 down, 0 none): a rating change starting within movement_window_frames
     after the face onset. Each rating change is credited to the latest face onset at or
     before it (face onsets are visited from last to first and each takes the first rating
     change not yet credited; upward changes are checked before downward ones).
  4  for downward changes the PE signs are flipped, so positive PE always means evidence
     in the direction of the change; rows without a change keep the raw PE.
  5  logistic regression of change (0/1) on the 5 bins (plus intercept), with change and
     no-change rows weighted to equal total weight.

Writes to output/tables/<cohort>/:
    timebinned_blockbetas.csv   one row per block
    timebinned_condition.csv    one row per subject x drug x dimension x volatility (IKS, BW-2)
    timebinned_kernel_quality.csv  rows kept and dropped (and why), fit status of the blocks
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.kernels.kernel_functions import (BLOCK_KEYS, beta_columns, block_parts,  # noqa: E402
                                     class_weights, condition_table, face_onsets, quality_table)
from utils.config import COHORTS, load_config, out_dir                      # noqa: E402
from utils.data import load_frames, probe_table                             # noqa: E402


def find_rating_changes(rating, merge_frames):
    """Start positions of upward and downward rating changes.

    A step less than merge_frames after the previous step is merged into a run; the run is
    kept at its first step, with the direction of the net change of the whole run.
    """
    change = np.r_[0.0, np.diff(rating)]
    idx = np.flatnonzero(change != 0)
    if idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    step = change[idx].copy()
    discard = [] # steps merged into an earlier step
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] < merge_frames:
            discard.append(i)
            # keep only the first step of the run
            keeper = i - 1
            while keeper in discard and keeper > 0:
                keeper -= 1
            # net change of the run so far: rating after this step - rating before the first step
            step[keeper] = rating[idx[i]] - rating[max(idx[keeper] - 1, 0)]
    keep = np.ones(len(idx), dtype=bool)
    keep[discard] = False
    starts, direction = idx[keep], np.sign(step[keep])
    return starts[direction > 0], starts[direction < 0]


def build_regression_rows(part, k, n, counts):
    """Outputs one row per face onset of one block part: outcome (+1 up, -1 down, 0 no rating
    change) and the mean PE in n time bins before the face onset (pe_1 oldest ... pe_n newest)."""
    rating = part["rating"].to_numpy(float)
    pe = part["trueVal"].to_numpy(float) - rating          # PE at each frame
    onsets = face_onsets(part["currentFace"].to_numpy())[0][1:]   # face onsets; excluding the first face of the part
    up_starts, down_starts = find_rating_changes(rating, k["merge_frames"])   # rating change onsets
    w, lookback = k["bin_width_frames"], k["bin_width_frames"] * n   # bin width, total look-back

    # usable face onsets: faces with a full look-back period, and not the last face onset of the part
    # (its response window of movement_window_frames always runs past the end of the part)
    usable = []
    for i, onset in enumerate(onsets):
        if onset < lookback:
            counts["face onsets dropped: incomplete look-back"] += 1
        elif i == len(onsets) - 1:
            counts["face onsets dropped: last face onset of the part"] += 1
        else:
            s = onset - lookback # first frame of the oldest bin
            usable.append((onset, [pe[s + b * w: s + (b + 1) * w].mean() for b in range(n)]))

    # outcome of each face onset: a rating change starting within movement_window_frames after it.
    # We iterate through face onsets from last (end of part/block) to first and each takes the first rating change not yet
    # credited, so that a rating change is credited to the latest face onset at or before it.
    up_credited = np.zeros(up_starts.size, bool) # True once that rating change has a face onset
    down_credited = np.zeros(down_starts.size, bool)
    outcome = np.zeros(len(usable), int) # up = +1, down = -1, no change = 0
    for j in range(len(usable) - 1, -1, -1):
        onset = usable[j][0]
        limit = onset + k["movement_window_frames"] # movement check window limit
        up_in_window = np.flatnonzero((up_starts >= onset) & (up_starts <= limit) & ~up_credited)
        down_in_window = np.flatnonzero((down_starts >= onset) & (down_starts <= limit) & ~down_credited)
        if up_in_window.size:
            outcome[j], up_credited[up_in_window[0]] = 1, True
        elif down_in_window.size:
            outcome[j], down_credited[down_in_window[0]] = -1, True
    counts["rating changes not credited to any face onset"] += int((~up_credited).sum() + (~down_credited).sum())
    counts["face onsets used"] += len(usable)
    return pd.DataFrame([[o, *bins] for (_, bins), o in zip(usable, outcome)],
                        columns=["outcome"] + [f"pe_{b}" for b in range(1, n + 1)])


def fit_block(D, n):
    """Class-weighted logistic regression of change on the direction-aligned PE bins."""
    X = D[[f"pe_{b}" for b in range(1, n + 1)]].to_numpy(float).copy()
    X[D["outcome"].to_numpy() == -1] *= -1        # downward changes: flip PE so + = evidence for the change
    move = (D["outcome"] != 0).to_numpy(int)      # 1 = a rating change followed the face onset
    res = dict(n_rows=len(D), n_move=int(move.sum()))
    if move.min() == move.max():                   # all 0 or all 1: logistic regression not possible
        return {**res, "fit_status": "skipped: only one outcome"}
    # logistic regression; change and no-change rows weighted to equal total weight
    fit = sm.GLM(move, sm.add_constant(X, has_constant="add"), family=sm.families.Binomial(),
                 freq_weights=class_weights(move == 1)).fit(maxiter=200)
    b = np.asarray(fit.params, float)
    return {**res, "fit_status": "ok", "intercept": b[0], **dict(zip(beta_columns(n), b[1:])),
            "converged": bool(fit.converged)}


def main(cohort):
    cfg = load_config(cohort)
    k, n = cfg["kernels"]["timebinned"], cfg["kernels"]["n_regressors"]
    frames = load_frames(cfg)
    counts, out = Counter(), []
    # split each block in two at the mid-block confidence check, build the rows for each part,
    # and pool both parts into one regression per block
    for meta, parts in block_parts(frames, probe_table(frames), cfg, "timebinned"):
        D = pd.concat([build_regression_rows(p, k, n, counts) for p in parts], ignore_index=True)
        res = fit_block(D, n)
        if res["fit_status"] != "ok":
            print(f"WARNING: block {meta['subID']}-{meta['visit']}-{meta['block']}: {res['fit_status']}")
        elif not res["converged"]:
            print(f"WARNING: block {meta['subID']}-{meta['visit']}-{meta['block']}: fit did not converge")
        out.append({**meta, **res})

    for what, c in sorted(counts.items()):
        print(f"{what}: {c:,}")
    bb = pd.DataFrame(out).reindex(columns=BLOCK_KEYS + ["n_rows", "n_move", "fit_status", "intercept"]
                                   + beta_columns(n) + ["converged"])
    d = out_dir("tables", cohort)
    bb.to_csv(d / "timebinned_blockbetas.csv", index=False)
    condition_table(bb).to_csv(d / "timebinned_condition.csv", index=False)
    quality_table(counts, bb).to_csv(d / "timebinned_kernel_quality.csv", index=False)
    print(f"{(bb.fit_status == 'ok').sum()} of {len(bb)} blocks fitted, {bb.n_rows.sum():,} face onsets, "
             f"{bb.n_move.sum():,} with a rating change")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    main(ap.parse_args().cohort)
