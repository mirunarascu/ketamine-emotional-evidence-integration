"""Face-wise linear integration kernel (main kernel analysis; Fig 4, 5, S1).

    python analysis/kernels/01_facewise_linear.py --cohort in_person

For every rating change: how much did each of the 5 faces before it push the rating, in the
direction of that face's prediction error (PE = face value - current rating)?

Steps, per block (split in two at the mid-block confidence check):
  1  rating changes: rating steps less than merge_frames apart are one change; the outcome
     is the net number of steps, clipped to -2..+2 (changes whose steps cancel are dropped).
  2  each change belongs to the face on screen when it started; if two changes start on the
     same face only the first is kept.
  3  change rows: walk back from nondecision_frames before the change, face by face. The face
     containing that point is cut there; faces with fewer than min_face_frames usable frames
     are skipped. The mean PE of the first 5 faces that pass are the regressors; y = outcome.
  4  no-change rows: every face without a change and at least min_face_frames long, with the
     5 usable faces before its onset as regressors; y = 0. Faces between a change row's
     newest regressor and the face the change started on are left out, because their
     regressors would be the same 5 faces as the change row.
  5  weighted least squares of y on the 5 regressors (plus intercept), with change and
     no-change rows weighted to equal total weight.
Regressors are ordered oldest first: beta_r_1 = face -5, beta_r_5 = face -1.

Writes to output/tables/<cohort>/:
    facewise_blockbetas.csv   one row per block
    facewise_condition.csv    one row per subject x drug x dimension x volatility (IKS, BW-2)
    facewise_kernel_quality.csv  rows kept and dropped (and why), fit status of the blocks
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.kernels.kernel_functions import (BLOCK_KEYS, beta_columns, block_parts,  # noqa: E402
                                     class_weights, condition_table, face_onsets, quality_table)
from utils.config import COHORTS, load_config, out_dir                      # noqa: E402
from utils.data import load_frames, probe_table                             # noqa: E402

MAX_OUTCOME = 2


def find_rating_changes(rating, merge_frames, counts):
    """Outputs (start frame, outcome) of each rating change"""
    step = np.r_[0.0, np.diff(rating)]
    idx = np.flatnonzero(step != 0)
    if idx.size == 0:
        return []
    sign = np.sign(step[idx])
    # merge consecutive rating changes into one rating change while each comes less than merge_frames after the previous one
    groups = np.split(np.arange(idx.size), np.flatnonzero(np.diff(idx) >= merge_frames) + 1)
    out = []
    for g in groups:
        net_rating_change = int(sign[g].sum())
        counts["rating changes"] += 1
        if net_rating_change == 0:
            counts["rating changes dropped: steps cancel out"] += 1
            continue
        if abs(net_rating_change) > MAX_OUTCOME:
            counts["rating changes clipped to +-2"] += 1
        out.append((int(idx[g[0]]), float(np.clip(net_rating_change, -MAX_OUTCOME, MAX_OUTCOME))))
    return out


def usable_faces(onsets, offsets, pe, cutoff, n, min_face):
    """Mean PE of the n usable faces before cutoff (oldest first), or None if there are fewer.
    
    onsets, offsets: first frame of each face, and the frame just after its last frame
    pe: PE at each frame of the block part (trueVal - rating)
    cutoff: frames from here on are not used (rating change onset - nondecision frames, or a face onset)
    n: number of face regressors (kernels: n_regressors in the config, 5)
    min_face: faces with fewer usable frames before cutoff are skipped (min_face_frames, 13)

    """
    if cutoff <= 0:
        return None
    j = int(np.searchsorted(onsets, cutoff - 1, side="right") - 1)   # face holding the last frame before cutoff
    pes = []
    # walk back from the face holding cutoff-1, skipping faces with fewer than min_face usable frames, until n usable faces are found
    while j >= 0 and len(pes) < n:
        lo, hi = int(onsets[j]), int(min(offsets[j], cutoff))
        if hi - lo >= min_face:
            pes.append(pe[lo:hi].mean()) # mean PE over the face's usable frames
        j -= 1
    # reverse PE order to oldest first towards newest (face -5, -4, -3, -2, -1)
    return pes[::-1] if len(pes) == n else None


def newest_usable_face(onsets, offsets, cutoff, min_face):
    """Index of the most recent face before cutoff that has at least min_face frames before
    cutoff (-1 if none). This is the face -1 of a change row; the faces after it and before the
    face during which the rating change started are 'covered' and not used as no-change rows."""
    if cutoff <= 0:
        return -1
    j = int(np.searchsorted(onsets, cutoff - 1, side="right") - 1)
    while j >= 0 and int(min(offsets[j], cutoff)) - int(onsets[j]) < min_face:
        j -= 1
    return j


def build_regression_rows(part, k, n, counts):
    """Regression rows (y, 5 PEs) of one block part: change rows first, then no-change rows."""
    rating = part["rating"].to_numpy(float)
    pe = part["trueVal"].to_numpy(float) - rating # PE at each frame
    onsets, offsets = face_onsets(part["currentFace"].to_numpy())
    dur = offsets - onsets

    owner = {} # face that owns each rating change row: {face index: (rating change start frame, outcome)}
    for start, outcome in find_rating_changes(rating, k["merge_frames"], counts):
        # find the face that was on screen when the rating change started
        j = int(np.searchsorted(onsets, start, side="right") - 1)
        if j in owner:
            counts["rating changes dropped: second change on the same face"] += 1
            continue
        owner[j] = (start, outcome)

    rows = {"change": [], "still": []}

    def add_row(kind, y, cutoff):
        pes = usable_faces(onsets, offsets, pe, cutoff, n, k["min_face_frames"])
        if pes is None:
            counts[f"{kind} rows dropped: fewer than {n} usable faces"] += 1
        else:
            rows[kind].append([y, *pes])

    # find the face(s) between the last evidence face and the movement containing face, 
    # and mark them as covered so they are not used for no-change rows
    covered = set()
    for j, (start, outcome) in owner.items():
        # evidence from the 12 nondecision frames (200ms) before the rating change is not used
        cutoff = start - k["nondecision_frames"]
        # add rating change regression rows
        add_row("change", outcome, cutoff)
        covered.update(range(newest_usable_face(onsets, offsets, cutoff, k["min_face_frames"]) + 1, j))

    # no-change/still rows (y = 0): every face that is not a change row, is at least min_face_frames long, and
    # has at least n usable faces before it
    for j in range(len(onsets)):
        if j in owner:
            continue
        if dur[j] < k["min_face_frames"]:
            counts["still rows dropped: face too short"] += 1
        elif j in covered:
            counts["still rows dropped: same regressors as a change row"] += 1
        else:
            add_row("still", 0.0, int(onsets[j]))
    return rows


def weighted_ols(X, y, w):
    """Weighted least squares with an intercept. Ordinary least squares 
    after multiplying each row by the square root of its weight."""
    A = np.column_stack([np.ones(len(X)), X]) # add a column of ones for the intercept
    # check if we have more rows than coefficients
    if A.shape[0] <= A.shape[1]:
        return None
    # get the square root of the weights for movement and no-movement rows, to use in the least squares fit
    s = np.sqrt(w)
    # check for collinearity between regressors: if the smallest singular value is 0 (rank deficient)
    # or the ratio of largest to smallest is > 1e8 (nearly collinear)
    sv = np.linalg.svd(A * s[:, None], compute_uv=False)
    if sv[-1] <= 0 or sv[0] / sv[-1] > 1e8:
        return None
    # least squares fit on the weighted rows
    b = np.linalg.lstsq(A * s[:, None], y * s, rcond=None)[0]
    return b if np.all(np.isfinite(b)) else None


def main(cohort):
    cfg = load_config(cohort)
    k, n = cfg["kernels"]["facewise"], cfg["kernels"]["n_regressors"]
    frames = load_frames(cfg)
    counts, out = Counter(), []
    # task blocks are interrupted by a mid-block confidence check. We split each block in two at that check, 
    # build the regressors for each part, and pool both parts into one regression per block
    for meta, parts in block_parts(frames, probe_table(frames), cfg, "facewise"):
        rows = {"change": [], "still": []}
        for part in parts:
            for kind, r in build_regression_rows(part, k, n, counts).items():
                rows[kind] += r
        sizes = dict(n_rows=len(rows["change"]) + len(rows["still"]), n_move=len(rows["change"]),
                     n_still=len(rows["still"]))
        # blocks that cannot be fitted stay in the table with their fit_status and no betas
        if len(rows["change"]) < n + 2 or not rows["still"]:
            print(f"WARNING: block {meta['subID']}-{meta['visit']}-{meta['block']} skipped: too few rows")
            out.append(dict(**meta, **sizes, fit_status="skipped: too few rows"))
            continue
        R = np.array(rows["change"] + rows["still"], float)
        is_change = np.r_[np.ones(len(rows["change"]), bool), np.zeros(len(rows["still"]), bool)]
        beta = weighted_ols(R[:, 1:], R[:, 0], class_weights(is_change))
        if beta is None:
            print(f"WARNING: block {meta['subID']}-{meta['visit']}-{meta['block']}: fit failed")
            out.append(dict(**meta, **sizes, fit_status="skipped: fit failed"))
            continue
        out.append(dict(**meta, **sizes, fit_status="ok",
                        intercept=beta[0], **dict(zip(beta_columns(n), beta[1:]))))

    for what, c in sorted(counts.items()):
        print(f"{what}: {c:,}")
    bb = pd.DataFrame(out).reindex(columns=BLOCK_KEYS + ["n_rows", "n_move", "fit_status", "intercept"]
                                   + beta_columns(n) + ["n_still"])
    d = out_dir("tables", cohort)
    bb.to_csv(d / "facewise_blockbetas.csv", index=False)
    condition_table(bb).to_csv(d / "facewise_condition.csv", index=False)
    counts["change rows used"], counts["still rows used"] = int(bb.n_move.sum()), int(bb.n_still.sum())
    quality_table(counts, bb).to_csv(d / "facewise_kernel_quality.csv", index=False)
    print(f"{(bb.fit_status == 'ok').sum()} of {len(bb)} blocks fitted, {bb.n_move.sum():,} change rows, "
             f"{bb.n_still.sum():,} no-change rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    main(ap.parse_args().cohort)
