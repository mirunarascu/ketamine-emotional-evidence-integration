"""Shuffled-ratings null for Fig 2B-C: shuffled rating traces per subject x block, scored like the real data.

    python analysis/behaviour/03_null_shuffles.py --cohort in_person          (~25 min on 8 cores; online ~40 min)
    python analysis/behaviour/03_null_shuffles.py --cohort in_person --n 3    (quick test)
    python analysis/behaviour/03_null_shuffles.py --cohort online             (not reported in the paper)

One shuffle, per block:
  1. split the rating trace into runs of the same rating;
  2. put the runs in a random order, keeping each run's value and length;
  3. rebuild the trace. True means, frames and the confidence question are unchanged.
The code redraws the order until no two neighbouring runs have the same value (at most
2000 tries), but in almost every block no such order is found, and one more random order is
then used without this check. Neighbouring runs with the same value then join into one longer run, so the shuffled
trace keeps the time spent at each rating but has fewer rating changes than the real one.
Each shuffled data set is scored with the same code as the real data (change_point_table),
and averaged in the same way: per subject x drug, then over drug.

Writes output/tables/<cohort>/null_shuffles.csv (one row per shuffle x subject x dimension).
Seeds: numpy default_rng([seed, shuffle number]), seed from config/analysis_config.yaml.
"""
import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.behaviour.behaviour_measures import BLOCK_KEYS, change_point_table, subject_means  # noqa: E402
from utils.config import COHORTS, load_config, out_dir                # noqa: E402
from utils.data import load_frames, probe_table                       # noqa: E402

MAX_TRIES = 2000
W = {}        # data held by each worker process


def shuffle_runs(values, rng):
    """Rating trace of one block with its runs in a new random order.

    Example: a block rated 4 for 300 frames, 3 for 60, 2 for 500, 3 for 100, 4 for 20 and
    5 for 600 might become 4 for 20, 3 for 60, 5 for 600, 4 for 300, 2 for 500 and 3 for 100.
    Each run keeps its value and length; only the order changes. If two runs with the same
    value end up next to each other (e.g. 4 for 300, then 4 for 20), they become one run
    (4 for 320 frames).
    """
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    durs = np.diff(np.r_[starts, len(values)])
    vals = values[starts]
    for _ in range(MAX_TRIES):
        order = rng.permutation(len(vals))
        v = vals[order]
        if np.all(v[1:] != v[:-1]) and not np.array_equal(v, vals):
            break
    else:
        order = rng.permutation(len(vals))
    return np.repeat(vals[order], durs[order])


def init_worker(cohort):
    cfg = load_config(cohort)
    fr = load_frames(cfg).sort_values(BLOCK_KEYS + ["currentFrame"]).reset_index(drop=True)
    W.update(cfg=cfg, frames=fr, probes=probe_table(fr),
             blocks=list(fr.groupby(BLOCK_KEYS, sort=False).indices.values()),
             rating=fr["rating"].to_numpy(float), true=fr["trueMeanVal"].to_numpy(float))


def one_shuffle(i):
    """Shuffle number i: shuffle every block, then score it like the real data.

    The random generator is seeded with [seed, i], so each shuffle can be repeated on its own.
    Returns one row per subject x dimension: change-point accuracy (analysed change points
    only) and absolute rating error, each averaged per subject x drug, then over drug.
    """
    rng = np.random.default_rng([W["cfg"]["null_model"]["seed"], i])
    r = W["rating"].copy()
    for ix in W["blocks"]:
        r[ix] = shuffle_runs(W["rating"][ix], rng)
    fr = W["frames"].assign(rating=r)
    cp = change_point_table(fr, W["probes"], W["cfg"])
    acc = subject_means(cp[cp["admissible"]], "detected_correctly")
    fr["re_abs_null"] = np.abs(r - W["true"])
    blk = fr.groupby(BLOCK_KEYS + ["dimension"], as_index=False)["re_abs_null"].mean()
    out = pd.concat([acc.rename("null_cp_accuracy"),
                     subject_means(blk, "re_abs_null").rename("null_RE_abs")], axis=1).reset_index()
    out.insert(1, "shuffle", i)
    return out


def main(cohort, n, n_jobs):
    cfg = load_config(cohort)
    n = n or cfg["null_model"]["n_shuffles"]
    print(f"{cohort}: {n} shuffles on {n_jobs} cores, seed {cfg['null_model']['seed']}")
    t0, parts = time.time(), []
    # Run the shuffles in parallel on n_jobs processes. Each process loads the data once
    # (init_worker), then gets shuffle numbers 1..n one at a time and returns one table per
    # shuffle. Shuffles come back in whatever order they finish; the seed depends only on the
    # shuffle number, so the results are the same whatever the order or number of processes.
    with get_context("spawn").Pool(n_jobs, initializer=init_worker, initargs=(cohort,)) as pool:
        for k, df in enumerate(pool.imap_unordered(one_shuffle, range(1, n + 1)), 1):
            parts.append(df)
            if k % 50 == 0 or k == n:
                print(f"{k}/{n} shuffles ({time.time() - t0:.0f} s)")
    # One row per shuffle x subject x dimension. null_cp_accuracy: mean over the analysed change
    # points of that dimension, per subject x drug, then over drug. null_RE_abs: mean absolute
    # rating error per block, averaged over that dimension's blocks per subject x drug, then
    # over drug. Same averaging as for the real data.
    res = pd.concat(parts).sort_values(["shuffle", "subID", "dimension"])
    path = out_dir("tables", cohort) / "null_shuffles.csv"
    res.to_csv(path, index=False)
    print(f"wrote {path} ({len(res):,} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    ap.add_argument("--n", type=int, default=None, help="number of shuffles (default: config)")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    main(a.cohort, a.n, a.n_jobs)
