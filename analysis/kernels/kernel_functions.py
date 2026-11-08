"""Shared by the two kernels: splitting blocks at the mid-block question, face onsets,
class weights, and the kernel summary measures (IKS, BW-2) per condition cell."""
import numpy as np
import pandas as pd

BLOCK_KEYS = ["kernel", "cohort", "subID", "visit", "block", "drug", "drug_label",
              "dimension", "volatility"]
CELL = ["subID", "drug_label", "dimension", "volatility"]


def beta_columns(n):
    """beta_r_1 is the oldest regressor (-n), beta_r_n the newest (-1)."""
    return [f"beta_r_{k}" for k in range(1, n + 1)]


def block_parts(frames, probes, cfg, kernel):
    """Yield (block info, [part, ...]) per block, split at the mid-block question.

    The question frame itself is dropped, and every regressor is built within one part,
    so nothing reaches across the pause. The parts of a block are then fitted together.
    """
    probe = probes.set_index(["subID", "visit", "block"])["conf_frame"]
    drug_names = {**cfg["coding"]["drug"], -1: "none"}
    for (sub, visit, blk), block in frames.groupby(["subID", "visit", "block"], sort=False):
        block = block.sort_values("currentFrame")
        drug = int(block["drug"].iat[0])
        meta = dict(kernel=kernel, cohort=cfg["cohort"]["name"], subID=int(sub), visit=int(visit),
                    block=int(blk), drug=drug, drug_label=drug_names[drug],
                    dimension=block["dimension"].iat[0], volatility=block["volatility"].iat[0])
        conf = probe.get((sub, visit, blk))
        parts = [block] if conf is None else [block[block.currentFrame < conf],
                                              block[block.currentFrame > conf]]
        yield meta, [p for p in parts if len(p) >= 2]


def face_onsets(face_id):
    """Onset and (exclusive) offset of every face in a part, as 0-based frame positions."""
    onsets = np.flatnonzero(np.r_[True, np.diff(face_id) != 0])
    return onsets, np.r_[onsets[1:], len(face_id)]


def class_weights(is_move):
    """Movement and no-movement rows get equal total weight: N/(2 n_move) and N/(2 n_still)."""
    n, n_m = len(is_move), int(is_move.sum())
    return np.where(is_move, n / (2.0 * n_m), n / (2.0 * (n - n_m)))


def quality_table(counts, bb):
    """Counts of rows kept and dropped (and why), and the fit status of every block.
    Saved as <kernel>_kernel_quality.csv."""
    rows = sorted(counts.items())
    rows.append(("blocks", len(bb)))
    rows += [(f"blocks fit_status: {s}", c) for s, c in bb["fit_status"].value_counts().sort_index().items()]
    if "converged" in bb:
        rows.append(("blocks fitted but not converged", int(((bb["fit_status"] == "ok") & (bb["converged"] == False)).sum())))  # noqa: E712
    return pd.DataFrame(rows, columns=["what", "n"])


def condition_table(bb):
    """One row per subject x drug x dimension x volatility: mean betas over the cell's blocks,
    IKS = beta(-1) - beta(-2), and BW-2 = beta(-2). This is the input to the statistics."""
    bb = bb[bb["fit_status"] == "ok"]
    cols = sorted([c for c in bb.columns if c.startswith("beta_r_")], key=lambda c: int(c.split("_")[-1]))
    bb = bb.assign(iks=bb[cols[-1]] - bb[cols[-2]], bw2=bb[cols[-2]])
    out = bb.groupby(["kernel"] + CELL, as_index=False).agg(
        **{f"b{k}": (c, "mean") for k, c in enumerate(cols, start=1)},
        iks=("iks", "mean"), bw2=("bw2", "mean"), n_blocks=("block", "count"))
    return out
