"""Behavioural measures computed from the frame table.

change_point_table      one row per change point: detected or not, RT
rating_change_counts    one row per block: number of rating changes
occupancy_table         one row per block x rating level: time spent at each level (Fig S2D)
detection_curves        proportion of change points detected by each time after the change point
                        (Kaplan-Meier; Fig 2A, S1A, S3E)
rating_error_*          rating error per block, per block x emotion level and per condition cell
"""
import numpy as np
import pandas as pd

BLOCK_KEYS = ["subID", "visit", "drug", "condID", "stabID", "block"]


def add_labels(table, cfg):
    """dimension / volatility / drug_label columns from the codes."""
    table["dimension"] = table["condID"].map(cfg["coding"]["condID"])
    table["volatility"] = table["stabID"].map(cfg["coding"]["stabID"])
    table["drug_label"] = table["drug"].map({**cfg["coding"]["drug"], -1: "none"})
    return table


def emotion_lookup(cfg):
    """{dimension: {true-mean value: emotion level}}"""
    return {dim: {float(v): label for label, vals in levels.items() for v in vals}
            for dim, levels in cfg["emotion_levels"].items()}


def change_point_table(frames, probes, cfg):
    """Builds the change-point table (cp_table.csv) used for all change-point accuracy and RT
    analyses: one row per change point (a change in the true mean) in each block.

    Response window: from cp + buffer to the earliest of next cp + buffer, cp + 7 s and
    the end of the block. A change point counts as detected if any rating change in the
    window goes in the direction of the change (earlier wrong-direction changes are
    skipped); RT is the time from the change point to that rating change.

    Change points are not analysed (admissible = False) if
      - their window is shorter than min_window_s ('window_lt_min'), which might happen at the end of the block, or
      - their window is interrupted by the mid-block confidence question
        ('probe_interrupted', see mark_probe_interrupted).
    They stay in the table so the exclusions can be checked.
    """
    cp_cfg = cfg["change_points"]
    fps = cfg["frame_rate_hz"]
    buffer_f = cfg["cohort"]["buffer_frames"]
    win = cp_cfg["max_window_frames"]
    min_win = int(round(cp_cfg["min_window_s"] * fps))
    emotion = emotion_lookup(cfg)

    rows = []
    for keys, block in frames.groupby(BLOCK_KEYS, sort=False):
        block = block.sort_values("currentFrame")
        frame_no = block["currentFrame"].to_numpy()
        rating = block["rating"].to_numpy(float)
        true_mean = block["trueMeanVal"].to_numpy(float)
        dim = block["dimension"].iat[0]
        last_frame = int(frame_no[-1])

        cp_idx = np.flatnonzero(np.diff(true_mean) != 0) + 1

        # onset of each participant rating change
        delta = np.r_[0.0, np.diff(rating)]
        change_pos = np.flatnonzero(delta != 0)
        run_start = change_pos[np.r_[True, np.diff(change_pos) > 1]]
        onset_frame = frame_no[run_start]
        onset_sign = np.sign(delta[run_start])

        for i, idx in enumerate(cp_idx):
            cp_frame = int(frame_no[idx])
            prev_mean, curr_mean = float(true_mean[idx - 1]), float(true_mean[idx])
            jump = curr_mean - prev_mean
            # define valid window for searching for a correct (same direction as true mean change) participant rating change
            w_start = cp_frame + buffer_f
            next_cp = int(frame_no[cp_idx[i + 1]]) + buffer_f if i + 1 < len(cp_idx) else last_frame
            cap = cp_frame + win
            # the valid window ends at the earliest of next CP + buffer, maximum CP cap (7 seconds) or last frame of the block
            w_end = min(next_cp, cap, last_frame)
            if w_end == last_frame:
                reason = "block_end"
            elif cap <= next_cp:
                reason = "fixed_cap"
            else:
                reason = "next_cp"

            in_window = (onset_frame >= w_start) & (onset_frame <= w_end)
            signs = onset_sign[in_window]
            matching = np.flatnonzero(signs == np.sign(jump))
            if matching.size:
                detected, n_wrong = 1, int(matching[0])
                rt_frames = int(onset_frame[in_window][matching[0]]) - cp_frame
            else:
                detected, n_wrong, rt_frames = 0, int(signs.size), np.nan

            # CP admissible only if the window is at least min_window_s (1 s) long; short windows
            # only happen for change points just before the end of the block
            ok = (w_end - w_start) >= min_win
            rows.append((*keys, dim, i + 1, cp_frame, prev_mean, curr_mean, jump, abs(jump),
                         "up" if jump > 0 else "down",
                         emotion[dim][prev_mean], emotion[dim][curr_mean],
                         w_start, w_end, reason, (w_end - w_start) / fps, buffer_f,
                         ok, "none" if ok else "window_lt_min",
                         detected, rt_frames, rt_frames / fps, int(signs.size), n_wrong))

    cols = BLOCK_KEYS + ["dimension", "cp_index", "cp_frame", "prev_mean", "curr_mean",
                         "jump_size", "cp_magnitude", "cp_direction", "prev_emotion_level",
                         "emotion_level", "w_start", "w_end", "w_end_reason", "window_duration_s",
                         "buffer_frames", "admissible", "exclusion_reason", "detected_correctly",
                         "rt_first_correct_frames", "rt_first_correct_s", "n_onsets_in_window",
                         "n_wrong_direction_before_first_correct"]
    table = add_labels(pd.DataFrame(rows, columns=cols), cfg)
    if cp_cfg["exclude_probe_interrupted"]:
        table = mark_probe_interrupted(table, probes)
    return table


def mark_probe_interrupted(table, probes):
    """Exclude the change point whose response window is interrupted by the mid-block question.

    This is the last change point before the question, unless its window had already closed
    before the question, or a change point falls exactly on the question frame. So at most
    one change point per block is excluded.
    Example: question at 4500, change points at 4210 and 4460 -> 4460 is excluded.
    """
    key = ["subID", "visit", "block"]
    t = table.merge(probes[key + ["conf_frame"]].rename(columns={"conf_frame": "probe_frame"}),
                    on=key, how="left", validate="many_to_one")
    if t["probe_frame"].isna().any():
        raise ValueError("a block with change points has no mid-block question")

    last_before = (t[t["cp_frame"] < t["probe_frame"]].groupby(key)["cp_frame"].max()
                   .rename("last_cp_before"))
    t = t.join(last_before, on=key)
    t["probe_interrupted"] = (t["cp_frame"] == t["last_cp_before"]) & (t["probe_frame"] <= t["w_end"])
    cp_on_probe = t.loc[t["cp_frame"] == t["probe_frame"], key].drop_duplicates()
    if len(cp_on_probe):
        on = t.set_index(key).index.isin(cp_on_probe.set_index(key).index)
        t.loc[on, "probe_interrupted"] = False
    t = t.drop(columns="last_cp_before")

    drop = t["probe_interrupted"] & t["admissible"]
    t.loc[drop, "admissible"] = False
    t.loc[drop, "exclusion_reason"] = "probe_interrupted"
    return t


def detection_times(frames, cp, fps):
    """One row per analysable change point: time_s of the first correct rating change
    (detected = 1), or the time the change point ended without one (detected = 0).

    Unlike cp_table, each change point is followed from its own frame (no buffer, no 7-s cap)
    until the next change point, the mid-block question (for change points before it) or
    the end of the block. Responses are found in the same way as in change_point_table.
    """
    BLOCK = ["subID", "visit", "block"]
    frames = frames.sort_values(BLOCK + ["currentFrame"])
    cp = cp.sort_values(BLOCK + ["cp_index"])
    cp["next_cp_frame"] = cp.groupby(BLOCK)["cp_frame"].shift(-1)      # from ALL change points
    cp_by_block = dict(tuple(cp[cp["admissible"]].groupby(BLOCK)))
    rows = []
    for keys, blk in frames.groupby(BLOCK, sort=False):
        if keys not in cp_by_block:
            continue
        frame_no = blk["currentFrame"].to_numpy()
        rating = blk["rating"].to_numpy(float)
        # rating-change onsets
        delta = np.r_[0.0, np.diff(rating)]
        changing = np.flatnonzero(delta != 0)
        start = changing[np.r_[True, np.diff(changing) > 1]] 
        onset, sign = frame_no[start], np.sign(delta[start])
        for r in cp_by_block[keys].itertuples():
            # Censoring: this change point is followed until the next change point, the
            # mid-block question (only for change points before it) or the block end,
            # whichever comes first.
            end = frame_no[-1] + 1                                      # block end
            if not np.isnan(r.next_cp_frame):
                end = min(end, int(r.next_cp_frame))                    # next change point
            if r.cp_frame < r.probe_frame:
                end = min(end, int(r.probe_frame))                      # mid-block question
            ok = (onset >= r.cp_frame) & (onset < end) & (sign == np.sign(r.jump_size))
            detected = ok.any()
            # detected: time to the first correct rating change onset; not detected
            # (censored): time to the last frame followed (end - 1)
            frames_to = (onset[ok][0] if detected else end - 1) - r.cp_frame
            rows.append(dict(subID=keys[0], drug_label=r.drug_label, dimension=r.dimension,
                             volatility=r.volatility, time_s=frames_to / fps, detected=int(detected)))
    return pd.DataFrame(rows)


def km_detected(time, detected, grid):
    """1 - Kaplan-Meier survival at each time in grid (NaN after the last observed time).

    time: per change point, the time of its first correct response (detected = 1) or the
          time it ended without one (detected = 0, censored), from the change point's onset.
    grid: the times at which the curve is read off (0 to 7 s, one per frame).
    """
    event_times = np.unique(time[detected == 1])          # detection times, smallest first

    # s = proportion of change points not yet detected ("survival"); 1 at time 0.
    # At each detection time u: (detected at u) / (still followed and not yet detected at u),
    # s is multiplied by 1 minus that share. A censored change point counts in the
    # denominator until it ends, then drops out (it counts as neither detected nor missed).
    surv, s = [], 1.0
    for u in event_times:
        s *= 1.0 - np.sum((time == u) & (detected == 1)) / np.sum(time >= u)
        surv.append(s)

    # For each grid time, idx is the position of the last detection time at or before it
    # (-1 if no change point has been detected yet). s at that grid time is the s after
    # that detection, or 1 (nothing detected yet) before the first detection.
    idx = np.searchsorted(event_times, grid, side="right") - 1
    s_at_grid = np.ones(len(grid))
    seen = idx >= 0                         # grid times at or after the first detection
    s_at_grid[seen] = np.array(surv)[idx[seen]]
    out = 1.0 - s_at_grid                   # proportion detected
    out[grid > time.max()] = np.nan         # missing after the last change point stopped being followed
    return out


def group_curves(times, group, grid, min_subjects):
    """Curve per participant x drug x dimension x group -> mean over drug -> group mean +/- 95% CI.

    group: the column the curves are split by, "volatility" (stable vs volatile) or "all"
    (a column with the same value in every row, so one curve per dimension).
    Time points with fewer than min_subjects participants are dropped.
    A participant's curve is NaN after their longest-followed change point (km_detected),
    so late time points average fewer participants (column n_subjects).
    """
    curves = [pd.DataFrame({"subID": s, "dimension": dim, group: lev, "drug_label": drug, "time_s": grid,
                            "value": km_detected(g["time_s"].to_numpy(), g["detected"].to_numpy(), grid)})
              for (s, drug, dim, lev), g in times.groupby(["subID", "drug_label", "dimension", group])]
    subj = pd.concat(curves).groupby(["subID", "dimension", group, "time_s"])["value"].mean()
    grp = subj.groupby(["dimension", group, "time_s"]).agg(mean="mean", sd="std", n_subjects="count").reset_index()
    grp["ci95"] = 1.96 * grp["sd"] / np.sqrt(grp["n_subjects"])
    return grp[grp["n_subjects"] >= min_subjects]


def detection_curves(frames, cp, cfg):
    """Group detection curves, by dimension and by dimension x volatility.

    Each curve is 1 - the Kaplan-Meier survival function, so change points that end without a
    correct response count as censored. Curves are made per participant x drug session x
    dimension (x volatility), averaged over drug sessions, then over participants (mean and
    95% CI = 1.96 x SEM). Time points with fewer than min_subjects participants are dropped.
    Settings: config/analysis_config.yaml, detection_curve.
    """
    settings, fps = cfg["detection_curve"], cfg["frame_rate_hz"]
    times = detection_times(frames, cp, fps).assign(all="all")
    print(f"detection curves: {len(times):,} change points, {times['detected'].mean():.1%} detected before they ended")
    grid = np.round(np.arange(0, settings["max_time_s"] + 1e-9, 1 / fps), 6)
    by_dim = group_curves(times, "all", grid, settings["min_subjects"]).drop(columns="all")
    by_vol = group_curves(times, "volatility", grid, settings["min_subjects"])
    return by_dim, by_vol


def rating_change_counts(frames, cfg):
    """One row per block: frames where the participant's rating changed, and true-mean changes."""
    frames = frames.sort_values(BLOCK_KEYS + ["currentFrame"], kind="mergesort")
    g = frames.groupby(BLOCK_KEYS, sort=False)
    frames = frames.assign(_rating_change=g["rating"].diff().fillna(0).ne(0),
                           _true_change=g["trueMeanVal"].diff().fillna(0).ne(0))
    agg = frames.groupby(BLOCK_KEYS, as_index=False).agg(
        n_rating_changes=("_rating_change", "sum"),
        n_true_changes=("_true_change", "sum"),
        n_frames=("currentFrame", "size"))
    agg["rating_changes_per_cp"] = agg["n_rating_changes"] / agg["n_true_changes"].replace(0, np.nan)
    agg["rating_changes_per_minute"] = agg["n_rating_changes"] / (agg["n_frames"] / cfg["frame_rate_hz"] / 60)
    return add_labels(agg, cfg)


def occupancy_table(frames, cfg):
    """One row per block x rating level (valence 2-6, arousal 2/4/6).

    occupancy_ratio = frames the participant's rating was at a level
                      / frames the true mean was at that level.
    Above 1 means the participant spent more time at that level than the true mean did.
    """
    rated = (frames.groupby(BLOCK_KEYS + ["rating"], as_index=False).size()
             .rename(columns={"rating": "level", "size": "n_frames_rating"}))
    true = (frames.groupby(BLOCK_KEYS + ["trueMeanVal"], as_index=False).size()
            .rename(columns={"trueMeanVal": "level", "size": "n_frames_true"}))
    occ = pd.merge(true, rated, on=BLOCK_KEYS + ["level"], how="outer")
    occ[["n_frames_rating", "n_frames_true"]] = occ[["n_frames_rating", "n_frames_true"]].fillna(0)
    occ["occupancy_ratio"] = occ["n_frames_rating"] / occ["n_frames_true"].replace(0, np.nan)
    occ["time_s_rating"] = occ["n_frames_rating"] / cfg["frame_rate_hz"]
    occ["time_s_true"] = occ["n_frames_true"] / cfg["frame_rate_hz"]
    return add_labels(occ, cfg).sort_values(BLOCK_KEYS + ["level"]).reset_index(drop=True)


def rating_error_block_emotion(frames, cfg):
    """Mean rating error per block x emotion level of the true mean."""
    keys = BLOCK_KEYS + ["emotion_level_frame"]
    agg = frames.groupby(keys, as_index=False).agg(
        RE_signed=("RE_signed", "mean"), RE_abs=("RE_abs", "mean"), RE_flipped=("RE_flipped", "mean"),
        rating=("rating", "mean"), trueMeanVal=("trueMeanVal", "mean"),
        n_frames=("currentFrame", "size"))
    return add_labels(agg.rename(columns={"emotion_level_frame": "emotion_level"}), cfg)


def rating_error_block(frames, cfg):
    """Mean rating error per block."""
    agg = frames.groupby(BLOCK_KEYS, as_index=False).agg(
        RE_signed=("RE_signed", "mean"), RE_abs=("RE_abs", "mean"), RE_flipped=("RE_flipped", "mean"),
        n_frames=("currentFrame", "size"))
    return add_labels(agg, cfg)


def rating_error_condition_emotion(block_emotion):
    """Average of the three blocks of each subject x drug x dimension x volatility x emotion level."""
    keys = ["subID", "drug", "drug_label", "condID", "dimension", "stabID", "volatility", "emotion_level"]
    return block_emotion.groupby(keys, as_index=False).agg(
        RE_signed_mean=("RE_signed", "mean"), RE_abs_mean=("RE_abs", "mean"),
        RE_flipped_mean=("RE_flipped", "mean"), n_blocks=("block", "nunique"),
        n_frames=("n_frames", "sum"))


def diff_re_condition(cond_emotion):
    """Flipped rating error at the positive/high end minus the negative/low end, per condition cell."""
    hi = {"valence": "positive", "arousal": "high"}
    lo = {"valence": "negative", "arousal": "low"}
    out = []
    for dim, sub in cond_emotion.groupby("dimension"):
        wide = sub.pivot_table(index=["subID", "drug", "drug_label", "condID", "stabID", "volatility"],
                               columns="emotion_level", values="RE_flipped_mean").reset_index()
        wide["DiffRE"] = wide[hi[dim]] - wide[lo[dim]]
        wide["dimension"] = dim
        out.append(wide[["subID", "drug", "drug_label", "condID", "stabID", "volatility",
                         "dimension", "DiffRE"]])
    return pd.concat(out, ignore_index=True)


def subject_means(df, col):
    """subject x drug x dimension mean, then the mean over drug."""
    return df.groupby(["subID", "drug", "dimension"])[col].mean().groupby(["subID", "dimension"]).mean()
