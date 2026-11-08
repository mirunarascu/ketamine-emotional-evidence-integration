"""Read the shared data files and add the columns the analyses use"""
import pandas as pd

from utils.config import data_dir

NO_DRUG = -1          # drug code for the online cohort


def load_subjects(cohort):
    return pd.read_csv(data_dir() / f"subjects_{cohort}.csv")


def load_frames(cfg):
    """Frame table of one cohort, with drug, condition codes, rating error and emotion level

    Rating error is computed as:
        RE_signed  = rating - true mean
        RE_abs     = |RE_signed|
        RE_flipped = RE_signed with the sign flipped for positive valence and high arousal
    """
    cohort = cfg["cohort"]["name"]
    frames = pd.read_csv(data_dir() / f"frames_{cohort}.csv.gz")
    print(f"{cohort}: {len(frames):,} frames, {frames.subID.nunique()} participants")

    # condition codes used by the measure and kernel code
    code = {v: k for k, v in cfg["coding"]["condID"].items()}
    frames["condID"] = frames["dimension"].map(code)
    code = {v: k for k, v in cfg["coding"]["stabID"].items()}
    frames["stabID"] = frames["volatility"].map(code)

    if cfg["cohort"]["has_drug"]:
        subj = load_subjects(cohort)[["subID", "visit", "drug"]].rename(columns={"drug": "drug_label"})
        frames = frames.merge(subj, on=["subID", "visit"], how="left", validate="many_to_one")
        if frames["drug_label"].isna().any():
            raise ValueError("some frames have no drug condition in the subjects file")
        code = {v: k for k, v in cfg["coding"]["drug"].items()}
        frames["drug"] = frames["drug_label"].map(code)
    else:
        frames["drug"] = NO_DRUG

    # emotion level of the true mean in each frame
    frames["emotion_level_frame"] = pd.NA
    for dim, levels in cfg["emotion_levels"].items():
        for label, values in levels.items():
            sel = frames["dimension"].eq(dim) & frames["trueMeanVal"].isin(values)
            frames.loc[sel, "emotion_level_frame"] = label
    n_missing = frames["emotion_level_frame"].isna().sum()
    if n_missing:
        raise ValueError(f"{n_missing} frames have a true mean outside the design levels")
    # ratings must be on the same scale as the true mean (valence 2-6, arousal 2/4/6)
    n_bad = 0
    for dim, levels in cfg["emotion_levels"].items():
        values = [v for vals in levels.values() for v in vals]
        n_bad += (frames["dimension"].eq(dim) & ~frames["rating"].isin(values)).sum()
    if n_bad:
        raise ValueError(f"{n_bad} frames have a rating outside the true-mean scale")

    frames["RE_signed"] = frames["rating"] - frames["trueMeanVal"]
    frames["RE_abs"] = frames["RE_signed"].abs()
    flip = frames["emotion_level_frame"].isin(cfg["rating_error_flip"])
    frames["RE_flipped"] = frames["RE_signed"].where(~flip, -frames["RE_signed"])
    return frames


def probe_table(frames):
    """Mid-block confidence probe frame of each block (one row per block)"""
    p = frames.loc[frames["probe"] == 1, ["subID", "visit", "block", "currentFrame"]]
    p = p.rename(columns={"currentFrame": "conf_frame"}).assign(kind="mid_block")
    if p.duplicated(["subID", "visit", "block"]).any():
        raise ValueError("more than one mid-block probe in a block")
    return p
