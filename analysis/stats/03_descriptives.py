"""Descriptive numbers reported in the text: sample (Methods) and mean RT (Results, in person).

    python analysis/stats/03_descriptives.py --cohort in_person

RT: mean of each participant's correct RTs (all analysable, correctly detected change
points, both visits), then mean and SD across participants.

Needs output/tables/<cohort>/cp_table.csv (01_make_behaviour_tables.py) and the subjects file.
Writes output/stats/<cohort>/descriptives.csv.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.config import COHORTS, out_dir   # noqa: E402
from utils.data import load_subjects        # noqa: E402


def main(cohort):
    rows = []

    # sample: one row per participant
    subj = load_subjects(cohort).query("visit == 1")
    sex = subj["gender"] if "gender" in subj else subj["sex"]
    rows.append(("participants", "all", len(subj), None))
    rows.append(("female", "all", int(sex.isin(["F", "Female"]).sum()), None))
    rows.append(("age (years), mean / SD", "all", subj["age"].mean(), subj["age"].std()))
    rows.append(("age (years), min / max", "all", subj["age"].min(), subj["age"].max()))

    # RT of correctly detected change points, participant means
    if cohort == "in_person":
        cp = pd.read_csv(out_dir("tables", cohort) / "cp_table.csv")
        cp = cp[cp["admissible"] & cp["detected_correctly"].eq(1)]
        for dim, d in [("all", cp), *cp.groupby("dimension")]:
            m = d.groupby("subID")["rt_first_correct_s"].mean()
            rows.append(("correct RT (s), mean / SD of participant means", dim, m.mean(), m.std()))

    out = pd.DataFrame(rows, columns=["measure", "condition", "value", "value_2"])
    out.to_csv(out_dir("stats", cohort) / "descriptives.csv", index=False)
    print("\n" + out.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=COHORTS)
    main(ap.parse_args().cohort)
