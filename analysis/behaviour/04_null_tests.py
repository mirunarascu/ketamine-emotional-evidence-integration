"""Real vs shuffled null (Fig 2B-C): paired t-test per dimension.

    python analysis/behaviour/04_null_tests.py --cohort in_person
    python analysis/behaviour/04_null_tests.py --cohort online       (not reported in the paper)

For each participant and dimension, the real value is compared with the participant's
mean over all shuffles. Both are averaged the same way (subject x drug, then over drug),
and real accuracy uses the same analysed change points as the null.
Writes output/tables/<cohort>/null_comparison_tests.csv and null_subject_values.csv (for the figure).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.behaviour.behaviour_measures import subject_means      # noqa: E402
from utils.config import COHORTS, out_dir                   # noqa: E402


def main(cohort):
    t = out_dir("tables", cohort)
    null = pd.read_csv(t / "null_shuffles.csv")
    cp = pd.read_csv(t / "cp_table.csv")
    rb = pd.read_csv(t / "rating_error_block.csv")
    n_shuffles = null["shuffle"].nunique()

    real = pd.concat([subject_means(cp[cp["admissible"]], "detected_correctly"),
                      subject_means(rb, "RE_abs")], axis=1)
    nmean = null.groupby(["subID", "dimension"])[["null_cp_accuracy", "null_RE_abs"]].mean()
    nmean.columns = ["detected_correctly", "RE_abs"]

    rows, values = [], []
    for dv in ("RE_abs", "detected_correctly"):
        for dim in ("valence", "arousal"):
            r, n = real[dv].xs(dim, level="dimension").align(nmean[dv].xs(dim, level="dimension"), join="inner")
            if len(r) != real[dv].xs(dim, level="dimension").size:
                print(f"WARNING: {dv} {dim}: {real[dv].xs(dim, level='dimension').size - len(r)} participants have no null")
            d = r - n
            se = d.std(ddof=1) / np.sqrt(len(d))
            tt = stats.ttest_rel(r, n)
            q = stats.t.ppf(.975, len(d) - 1)   # t value for a 95% CI (two-sided), df = n - 1
            rows.append(dict(measure=dv, dimension=dim, n_subjects=len(d), mean_real=r.mean(),
                             mean_null=n.mean(), mean_diff=d.mean(), ci_low=d.mean() - q * se,
                             ci_high=d.mean() + q * se, t=tt.statistic, df=len(d) - 1, p=tt.pvalue,
                             d_z=d.mean() / d.std(ddof=1), n_shuffles=n_shuffles))
            values.append(pd.DataFrame({"subID": r.index, "dimension": dim, "measure": dv,
                                        "real": r.values, "null": n.values}))

    tests = pd.DataFrame(rows)
    tests.to_csv(t / "null_comparison_tests.csv", index=False)
    pd.concat(values).to_csv(t / "null_subject_values.csv", index=False)
    for r in tests.itertuples():
        print(f"{r.measure} {r.dimension}: real {r.mean_real:.3f} vs null {r.mean_null:.3f}, "
                 f"t({r.df}) = {r.t:.2f}, p = {r.p:.2g}, d_z = {r.d_z:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    main(ap.parse_args().cohort)
