# ketamine-emotional-evidence-integration

Analysis code for Rascu et al., "Context-dependent effects of ketamine on emotional evidence
integration". 

Preprint: bioRxiv (2026), https://www.biorxiv.org/content/10.64898/2026.10.30.762604v1

Participants continuously rated the average valence or arousal of a rapid stream of faces, in
stable and volatile blocks. An in-person sample (n = 29) completed the task in 2 visits, one week apart,
1.5-hours following ketamine and placebo infusion (within-subject, counterbalanced); 
an online sample (n = 92) did the task in one session. The code computes the behavioural measures 
(change-point accuracy and reaction time, rating error, rating changes), the evidence-integration kernels, 
all statistics and all figure panels reported in the paper and its supplement, from the shared data.

## Repository structure

```
analysis/
  behaviour/      behavioural measures (change points, rating error, rating changes),
                  detection curves and the shuffled-rating null model
  kernels/        evidence-integration kernels (face-wise linear, time-binned logistic)
  stats/          mixed models and tests for all tables and numbers in the text (R),
                  descriptives, questionnaires
  figures/        all figure panels
utils/            shared code: settings, reading the data, plotting style
config/           analysis_config.yaml (every setting that affects the results)
data/             put the shared data files here (not included, see Data)
task/             task code (to be added)
output/           analysis outputs tables/, stats/, figures/ (added when run)
run_all.sh        runs the whole analysis in order
environment.yml   software versions used for the paper
```

## Data

The data will be made available upon publication. Put the files in `data/`, or
set the environment variable `PEDUKS_DATA_DIR` to their folder (e.g.
`export PEDUKS_DATA_DIR=/path/to/data` in the terminal before running).

| File | Content |
|---|---|
| `frames_in_person.csv.gz`, `frames_online.csv.gz` | one row per screen frame (60 Hz): face shown, its value, the true mean, the participant's rating |
| `subjects_in_person.csv`, `subjects_online.csv` | drug condition per visit (in person), age, gender/sex |
| `questionnaires_in_person.csv` | 4-item BPRS, CADSS-6, PANAS |

## Requirements

Python 3.11 (pandas, numpy, scipy, statsmodels, matplotlib, pyyaml) and R 4.5 (lme4, lmerTest,
pbkrtest, emmeans, yaml). `environment.yml` lists the exact versions used for the paper. With
conda (or mamba) installed, create the environment once and activate it before running:

```
conda env create -f environment.yml
conda activate peduks-analysis
```



## How to run

Run everything from this folder. The simplest way is one command, which runs all steps below
in order for both cohorts (about 1.5 h on 8 cores, mostly the null shuffles):

```
bash run_all.sh            # full run (1000 shuffles per cohort, as in the paper)
bash run_all.sh --quick    # test that everything runs: only 3 shuffles, so the
                           # real-vs-null results are not the reported ones
N_JOBS=4 bash run_all.sh   # number of cores for the shuffles (default 8)
```

It stops at the first error. It needs bash (macOS, Linux, or Git Bash/WSL on Windows). The
scripts can also be run one by one, as below. Scripts with `--cohort` are run once with
`in_person` and once with `online`. The order matters, because each step reads the tables
written by the steps before.

```
# 1. behavioural measures
python analysis/behaviour/01_make_behaviour_tables.py --cohort in_person
python analysis/behaviour/02_task_descriptives.py --cohort in_person
python analysis/behaviour/03_null_shuffles.py --cohort in_person      # ~25 min on 8 cores (online ~40 min)
python analysis/behaviour/04_null_tests.py --cohort in_person

# 2. integration kernels
python analysis/kernels/01_facewise_linear.py --cohort in_person
python analysis/kernels/02_timebinned_logistic.py --cohort in_person

# 3. statistics
Rscript analysis/stats/01_supplement_tables.R --cohort in_person      # ~7 min (online ~1.5 min)
Rscript analysis/stats/02_covariates_S9.R
python analysis/stats/03_descriptives.py --cohort in_person
Rscript analysis/stats/04_questionnaire_stats.R

# 4. figures
python analysis/figures/01_behaviour_panels.py --cohort in_person
python analysis/figures/02_detection_curves.py --cohort in_person
python analysis/figures/03_kernel_panels.py --cohort in_person
python analysis/figures/04_questionnaire_panels.py
python analysis/figures/05_example_blocks.py
```

The online cohort uses the same commands with `--cohort online`, except for Table S9
(`02_covariates_S9.R`, drug models), the questionnaires (`04_questionnaire_stats.R`,
`04_questionnaire_panels.py`) and the Fig 1 example blocks (`05_example_blocks.py`), which
are in person only. 

Main settings (response windows, kernel settings, number of shuffles,
random seed, detection-curve settings) are in `config/analysis_config.yaml`.

## Outputs

Everything is written to `output/`:

- `output/tables/<cohort>/`: behavioural and kernel measure tables (CSV)
- `output/stats/<cohort>/`: model results (`all_rows.csv` with every number; one
  `table_<id>.txt` per supplementary table, with APA-style text for each row)
- `output/figures/<Fig>/`: one PNG per figure panel, with the plotted values in `source_data/`;
  panels that are not in the paper are in `output/figures/extra/`. For vector files, add `svg`
  to `figures: formats` in `config/analysis_config.yaml`. The Fig 1 example blocks (participant,
  visit, blocks) are chosen at the top of `analysis/figures/05_example_blocks.py`.

## AI use disclosure
Generative AI tools, including ChatGPT and Codex (OpenAI) and Claude Opus 5.5 (Anthropic), were used to assist with code development, debugging, visualisation, documentation, and refactoring. The analyses reported in the manuscript were developed and initially implemented by the authors. AI assistance was used in the subsequent optimisation and restructuring of the codebase, which was reviewed and validated by MR. All scientific decisions, analytical methods, and interpretations of results remain the responsibility of the authors.
