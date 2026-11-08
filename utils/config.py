"""Settings and folders"""
import os
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "config"
OUTPUT_DIR = REPO / "output"
COHORTS = ("in_person", "online")


def load_config(cohort):
    """Analysis settings with the cohort-specific values added under cfg['cohort']."""
    if cohort not in COHORTS:
        raise ValueError(f"unknown cohort {cohort!r}, use one of {COHORTS}")
    cfg = yaml.safe_load(open(CONFIG_DIR / "analysis_config.yaml"))
    cfg["cohort"] = cfg["cohorts"][cohort]
    cfg["cohort"]["name"] = cohort
    return cfg


def data_dir():
    """Folder with the shared data files: $PEDUKS_DATA_DIR if set, else data/ in the repository."""
    d = Path(os.environ["PEDUKS_DATA_DIR"]) if os.environ.get("PEDUKS_DATA_DIR") else REPO / "data"
    if not d.exists():
        raise FileNotFoundError(f"data folder {d} not found (see Data in README.md)")
    return d


def out_dir(*parts):
    """output/<parts>, created if needed."""
    d = OUTPUT_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
