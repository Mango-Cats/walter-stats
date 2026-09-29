"""Cohen's kappa bootstrap analysis configuration."""

from pathlib import Path

# Path to the CSV file containing annotations
CSV: str | Path = ".data/annotation_sheet.csv"

# Column name containing the LLM predictions
LLM_COL: str = "LLM"

# Human annotator column names (Cohen's kappa uses 2 annotators: Human 1 and Human 2)
HUMAN_1: str = "A1"
HUMAN_2: str = "A2"
ANNOTATORS: list[str] = [HUMAN_1, HUMAN_2]

# Number of bootstrap resamples
N_BOOT: int = 10000

# Random seed for bootstrap reproducibility
SEED: int = 0

# Optional column name in CSV to stratify bootstrap resampling on (e.g., 'LLM' or None)
STRATA: str | None = None

# Optional minimum acceptable kappa threshold for lower bootstrap CI bound verification
THRESHOLD: float | None = 0.6

# Optional non-inferiority margin on kappa difference (passes if upper CI bound < margin)
MARGIN: float | None = 0.1

# Optional path to save evaluation results in JSON format (set to None to disable)
OUT: str | Path | None = ".results/kappa_bootstrap.json"
