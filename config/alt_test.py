"""Alternative Annotator Test (alt-test) configuration."""

from pathlib import Path

# Path to the CSV file containing annotations
CSV: str | Path = ".data/annotation_sheet.csv"

# Column name containing the LLM predictions
LLM_COL: str = "LLM"

# Human annotator column names
HUMAN_1: str = "A1"
HUMAN_2: str = "A2"
HUMAN_3: str = "A3"
ANNOTATORS: list[str] = [HUMAN_1, HUMAN_2, HUMAN_3]

# Equivalence margin threshold (higher values represent more lenient criteria)
EPSILON: float = 0.15

# False Discovery Rate significance level for Benjamini-Yekutieli procedure
Q: float = 0.05

# Minimum winning rate required for LLM approval
OMEGA_THRESHOLD: float = 0.5

# Whether to apply inverse-probability weighting correction for imbalanced labels
IPW: bool = False

# Random seed for majority-vote tie-breaking under IPW
SEED: int = 0

# Optional path to save evaluation results in JSON format (set to None to disable)
OUT: str | Path | None = ".results/alt_test.json"
