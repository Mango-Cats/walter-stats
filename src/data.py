"""Data loading and validation for binary LLM and human annotations."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class BinaryAnnotationData:
    """Validated dataset containing binary ratings from an LLM and human annotators.

    Attributes:
        df: Processed DataFrame containing annotations and optional metadata columns.
        ann_cols: Names of the detected human annotator columns, sorted numerically.
        llm: 1D array of binary ratings produced by the LLM.
        H: 2D array of binary ratings produced by human annotators of shape (n_items, n_annotators).
    """

    df: pd.DataFrame
    ann_cols: list[str]
    llm: npt.NDArray[np.int_]
    H: npt.NDArray[np.int_]

    @property
    def annotator_dict(self) -> dict[str, npt.NDArray[np.int_]]:
        """Map of annotator column name to 1D binary rating array."""
        return {col: self.H[:, i] for i, col in enumerate(self.ann_cols)}

    @property
    def n_items(self) -> int:
        """Total number of evaluated items."""
        return len(self.df)

    @property
    def n_annotators(self) -> int:
        """Total number of human annotators."""
        return len(self.ann_cols)


def load_binary_csv(
    path: str | Path,
    llm_col: str = "label",
    annotator_cols: Sequence[str] | None = None,
) -> BinaryAnnotationData:
    """Load and validate a CSV containing binary annotations from an LLM and human annotators.

    Args:
        path: File system path to the CSV file.
        llm_col: Column name containing the LLM predictions. Defaults to 'label'.
        annotator_cols: Optional sequence of human annotator column names.
            If None, columns matching the regex pattern '[aA]\\d+' are detected automatically.

    Returns:
        BinaryAnnotationData container with the validated DataFrame and NumPy arrays.

    Raises:
        FileNotFoundError: If the CSV file cannot be located.
        ValueError: If the CSV is empty, lacks required columns, contains fewer than two
            annotator columns, or contains values other than 0 and 1.
    """
    file_path = Path(path)
    if not file_path.exists():
        candidate_hidden = Path(".data") / path
        if candidate_hidden.exists():
            file_path = candidate_hidden
        else:
            candidate_regular = Path("data") / path
            if candidate_regular.exists():
                file_path = candidate_regular
            else:
                raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(file_path, dtype=str)
    except pd.errors.EmptyDataError:
        raise ValueError("CSV has no data rows.")

    if df.empty:
        raise ValueError("CSV has no data rows.")
    if llm_col not in df.columns:
        raise ValueError(f"missing required column '{llm_col}' (the LLM label).")

    if annotator_cols is not None:
        ann_cols: list[str] = list(annotator_cols)
        missing = [c for c in ann_cols if c not in df.columns]
        if missing:
            raise ValueError(f"annotator column(s) not found in CSV: {missing}")
    else:
        ann_cols = sorted(
            (c for c in df.columns if re.fullmatch(r"[aA]\d+", c)),
            key=lambda c: int(c[1:]),
        )

    if len(ann_cols) < 2:
        raise ValueError(f"need at least 2 annotator columns; found {ann_cols}")

    problems: list[tuple[int, str, str]] = []
    for c in [llm_col] + ann_cols:
        vals = df[c].fillna("").str.strip()
        invalid_indices = np.flatnonzero(~vals.isin(["0", "1"]).to_numpy())
        for i in invalid_indices:
            shown = "<empty>" if vals.iloc[i] == "" else repr(df[c].iloc[i])
            problems.append((int(i) + 2, c, shown))
        df[c] = vals

    if problems:
        problems.sort()
        lines = "\n".join(f"  CSV line {ln}, column '{c}': {v}" for ln, c, v in problems[:50])
        more = f"\n  ... and {len(problems) - 50} more" if len(problems) > 50 else ""
        ann_names = "/".join(ann_cols)
        raise ValueError(
            f"{len(problems)} invalid label(s); every value in {llm_col}/{ann_names} "
            f"must be 0 or 1.\n{lines}{more}"
        )

    llm: npt.NDArray[np.int_] = df[llm_col].astype(int).to_numpy()
    H: npt.NDArray[np.int_] = df[ann_cols].astype(int).to_numpy()

    return BinaryAnnotationData(df=df, ann_cols=ann_cols, llm=llm, H=H)
