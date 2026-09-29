#!/usr/bin/env python3
"""Command-line runner for the Alternative Annotator Test (alt-test).

Executes the Alternative Annotator Test using parameters defined in config/alt_test.py.

References:
    Calderon, N., Reichart, R., & Dror, R. (2025). The Alternative Annotator Test
    for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators
    with LLMs. In Proceedings of the 63rd Annual Meeting of the Association for
    Computational Linguistics (Volume 1: Long Papers), pages 16051-16081.
    https://doi.org/10.18653/v1/2025.acl-long.875

    Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery
    rate in multiple testing under dependency. The Annals of Statistics, 29(4),
    1165-1188. https://doi.org/10.1214/aos/1013699998
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import alt_test as cfg
from src.alt_test import AnnotatorResult, alt_test, format_alt_test_report
from src.data import BinaryAnnotationData, load_binary_csv


def main() -> None:
    """Execute the Alternative Annotator Test using settings from config/alt_test.py."""
    try:
        data: BinaryAnnotationData = load_binary_csv(
            path=cfg.CSV,
            llm_col=cfg.LLM_COL,
            annotator_cols=cfg.ANNOTATORS,
        )
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    llm = data.llm
    h_matrix = data.H
    ann_cols: list[str] = data.ann_cols
    n_items, _ = h_matrix.shape

    if n_items < 30:
        sys.exit(f"ERROR: {n_items} items; the t-test needs at least 30 (paper Appendix A).")

    try:
        rows: list[AnnotatorResult]
        omega: float
        rho: float
        rows, omega, rho = alt_test(
            llm=llm,
            H=h_matrix,
            eps=cfg.EPSILON,
            q=cfg.Q,
            ipw=cfg.IPW,
            seed=cfg.SEED,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    report: str = format_alt_test_report(
        ann_cols=ann_cols,
        rows=rows,
        omega=omega,
        rho=rho,
        llm=llm,
        H=h_matrix,
        eps=cfg.EPSILON,
        q=cfg.Q,
        omega_threshold=cfg.OMEGA_THRESHOLD,
        ipw=cfg.IPW,
    )
    print(report)

    if cfg.OUT:
        out_file = Path(cfg.OUT)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        passed: bool = omega >= cfg.OMEGA_THRESHOLD
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n": n_items,
                    "annotators": ann_cols,
                    "epsilon": cfg.EPSILON,
                    "q": cfg.Q,
                    "omega_threshold": cfg.OMEGA_THRESHOLD,
                    "ipw": cfg.IPW,
                    "per_annotator": dict(zip(ann_cols, rows)),
                    "winning_rate": omega,
                    "advantage_probability": rho,
                    "passed": passed,
                },
                f,
                indent=2,
            )
        print(f"Saved JSON to {out_file}")


if __name__ == "__main__":
    main()
