#!/usr/bin/env python3
"""Command-line runner for Cohen's kappa agreement analysis with bootstrap CIs.

Calculates pairwise Cohen's kappa, analytic standard errors, and item-level
bootstrap confidence intervals using parameters defined in config/kappa.py.

References:
    Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational
    and Psychological Measurement, 20(1), 37-46.
    https://doi.org/10.1177/001316446002000104

    Fleiss, J. L., Cohen, J., & Everitt, B. S. (1969). Large sample standard
    errors of kappa and weighted kappa. Psychological Bulletin, 72(5), 323-327.
    https://doi.org/10.1037/h0028106

    Sim, J., & Wright, C. C. (2005). The kappa statistic in reliability studies:
    Use, interpretation, and sample size requirements. Physical Therapy, 85(3),
    257-268. https://doi.org/10.1093/ptj/85.3.257
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import kappa as cfg
from src.data import BinaryAnnotationData, load_binary_csv
from src.kappa import (
    KappaAnalysisResult,
    format_kappa_report,
    run_kappa_analysis,
)


def main() -> None:
    """Execute Cohen's kappa agreement analysis using settings from config/kappa.py."""
    try:
        data: BinaryAnnotationData = load_binary_csv(
            path=cfg.CSV,
            llm_col=cfg.LLM_COL,
            annotator_cols=cfg.ANNOTATORS,
        )
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")

    try:
        analysis: KappaAnalysisResult = run_kappa_analysis(
            df=data.df,
            ann_cols=data.ann_cols,
            llm_col=cfg.LLM_COL,
            n_boot=cfg.N_BOOT,
            seed=cfg.SEED,
            strata_col=cfg.STRATA,
            threshold=cfg.THRESHOLD,
            margin=cfg.MARGIN,
        )
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    report: str = format_kappa_report(
        analysis=analysis,
        df=data.df,
        ann_cols=data.ann_cols,
        llm_col=cfg.LLM_COL,
        n_boot=cfg.N_BOOT,
        seed=cfg.SEED,
        strata_col=cfg.STRATA,
        threshold=cfg.THRESHOLD,
        margin=cfg.MARGIN,
    )
    print(report)

    if cfg.OUT:
        out_file = Path(cfg.OUT)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "n": analysis["n"],
                    "annotators": analysis["annotators"],
                    "pairs": analysis["pairs"],
                    "human_human_mean_kappa": analysis["human_human_mean_kappa"],
                    "llm_human_mean_kappa": analysis["llm_human_mean_kappa"],
                    "difference": analysis["difference"],
                    "margin": analysis["margin"],
                    "bootstrap": analysis["bootstrap"],
                },
                f,
                indent=2,
            )
        print(f"\nSaved JSON to {out_file}")


if __name__ == "__main__":
    main()
