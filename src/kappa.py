"""Cohen's kappa agreement analysis with bootstrap confidence intervals.

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

    Byrt, T., Bishop, J., & Carlin, J. B. (1993). Bias, prevalence and kappa.
    Journal of Clinical Epidemiology, 46(5), 423-429.
    https://doi.org/10.1016/0895-4356(93)90018-V

    Cicchetti, D. V., & Feinstein, A. R. (1990). High agreement but low kappa:
    II. Resolving the paradoxes. Journal of Clinical Epidemiology, 43(6), 551-558.
    https://doi.org/10.1016/0895-4356(90)90159-M
"""

import itertools
from typing import Sequence, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd


class PairStatistics(TypedDict):
    """Pairwise agreement metrics for binary classifications."""

    kappa: float
    analytic_se: float
    analytic_ci95: list[float]
    observed_agreement: float
    max_kappa: float
    pabak: float
    prevalence_index: float
    bias_index: float
    positive_agreement: float
    negative_agreement: float
    positive_rate_rater1: float
    positive_rate_rater2: float
    table: list[list[int]]


class BootstrapEstimate(TypedDict):
    """Percentile bootstrap confidence interval and diagnostics."""

    ci95: list[float]
    degenerate_resamples: int


class KappaAnalysisResult(TypedDict):
    """Aggregated outcomes of Cohen's kappa agreement analysis."""

    n: int
    annotators: list[str]
    pairs: dict[str, PairStatistics]
    human_human_mean_kappa: float
    llm_human_mean_kappa: float
    difference: float
    margin: float | None
    bootstrap: dict[str, BootstrapEstimate]
    non_inferiority_passed: bool | None


def kappa_from_counts(c: npt.NDArray[np.number]) -> npt.NDArray[np.float64]:
    """Calculate Cohen's kappa from 2x2 contingency table counts.

    The trailing dimension of c must have length 4, corresponding to cell counts
    for outcomes (0,0), (0,1), (1,0), and (1,1). Returns NaN where expected agreement
    equals 1.0.

    Args:
        c: Array of counts with shape (..., 4).

    Returns:
        Array of calculated kappa coefficients with shape (...).
    """
    counts: npt.NDArray[np.float64] = np.asarray(c, dtype=np.float64)
    total: npt.NDArray[np.float64] = counts.sum(axis=-1)
    po: npt.NDArray[np.float64] = (counts[..., 0] + counts[..., 3]) / total
    r1: npt.NDArray[np.float64] = (counts[..., 2] + counts[..., 3]) / total
    r2: npt.NDArray[np.float64] = (counts[..., 1] + counts[..., 3]) / total
    pe: npt.NDArray[np.float64] = r1 * r2 + (1.0 - r1) * (1.0 - r2)

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa: npt.NDArray[np.float64] = (po - pe) / (1.0 - pe)
    return np.where(np.isclose(pe, 1.0), np.nan, kappa)


def analytic_se(t: npt.NDArray[np.number]) -> float:
    """Calculate non-null large-sample standard error of Cohen's kappa.

    Args:
        t: 2x2 contingency matrix of observed classification counts.

    Returns:
        Asymptotic standard error, or NaN if expected agreement is 1.0.

    References:
        Fleiss, J. L., Cohen, J., & Everitt, B. S. (1969). Large sample standard
        errors of kappa and weighted kappa. Psychological Bulletin, 72(5), 323-327.
    """
    table: npt.NDArray[np.float64] = np.asarray(t, dtype=np.float64)
    n: float = float(table.sum())
    p: npt.NDArray[np.float64] = table / n
    pr: npt.NDArray[np.float64] = p.sum(axis=1)
    pc: npt.NDArray[np.float64] = p.sum(axis=0)
    po: float = float(np.trace(p))
    pe: float = float((pr * pc).sum())

    if np.isclose(pe, 1.0):
        return float("nan")

    k: float = (po - pe) / (1.0 - pe)
    diag: float = sum(p[i, i] * (1.0 - (pr[i] + pc[i]) * (1.0 - k)) ** 2 for i in range(2))
    off: float = (1.0 - k) ** 2 * sum(
        p[i, j] * (pc[i] + pr[j]) ** 2 for i in range(2) for j in range(2) if i != j
    )
    var: float = (diag + off - (k - pe * (1.0 - k)) ** 2) / (n * (1.0 - pe) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def pair_stats(
    r1: npt.NDArray[np.integer],
    r2: npt.NDArray[np.integer],
) -> PairStatistics:
    """Compute comprehensive pairwise agreement statistics between two binary raters.

    Args:
        r1: 1D array of binary ratings from the first rater.
        r2: 1D array of binary ratings from the second rater.

    Returns:
        PairStatistics mapping containing kappa, analytic confidence interval,
        observed agreement, PABAK, prevalence index, bias index, and specific agreements.
    """
    code: npt.NDArray[np.int_] = 2 * r1 + r2
    counts: npt.NDArray[np.int_] = np.bincount(code, minlength=4)
    table: npt.NDArray[np.float64] = counts.reshape(2, 2).astype(np.float64)
    n: float = float(table.sum())

    a: float = table[0, 0]
    b: float = table[0, 1]
    cc: float = table[1, 0]
    d: float = table[1, 1]

    po: float = (a + d) / n
    pr: npt.NDArray[np.float64] = table.sum(axis=1) / n
    pc: npt.NDArray[np.float64] = table.sum(axis=0) / n
    pe: float = float((pr * pc).sum())
    k: float = float(kappa_from_counts(counts))
    se: float = analytic_se(table)

    pos_denom: float = 2.0 * d + b + cc
    neg_denom: float = 2.0 * a + b + cc

    max_k: float = (
        float((np.minimum(pr, pc).sum() - pe) / (1.0 - pe))
        if not np.isclose(pe, 1.0)
        else float("nan")
    )

    return {
        "kappa": k,
        "analytic_se": se,
        "analytic_ci95": [k - 1.96 * se, k + 1.96 * se],
        "observed_agreement": po,
        "max_kappa": max_k,
        "pabak": float(2.0 * po - 1.0),
        "prevalence_index": float(abs(a - d) / n),
        "bias_index": float(abs(b - cc) / n),
        "positive_agreement": float(2.0 * d / pos_denom) if pos_denom > 0 else float("nan"),
        "negative_agreement": float(2.0 * a / neg_denom) if neg_denom > 0 else float("nan"),
        "positive_rate_rater1": float(pr[1]),
        "positive_rate_rater2": float(pc[1]),
        "table": table.astype(int).tolist(),
    }


def bootstrap(
    codes: dict[str, npt.NDArray[np.integer]],
    hh_names: Sequence[str],
    hl_names: Sequence[str],
    n_boot: int = 10000,
    seed: int = 0,
    strata: Sequence[npt.NDArray[np.integer]] | None = None,
    chunk: int = 500,
) -> dict[str, BootstrapEstimate]:
    """Resample items with replacement to calculate 95% bootstrap confidence intervals.

    Args:
        codes: Dictionary mapping pair identifiers to encoded pairwise rating arrays.
        hh_names: Identifiers corresponding to human-human pairs.
        hl_names: Identifiers corresponding to LLM-human pairs.
        n_boot: Total number of bootstrap iterations. Defaults to 10000.
        seed: Random seed for reproducibility. Defaults to 0.
        strata: Optional list of item indices corresponding to stratification groups.
        chunk: Batch size for vectorized resampling. Defaults to 500.

    Returns:
        Dictionary mapping pair identifiers and aggregate comparisons to their
        95% bootstrap confidence intervals and degenerate sample counts.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    n_items: int = len(next(iter(codes.values())))
    names: list[str] = list(codes.keys())

    x_matrix: npt.NDArray[np.int_] = np.stack([codes[k] for k in names])
    onehot: npt.NDArray[np.int32] = np.stack([(x_matrix == v) for v in range(4)], axis=-1).astype(
        np.int32
    )
    draws: dict[str, list[npt.NDArray[np.float64]]] = {k: [] for k in names}
    completed: int = 0

    while completed < n_boot:
        b: int = min(chunk, n_boot - completed)
        if strata is None:
            idx = rng.integers(0, n_items, (b, n_items))
        else:
            idx = np.hstack([group[rng.integers(0, len(group), (b, len(group)))] for group in strata])

        batch_counts = onehot[:, idx, :].sum(axis=2)
        ks = kappa_from_counts(batch_counts)
        for p_idx, k in enumerate(names):
            draws[k].append(ks[p_idx])
        completed += b

    concatenated: dict[str, npt.NDArray[np.float64]] = {
        k: np.concatenate(v) for k, v in draws.items()
    }
    concatenated["human-human (mean)"] = np.mean([concatenated[k] for k in hh_names], axis=0)
    concatenated["LLM-human (mean)"] = np.mean([concatenated[k] for k in hl_names], axis=0)
    concatenated["difference"] = (
        concatenated["human-human (mean)"] - concatenated["LLM-human (mean)"]
    )

    out: dict[str, BootstrapEstimate] = {}
    for k, v in concatenated.items():
        degenerate_count: int = int(np.isnan(v).sum())
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        out[k] = {
            "ci95": [float(lo), float(hi)],
            "degenerate_resamples": degenerate_count,
        }
    return out


def run_kappa_analysis(
    df: pd.DataFrame,
    ann_cols: Sequence[str],
    llm_col: str = "label",
    n_boot: int = 10000,
    seed: int = 0,
    strata_col: str | None = None,
    threshold: float | None = None,
    margin: float | None = None,
) -> KappaAnalysisResult:
    """Coordinate pairwise statistics, bootstrap resampling, and non-inferiority evaluation.

    Args:
        df: DataFrame containing the annotations.
        ann_cols: Names of all human annotator columns.
        llm_col: Column name containing the LLM predictions. Defaults to 'label'.
        n_boot: Total bootstrap resamples. Defaults to 10000.
        seed: Random seed for bootstrap. Defaults to 0.
        strata_col: Optional column name for stratified resampling.
        threshold: Minimum acceptable kappa value for lower CI bound comparison.
        margin: Non-inferiority tolerance margin for the mean kappa difference.

    Returns:
        KappaAnalysisResult structure containing pairwise metrics, bootstrap CIs,
        and hypothesis evaluation outcomes.

    Raises:
        ValueError: If strata_col is specified but absent from df.
    """
    llm: npt.NDArray[np.int_] = df[llm_col].astype(int).to_numpy()
    h_dict: dict[str, npt.NDArray[np.int_]] = {c: df[c].astype(int).to_numpy() for c in ann_cols}

    hh_pairs: dict[str, tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]] = {
        f"{x}-{y}": (h_dict[x], h_dict[y]) for x, y in itertools.combinations(ann_cols, 2)
    }
    hl_pairs: dict[str, tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]] = {
        f"LLM-{x}": (h_dict[x], llm) for x in ann_cols
    }
    all_pairs = {**hh_pairs, **hl_pairs}

    stats_: dict[str, PairStatistics] = {k: pair_stats(*v) for k, v in all_pairs.items()}
    codes: dict[str, npt.NDArray[np.int_]] = {k: 2 * v[0] + v[1] for k, v in all_pairs.items()}

    strata: list[npt.NDArray[np.intp]] | None = None
    if strata_col:
        if strata_col not in df.columns:
            raise ValueError(f"--strata column not found: {strata_col}")
        col = df[strata_col].to_numpy()
        strata = [np.flatnonzero(col == val) for val in pd.unique(col)]

    boot = bootstrap(codes, list(hh_pairs.keys()), list(hl_pairs.keys()), n_boot, seed, strata)

    k_hh: float = float(np.mean([stats_[k]["kappa"] for k in hh_pairs]))
    k_hl: float = float(np.mean([stats_[k]["kappa"] for k in hl_pairs]))
    diff: float = k_hh - k_hl

    non_inf_passed: bool | None = None
    if margin is not None:
        dci = boot["difference"]["ci95"]
        non_inf_passed = bool(dci[1] < margin)

    return {
        "n": len(df),
        "annotators": list(ann_cols),
        "pairs": stats_,
        "human_human_mean_kappa": k_hh,
        "llm_human_mean_kappa": k_hl,
        "difference": diff,
        "margin": margin,
        "bootstrap": boot,
        "non_inferiority_passed": non_inf_passed,
    }


def format_kappa_report(
    analysis: KappaAnalysisResult,
    df: pd.DataFrame,
    ann_cols: Sequence[str],
    llm_col: str = "label",
    n_boot: int = 10000,
    seed: int = 0,
    strata_col: str | None = None,
    threshold: float | None = None,
    margin: float | None = None,
) -> str:
    """Format human-readable terminal report matching the kappa analysis output.

    Args:
        analysis: Result structure from run_kappa_analysis.
        df: Input DataFrame containing the ratings.
        ann_cols: Names of all human annotators.
        llm_col: Column name containing the LLM predictions. Defaults to 'label'.
        n_boot: Number of bootstrap iterations.
        seed: Random seed used in resampling.
        strata_col: Name of column used for stratification, if any.
        threshold: Minimum acceptable kappa value, if specified.
        margin: Non-inferiority margin, if specified.

    Returns:
        Multi-line formatted summary report string.
    """
    llm: npt.NDArray[np.int_] = df[llm_col].astype(int).to_numpy()
    h_dict: dict[str, npt.NDArray[np.int_]] = {c: df[c].astype(int).to_numpy() for c in ann_cols}
    stats_ = analysis["pairs"]
    boot = analysis["bootstrap"]
    k_hh = analysis["human_human_mean_kappa"]
    k_hl = analysis["llm_human_mean_kappa"]
    diff = analysis["difference"]

    hh_keys: list[str] = [f"{x}-{y}" for x, y in itertools.combinations(ann_cols, 2)]
    hl_keys: list[str] = [f"LLM-{x}" for x in ann_cols]

    lines: list[str] = [
        f"Items: {len(df)}   Annotators: {len(ann_cols)} ({', '.join(ann_cols)})   all labels validated as 0/1",
        (
            f"Bootstrap: {n_boot} resamples of items with replacement, seed={seed}"
            + (f", stratified on '{strata_col}'" if strata_col else "")
        ),
        (
            f"Positive rate: LLM {float(llm.mean()):.3f}; "
            + ", ".join(f"{c} {float(h_dict[c].mean()):.3f}" for c in ann_cols)
            + "\n"
        ),
    ]

    def format_line(name: str, k: float, ci: list[float]) -> str:
        s = f"  {name:<20} kappa = {k:6.3f}   95% bootstrap CI [{ci[0]:6.3f}, {ci[1]:6.3f}]"
        if threshold is not None:
            s += f"   lower bound > {threshold}: {'yes' if ci[0] > threshold else 'NO'}"
        return s

    lines.append("Kappa estimates")
    for k in hh_keys:
        lines.append(format_line(k, stats_[k]["kappa"], boot[k]["ci95"]))
    lines.append(format_line("human-human (mean)", k_hh, boot["human-human (mean)"]["ci95"]))
    lines.append("")
    for k in hl_keys:
        lines.append(format_line(k, stats_[k]["kappa"], boot[k]["ci95"]))
    lines.append(format_line("LLM-human (mean)", k_hl, boot["LLM-human (mean)"]["ci95"]))

    dci = boot["difference"]["ci95"]
    lines.append(
        f"\n  Difference (mean human-human minus mean LLM-human) = {diff:6.3f}   "
        f"95% bootstrap CI [{dci[0]:6.3f}, {dci[1]:6.3f}]"
    )
    if margin is not None:
        ok = dci[1] < margin
        lines.append(
            f"  Non-inferiority, margin {margin}: upper bound {dci[1]:.3f} "
            f"{'<' if ok else '>='} {margin} -> {'PASSED' if ok else 'FAILED'}"
        )
    else:
        lines.append(
            "  (CI containing 0 = no detectable gap; it is NOT proof of equivalence. Use --margin.)"
        )

    lines.append("\nPer-pair details")
    for k, st in stats_.items():
        r1, r2 = k.split("-")
        lines.append(f"\n  {k}")
        lines.append(
            f"    observed agreement {st['observed_agreement']:.3f}   PABAK {st['pabak']:.3f}   "
            f"max attainable kappa {st['max_kappa']:.3f}"
        )
        lines.append(
            f"    analytic SE {st['analytic_se']:.3f}   analytic 95% CI "
            f"[{st['analytic_ci95'][0]:.3f}, {st['analytic_ci95'][1]:.3f}]"
        )
        lines.append(
            f"    positive agreement {st['positive_agreement']:.3f}   negative agreement {st['negative_agreement']:.3f}"
        )
        lines.append(
            f"    prevalence index {st['prevalence_index']:.3f}   bias index {st['bias_index']:.3f}   "
            f"positive rate {r1} {st['positive_rate_rater1']:.3f} vs {r2} {st['positive_rate_rater2']:.3f}"
        )
        t = st["table"]
        lines.append(
            f"    table (rows = {r1}, cols = {r2}):   0: [{t[0][0]:5d} {t[0][1]:5d}]   1: [{t[1][0]:5d} {t[1][1]:5d}]"
        )

    bad = {k: v["degenerate_resamples"] for k, v in boot.items() if v["degenerate_resamples"]}
    if bad:
        lines.append(
            f"\nWARNING: some resamples gave undefined kappa (excluded): {bad}. "
            "Interpret CIs cautiously."
        )

    return "\n".join(lines)
