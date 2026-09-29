"""Alternative Annotator Test (alt-test) for binary LLM annotations.

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

from typing import Sequence, TypedDict

import numpy as np
import numpy.typing as npt
from scipy import stats


class AnnotatorResult(TypedDict):
    """Statistical outcomes for a single evaluated annotator."""

    rho_f: float
    rho_h: float
    d_mean: float
    t: float
    p: float
    rejected: bool


def by_procedure(
    p_values: Sequence[float] | npt.NDArray[np.floating],
    q: float,
) -> list[int]:
    """Control False Discovery Rate under arbitrary dependency using Benjamini-Yekutieli.

    Applies the step-up procedure to identify hypotheses where null can be rejected
    at FDR level q.

    Args:
        p_values: Collection of p-values to evaluate.
        q: Target False Discovery Rate bound in (0, 1).

    Returns:
        Sorted list of original indices corresponding to rejected null hypotheses.

    References:
        Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery
        rate in multiple testing under dependency. The Annals of Statistics, 29(4),
        1165-1188.
    """
    p: npt.NDArray[np.float64] = np.asarray(p_values, dtype=np.float64)
    m: int = len(p)
    if m == 0:
        return []

    order: npt.NDArray[np.intp] = np.argsort(p)
    harmonic_sum: float = float(np.sum(1.0 / np.arange(1, m + 1)))
    thresholds: npt.NDArray[np.float64] = np.arange(1, m + 1) / m * q / harmonic_sum
    below: npt.NDArray[np.intp] = np.flatnonzero(p[order] <= thresholds)

    if below.size == 0:
        return []
    return sorted(order[: below[-1] + 1].tolist())


def one_sided_p(
    d: npt.NDArray[np.floating | np.integer],
    eps: float,
    w: npt.NDArray[np.floating] | None = None,
) -> tuple[float, float, float]:
    """Calculate one-sided one-sample t-test against margin epsilon.

    Tests H0: mean(d) >= eps against H1: mean(d) < eps. When w is provided,
    computes weighted sample statistics and Kish's effective sample size.
    Returns (0.0, mean, NaN) if sample standard deviation is zero and mean < eps,
    or (1.0, mean, NaN) if mean >= eps.

    Args:
        d: 1D array of pairwise score differences (W^h - W^f).
        eps: Equivalence margin threshold.
        w: Optional 1D non-negative array of inverse-probability sample weights.

    Returns:
        tuple containing:
            - p-value: Cumulative probability under the Student's t distribution.
            - sample_mean: Observed (weighted) mean difference.
            - t_statistic: Standardized t statistic, or NaN if variance is zero.
    """
    d_float: npt.NDArray[np.float64] = np.asarray(d, dtype=np.float64)

    if w is None:
        n_eff: float = float(len(d_float))
        mean: float = float(np.mean(d_float))
        sd: float = float(np.std(d_float, ddof=1))
    else:
        w_float: npt.NDArray[np.float64] = np.asarray(w, dtype=np.float64)
        sum_w: float = float(np.sum(w_float))
        sum_w_sq: float = float(np.sum(w_float**2))
        n_eff = (sum_w**2) / sum_w_sq
        mean = float(np.sum(w_float * d_float) / sum_w)
        sd = float(np.sqrt(np.sum(w_float * (d_float - mean) ** 2) / sum_w))

    if sd == 0.0:
        p_val: float = 0.0 if mean < eps else 1.0
        return p_val, mean, float("nan")

    t_stat: float = (mean - eps) / (sd / np.sqrt(n_eff))
    p_val = float(stats.t.cdf(t_stat, df=n_eff - 1))
    return p_val, mean, t_stat


def alt_test(
    llm: npt.NDArray[np.integer],
    H: npt.NDArray[np.integer],
    eps: float = 0.2,
    q: float = 0.05,
    ipw: bool = False,
    seed: int = 0,
) -> tuple[list[AnnotatorResult], float, float]:
    """Execute the Alternative Annotator Test across human annotators.

    For each human annotator, compares leave-one-out agreement of the LLM against
    the left-out annotator with respect to remaining annotators. Applies Benjamini-
    Yekutieli FDR control to determine annotators over whom the LLM statistically
    establishes superiority or non-inferiority.

    Args:
        llm: 1D array of binary annotations (0 or 1) produced by the LLM of shape (n_items,).
        H: 2D array of binary annotations produced by human raters of shape (n_items, n_annotators).
        eps: Cost-benefit tolerance margin. Defaults to 0.2.
        q: Target False Discovery Rate bound for Benjamini-Yekutieli. Defaults to 0.05.
        ipw: Whether to apply inverse-probability weighting based on leave-one-out majority vote.
        seed: Random seed used for resolving ties during majority vote in IPW.

    Returns:
        tuple containing:
            - rows: Detailed statistical results per evaluated human annotator.
            - omega: Fraction of rejected null hypotheses (winning rate).
            - rho: Average advantage probability of the LLM across all annotators.

    Raises:
        ValueError: If IPW is requested and any majority-vote partition degenerates to a single class.

    References:
        Calderon, N., Reichart, R., & Dror, R. (2025). The Alternative Annotator Test
        for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators
        with LLMs. In Proceedings of ACL 2025, pages 16051-16081.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    n_items, n_annotators = H.shape
    rows: list[AnnotatorResult] = []

    for j in range(n_annotators):
        rest = np.delete(H, j, axis=1)
        s_llm = (rest == llm[:, None]).mean(axis=1)
        s_hum = (rest == H[:, [j]]).mean(axis=1)
        wf = (s_llm >= s_hum).astype(np.float64)
        wh = (s_hum >= s_llm).astype(np.float64)
        d = wh - wf

        if ipw:
            ones = rest.sum(axis=1)
            zeros = rest.shape[1] - ones
            mv = np.where(ones > zeros, 1, np.where(zeros > ones, 0, rng.integers(0, 2, n_items)))
            counts = np.bincount(mv, minlength=2)
            if (counts == 0).any():
                raise ValueError(
                    f"--ipw impossible for annotator index {j}: leave-one-out majority vote has only one class."
                )
            pi = (n_items / counts)[mv]
            rho_f = float((pi * wf).sum() / pi.sum())
            rho_h = float((pi * wh).sum() / pi.sum())
            p, dbar, t = one_sided_p(d, eps, pi)
        else:
            rho_f = float(wf.mean())
            rho_h = float(wh.mean())
            p, dbar, t = one_sided_p(d, eps)

        rows.append({
            "rho_f": rho_f,
            "rho_h": rho_h,
            "d_mean": dbar,
            "t": t,
            "p": p,
            "rejected": False,
        })

    rejected_indices: list[int] = by_procedure([r["p"] for r in rows], q)
    for j, row in enumerate(rows):
        row["rejected"] = j in rejected_indices

    omega: float = len(rejected_indices) / n_annotators
    rho: float = float(np.mean([r["rho_f"] for r in rows]))
    return rows, omega, rho


def format_alt_test_report(
    ann_cols: Sequence[str],
    rows: Sequence[AnnotatorResult],
    omega: float,
    rho: float,
    llm: npt.NDArray[np.integer],
    H: npt.NDArray[np.integer],
    eps: float,
    q: float,
    omega_threshold: float,
    ipw: bool,
) -> str:
    """Generate a formatted terminal text summary of Alt-Test results.

    Args:
        ann_cols: Names of the human annotators.
        rows: Per-annotator statistical results.
        omega: Observed winning rate.
        rho: Observed average advantage probability.
        llm: 1D array of binary ratings from the LLM.
        H: 2D array of binary ratings from human annotators.
        eps: Equivalence margin used during testing.
        q: FDR significance level used during testing.
        omega_threshold: Minimum winning rate required to pass.
        ipw: Whether inverse-probability weighting was enabled.

    Returns:
        Multi-line string containing the formatted diagnostic report.
    """
    n, m = H.shape
    passed = omega >= omega_threshold
    lines: list[str] = [
        f"Items: {n}   Annotators: {m} ({', '.join(ann_cols)})",
        (
            f"epsilon = {eps}   FDR q = {q} (Benjamini-Yekutieli)   "
            f"pass if omega >= {omega_threshold}   IPW: {'on' if ipw else 'off'}"
        ),
        (
            f"LLM positive rate: {float(llm.mean()):.3f}   annotator positive rates: "
            + ", ".join(f"{c} {float(H[:, j].mean()):.3f}" for j, c in enumerate(ann_cols))
        ),
    ]

    if m < 3:
        lines.append("NOTE: with 2 annotators the paper considers the test less reliable (Appendix A).")
    if eps > 0.2:
        lines.append(
            f"NOTE: epsilon {eps} is above the paper's largest recommended value (0.2, experts); "
            "justify it explicitly."
        )

    lines.append("\n  excluded   rho_f (LLM)  rho_h (human)  mean d     t        p        H0 rejected")
    for c, r in zip(ann_cols, rows):
        lines.append(
            f"  {c:<9}  {r['rho_f']:.3f}        {r['rho_h']:.3f}          {r['d_mean']:+.3f}   "
            f"{r['t']:7.2f}  {r['p']:.2e}  {'yes (LLM wins)' if r['rejected'] else 'no'}"
        )

    lines.append(f"\nWinning rate omega = {omega:.3f}   Average advantage probability rho = {rho:.3f}")
    lines.append(
        f"RESULT: {'PASSED' if passed else 'FAILED'} "
        f"(omega {omega:.3f} {'>=' if passed else '<'} {omega_threshold})"
    )

    return "\n".join(lines)
