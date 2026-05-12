#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Statistical helper functions for Leakly.
'''
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def adjust_pvalues(
    p_values: Sequence[float],
    method: str = "fdr_bh",
) -> list[float]:
    """
    Adjust p-values for multiple comparisons.

    Parameters
    ----------
    p_values:
        Raw p-values.
    method:
        Correction method, such as ``"bonferroni"`` or ``"fdr_bh"``.

    Returns
    -------
    list[float]
        Adjusted p-values in the original input order.
    """
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p-values must be finite values between 0 and 1")

    method = method.lower()
    n_values = values.size
    if method == "bonferroni":
        return np.minimum(values * n_values, 1.0).tolist()
    if method == "fdr_bh":
        order = np.argsort(values)
        sorted_values = values[order]
        adjusted_sorted = np.empty_like(sorted_values)
        cumulative_min = 1.0
        for index in range(n_values - 1, -1, -1):
            rank = index + 1
            adjusted_value = sorted_values[index] * n_values / rank
            cumulative_min = min(cumulative_min, adjusted_value)
            adjusted_sorted[index] = cumulative_min
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = np.minimum(adjusted_sorted, 1.0)
        return adjusted.tolist()
    raise ValueError(f"Unsupported p-value adjustment method: {method}")


def permutation_test_auc(
    observed_auc: float,
    permuted_aucs: Sequence[float],
    alternative: str = "greater",
) -> float:
    """
    Compute a permutation-test p-value for an observed AUC.

    Parameters
    ----------
    observed_auc:
        AUC from the unpermuted pipeline.
    permuted_aucs:
        AUC values from permuted-label pipeline runs.
    alternative:
        Alternative hypothesis direction.

    Returns
    -------
    float
        Permutation-test p-value.
    """
    observed = float(observed_auc)
    permuted = np.asarray(list(permuted_aucs), dtype=float)
    if permuted.size == 0:
        raise ValueError("permuted_aucs must contain at least one value")
    if not np.isfinite(observed) or np.any(~np.isfinite(permuted)):
        raise ValueError("AUC values must be finite")

    alternative = alternative.lower()
    if alternative == "greater":
        count = np.sum(permuted >= observed)
    elif alternative == "less":
        count = np.sum(permuted <= observed)
    elif alternative == "two-sided":
        center = float(np.mean(permuted))
        count = np.sum(np.abs(permuted - center) >= abs(observed - center))
    else:
        raise ValueError(
            "alternative must be 'greater', 'less', or 'two-sided'")
    return float((count + 1) / (permuted.size + 1))


def summarize_scores(scores: Sequence[float]) -> dict[str, float]:
    """
    Summarize a sequence of evaluation scores.

    Parameters
    ----------
    scores:
        Evaluation scores.

    Returns
    -------
    dict[str, float]
        Summary statistics such as mean, median, and standard deviation.
    """
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("scores must contain at least one value")
    if np.any(~np.isfinite(values)):
        raise ValueError("scores must be finite")
    return {
        "n": float(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
