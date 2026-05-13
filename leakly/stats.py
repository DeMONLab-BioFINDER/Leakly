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


def permutation_test(
    value2compare: float,
    permuted_values: Sequence[float],
    alternative: str = "less",
) -> float:
    """
    Compute a permutation-test p-value for an observed value.

    Args:
        value2compare (float): Value to compare against permuted values.
        permuted_values (Sequence[float]): Permuted values for comparison.
        alternative (str, optional): 
            Alternative hypothesis direction. Defaults to "less".

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        float: _description_
    """
    observed = float(value2compare)
    permuted = np.asarray(list(permuted_values), dtype=float)
    if permuted.size == 0:
        raise ValueError("permuted_values must contain at least one value")
    if not np.isfinite(observed) or np.any(~np.isfinite(permuted)):
        raise ValueError("Values must be finite")

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
