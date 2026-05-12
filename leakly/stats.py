#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Helper functions for statistical analysis.

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from collections.abc import Sequence
import numpy as np
from statistics import mean, median, pstdev


def adjust_pvalues(p_values: Sequence[float], method: str) -> list[float]:
    """
    Adjust p-values for multiple testing.

    Args:
        p_values (Sequence[float]): List of p-values to adjust.
        method (str): Method for p-value adjustment. 
        Supported methods: "bonferroni", "fdr_bh"

    Returns:
        list[float]: List of adjusted p-values.
    """
    values = [float(value) for value in p_values]
    if not values:
        return []

    values_array = np.asarray(values, dtype=float)
    n_values = len(values)
    invalid_mask = (
        ~np.isfinite(values_array)
        | (values_array < 0.0)
        | (values_array > 1.0)
    )
    if np.any(invalid_mask):
        raise ValueError("p-values must be finite numbers between 0 and 1")
    

    if method == "bonferroni":
        return [min(value * n_values, 1.0) for value in values]
    
    elif method == "fdr_bh":
        sorted_indices = np.argsort(values_array)
        sorted_values = values_array[sorted_indices]
        adjusted = np.empty_like(sorted_values)
        cumulative_min = 1.0
        for i in reversed(range(n_values)):
            rank = i + 1
            adjusted_value = sorted_values[i] * n_values / rank
            cumulative_min = min(cumulative_min, adjusted_value)
            adjusted[i] = cumulative_min
        adjusted_pvalues = np.empty_like(adjusted)
        adjusted_pvalues[sorted_indices] = np.minimum(adjusted, 1.0)
        return adjusted_pvalues.tolist()
    
    else:
        raise ValueError(f"Unsupported correction method: {method}")