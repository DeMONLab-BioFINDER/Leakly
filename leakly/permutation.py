#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Permutation utilities for leakage checks.

Usage
-----
```python
from leakly import permute_label

permuted_y = permute_label(
    y, 
    perc_permutation=0.1, 
    random_state=42)
```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''

from __future__ import annotations
from typing import Any
import numpy as np


def permute_label(
    y: Any,
    perc_permutation: float = 1.0,
    random_state: int | None = None,
) -> Any:
    """
    Randomly shuffle a percentage of target labels.

    Args:
        y (Any): Target labels. 
            Supports numpy arrays, lists, tuples, pandas Series,
            and one-column pandas DataFrames.
        perc_permutation (float, optional): 
            Fraction of samples whose labels are shuffled. Defaults to 1.0.
        random_state (int | None, optional): 
            Optional seed for reproducible permutations. Defaults to None.

    Returns:
        Any: Permuted target labels.
    """
    percentage = _validate_percentage(perc_permutation)
    y_array = np.asarray(y)
    original_shape = y_array.shape

    if y_array.size == 0:
        raise ValueError("y must contain at least one label")
    if y_array.ndim == 0:
        raise ValueError("y must be one-dimensional or a one-column array")
    if y_array.ndim > 2:
        raise ValueError("y must be one-dimensional or a one-column array")
    if y_array.ndim == 2 and 1 not in y_array.shape:
        raise ValueError("two-dimensional y must have exactly one column")

    flat_y = y_array.reshape(-1)
    n_samples = flat_y.size
    n_permuted = int(round(n_samples * percentage))

    permuted = flat_y.copy()
    if n_permuted > 0:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(n_samples, size=n_permuted, replace=False)
        permuted[indices] = rng.permutation(permuted[indices])

    return _restore_label_type(y, permuted.reshape(original_shape))


def _validate_percentage(perc_permutation: float) -> float:
    """
    Validate the percentage of samples to permute.

    Args:
        perc_permutation (float): The percentage of samples to permute.

    Returns:
        float: The validated percentage.
    """
    try:
        percentage = float(perc_permutation)
    except (TypeError, ValueError) as exc:
        raise ValueError("perc_permutation must be a number") from exc
    if not 0.0 <= percentage <= 1.0:
        raise ValueError("perc_permutation must be between 0 and 1")
    return percentage


def _restore_label_type(original_y: Any, y_array: np.ndarray) -> Any:
    """
    Restore the type of the permuted labels to match the original input.

    Args:
        original_y (Any): The original target labels.
        y_array (np.ndarray): The permuted labels as a numpy array.

    Returns:
        Any: The permuted labels in the same type as the original input.
    """
    if isinstance(original_y, np.ndarray):
        return y_array
    if isinstance(original_y, list):
        return y_array.reshape(-1).tolist()
    if isinstance(original_y, tuple):
        return tuple(y_array.reshape(-1).tolist())
    if hasattr(original_y, "iloc") and hasattr(original_y, "copy"):
        restored = original_y.copy()
        if getattr(restored, "ndim", 1) == 1:
            restored.iloc[:] = y_array.reshape(-1)
        else:
            restored.iloc[:, 0] = y_array.reshape(-1)
        return restored
    return y_array
