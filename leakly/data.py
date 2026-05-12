#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Data utilities for Leakly.

The package keeps data handling array-based: ``X`` for features, ``y`` for the
target, and optional ``covariates`` for confounding variables.
'''
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_array, check_X_y, column_or_1d
from .config import DataProcConfig, SplitConfig


ArrayLike = Any


def validate_data(
    X: ArrayLike,
    y: ArrayLike,
    covariates: ArrayLike | None = None,
) -> None:
    """
    Validate feature, target, and optional covariate arrays.
    """
    x, _ = check_X_y(
        X,
        y,
        dtype=float,
        ensure_all_finite="allow-nan",
    )
    if covariates is None:
        return

    covariate_array = np.asarray(covariates)
    if covariate_array.ndim == 1:
        covariate_array = covariate_array.reshape(-1, 1)
    if covariate_array.ndim != 2:
        raise ValueError("covariates must be one- or two-dimensional")
    if covariate_array.shape[0] != x.shape[0]:
        raise ValueError("covariates and X must have the same row count")
    if pd.isna(covariate_array).any():
        raise ValueError("covariates contain missing values")


def data_split(
    X: ArrayLike,
    y: ArrayLike,
    covariates: ArrayLike | None = None,
    config: SplitConfig | None = None,
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike | None,
    ArrayLike | None,
]:
    """
    Split ``X``, ``y``, and optional ``covariates`` into train/test partitions.
    """
    if isinstance(covariates, SplitConfig) and config is None:
        config = covariates
        covariates = None

    config = config or SplitConfig()
    if config.method == "predefined":
        raise ValueError("predefined splits require explicit indices")
    if config.method != "train_test":
        raise ValueError(f"Unsupported split method: {config.method}")
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    validate_data(X, y, covariates)

    split_kwargs = {
        "test_size": config.test_fraction,
        "random_state": config.random_state,
        "stratify": column_or_1d(y) if config.stratify else None,
    }
    if covariates is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, **split_kwargs)
        return X_train, X_test, y_train, y_test, None, None

    return train_test_split(X, y, covariates, **split_kwargs)


def subset_rows(values: ArrayLike | None, indices: ArrayLike) -> ArrayLike | None:
    """
    Select rows from numpy arrays, pandas objects, or simple sequences.
    """
    if values is None:
        return None
    row_indices = np.asarray(indices, dtype=int)
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.iloc[row_indices]
    return np.asarray(values)[row_indices]


def fit_imputer(
    X_train: ArrayLike,
    config: DataProcConfig | None = None,
) -> Any:
    """
    Fit an sklearn imputer on training features only.
    """
    config = config or DataProcConfig()
    method = config.imputation_method
    if method is None or method == "none":
        return None
    if method == "knn":
        imputer = KNNImputer(n_neighbors=config.n_neighbors)
    elif method in {"mean", "median", "most_frequent"}:
        imputer = SimpleImputer(strategy=method)
    else:
        raise ValueError(f"Unsupported imputation method: {method}")
    return imputer.fit(
        check_array(X_train, dtype=float, ensure_all_finite="allow-nan"))


def transform_imputer(imputer: Any, X: ArrayLike) -> np.ndarray:
    """
    Apply a fitted sklearn imputer.
    """
    x = check_array(X, dtype=float, ensure_all_finite="allow-nan")
    return x if imputer is None else imputer.transform(x)


def fit_normalizer(
    X_train: ArrayLike,
    config: DataProcConfig | None = None,
) -> Any:
    """
    Fit an sklearn scaler on training features only.
    """
    config = config or DataProcConfig()
    method = config.normalization_method
    if method is None or method == "none":
        return None
    if method == "zscore":
        normalizer = StandardScaler(
            with_mean=config.with_mean,
            with_std=config.with_std,
        )
    elif method == "minmax":
        normalizer = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return normalizer.fit(
        check_array(X_train, dtype=float, ensure_all_finite="allow-nan"))


def transform_normalizer(normalizer: Any, X: ArrayLike) -> np.ndarray:
    """
    Apply a fitted sklearn scaler.
    """
    x = check_array(X, dtype=float, ensure_all_finite="allow-nan")
    return x if normalizer is None else normalizer.transform(x)
