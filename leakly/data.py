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

from .config import ImputationConfig, NormalizationConfig, SplitConfig


ArrayLike = Any


def validate_arrays(
    X: ArrayLike,
    y: ArrayLike,
    covariates: ArrayLike | None = None,
) -> None:
    """
    Validate feature, target, and covariate arrays.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target vector.
    covariates:
        Optional covariate matrix.

    Returns
    -------
    None
    """
    x = np.asarray(X)
    y_array = np.asarray(y)
    if x.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix")
    if y_array.ndim != 1:
        y_array = y_array.reshape(-1)
    if x.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    numeric_x = x.astype(float)
    if np.any(np.isinf(numeric_x)):
        raise ValueError("X contains infinite values")
    if pd.isna(y_array).any():
        raise ValueError("y contains missing values")

    if covariates is not None:
        covariate_array = np.asarray(covariates)
        if covariate_array.ndim == 1:
            covariate_array = covariate_array.reshape(-1, 1)
        if covariate_array.ndim != 2:
            raise ValueError("covariates must be one- or two-dimensional")
        if covariate_array.shape[0] != x.shape[0]:
            raise ValueError("covariates and X must have the same row count")
        if pd.isna(covariate_array).any():
            raise ValueError("covariates contain missing values")


def train_test_split_indices(
    y: ArrayLike,
    config: SplitConfig,
) -> tuple[list[int], list[int]]:
    """
    Create train/test sample indices.

    Parameters
    ----------
    y:
        Target vector used for optional stratification.
    config:
        Split configuration.

    Returns
    -------
    tuple[list[int], list[int]]
        Training indices and test indices.
    """
    y_array = np.asarray(y).reshape(-1)
    if y_array.size < 2:
        raise ValueError("At least two samples are required for a train/test split")
    if config.method == "predefined":
        raise ValueError("predefined splits require explicit indices")
    if config.method != "train_test":
        raise ValueError(f"Unsupported split method: {config.method}")
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    indices = np.arange(y_array.size)
    stratify = y_array if _can_stratify(y_array, config.test_fraction) else None
    if not config.stratify:
        stratify = None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_fraction,
        random_state=config.random_state,
        stratify=stratify,
    )
    return train_idx.tolist(), test_idx.tolist()


def subset_rows(X: ArrayLike, indices: list[int]) -> ArrayLike:
    """
    Select rows from an array-like object.

    Parameters
    ----------
    X:
        Array-like object.
    indices:
        Row indices to select.

    Returns
    -------
    ArrayLike
        Row subset.
    """
    if X is None:
        return None
    return np.asarray(X)[np.asarray(indices, dtype=int)]


def fit_imputer(
    X_train: ArrayLike,
    config: ImputationConfig,
) -> Any:
    """
    Fit an imputer on training features only.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    config:
        Imputation configuration.

    Returns
    -------
    Any
        Fitted imputer object.
    """
    config = config or ImputationConfig()
    method = config.method
    if method == "none":
        return None
    if method == "knn":
        imputer = KNNImputer(n_neighbors=config.n_neighbors)
    elif method in {"mean", "median", "most_frequent"}:
        imputer = SimpleImputer(strategy=method)
    else:
        raise ValueError(f"Unsupported imputation method: {method}")
    return imputer.fit(np.asarray(X_train, dtype=float))


def transform_imputer(imputer: Any, X: ArrayLike) -> ArrayLike:
    """
    Apply a fitted imputer to a feature matrix.

    Parameters
    ----------
    imputer:
        Fitted imputer object.
    X:
        Feature matrix to transform.

    Returns
    -------
    ArrayLike
        Imputed feature matrix.
    """
    x = np.asarray(X, dtype=float)
    if imputer is None:
        return x
    return imputer.transform(x)


def fit_normalizer(
    X_train: ArrayLike,
    config: NormalizationConfig,
) -> Any:
    """
    Fit a normalizer on training features only.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    config:
        Normalization configuration.

    Returns
    -------
    Any
        Fitted normalizer object.
    """
    config = config or NormalizationConfig()
    method = config.method
    if method == "none":
        return None
    if method == "zscore":
        normalizer = StandardScaler(
            with_mean=config.with_mean, with_std=config.with_std)
    elif method == "minmax":
        normalizer = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return normalizer.fit(np.asarray(X_train, dtype=float))


def transform_normalizer(normalizer: Any, X: ArrayLike) -> ArrayLike:
    """
    Apply a fitted normalizer to a feature matrix.

    Parameters
    ----------
    normalizer:
        Fitted normalizer object.
    X:
        Feature matrix to transform.

    Returns
    -------
    ArrayLike
        Normalized feature matrix.
    """
    x = np.asarray(X, dtype=float)
    if normalizer is None:
        return x
    return normalizer.transform(x)


def encode_covariates(
    covariates: ArrayLike | None,
    covariate_names: list[str] | None = None,
) -> tuple[ArrayLike | None, list[str] | None]:
    """
    Encode covariates for downstream modeling.

    Parameters
    ----------
    covariates:
        Optional raw covariate matrix.
    covariate_names:
        Optional covariate names.

    Returns
    -------
    tuple[ArrayLike | None, list[str] | None]
        Encoded covariates and output covariate names.
    """
    if covariates is None:
        return None, None
    frame = _covariates_to_frame(covariates, covariate_names)
    categorical_columns = [
        column for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        )
    ]
    encoded = pd.get_dummies(
        frame, columns=categorical_columns, drop_first=True, dtype=float)
    return encoded.to_numpy(dtype=float), encoded.columns.tolist()


def combine_features_and_covariates(
    X: ArrayLike,
    covariates: ArrayLike | None = None,
) -> ArrayLike:
    """
    Combine feature and covariate matrices when a model should use both.

    Parameters
    ----------
    X:
        Feature matrix.
    covariates:
        Optional covariate matrix.

    Returns
    -------
    ArrayLike
        Combined model matrix.
    """
    x = np.asarray(X, dtype=float)
    if covariates is None:
        return x
    covariate_array = np.asarray(covariates, dtype=float)
    if covariate_array.ndim == 1:
        covariate_array = covariate_array.reshape(-1, 1)
    if covariate_array.shape[0] != x.shape[0]:
        raise ValueError("X and covariates must have the same row count")
    if covariate_array.shape[1] == 0:
        return x
    return np.hstack([x, covariate_array])


def _can_stratify(y: np.ndarray, test_fraction: float) -> bool:
    """Return whether sklearn stratification is feasible for this target."""
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        return False
    n_test = int(np.ceil(y.size * test_fraction))
    n_train = y.size - n_test
    return counts.min() >= 2 and n_test >= classes.size and n_train >= classes.size


def _covariates_to_frame(
    covariates: ArrayLike,
    covariate_names: list[str] | None = None,
) -> pd.DataFrame:
    """Convert covariates into a named pandas DataFrame."""
    array = np.asarray(covariates)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("covariates must be one- or two-dimensional")
    if covariate_names is None:
        covariate_names = [
            f"covariate_{index + 1}" for index in range(array.shape[1])]
    if len(covariate_names) != array.shape[1]:
        raise ValueError(
            "covariate_names length must match the number of covariate columns")
    frame = pd.DataFrame(array, columns=covariate_names)
    if frame.isna().any().any():
        raise ValueError("covariates contain missing values")
    return frame
