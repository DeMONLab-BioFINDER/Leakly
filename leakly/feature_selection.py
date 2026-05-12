#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Feature selection methods for Leakly.

This module currently supports one feature selector: ``LinearRegressionDAA``.
The public wrapper ``feature_selection_method`` dispatches to that method and
raises an error for any other requested feature-selection method.
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .config import FeatureSelectionConfig
from .stats import adjust_pvalues


ArrayLike = Any


@dataclass(slots=True)
class BiomarkerRecord:
    """DAA summary statistics for one feature."""

    feature_name: str
    feature_index: int
    effect_size: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None


class BaseDAAMethod(ABC):
    """Base class for Differential Abundance Analysis feature selectors."""

    def __init__(self, config: FeatureSelectionConfig | None = None) -> None:
        """Store method configuration, using defaults when none is provided."""
        self.config = config or FeatureSelectionConfig()

    @property
    def method_name(self) -> str:
        """Return the method name."""
        return self.__class__.__name__

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariates: ArrayLike | None = None,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> "BaseDAAMethod":
        """Fit the DAA method to the provided arrays."""

    @abstractmethod
    def select_features(self) -> tuple[list[str], list[int]]:
        """Return selected feature names and indices."""

    def run(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariates: ArrayLike | None = None,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> tuple[list[str], list[int]]:
        """
        Fit the method and return selected features.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target vector.
        covariates:
            Optional covariate matrix.
        feature_names:
            Optional feature names.
        covariate_names:
            Optional covariate names.

        Returns
        -------
        tuple[list[str], list[int]]
            Selected feature names and selected feature indices.
        """
        self.fit(X, y, covariates, feature_names, covariate_names)
        return self.select_features()


class LinearRegressionDAA(BaseDAAMethod):
    """Differential Abundance Analysis using linear regression."""

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariates: ArrayLike | None = None,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> "LinearRegressionDAA":
        """
        Fit linear-regression DAA.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target vector.
        covariates:
            Optional covariate matrix.
        feature_names:
            Optional feature names.
        covariate_names:
            Optional covariate names.

        Returns
        -------
        LinearRegressionDAA
            Fitted DAA method.
        """
        x, feature_names = _create_feature_matrix(X, feature_names)
        n_samples = x.shape[0]
        full_design, reduced_design, target_slice = _build_design_matrices(
            y, n_samples, covariates, covariate_names)

        if full_design.shape[0] != n_samples:
            raise ValueError("Design matrix row count does not match X")

        records: list[BiomarkerRecord] = []
        p_values: list[float] = []

        for feature_index, feature_name in enumerate(feature_names):
            response = x[:, feature_index]
            beta_full, sse_full, rank_full = _fit_ols(full_design, response)
            _, sse_reduced, rank_reduced = _fit_ols(reduced_design, response)

            df_num = rank_full - rank_reduced
            df_den = n_samples - rank_full
            p_value = _partial_f_pvalue(sse_full, sse_reduced, df_num, df_den)

            target_coefficients = beta_full[target_slice]
            if target_coefficients.size != 1:
                raise ValueError(
                    "Only binary or continuous targets are supported")
            effect_size = float(target_coefficients[0])

            records.append(
                BiomarkerRecord(
                    feature_name=feature_name,
                    feature_index=feature_index,
                    effect_size=effect_size,
                    p_value=p_value,
                )
            )
            p_values.append(p_value)

        adjusted = adjust_pvalues(p_values, method=self.config.correction_method)
        for record, adjusted_p_value in zip(records, adjusted):
            record.adjusted_p_value = adjusted_p_value

        self._records = records
        self._metadata = {
            "alpha": self.config.alpha,
            "correction_method": self.config.correction_method,
            "minimum_effect_size": self.config.minimum_effect_size,
            "n_samples": x.shape[0],
            "n_features": len(feature_names),
        }
        return self

    def select_features(self) -> tuple[list[str], list[int]]:
        """
        Select features based on fitted DAA records.

        Returns
        -------
        tuple[list[str], list[int]]
            Selected feature names and selected feature indices.
        """
        if not hasattr(self, "_records"):
            raise RuntimeError("DAA method has not been fitted")

        records = sorted(
            self._records,
            key=lambda record: (
                float("inf") if record.adjusted_p_value is None
                else record.adjusted_p_value,
                record.feature_index,
            ),
        )

        minimum_effect_size = self.config.minimum_effect_size
        if minimum_effect_size is not None:
            records = [
                record for record in records
                if record.effect_size is not None
                and abs(record.effect_size) >= minimum_effect_size
            ]

        significant = [
            record for record in records
            if record.adjusted_p_value is not None
            and record.adjusted_p_value <= self.config.alpha
        ]
        if significant:
            records = significant
        elif self.config.top_ranks is None:
            records = []

        if self.config.top_ranks is not None:
            records = records[: int(self.config.top_ranks)]

        return (
            [record.feature_name for record in records],
            [record.feature_index for record in records],
        )


def feature_selection_method(
    X: ArrayLike,
    y: ArrayLike,
    covariates: ArrayLike | None = None,
    config: FeatureSelectionConfig | None = None,
    feature_names: list[str] | None = None,
    covariate_names: list[str] | None = None,
) -> list[int]:
    """
    Run the configured feature selector.

    Only ``LinearRegressionDAA`` is supported. Any other method name raises
    ``ValueError``.

    Parameters
    ----------
    X:
        Training feature matrix.
    y:
        Training target vector.
    covariates:
        Optional training covariate matrix.
    config:
        Feature selection configuration.
    feature_names:
        Optional feature names.
    covariate_names:
        Optional covariate names.

    Returns
    -------
    list[int]
        Selected feature indices.
    """
    method_name, daa_config = _resolve_feature_selection_config(config)
    if method_name != "LinearRegressionDAA":
        raise ValueError(
            "Unsupported feature selection method: "
            f"{method_name!r}. Only 'LinearRegressionDAA' is supported."
        )

    _, selected_indices = LinearRegressionDAA(daa_config).run(
        X, y, covariates, feature_names, covariate_names)
    return selected_indices


def _resolve_feature_selection_config(
    config: FeatureSelectionConfig | None,
) -> tuple[str, FeatureSelectionConfig]:
    """Return the requested method name and feature-selection config."""
    if config is None:
        config = FeatureSelectionConfig()
    if not isinstance(config, FeatureSelectionConfig):
        raise TypeError("config must be a FeatureSelectionConfig")
    return config.method, config


def _infer_variable_type(variable_vec: ArrayLike) -> str:
    """Infer whether a vector should be treated as continuous or categorical."""
    variable_series = pd.Series(variable_vec)
    if (
        pd.api.types.is_bool_dtype(variable_series)
        or pd.api.types.is_categorical_dtype(variable_series)
        or isinstance(variable_series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(variable_series)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(variable_series):
        return "continuous"
    raise ValueError(f"Unsupported variable type: {variable_series.dtype}")


def _create_feature_matrix(
    X: ArrayLike,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return a numeric feature matrix and validated feature names."""
    if X is None:
        raise ValueError("X cannot be None")
    x = np.asarray(X, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains NaN or infinite values")

    names = list(feature_names or [])
    if not names:
        names = [f"feature_{index + 1}" for index in range(x.shape[1])]
    if len(names) != x.shape[1]:
        raise ValueError(
            "feature_names length must match the number of columns in X")
    return x, names


def _encode_covariates(
    covariates: ArrayLike | None,
    n_samples: int,
    covariate_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Encode covariates as a numeric design matrix."""
    if covariates is None:
        return np.empty((n_samples, 0), dtype=float), []

    covariate_array = np.asarray(covariates)
    if covariate_array.ndim == 1:
        covariate_array = covariate_array.reshape(-1, 1)
    if covariate_array.ndim != 2:
        raise ValueError("covariates must be one- or two-dimensional")
    if covariate_array.shape[0] != n_samples:
        raise ValueError("covariates row count must match X")

    names = list(covariate_names or [
        f"covariate_{index + 1}"
        for index in range(covariate_array.shape[1])
    ])
    if len(names) != covariate_array.shape[1]:
        raise ValueError(
            "covariate_names length must match the number of covariate columns")

    encoded_columns: list[np.ndarray] = []
    encoded_names: list[str] = []
    for column_index, covariate_name in enumerate(names):
        covariate_vector = covariate_array[:, column_index]
        if pd.isna(covariate_vector).any():
            raise ValueError("covariates contain missing values")

        variable_type = _infer_variable_type(covariate_vector)
        if variable_type == "continuous":
            column = np.asarray(covariate_vector, dtype=float).reshape(-1, 1)
            if not np.all(np.isfinite(column)):
                raise ValueError("covariates contain infinite values")
            encoded_columns.append(column)
            encoded_names.append(covariate_name)
        else:
            dummies = pd.get_dummies(
                covariate_vector, prefix=covariate_name, drop_first=True,
                dtype=float)
            if dummies.shape[1] > 0:
                encoded_columns.append(dummies.to_numpy(dtype=float))
                encoded_names.extend(dummies.columns.tolist())

    if not encoded_columns:
        return np.empty((n_samples, 0), dtype=float), []
    return np.hstack(encoded_columns), encoded_names


def _encode_target(vector: ArrayLike, length: int, name: str) -> np.ndarray:
    """Encode a binary categorical or continuous target as a numeric column."""
    if vector is None:
        raise ValueError(f"{name} cannot be None")
    target = np.asarray(vector).reshape(-1)
    if target.size != length:
        raise ValueError(f"{name} length must match X row count")
    if pd.isna(target).any():
        raise ValueError(f"{name} contains missing values")

    variable_type = _infer_variable_type(target)
    if variable_type == "continuous":
        encoded = np.asarray(target, dtype=float).reshape(-1, 1)
        if not np.all(np.isfinite(encoded)):
            raise ValueError(f"{name} contains infinite values")
        return encoded

    dummies = pd.get_dummies(target, drop_first=True, dtype=float)
    if dummies.shape[1] != 1:
        categories = pd.unique(target).tolist()
        raise ValueError(
            f"{name} has {len(categories)} categories {categories}; "
            "only binary categorical targets are supported"
        )
    return dummies.to_numpy(dtype=float)


def _build_design_matrices(
    y: ArrayLike,
    n_samples: int,
    covariates: ArrayLike | None = None,
    covariate_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, slice]:
    """Build full and reduced OLS design matrices."""
    target_matrix = _encode_target(y, length=n_samples, name="y")
    covariate_matrix, _ = _encode_covariates(
        covariates, n_samples, covariate_names)

    intercept = np.ones((n_samples, 1), dtype=float)
    reduced = np.hstack([intercept, covariate_matrix])
    full = np.hstack([intercept, target_matrix, covariate_matrix])
    target_slice = slice(1, 1 + target_matrix.shape[1])
    return full, reduced, target_slice


def _fit_ols(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Fit OLS and return coefficients, SSE, and design rank."""
    beta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ beta
    sse = float(np.sum(residuals**2))
    return beta, sse, int(rank)


def _partial_f_pvalue(
    sse_full: float,
    sse_reduced: float,
    df_num: int,
    df_den: int,
) -> float:
    """Return the partial F-test p-value for nested OLS models."""
    if df_num <= 0 or df_den <= 0:
        return 1.0

    numerator = max(sse_reduced - sse_full, 0.0) / df_num
    denominator = sse_full / df_den
    if denominator <= 0.0:
        f_statistic = float("inf") if numerator > 0.0 else 0.0
    else:
        f_statistic = numerator / denominator

    p_value = float(scipy_stats.f.sf(f_statistic, df_num, df_den))
    return p_value if isfinite(p_value) else 1.0
