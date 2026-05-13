#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Feature selection.

Notes[2024-05-12]
-----
- Only supports linear regression differential abundance analysis for now.

Usage
-----
```python
from leakly import FeatureSelectionConfig, feature_selection

selected_features, selected_indices = feature_selection(
    X, y, covariates, 
    feature_names=feature_names,
    config=FeatureSelectionConfig(
        method="LinearRegressionDAA",
        alpha=0.05,
        correction_method="fdr_bh",
        minimum_effect_size=None,
        top_ranks=None,
    ),
)
```
Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any
ArrayLike = Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        """Fallback iterator when tqdm is not installed."""
        return iterable

from .config import FeatureSelectionConfig
from .stats import adjust_pvalues


def feature_selection(
    X: ArrayLike,
    y: ArrayLike,
    covariates: ArrayLike | None = None,
    config: FeatureSelectionConfig | None = None,
    feature_names: list[str] | None = None,
    covariate_names: list[str] | None = None,
    show_progress: bool = False,
) -> tuple[list[str], list[int]]:
    """
    Run the configured feature selector.

    Args:
        X (ArrayLike): 
            Feature matrix.
        y (ArrayLike): 
            Target vector.
        covariates (ArrayLike | None, optional): 
            Optional covariate matrix. Defaults to None.
        config (FeatureSelectionConfig | None, optional): 
            Feature selection configuration. Defaults to None.
        feature_names (list[str] | None, optional): 
            Optional feature names. Defaults to None.
        covariate_names (list[str] | None, optional): 
            Optional covariate names. Defaults to None.
        show_progress (bool, optional):
            Whether to show a progress bar. Defaults to False.

    Raises:
        ValueError: 
            If the specified feature selection method is not supported.

    Returns:
        tuple[list[str], list[int]]: 
            Selected feature names and their corresponding indices.
        tuple[list[str], list[int]]: _description_
    """
    method_name, selection_config = _resolve_feature_selection_config(config)
    if selection_config.selected_feature_names is not None:
        return _select_configured_feature_names(
            X,
            feature_names,
            selection_config.selected_feature_names,
        )

    if method_name != "LinearRegressionDAA":
        raise ValueError(
            "Unsupported feature selection method: "
            f"{method_name!r}. Only 'LinearRegressionDAA' is supported."
        )

    selected_features, selected_indices = LinearRegressionDAA(
        alpha=selection_config.alpha,
        correction_method=selection_config.correction_method,
        minimum_effect_size=selection_config.minimum_effect_size,
        top_ranks=selection_config.top_ranks,
        show_progress=show_progress,
    ).run(
        X, y, covariates, feature_names, covariate_names)
    return selected_features, selected_indices


### Internal helper functions and classes ###
@dataclass(slots=True)
class FeatureRecord:
    """DAA summary statistics for one feature."""

    feature_name: str
    feature_index: int
    effect_size: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None


class LinearRegressionDAA:
    """Differential Abundance Analysis using linear regression."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        correction_method: str = "fdr_bh",
        minimum_effect_size: float | None = 0.0,
        top_ranks: int | None = 10,
        show_progress: bool = False,
    ) -> None:
        """
        Initialize linear-regression DAA settings.

        Parameters
        ----------
        alpha:
            Significance threshold applied to adjusted p-values.
        correction_method:
            Multiple-testing correction method.
        minimum_effect_size:
            Optional minimum absolute effect size.
        top_ranks:
            Optional maximum number of selected features.
        show_progress:
            Whether to show a progress bar while fitting.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if minimum_effect_size is not None and minimum_effect_size < 0:
            raise ValueError("minimum_effect_size must be non-negative")
        if top_ranks is not None and top_ranks < 1:
            raise ValueError("top_ranks must be at least 1")

        self.alpha = alpha
        self.correction_method = correction_method
        self.minimum_effect_size = minimum_effect_size
        self.top_ranks = top_ranks
        self.show_progress = show_progress

    @property
    def method_name(self) -> str:
        """Return the method name."""
        return self.__class__.__name__

    def run(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariates: ArrayLike | None = None,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> tuple[list[str], list[int]]:
        """
        Run the configured feature selector.

        Args:
            X (ArrayLike): Feature matrix.
            y (ArrayLike): Target vector.
            covariates (ArrayLike | None, optional): 
                Optional covariate matrix. Defaults to None.
            feature_names (list[str] | None, optional): 
                Optional feature names. Defaults to None.
            covariate_names (list[str] | None, optional): 
                Optional covariate names. Defaults to None.

        Returns:
            tuple[list[str], list[int]]: 
                Selected feature names and their corresponding indices.
        """
        self.fit(X, y, covariates, feature_names, covariate_names)
        return self.select_features()


    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariates: ArrayLike | None = None,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> "LinearRegressionDAA":
        """
        Run linear regression DAA for feature selection.

        Args:
            X (ArrayLike): 
                Feature matrix.
            y (ArrayLike): 
                Target vector.
            covariates (ArrayLike | None, optional): 
                Optional covariate matrix. Defaults to None.
            feature_names (list[str] | None, optional): 
                Optional feature names. Defaults to None.
            covariate_names (list[str] | None, optional): 
                Optional covariate names. Defaults to None.

        Returns:
            LinearRegressionDAA: Fitted DAA object.
        """
        x, feature_names = _create_feature_matrix(X, feature_names)
        n_samples = x.shape[0]
        full_design, reduced_design, target_slice = _build_design_matrices(
            y, n_samples, covariates, covariate_names)

        if full_design.shape[0] != n_samples:
            raise ValueError("Design matrix row count does not match X")

        records: list[FeatureRecord] = []
        p_values: list[float] = []

        feature_iterator = tqdm(
            enumerate(feature_names),
            total=len(feature_names),
            desc="LinearRegressionDAA",
            unit="feature",
            disable=(not self.show_progress),
        )
        for feature_index, feature_name in feature_iterator:
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
                FeatureRecord(
                    feature_name=feature_name,
                    feature_index=feature_index,
                    effect_size=effect_size,
                    p_value=p_value,
                )
            )
            p_values.append(p_value)
        # adjust p-values for multiple testing and store in records
        adjusted = adjust_pvalues(
            p_values, method=self.correction_method)
        for record, adjusted_p_value in zip(records, adjusted):
            record.adjusted_p_value = adjusted_p_value

        self._records = records
        self._metadata = {
            "alpha": self.alpha,
            "correction_method": self.correction_method,
            "minimum_effect_size": self.minimum_effect_size,
            "n_samples": x.shape[0],
            "n_features": len(feature_names),
        }
        return self

    def select_features(self) -> tuple[list[str], list[int]]:
        """
        Select features based on fitted DAA records.

        Returns:
            tuple[list[str], list[int]]: 
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

        minimum_effect_size = self.minimum_effect_size
        if minimum_effect_size is not None:
            records = [
                record for record in records
                if record.effect_size is not None
                and abs(record.effect_size) >= minimum_effect_size
            ]

        significant = [
            record for record in records
            if record.adjusted_p_value is not None
            and record.adjusted_p_value <= self.alpha
        ]
        if significant:
            records = significant
        elif self.top_ranks is None:
            records = []

        if self.top_ranks is not None:
            records = records[: int(self.top_ranks)]

        return (
            [record.feature_name for record in records],
            [record.feature_index for record in records],
        )


def _resolve_feature_selection_config(
    config: FeatureSelectionConfig | None,
) -> tuple[str, FeatureSelectionConfig]:
    """
    Resolve the feature selection method and configuration.

    Args:
        config (FeatureSelectionConfig | None): 
            The feature selection configuration.

    Returns:
        tuple[str, FeatureSelectionConfig]: 
            The resolved method name and configuration.
    """

    if config is None:
        config = FeatureSelectionConfig()
    if not isinstance(config, FeatureSelectionConfig):
        raise TypeError("config must be a FeatureSelectionConfig")
    return config.method, config


def _select_configured_feature_names(
    X: ArrayLike,
    feature_names: list[str] | None,
    selected_feature_names: list[str],
) -> tuple[list[str], list[int]]:
    """
    Resolve an explicit user-supplied feature list to feature indices.
    """
    _, names = _create_feature_matrix(X, feature_names)
    name_to_index = {name: index for index, name in enumerate(names)}
    if len(name_to_index) != len(names):
        raise ValueError("feature_names must be unique")

    selected_names = list(selected_feature_names)
    missing = [name for name in selected_names if name not in name_to_index]
    if missing:
        raise ValueError(
            "selected_feature_names contains unknown features: "
            f"{missing}"
        )
    return selected_names, [name_to_index[name] for name in selected_names]


def _infer_variable_type(variable_vec: ArrayLike) -> str:
    """
    Infer whether a vector should be treated as continuous or categorical.

    Args:
        variable_vec (ArrayLike): The variable vector to infer the type of.

    Returns:
        str: The inferred variable type, either "continuous" or "categorical".
    """
    variable_series = pd.Series(variable_vec)
    if (
        pd.api.types.is_bool_dtype(variable_series)
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
    """
    Create a numeric feature matrix.

    Args:
        X (ArrayLike): The feature matrix.
        feature_names (list[str] | None, optional): 
            The feature names. Defaults to None.

    Returns:
        tuple[np.ndarray, list[str]]: 
            The numeric feature matrix and validated feature names.
    """
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
    """
    Encode covariates as a numeric design matrix.

    Args:
        covariates (ArrayLike | None): T
            The covariates to encode.
        n_samples (int): 
            The number of samples.
        covariate_names (list[str] | None, optional): 
            The names of the covariates. Defaults to None.

    Returns:
        tuple[np.ndarray, list[str]]: 
            The encoded covariates and their names.
    """
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
            "covariate_names length NOT match the number of covariate columns")

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
            column = StandardScaler().fit_transform(column)
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


def _encode_target(vector: ArrayLike, 
                   length: int, 
                   name: str) -> np.ndarray:
    """
    Encode a binary categorical or continuous target as a numeric column.

    Args:
        vector (ArrayLike): The target vector to encode.
        length (int): The number of samples.
        name (str): The name of the target.

    Returns:
        np.ndarray: The encoded target.
    """
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
    """
    Build full and reduced OLS design matrices.

    Args:
        y (ArrayLike): The target vector.
        n_samples (int): The number of samples.
        covariates (ArrayLike | None, optional): 
            The covariates matrix. Defaults to None.
        covariate_names (list[str] | None, optional): 
            The names of the covariates. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, slice]: 
            The full and reduced design matrices and 
            the slice for the target variables.
    """
    target_matrix = _encode_target(y, length=n_samples, name="y")
    covariate_matrix, _ = _encode_covariates(
        covariates, n_samples, covariate_names)

    intercept = np.ones((n_samples, 1), dtype=float)
    reduced = np.hstack([intercept, covariate_matrix])
    full = np.hstack([intercept, target_matrix, covariate_matrix])
    target_slice = slice(1, 1 + target_matrix.shape[1])
    return full, reduced, target_slice


def _fit_ols(design: np.ndarray, 
             response: np.ndarray) -> tuple[np.ndarray, float, int]:
    """
    Fit OLS and return coefficients, SSE, and design rank.

    Args:
        design (np.ndarray): The design matrix.
        response (np.ndarray): The response vector.

    Returns:
        tuple[np.ndarray, float, int]: 
            The fitted coefficients, SSE, and design rank.
    """
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
    """
    Return the partial F-test p-value for nested OLS models.

    Args:
        sse_full (float): The SSE of the full model.
        sse_reduced (float): The SSE of the reduced model.
        df_num (int): The degrees of freedom for the numerator.
        df_den (int): The degrees of freedom for the denominator.

    Returns:
        float: The partial F-test p-value.
    """
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
