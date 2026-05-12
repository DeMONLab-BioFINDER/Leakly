#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Feature selection using Differential Abundance Analysis (DAA).

Notes[2024-05-11]
-----
- Current implementation only supports linear regression.

Usage
-----
```python
from leakly import DAAConfig
from leakly import Dataset
from leakly import LinearRegressionDAA

# create a dataset
data = Dataset(
    X=[[1.0, 2.0], [1.5, 1.8], [0.5, 2.2]],
    y=[0, 1, 0],
    feature_names=["feature1", "feature2"],
    covariates=[[25, 0], [30, 1], [22, 0]],
    covariate_names=["age", "sex"]
)

# create a DAA method with default configuration
config = DAAConfig(alpha=0.05, 
                   correction_method="fdr_bh", 
                   min_effect_size=0.0)
daa_method = LinearRegressionDAA(config)

# run DAA to select biomarkers
selected_features, selected_indices = daa_method.run(data)
```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from math import isfinite

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Any
from .config import DAAConfig
from .data import Dataset
from .stats import adjust_pvalues


class BaseDAAMethod(ABC):
    """
    Base class for Differential Abundance Analysis (DAA).

    Args:
        ABC (_type_): _description_
    """
    def __init__(self, config: DAAConfig | None = None) -> None:
        """
        Store method configuration, using defaults when none is provided.
        """
        self.config = config

    @property
    def method_name(self) -> str:
        """Return the method name."""

        return self.__class__.__name__

    @abstractmethod
    def fit(self, data: Dataset) -> BaseDAAMethod:
        """Fit the DAA method to the provided data."""

    @abstractmethod
    def select_biomarkers(self) -> List:
        """Return selected biomarkers and their summary statistics."""


    def run(self, data: Dataset) -> List:
        """Fit the method and return the DAA result."""

        self.fit(data)
        return self.select_biomarkers()


def _infer_variable_types(variable_vec) -> str:
    """
    Infer the types of variables in a numpy vector.

    Args:
        variable_vec (_type_): _description_
    """
    # covert to pandas Series for easier type inference
    variable_series = pd.Series(variable_vec)
    if pd.api.types.is_numeric_dtype(variable_series):
        return "continuous"
    elif pd.api.types.is_bool_dtype(variable_series):
        return "categorical"
    elif pd.api.types.is_categorical_dtype(variable_series):
        return "categorical"
    elif isinstance(variable_series.dtype, pd.CategoricalDtype):
        return "categorical"
    else:
        raise ValueError(
            "Unsupported variable type: {}".format(variable_series.dtype))


def _create_feature_matrix(data: Dataset) -> tuple[np.ndarray, list[str]]:
    """
    Return a numeric feature matrix and validated feature names.
    
    Args:
        data (Dataset): The input dataset class.
    
    Returns:
        tuple[np.ndarray, list[str]]: 
        A tuple containing the feature matrix and the feature names.

    """
    if data.X is None:
        raise ValueError("No available data.X in the input dataset!")
    x = np.asarray(data.X, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("data.X must be a two-dimensional matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("data.X contains NaN or infinite values")

    feature_names = list(data.feature_names)
    if not feature_names:
        feature_names = [
            f"feature_{index + 1}" for index in range(x.shape[1])]
    if len(feature_names) != x.shape[1]:
        raise ValueError(
            "feature_names length must match the number of columns in X")

    return x, feature_names


def _encode_covariates(data: Dataset) -> tuple[np.ndarray, list[str]]:
    """
    Encode covariates into a numeric matrix and return the covariate names.

    Args:
        data (Dataset): The input dataset containing covariates.

    Returns:
        tuple[np.ndarray, list[str]]: 
        A tuple containing the covariate matrix and the covariate names.
    """
    # if no covariates, return an empty matrix and an empty list of names
    if data.covariates is None:
        return np.empty((len(data.X), 0), dtype=float), []
    
    if not np.all(np.isfinite(data.covariates)):
        raise ValueError("data.covariates contains NaN or infinite values")
    
    if data.covariate_names is None:
        raise ValueError("Covariate names not available!")
    
    covariate_names = list(data.covariate_names)

    for i, _ in enumerate(covariate_names):
        covariate_vec = data.covariates[:, i]
        var_type = _infer_variable_types(covariate_vec)
        if var_type == "continuous":
            # already numeric, just ensure it's a 2D array
            covariate_vec = np.asarray(
                covariate_vec, dtype=float).reshape(-1, 1)
        elif var_type == "categorical":
            # use one-hot encoding for categorical variables
            covariate_vec = pd.get_dummies(
                covariate_vec, drop_first=True).values
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")
        
        if i == 0:
            covariate_matrix = covariate_vec
        else:
            covariate_matrix = np.hstack((covariate_matrix, covariate_vec))
    
    return covariate_matrix, covariate_names


def _encode_target(vector: np.ndarray, length: int, name: str) -> np.ndarray:
    """
    Encode the target variable into a numeric matrix.

    Args:
        vector (np.ndarray): The target variable vector.
        length (int): The length of the target variable.
        name (str): The name of the target variable.

    Returns:
        np.ndarray: The encoded target variable matrix.
    """
    if name is None:
        raise ValueError(f"{name} cannot be None")
    assert len(vector) == length, \
        f"Length of {name} NOT match the number of samples"
    assert np.sum(np.isnan(vector)) == 0, \
        f"{name} contains NaN values"
    
    var_type = _infer_variable_types(vector)
    if var_type == "continuous":
        target_matrix = np.asarray(vector, dtype=float).reshape(-1, 1)
    elif var_type == "categorical":
        dummies = pd.get_dummies(vector, drop_first=True)
        if dummies.shape[1] != 1:
            categories = pd.unique(vector).tolist()
            raise ValueError(
                f"{name} has {len(categories)} categories {categories}; "
                "only binary (two-category) targets are supported"
            )
        target_matrix = dummies.values
    else:
        raise ValueError(f"Unsupported variable type: {var_type}")

    return target_matrix


def _build_design_matrices(
        data: Dataset) -> tuple[np.ndarray, np.ndarray, slice]:
    """
    Build design matrices for OLS.

    Args:
        data (Dataset): 
        The input dataset containing features, target, and covariates.

    Returns:
        tuple[np.ndarray, np.ndarray, slice]: 
        A tuple containing the full, reduced design matrix, and target slice.
    """
    target_matrix = _encode_target(data.y, length=len(data.X), name="data.y")
    covariate_matrix, _ = _encode_covariates(data) 

    intercept = np.ones((target_matrix.shape[0], 1), dtype=float)
    reduced = np.hstack([intercept, covariate_matrix])
    full = np.hstack([intercept, target_matrix, covariate_matrix])
    target_slice = slice(1, 1 + target_matrix.shape[1])

    return full, reduced, target_slice



def _fit_ols(design: np.ndarray, 
             response: np.ndarray) -> tuple[np.ndarray, float, int]:
    """
    Fit ordinary least squares (OLS) and return coefficients, SSE, and rank.

    Args:
        design (np.ndarray): The design matrix for regression.
        response (np.ndarray): The response vector for regression.

    Returns:
        tuple[np.ndarray, float, int]: 
        The fitted coefficients, sum of squared errors, and rank.
    """
    beta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ beta
    sse = float(np.sum(residuals**2))
    return beta, sse, int(rank)


@dataclass(slots=True)
class BiomarkerRecord:
    """DAA summary statistics for one feature."""

    feature_name: str
    feature_index: int
    effect_size: float | None = None
    p_value: float | None = None
    adjusted_p_value: float | None = None


class LinearRegressionDAA(BaseDAAMethod):
    """
    Differential Abundance Analysis (DAA) using linear regression.
    """
    def fit(self, data: Dataset) -> LinearRegressionDAA:
        """
        Fit the linear regression model to the data.
        Args:
            data: Dataset
                The input dataset containing features, target, and covariates.
        Returns:
            LinearRegressionDAA: The fitted DAA method.
        """
        x, feature_names = _create_feature_matrix(data)
        n_samples = x.shape[0]
        full_design, reduced_design, target_slice = \
            _build_design_matrices(data)

        assert full_design.shape[0] == n_samples, \
            "Number of samples in design matrix does not match data.X"
        
        records: list[BiomarkerRecord] = []
        p_values: list[float] = []

        for feature_index, feature_name in enumerate(feature_names):
            # partial F-test 
            response = x[:, feature_index]
            beta_full, sse_full, rank_full = _fit_ols(full_design, response)
            _, sse_reduced, rank_reduced = _fit_ols(reduced_design, response)
            
            df_num = rank_full - rank_reduced
            df_den = n_samples - rank_full

            if df_num <= 0 or df_den <= 0:
                p_value = 1.0
            else:
                numerator = max(sse_reduced - sse_full, 0.0) / df_num
                denominator = sse_full / df_den
                if denominator <= 0.0:
                    f_statistic = float("inf") if numerator > 0.0 else 0.0
                else:
                    f_statistic = numerator / denominator
                p_value = float(scipy_stats.f.sf(f_statistic, df_num, df_den))
                if not isfinite(p_value):
                    p_value = 1.0

            target_coefficients = beta_full[target_slice]
            if target_coefficients.size == 1:
                effect_size = float(target_coefficients[0])
            else:
                raise ValueError(
                    "Only support binary or continuous targets;" \
                    "multiple target coefficients found"
                )

            records.append(
                BiomarkerRecord(
                    feature_name=feature_name,
                    feature_index=feature_index,
                    effect_size=effect_size,
                    p_value=p_value,
                )
            )
            p_values.append(p_value)

        # adjust p-values for multiple testing
        adjusted = adjust_pvalues(
            p_values, method=self.config.correction_method)
        for record, adjusted_p_value in zip(records, adjusted):
            record.adjusted_p_value = adjusted_p_value

        self._records = records
        self._metadata = {
            "alpha": self.config.alpha,
            "correction_method": self.config.correction_method,
            "min_effect_size": self.config.min_effect_size,
            "n_samples": x.shape[0],
            "n_features": len(feature_names),
        }
        return self

    def select_biomarkers(self) -> tuple[list[str], list[int]]:
        """
        Select biomarkers based on adjusted p-values.

        Returns:
            Tuple[List[str], List[int]]: 
            Lists of selected feature names and indices.
        """
        # naive implementations 
        # only select features with significant adjusted p-values
        # return feature names and indices of selected features

        selected_feature_names = []
        selected_feature_indices = []
        for record in self._records:
            if record.adjusted_p_value <= self.config.alpha:
                selected_feature_names.append(record.feature_name)
                selected_feature_indices.append(record.feature_index)
        return selected_feature_names, selected_feature_indices




