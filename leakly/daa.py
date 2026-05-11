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
from leakly import LR_DAA

```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import List
from .config import DAAConfig
from .data import Dataset


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


def _build_design_matrices(
        data: Dataset) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Build design matrices for OLS.

    Args:
        data (Dataset): The input dataset containing features, target, and covariates.

    Returns:
        tuple[np.ndarray, np.ndarray, list[str], list[str]]: 
        A tuple containing the feature matrix, covariate matrix, feature names, and covariate names.
    """
    



def _fit_ols(design: np.ndarray, 
             response: np.ndarray) -> tuple[np.ndarray, float, int]:
    """
    Fit ordinary least squares (OLS) and return coefficients, SSE, and rank.
        tuple[np.ndarray, np.ndarray, list[str], list[str]]: _description_
    """
    



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




