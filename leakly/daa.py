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


class LR_DAA(BaseDAAMethod):
    """
    Differential Abundance Analysis (DAA) using linear regression.
    """
    def fit(self, data: Dataset) -> LR_DAA:
        """
        Fit the linear regression model to the data.
        Args:
            data: Dataset
                The input dataset containing features, target, and covariates.
        Returns:
            LR_DAA: The fitted DAA method.
        """
        


