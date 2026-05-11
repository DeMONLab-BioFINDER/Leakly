#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Synthetic data generators for examples and unit tests.

Notes[2024-05-11]
-----
- Current covariates are homogeneously distributed across target classes.
- Current implementation only supports binary classification.

Usage
-----
```python
from leakly import SimulationConfig, simulate_dataset

simulated = simulate_dataset(
    SimulationConfig(
        n_samples=200,
        n_features=100,
        n_covariates=3,
        effect_fraction=0.1,
        effect_size=1.0,
        test_fraction=0.2,
        class_balance=0.5,
        random_state=42,
    )
)
```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .config import SimulationConfig
from .data import Dataset


@dataclass(slots=True)
class SimulatedDataset:
    """
    A data structure class for simulated datasets.

    Attributes
    ----------
    data: MLData
        The simulated dataset in the MLData format.
    signal_features: list[str]
        List of feature names that have a true signal (non-zero effect). 
    """
    data: Dataset
    signal_features: list[str]


def simulate_dataset(
        config: SimulationConfig) -> SimulatedDataset:
    """
    Simulate a dataset based on the provided configuration.

    Args:
        config: SimulationConfig 
            The simulation configuration.

    Returns:
        SimulatedDataset: The simulated dataset.
    """
    # check config parameters
    _validate_simulation_config(config)
    # set random seed
    rng = np.random.default_rng(config.random_state)

    # Simulations
    # 1. Determine signal features
    n_signal = max(
        0, int(round(config.n_features * config.effect_fraction)))
    feature_names = [
        f"feature_{index + 1}" for index in range(config.n_features)]
    
    # 2. Simulate y 
    y = np.zeros(config.n_samples, dtype=int)
    n_positive = int(round(config.n_samples * config.class_balance))
    n_positive = min(max(n_positive, 1), config.n_samples - 1)
    y[:n_positive] = 1
    rng.shuffle(y)
    
    # 3. Simulate X
    X = rng.normal(
        loc=0.0, scale=1.0, 
        size=(config.n_samples, config.n_features))
    if n_signal:
        X[:, :n_signal] += y[:, None] * config.effect_size

    # 4. Simulate covariates (randomly continuous or binary or categorical)
    # Homogeneously distributed across target classes (no confounding)
    covariates = np.zeros((config.n_samples, config.n_covariates))
    covariate_names = []
    for cov_index in range(config.n_covariates):
        cov_name = f"covariate_{cov_index + 1}"
        if rng.random() < 0.5:
            # continuous covariate
            cov_values = rng.normal(
                loc=0.0, scale=1.0, size=config.n_samples)
            covariates[:, cov_index] = cov_values
        else:
            # binary or categorical covariate with 2-4 categories
            n_categories = rng.integers(2, 5)
            cov_values = rng.integers(0, n_categories, size=config.n_samples)
            covariates[:, cov_index] = cov_values
        covariate_names.append(cov_name)

    # 4. Generate train/test split
    train_index = list(
        range(
            config.n_samples - int(
                round(config.n_samples * config.test_fraction))))
    test_index = list(
        range(
            config.n_samples - int(
                round(config.n_samples * config.test_fraction)), 
                config.n_samples))
    
    # 5. Generate SimulationDataset
    ml_data = Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        train_idx=train_index,
        test_idx=test_index,
        covariates=covariates,
        covariate_names=covariate_names
    )
    signal_features = feature_names[:n_signal]
    return SimulatedDataset(data=ml_data, signal_features=signal_features)


def _validate_simulation_config(
        config: SimulationConfig) -> None:
    """
    Validate the simulation configuration parameters.

    Args:
        config: SimulationConfig
            The simulation configuration to validate.

    Raises:
        ValueError: If any of the configuration parameters are invalid.
    """
    if config.n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if config.n_features < 1:
        raise ValueError("n_features must be at least 1")
    if not 0.0 <= config.effect_fraction <= 1.0:
        raise ValueError("effect_fraction must be between 0 and 1")
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0.0 < config.class_balance < 1.0:
        raise ValueError("class_balance must be between 0 and 1")
    if config.n_covariates < 0:
        raise ValueError("n_covariates cannot be negative")