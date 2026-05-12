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
        class_balance=0.5,
        random_state=42,
    )
)
```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
import numpy as np
from typing import Any
from dataclasses import dataclass
from .config import SimulationConfig


@dataclass(slots=True)
class SimulatedDataset:
    """
    A data structure class for simulated datasets.

    Attributes
    ----------
    X: Any
        Simulated feature matrix.
    y: Any
        Simulated target vector.
    covariates: Any | None
        Optional simulated covariate matrix.
    feature_names: list[str]
        Simulated feature names.
    covariate_names: list[str] | None
        Optional covariate names.
    signal_features: list[str]
        List of feature names that have a true signal (non-zero effect). 
    """
    X: Any
    y: Any
    covariates: Any | None
    feature_names: list[str]
    covariate_names: list[str] | None
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
    if config.feature_names is not None:
        if len(config.feature_names) != config.n_features:
            raise ValueError("feature_names length must match n_features")
        feature_names = list(config.feature_names)
    else:
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
    if config.n_covariates:
        covariates = np.zeros((config.n_samples, config.n_covariates))
        if config.covariate_names is not None:
            if len(config.covariate_names) != config.n_covariates:
                raise ValueError("covariate_names length must match n_covariates")
            covariate_names = list(config.covariate_names)
        else:
            covariate_names = [
                f"covariate_{index + 1}"
                for index in range(config.n_covariates)
            ]
        for cov_index, _ in enumerate(covariate_names):
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
    else:
        covariates = None
        covariate_names = None

    # 4. Generate SimulationDataset
    return SimulatedDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        covariates=covariates,
        covariate_names=covariate_names,
        signal_features=feature_names[:n_signal],
    )


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
    if not 0.0 < config.class_balance < 1.0:
        raise ValueError("class_balance must be between 0 and 1")
    if config.n_covariates < 0:
        raise ValueError("n_covariates cannot be negative")