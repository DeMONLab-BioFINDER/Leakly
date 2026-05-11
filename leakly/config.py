#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Configuration models and YAML template helpers for DAMLeak.

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

CovariateType = Literal["continuous", "categorical"]
CorrectionMethod = Literal["bonferroni", "fdr_bh"]
SimilarityMetric = Literal["jaccard", "overlap"]

#### Example Configuration YAML ####
EXAMPLE_CONFIG_YAML = """
input:
    data_path: data/input.csv
    feature_cols_path: data/features.txt
    target_col: diagnosis
    split_col: split
    train_label: train
    test_label: test
    selected_feature_cols_path: data/selected_features.txt
    sample_id_col: sample_id
    covariate_cols_path: data/covariate_columns.txt
"""



#### Input Configuration ####
@dataclass(slots=True, kw_only=True)
class InputConfig:
    """
    Configuration for input data. 

    Attributes
    ----------
    data_path : str
        Path to the input dataset (CSV or TSV).
    feature_cols_path: str
        Path to the file containing feature names (one per line).
    target_col: str
        Name of the column containing target labels..
    split_col: str | None
        Name of the column containing split labels (e.g., "train", "test").
    train_label: str
        Label used to identify training samples in the split column.
    test_label: str
        Label used to identify testing samples in the split column.
    selected_feature_cols_path: str
        Path to the file containing selected feature names (one per line).
    sample_id_col [optional]: str | None
        Name of the column containing sample IDs.
    covariate_cols_path [optional]: str | None
        Path to the file containing covariate names (one per line).
    """
    data_path: str | Path = "data/input.csv"
    feature_cols_path: str | Path = "data/features.txt"
    target_col: str = "diagnosis"
    split_col: str | None = "split"
    train_label: str | int = "train"
    test_label: str | int = "test"
    selected_feature_cols_path: str | Path = "data/selected_features.txt"
    sample_id_col: str | None = "sample_id"
    covariate_cols_path: str | None = None


def create_input_config(
        *,
        data_path: str | Path = "data/input.csv",
        feature_cols_path: str | Path = "data/features.txt",
        target_col: str = "diagnosis",
        split_col: str | None = "split",
        train_label: str | int = "train",
        test_label: str | int = "test",
        selected_feature_cols_path: str | Path = "data/selected_features.txt",
        sample_id_col: str | None = "sample_id",
        covariate_cols_path: str | None = None
) -> InputConfig:
    """
    Create an InputConfig instance with the specified parameters.

    Args:
        data_path : str
            Path to the input dataset (CSV or TSV).
        feature_cols_path: str
            Path to the file containing feature names (one per line).
        target_col: str
            Name of the column containing target labels..
        split_col: str | None
            Name of the column containing split labels (e.g., "train", "test").
        train_label: str
            Label used to identify training samples in the split column.
        test_label: str
            Label used to identify testing samples in the split column.
        selected_feature_cols_path: str
            Path to the file containing selected feature names (one per line).
        sample_id_col [optional]: str | None
            Name of the column containing sample IDs.
        covariate_cols_path [optional]: str | None
            Path to the file containing covariate names (one per line).

    Returns:
        InputConfig: An initialized instance of InputConfig.
    """
    return InputConfig(
        data_path=data_path,
        feature_cols_path=feature_cols_path,
        target_col=target_col,
        split_col=split_col,
        train_label=train_label,
        test_label=test_label,
        selected_feature_cols_path=selected_feature_cols_path,
        sample_id_col=sample_id_col,
        covariate_cols_path=covariate_cols_path
    )


### Configuration for Differential Abundance Analysis (DAA) ###
@dataclass(slots=True, kw_only=True)
class DAAConfig:
    """
    Configuration for Differential Abundance Analysis (DAA).
    
    Attributes
    ----------
    method: str
        Method to use for DAA (e.g., "linear_regression").
    alpha: float
        Significance level for feature selection (default: 0.05).
    minimum_effect_size: float | None
        Minimum effect size threshold for feature selection.
    top_ranks: float | None
        Top ranks threshold for feature selection (default: 10). 
    correction_method: str
        Method for multiple testing correction (default: "fdr_bh").
    """
    method: str = "linear_regression"
    alpha: float = 0.05
    minimum_effect_size: float | None = 0
    top_ranks: float | None = 10
    correction_method: str = "fdr_bh"


def create_daa_config(
        *,
        method: str = "linear_regression",
        alpha: float = 0.05,
        minimum_effect_size: float | None = 0,
        top_ranks: float | None = 10,
        correction_method: str = "fdr_bh"
) -> DAAConfig:
    """
    Create a DAAConfig instance with the specified parameters.

    Args:
        method: str
            Method to use for DAA (e.g., "linear_regression").
        alpha: float
            Significance level for feature selection (default: 0.05).
        minimum_effect_size: float | None
            Minimum effect size threshold for feature selection.
        top_ranks: float | None
            Top ranks threshold for feature selection (default: 10).
        correction_method: str
            Method for multiple testing correction (default: "fdr_bh").
    Returns:
        DAAConfig: An initialized instance of DAAConfig.
    """
    return DAAConfig(
        method=method,
        alpha=alpha,
        minimum_effect_size=minimum_effect_size,
        top_ranks=top_ranks,
        correction_method=correction_method
    )



#### Simulation Configuration ####
@dataclass(slots=True, kw_only=True)
class SimulationConfig:
    """
    Configuration for synthetic data simulation.

    Attributes
    ----------
    n_samples: int
        Total number of samples to simulate.
    n_features: int
        Total number of features to simulate.
    n_covariates: int
        Total number of covariates to simulate.
    effect_fraction: float
        Fraction of features that have a true effect (between 0 and 1).
    effect_size: float
        Effect size (Cohen's d) for the features with a true effect.
    test_fraction: float
        Fraction of samples to use as the test set (between 0 and 1).
    class_balance: float
        Proportion of positive class samples (between 0 and 1).
    feature_names [optional]: list[str] | None
        List of feature names. If None, default names will be generated.
    covariate_names [optional]: list[str] | None
        List of covariate names. If None, default names will be generated.
    random_state: int | None
        Random seed for reproducibility.
    """
    n_samples: int = 200
    n_features: int = 100
    n_covariates: int = 3
    effect_fraction: float = 0.1
    effect_size: float = 1.0
    test_fraction: float = 0.2
    class_balance: float = 0.5
    feature_names: list[str] | None = None
    covariate_names: list[str] | None = None
    random_state: int | None = None


def create_simulation_config(
        *,
        n_samples: int = 200,
        n_features: int = 100,
        n_covariates: int = 3,
        effect_fraction: float = 0.1,
        effect_size: float = 1.0,
        test_fraction: float = 0.2,
        class_balance: float = 0.5,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
        random_state: int | None = None
) -> SimulationConfig:
    """
    Create a SimulationConfig instance with the specified parameters.

    Args:
        n_samples: int
            Total number of samples to simulate.
        n_features: int
            Total number of features to simulate.
        n_covariates: int
            Total number of covariates to simulate.
        effect_fraction: float
            Fraction of features that have a true effect (between 0 and 1).
        effect_size: float
            Effect size (Cohen's d) for the features with a true effect.
        test_fraction: float
            Fraction of samples to use as the test set (between 0 and 1).
        class_balance: float
            Proportion of positive class samples (between 0 and 1).
        feature_names [optional]: list[str] | None
            List of feature names. If None, default names will be generated.
        covariate_names [optional]: list[str] | None
            List of covariate names. If None, default names will be generated.
        random_state: int | None
            Random seed for reproducibility

    Returns:
        SimulationConfig: An initialized instance of SimulationConfig.
    """
    return SimulationConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_covariates=n_covariates,
        effect_fraction=effect_fraction,
        effect_size=effect_size,
        test_fraction=test_fraction,
        class_balance=class_balance,
        feature_names=feature_names,
        covariate_names=covariate_names,
        random_state=random_state
    )
