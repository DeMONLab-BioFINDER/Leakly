#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Configuration models and YAML helpers for Leakly.

The dataclasses define the configuration surface for the package and can be
loaded from or saved to YAML files.
'''
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


CovariateType = Literal["continuous", "categorical"]
CorrectionMethod = Literal["bonferroni", "fdr_bh"]
EvaluationMetric = Literal["auc", "accuracy", "r2", "mse"]
FeatureSelectionMethod = Literal["LinearRegressionDAA"]
ImputationMethod = Literal["knn", "mean", "median", "most_frequent", "none"]
MLModelName = Literal["random_forest", "svm", "custom"]
NormalizationMethod = Literal["zscore", "minmax", "none"]
ProblemType = Literal["binary_classification", "regression"]
SplitMethod = Literal["train_test", "predefined"]


EXAMPLE_CONFIG_YAML = """
preproc:
    imputation_method: knn
    n_neighbors: 5
    normalization_method: zscore
    with_mean: true
    with_std: true
feature_selection:
    method: LinearRegressionDAA
    alpha: 0.05
    minimum_effect_size: 0.0
    top_ranks: 10
    correction_method: fdr_bh
split:
    method: train_test
    test_fraction: 0.2
ml:
    model: random_forest
    metric: auc
checker:
    perc_permutation: 1.0
    n_permutations: 100
"""


@dataclass(slots=True, kw_only=True)
class InputConfig:
    """
    Optional file/input metadata for command-line or YAML-driven workflows.

    The core package APIs operate on arrays directly, so these fields are only
    metadata for callers that want to load data from files themselves.
    """

    data_path: str | Path | None = None
    feature_cols_path: str | Path | None = None
    target_col: str | None = None
    split_col: str | None = None
    train_label: str | int | None = None
    test_label: str | int | None = None
    sample_id_col: str | None = None
    covariate_cols_path: str | Path | None = None


@dataclass(slots=True, kw_only=True)
class DataProcConfig:
    """
    General preprocessing options before model fitting.

    Attributes
    ----------
    imputation_method:
        Optional named missing-value imputation strategy.
    n_neighbors:
        Number of neighbors used by KNN imputation.
    normalization_method:
        Optional named feature normalization strategy.
    with_mean:
        Whether z-score normalization should center features.
    with_std:
        Whether z-score normalization should scale features.
    """
    imputation_method: ImputationMethod | None = "knn"
    n_neighbors: int = 5
    normalization_method: NormalizationMethod | None = "zscore"
    with_mean: bool = True
    with_std: bool = True

    def __post_init__(self) -> None:
        """Validate preprocessing options."""
        if self.imputation_method is not None and self.imputation_method not in {
            "knn",
            "mean",
            "median",
            "most_frequent",
            "none",
        }:
            raise ValueError(
                f"Unsupported imputation method: {self.imputation_method}")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if (
            self.normalization_method is not None
            and self.normalization_method not in {"zscore", "minmax", "none"}
        ):
            raise ValueError(
                f"Unsupported normalization method: {self.normalization_method}")


@dataclass(slots=True, kw_only=True)
class FeatureSelectionConfig:
    """
    Feature selection options used by the ML pipeline.

    Attributes
    ----------
    method:
        Feature selection method name.
    alpha:
        Significance threshold.
    minimum_effect_size:
        Optional effect-size threshold.
    top_ranks:
        Optional maximum number of selected features.
    correction_method:
        Multiple-testing correction method.
    selected_feature_names:
        Optional explicit feature subset supplied by the user.
    """

    method: FeatureSelectionMethod = "LinearRegressionDAA"
    alpha: float = 0.05
    minimum_effect_size: float | None = 0.0
    top_ranks: int | None = 10
    correction_method: CorrectionMethod = "fdr_bh"
    selected_feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate basic feature-selection options."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.minimum_effect_size is not None and self.minimum_effect_size < 0:
            raise ValueError("minimum_effect_size must be non-negative")
        if self.top_ranks is not None and self.top_ranks < 1:
            raise ValueError("top_ranks must be at least 1")


@dataclass(slots=True, kw_only=True)
class SplitConfig:
    """
    Train/test split options.

    Attributes
    ----------
    method:
        Split strategy name.
    test_fraction:
        Fraction of samples assigned to the test split.
    random_state:
        Optional random seed.
    stratify:
        Whether classification splits should preserve label proportions.
    """

    method: SplitMethod = "train_test"
    test_fraction: float = 0.2
    random_state: int | None = None
    stratify: bool = False


@dataclass(slots=True, kw_only=True)
class MLConfig:
    """
    Machine-learning model and evaluation options.

    Attributes
    ----------
    model:
        Model family name.
    problem_type:
        Prediction problem type.
    metric:
        Primary evaluation metric.
    random_state:
        Optional model random seed.
    model_params:
        Optional model-specific parameters.
    """

    model: MLModelName = "random_forest"
    problem_type: ProblemType = "binary_classification"
    metric: EvaluationMetric = "auc"
    random_state: int | None = 42
    model_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class CheckerConfig:
    """
    Leakage-check permutation options.

    Attributes
    ----------
    perc_permutation:
        Fraction of labels to permute in one checker run.
    n_permutations:
        Number of repeated permutation runs.
    random_state:
        Optional random seed.
    result_save_path:
        Optional result path. A ``%random_state`` token may be substituted.
    """

    perc_permutation: float = 1.0
    n_permutations: int = 100
    random_state: int | None = None
    result_save_path: str | Path | None = None


@dataclass(slots=True, kw_only=True)
class PipelineConfig:
    """
    Full configuration for the example ML pipeline.

    Attributes
    ----------
    input:
        Input data configuration.
    preproc:
        General preprocessing configuration.
    feature_selection:
        Feature selection configuration.
    split:
        Train/test split configuration.
    ml:
        Model and metric configuration.
    checker:
        Leakage checker configuration.
    """

    input: InputConfig = field(default_factory=InputConfig)
    preproc: DataProcConfig = field(default_factory=DataProcConfig)
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    split: SplitConfig = field(default_factory=SplitConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    checker: CheckerConfig = field(default_factory=CheckerConfig)

    def __post_init__(self) -> None:
        """Coerce nested dictionaries loaded from YAML into config objects."""
        nested = {
            "input": InputConfig,
            "preproc": DataProcConfig,
            "feature_selection": FeatureSelectionConfig,
            "split": SplitConfig,
            "ml": MLConfig,
            "checker": CheckerConfig,
        }
        for attr, cls in nested.items():
            value = getattr(self, attr)
            if isinstance(value, dict):
                setattr(self, attr, cls(**value))


@dataclass(slots=True, kw_only=True)
class SimulationConfig:
    """
    Synthetic-data simulation options.

    Attributes
    ----------
    n_samples:
        Number of simulated samples.
    n_features:
        Number of simulated features.
    n_covariates:
        Number of simulated covariates.
    effect_fraction:
        Fraction of features with true signal.
    effect_size:
        Effect size for signal features.
    class_balance:
        Fraction of positive-class samples.
    feature_names:
        Optional feature names.
    covariate_names:
        Optional covariate names.
    random_state:
        Optional random seed.
    """

    n_samples: int = 200
    n_features: int = 100
    n_covariates: int = 3
    effect_fraction: float = 0.1
    effect_size: float = 1.0
    class_balance: float = 0.5
    feature_names: list[str] | None = None
    covariate_names: list[str] | None = None
    random_state: int | None = None


def load_config_yaml(path: str | Path) -> PipelineConfig:
    """
    Load a pipeline configuration from a YAML file.

    Parameters
    ----------
    path:
        YAML file path.

    Returns
    -------
    PipelineConfig
        Parsed pipeline configuration.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Configuration YAML must contain a mapping at top level")
    return _pipeline_config_from_dict(loaded)


def save_config_yaml(config: PipelineConfig, path: str | Path) -> None:
    """
    Save a pipeline configuration to a YAML file.

    Parameters
    ----------
    config:
        Pipeline configuration to serialize.
    path:
        Output YAML file path.

    Returns
    -------
    None
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_make_yaml_safe(asdict(config)), file, sort_keys=False)


def create_default_config() -> PipelineConfig:
    """
    Create the default Leakly pipeline configuration.

    Returns
    -------
    PipelineConfig
        Default pipeline configuration.
    """
    return PipelineConfig()


def _pipeline_config_from_dict(values: dict[str, Any]) -> PipelineConfig:
    """
    Build a ``PipelineConfig`` from a nested dictionary.

    Parameters
    ----------
    values:
        Nested configuration dictionary, usually loaded from YAML.

    Returns
    -------
    PipelineConfig
        Parsed configuration with defaults for omitted sections.
    """
    values = dict(values)
    fs_values = dict(values.get("feature_selection", {}) or {})
    if "daa" in fs_values:
        raise ValueError(
            "Nested feature_selection.daa is no longer supported. "
            "Put alpha, minimum_effect_size, top_ranks, and "
            "correction_method directly under feature_selection."
        )
    if "min_effect_size" in fs_values:
        fs_values["minimum_effect_size"] = fs_values.pop("min_effect_size")

    return PipelineConfig(
        input=InputConfig(**(values.get("input", {}) or {})),
        preproc=DataProcConfig(**(values.get("preproc", {}) or {})),
        feature_selection=FeatureSelectionConfig(**fs_values),
        split=SplitConfig(**(values.get("split", {}) or {})),
        ml=MLConfig(**(values.get("ml", {}) or {})),
        checker=CheckerConfig(**(values.get("checker", {}) or {})),
    )


def _make_yaml_safe(value: Any) -> Any:
    """Convert dataclass values into plain YAML-safe Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _make_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_yaml_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_yaml_safe(item) for item in value]
    return value
