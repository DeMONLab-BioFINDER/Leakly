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


NO_LEAKAGE_PIPELINE_STEPS = [
    "data_split",
    "imputation",
    "normalization",
    "feature_selection",
    "model",
]

LEAKAGE_PIPELINE_STEPS = [
    "imputation",
    "normalization",
    "feature_selection",
    "data_split",
    "model",
]

EXAMPLE_LEAKAGE_PIPELINE_YAML = "Example_LeakgePipeline.yaml"
EXAMPLE_NON_LEAKAGE_PIPELINE_YAML = "Example_NonLeakgePipeline.yaml"

ImputationMethod = Literal["knn", "mean", "median", "most_frequent", "none"]
NormalizationMethod = Literal["zscore", "minmax", "none"]
FeatureSelectionMethod = Literal["LinearRegressionDAA"]
CorrectionMethod = Literal["fdr_bh", "bonferroni"]
SplitMethod = Literal["train_test", "predefined"]
MLModelName = Literal["random_forest", "svm", "custom"]
ProblemType = Literal["binary_classification", "regression"]
EvaluationMetric = Literal["auc", "accuracy", "r2", "mse"]


@dataclass(slots=True, kw_only=True)
class ImputationConfig:
    """
    Missing-value imputation options.

    Attributes
    ----------
    method:
        Optional imputation strategy.
    n_neighbors:
        Number of neighbors used by KNN imputation.
    """
    method: ImputationMethod | None = "knn"
    n_neighbors: int = 5

    def __post_init__(self) -> None:
        """Validate imputation options."""
        if self.method is not None and self.method not in {
            "knn",
            "mean",
            "median",
            "most_frequent",
            "none",
        }:
            raise ValueError(f"Unsupported imputation method: {self.method}")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")


@dataclass(slots=True, kw_only=True)
class NormalizationConfig:
    """
    Feature normalization options.

    Attributes
    ----------
    method:
        Optional normalization strategy.
    with_mean:
        Whether z-score normalization should center features.
    with_std:
        Whether z-score normalization should scale features.
    """
    method: NormalizationMethod | None = "zscore"
    with_mean: bool = True
    with_std: bool = True

    def __post_init__(self) -> None:
        """Validate normalization options."""
        if self.method is not None and self.method not in {
            "zscore",
            "minmax",
            "none",
        }:
            raise ValueError(
                f"Unsupported normalization method: {self.method}")


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
        if self.minimum_effect_size is not None and \
            self.minimum_effect_size < 0:
            raise ValueError("minimum_effect_size must be non-negative")
        if self.top_ranks is not None and self.top_ranks < 1:
            raise ValueError("top_ranks must be at least 1")
        if self.selected_feature_names is not None:
            if not isinstance(self.selected_feature_names, list) or not all(
                isinstance(name, str) for name in self.selected_feature_names
            ):
                raise ValueError(
                    "selected_feature_names must be a list of strings or null"
                )


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
    random_state: int | None = 42
    stratify: bool = False


@dataclass(slots=True, kw_only=True)
class ModelConfig:
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
class PipelineConfig:
    """
    Default configuration for the ordered ML pipeline.
    """

    pipeline: list[str] = field(
        default_factory=lambda: list(NO_LEAKAGE_PIPELINE_STEPS)
    )
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    normalization: NormalizationConfig = field(
        default_factory=NormalizationConfig)
    data_split: SplitConfig = field(default_factory=SplitConfig)
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        """Coerce nested dictionaries loaded from YAML into config objects."""
        if isinstance(self.pipeline, str):
            raise ValueError("pipeline must be a list of step names")
        self.pipeline = list(self.pipeline)
        nested = {
            "imputation": ImputationConfig,
            "normalization": NormalizationConfig,
            "data_split": SplitConfig,
            "feature_selection": FeatureSelectionConfig,
            "model": ModelConfig,
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
    random_state: int | None = 42


def create_default_config() -> PipelineConfig:
    """
    Create the default non-leaky ML pipeline configuration.
    """
    return PipelineConfig()


def example_config_dict(
    pipeline_steps: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a YAML-ready example pipeline configuration dictionary.
    """
    config = create_default_config()
    values = _make_yaml_safe(asdict(config))
    values["pipeline"] = list(pipeline_steps or NO_LEAKAGE_PIPELINE_STEPS)
    return values


def save_example_pipeline_configs(
    directory: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Generate example leakage and non-leakage pipeline YAML files.
    """
    output_dir = Path(directory) if directory is not None else Path(
        __file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    leakage_path = output_dir / EXAMPLE_LEAKAGE_PIPELINE_YAML
    no_leakage_path = output_dir / EXAMPLE_NON_LEAKAGE_PIPELINE_YAML

    with leakage_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            example_config_dict(LEAKAGE_PIPELINE_STEPS),
            file,
            sort_keys=False,
        )
    with no_leakage_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            example_config_dict(NO_LEAKAGE_PIPELINE_STEPS),
            file,
            sort_keys=False,
        )
    return leakage_path, no_leakage_path


def load_example_leakage_config() -> dict[str, Any]:
    """
    Load the example leakage pipeline configuration.
    """
    return _load_yaml_mapping(
        Path(__file__).parent / EXAMPLE_LEAKAGE_PIPELINE_YAML)


def load_example_nonleakage_config() -> dict[str, Any]:
    """
    Load the example non-leakage pipeline configuration.
    """
    return _load_yaml_mapping(
        Path(__file__).parent / EXAMPLE_NON_LEAKAGE_PIPELINE_YAML
    )


def load_config_yaml(path: str | Path) -> PipelineConfig:
    """
    Load a pipeline configuration from YAML.
    """
    values = _load_yaml_mapping(path)
    if "split" in values and "data_split" not in values:
        values["data_split"] = values.pop("split")
    if "ml" in values and "model" not in values:
        values["model"] = values.pop("ml")
    return PipelineConfig(**values)


def print_config(config: Any | None = None) -> None:
    """
    Pretty-print a pipeline configuration as YAML.
    """
    if config is None:
        values = example_config_dict()
    elif isinstance(config, PipelineConfig):
        values = _make_yaml_safe(asdict(config))
    elif isinstance(config, dict):
        values = _make_yaml_safe(config)
    elif hasattr(config, "config") and hasattr(config, "pipeline"):
        values = _make_yaml_safe(asdict(config.config))
        values["pipeline"] = list(config.pipeline)
    else:
        raise TypeError(
            "config must be a PipelineConfig, dict, MLPipeline, or None"
        )
    print(yaml.safe_dump(values, sort_keys=False), end="")


def save_config_yaml(config: PipelineConfig, path: str | Path) -> None:
    """
    Save a pipeline configuration to YAML.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_make_yaml_safe(asdict(config)), file, sort_keys=False)


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError("Configuration YAML must contain a mapping")
    return values


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
