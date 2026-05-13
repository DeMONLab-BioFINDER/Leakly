#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Public API for Leakly.

This module collects the high-level classes and helpers intended for package
users.
'''

from .config import (
    FeatureSelectionConfig,
    ImputationConfig,
    ModelConfig,
    NormalizationConfig,
    PipelineConfig,
    SimulationConfig,
    SplitConfig,
    create_default_config,
    load_example_leakage_config,
    load_example_nonleakage_config,
    load_config_yaml,
    print_config,
    save_example_pipeline_configs,
    save_config_yaml,
)
from .data import data_split
from .feature_selection import (
    LinearRegressionDAA,
    feature_selection,
)
from .ml_pipeline import MLPipeline
from .models import SklearnModel, create_model, ml_model
from .permutation import permute_label
from .simulation import SimulatedDataset, simulate_dataset
from .summary import SummaryPlotter

__all__ = [
    "FeatureSelectionConfig",
    "ImputationConfig",
    "LinearRegressionDAA",
    "MLPipeline",
    "ModelConfig",
    "NormalizationConfig",
    "PipelineConfig",
    "SimulatedDataset",
    "SimulationConfig",
    "SklearnModel",
    "SplitConfig",
    "SummaryPlotter",
    "create_model",
    "create_default_config",
    "data_split",
    "feature_selection",
    "load_example_leakage_config",
    "load_example_nonleakage_config",
    "load_config_yaml",
    "ml_model",
    "permute_label",
    "print_config",
    "save_example_pipeline_configs",
    "save_config_yaml",
    "simulate_dataset",
]
