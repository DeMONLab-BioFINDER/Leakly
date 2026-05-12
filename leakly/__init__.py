#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Public API for Leakly.

This module collects the high-level classes and helpers intended for package
users.
'''
from .checker import LeakageChecker, LeakageCheckerOneRun
from .config import (
    CheckerConfig,
    FeatureSelectionConfig,
    ImputationConfig,
    InputConfig,
    MLConfig,
    NormalizationConfig,
    PipelineConfig,
    PreprocConfig,
    SimulationConfig,
    SplitConfig,
    create_default_config,
    load_config_yaml,
    save_config_yaml,
)
from .feature_selection import (
    LinearRegressionDAA,
    feature_selection,
)
from .ml_pipeline import BaseMLPipeline, ExampleMLPipeline
from .models import BaseMLModel, RandomForestModel, SklearnModel, create_model
from .simulation import SimulatedDataset, simulate_dataset
from .summary import SummaryPlotter

__all__ = [
    "BaseMLModel",
    "BaseMLPipeline",
    "CheckerConfig",
    "ExampleMLPipeline",
    "FeatureSelectionConfig",
    "ImputationConfig",
    "InputConfig",
    "LeakageChecker",
    "LeakageCheckerOneRun",
    "LinearRegressionDAA",
    "MLConfig",
    "NormalizationConfig",
    "PipelineConfig",
    "PreprocConfig",
    "RandomForestModel",
    "SimulatedDataset",
    "SimulationConfig",
    "SklearnModel",
    "SplitConfig",
    "SummaryPlotter",
    "create_model",
    "create_default_config",
    "feature_selection",
    "load_config_yaml",
    "save_config_yaml",
    "simulate_dataset",
]
