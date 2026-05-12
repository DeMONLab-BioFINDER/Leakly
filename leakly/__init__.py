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
    DataProcConfig,
    FeatureSelectionConfig,
    InputConfig,
    MLConfig,
    PipelineConfig,
    SimulationConfig,
    SplitConfig,
    create_default_config,
    load_config_yaml,
    save_config_yaml,
)
from .data import data_split
from .feature_selection import (
    LinearRegressionDAA,
    feature_selection,
)
from .ml_pipeline import BaseMLPipeline, ExampleMLPipeline
from .models import SklearnModel, create_model, ml_model
from .simulation import SimulatedDataset, simulate_dataset
from .summary import SummaryPlotter

__all__ = [
    "BaseMLPipeline",
    "CheckerConfig",
    "DataProcConfig",
    "ExampleMLPipeline",
    "FeatureSelectionConfig",
    "InputConfig",
    "LeakageChecker",
    "LeakageCheckerOneRun",
    "LinearRegressionDAA",
    "MLConfig",
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
    "load_config_yaml",
    "ml_model",
    "save_config_yaml",
    "simulate_dataset",
]
