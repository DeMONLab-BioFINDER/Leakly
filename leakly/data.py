#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Data-related functions and classes.

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class Dataset:
    """
    A data structure class for in-memory computation.

    Attributes
    ----------
    X: Any
        feature matrix.
    y: Any
        Target vector.
    feature_names: list[str]
        List of feature names.
    train_idx: list[int]
        List of training sample indices.
    test_idx: list[int]
        List of testing sample indices.
    sample_ids [optional]: list[str] | None
        List of sample IDs.
    covariates [optional]: Any | None
        Covariate matrix.
    covariate_names [optional]: list[str] | None
        List of covariate names.
    """
    X: Any
    y: Any
    feature_names: list[str]
    train_idx: list[int]
    test_idx: list[int]
    sample_ids: list[str] | None = None
    covariates: Any | None = None
    covariate_names: list[str] | None = None
    

def _normalization():
    """Placeholder for normalization function."""
    pass

def _imputation():
    """Placeholder for imputation function."""
    pass

def _outlier_removal():
    """Placeholder for outlier removal function."""
    pass

