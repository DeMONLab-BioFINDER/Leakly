#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Machine-learning pipeline interfaces for Leakly.

The example pipeline is designed to split data before imputation,
normalization, feature selection, model fitting, and evaluation.
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import PipelineConfig
from .data import (
    fit_imputer,
    fit_normalizer,
    subset_rows,
    train_test_split_indices,
    transform_imputer,
    transform_normalizer,
    validate_data,
)
from .feature_selection import feature_selection
from .models import create_model


ArrayLike = Any


class BaseMLPipeline(ABC):
    """
    Base interface for a user-defined ML pipeline.

    Implementations should store the original target vector as ``self.y`` so
    leakage checkers can generate replacement labels for permutation runs.
    """

    @abstractmethod
    def run(self, y: ArrayLike | None = None) -> float:
        """
        Run the pipeline and return the held-out test score.

        Parameters
        ----------
        y:
            Optional replacement target vector, used by leakage checkers for
            permuted-label runs.

        Returns
        -------
        float
            Test set score.
        """
        pass


class ExampleMLPipeline(BaseMLPipeline):
    """
    Example non-leaky ML pipeline scaffold.
    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        covariates: ArrayLike | None = None,
        config: PipelineConfig | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Store pipeline inputs.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target vector.
        covariates:
            Optional covariate matrix.
        config:
            Optional full pipeline configuration.
        feature_names:
            Optional feature names.
        """
        self.X = X
        self.y = y
        self.covariates = covariates
        self.config = config or PipelineConfig()
        self.feature_names = feature_names

    def split_data(self, y: ArrayLike | None = None) -> dict[str, Any]:
        """
        Split arrays into train and test partitions.

        Parameters
        ----------
        y:
            Optional replacement target vector.

        Returns
        -------
        dict[str, Any]
            Train/test arrays and related split metadata.
        """
        target = self.y if y is None else y
        validate_data(self.X, target, self.covariates)
        train_idx, test_idx = train_test_split_indices(target, self.config.split)
        covariates = self.covariates
        return {
            "X_train": subset_rows(self.X, train_idx),
            "X_test": subset_rows(self.X, test_idx),
            "y_train": subset_rows(target, train_idx),
            "y_test": subset_rows(target, test_idx),
            "covariates_train": subset_rows(covariates, train_idx),
            "covariates_test": subset_rows(covariates, test_idx),
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    def preprocess_train_test(self, split_data: dict[str, Any]) -> dict[str, Any]:
        """
        Fit preprocessing on training data and transform train/test data.

        Parameters
        ----------
        split_data:
            Output from ``split_data``.

        Returns
        -------
        dict[str, Any]
            Preprocessed train/test arrays.
        """
        imputer = fit_imputer(split_data["X_train"], self.config.preproc)
        X_train = transform_imputer(imputer, split_data["X_train"])
        X_test = transform_imputer(imputer, split_data["X_test"])

        normalizer = fit_normalizer(X_train, self.config.preproc)
        X_train = transform_normalizer(normalizer, X_train)
        X_test = transform_normalizer(normalizer, X_test)

        processed = dict(split_data)
        processed.update(
            {
                "X_train": X_train,
                "X_test": X_test,
                "imputer": imputer,
                "normalizer": normalizer,
            }
        )
        return processed

    def select_features(self, processed_data: dict[str, Any]) -> dict[str, Any]:
        """
        Fit feature selection on training data and transform train/test data.

        Parameters
        ----------
        processed_data:
            Output from ``preprocess_train_test``.

        Returns
        -------
        dict[str, Any]
            Feature-selected train/test arrays and selection metadata.
        """
        X_train = np.asarray(processed_data["X_train"], dtype=float)
        X_test = np.asarray(processed_data["X_test"], dtype=float)
        selected_feature_names, selected_feature_indices = feature_selection(
            X_train,
            processed_data["y_train"],
            processed_data["covariates_train"],
            self.config.feature_selection,
            self.feature_names,
        )
        if not selected_feature_indices:
            raise ValueError("Feature selection returned no features")
        selected_data = dict(processed_data)
        selected_data.update(
            {
                "X_train_features": X_train[:, selected_feature_indices],
                "X_test_features": X_test[:, selected_feature_indices],
                "selected_feature_names": selected_feature_names,
                "selected_feature_indices": selected_feature_indices,
            }
        )
        return selected_data

    def fit_model(self, selected_data: dict[str, Any]) -> Any:
        """
        Fit the configured machine-learning model.

        Parameters
        ----------
        selected_data:
            Output from ``select_features``.

        Returns
        -------
        Any
            Fitted model object.
        """
        model = create_model(self.config.ml)
        model.fit(selected_data["X_train_features"], selected_data["y_train"])
        self.model_ = model
        return model

    def evaluate_model(self, model: Any, selected_data: dict[str, Any]) -> float:
        """
        Evaluate a fitted model on held-out test data.

        Parameters
        ----------
        model:
            Fitted model object.
        selected_data:
            Output from ``select_features``.

        Returns
        -------
        float
            Test set score.
        """
        return model.evaluate(
            selected_data["X_test_features"],
            selected_data["y_test"],
            self.config.ml.metric,
        )

    def run(self, y: ArrayLike | None = None) -> float:
        """
        Run split, preprocessing, feature selection, fitting, and evaluation.

        Parameters
        ----------
        y:
            Optional replacement target vector.

        Returns
        -------
        float
            Test set score.
        """
        split = self.split_data(y)
        processed = self.preprocess_train_test(split)
        selected = self.select_features(processed)
        model = self.fit_model(selected)
        score = self.evaluate_model(model, selected)
        self.train_idx_ = selected["train_idx"]
        self.test_idx_ = selected["test_idx"]
        self.selected_feature_indices_ = selected["selected_feature_indices"]
        self.test_score_ = score
        return float(score)
