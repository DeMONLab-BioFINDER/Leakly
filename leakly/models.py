#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Machine-learning model interfaces for Leakly.

Model wrappers are intentionally minimal and sklearn-compatible.
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.svm import SVC, SVR

from .config import MLConfig


ArrayLike = Any


class BaseMLModel(ABC):
    """
    Base interface for machine-learning models.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        """
        Store model configuration.

        Parameters
        ----------
        config:
            Optional model configuration.
        """
        self.config = config or MLConfig()

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> "BaseMLModel":
        """
        Fit the model.

        Parameters
        ----------
        X:
            Training model matrix.
        y:
            Training target vector.

        Returns
        -------
        BaseMLModel
            Fitted model.
        """
        pass

    @abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict target labels or values.

        Parameters
        ----------
        X:
            Model matrix.

        Returns
        -------
        ArrayLike
            Predictions.
        """
        pass

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities when supported by the estimator.

        Parameters
        ----------
        X:
            Model matrix.

        Returns
        -------
        ArrayLike
            Predicted probabilities.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba")

    def evaluate(self, X: ArrayLike, y: ArrayLike, metric: str = "auc") -> float:
        """
        Evaluate the model on held-out data.

        Parameters
        ----------
        X:
            Test model matrix.
        y:
            Test target vector.
        metric:
            Evaluation metric name.

        Returns
        -------
        float
            Evaluation score.
        """
        metric = metric.lower()
        y_true = np.asarray(y).reshape(-1)
        if metric == "auc":
            scores = self._score_for_auc(X)
            return float(roc_auc_score(y_true, scores))
        y_pred = self.predict(X)
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "r2":
            return float(r2_score(y_true, y_pred))
        if metric == "mse":
            return float(mean_squared_error(y_true, y_pred))
        raise ValueError(f"Unsupported evaluation metric: {metric}")

    def _score_for_auc(self, X: ArrayLike) -> ArrayLike:
        """Return continuous scores suitable for ROC AUC."""
        try:
            probabilities = np.asarray(self.predict_proba(X))
            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                return probabilities[:, 1]
            return probabilities.reshape(-1)
        except NotImplementedError:
            predictions = np.asarray(self.predict(X))
            return predictions.reshape(-1)


class SklearnModel(BaseMLModel):
    """
    Adapter for user-provided sklearn-compatible estimators.
    """

    def __init__(self, estimator: Any, config: MLConfig | None = None) -> None:
        """
        Store the wrapped estimator.

        Parameters
        ----------
        estimator:
            Object exposing sklearn-style ``fit`` and ``predict`` methods.
        config:
            Optional model configuration.
        """
        super().__init__(config=config)
        self.estimator = estimator

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SklearnModel":
        """
        Fit the wrapped estimator.

        Parameters
        ----------
        X:
            Training model matrix.
        y:
            Training target vector.

        Returns
        -------
        SklearnModel
            Fitted model wrapper.
        """
        self.estimator.fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict with the wrapped estimator.

        Parameters
        ----------
        X:
            Model matrix.

        Returns
        -------
        ArrayLike
            Predictions.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict probabilities with the wrapped estimator.

        Parameters
        ----------
        X:
            Model matrix.

        Returns
        -------
        ArrayLike
            Predicted probabilities.
        """
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                return scores
            return scores[:, -1]
        return super().predict_proba(X)


class RandomForestModel(SklearnModel):
    """
    Default random forest model wrapper.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        """
        Prepare a random forest model wrapper.

        Parameters
        ----------
        config:
            Optional model configuration.
        """
        config = config or MLConfig()
        params = dict(config.model_params)
        params.setdefault("random_state", config.random_state)
        params.setdefault("n_estimators", 200)
        if config.problem_type == "regression":
            estimator = RandomForestRegressor(**params)
        else:
            estimator = RandomForestClassifier(**params)
        super().__init__(estimator=estimator, config=config)


def create_model(config: MLConfig | None = None, estimator: Any | None = None) -> BaseMLModel:
    """
    Create a model wrapper from configuration or a custom estimator.

    Parameters
    ----------
    config:
        Optional model configuration.
    estimator:
        Optional user-provided sklearn-compatible estimator.

    Returns
    -------
    BaseMLModel
        Model wrapper.
    """
    config = config or MLConfig()
    if estimator is not None:
        return SklearnModel(estimator=estimator, config=config)
    if config.model == "random_forest":
        return RandomForestModel(config=config)
    if config.model == "svm":
        params = dict(config.model_params)
        if config.problem_type == "regression":
            estimator = SVR(**params)
        else:
            params.setdefault("probability", config.metric == "auc")
            params.setdefault("random_state", config.random_state)
            estimator = SVC(**params)
        return SklearnModel(estimator=estimator, config=config)
    if config.model == "custom":
        raise ValueError("A custom estimator must be provided for model='custom'")
    raise ValueError(f"Unsupported model: {config.model}")
