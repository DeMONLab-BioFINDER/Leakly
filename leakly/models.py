#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Machine learning model.

Notes[2024-05-12]
-----
- Uses scikit-learn compatible estimators.
- Default model is random forest.

Usage
-----
```python
from leakly import ModelConfig, ml_model

test_auc = ml_model(
    X_train_processed,
    y_train,
    X_test_processed,
    y_test,
    ModelConfig(
        model="random_forest",
        problem_type="binary_classification",
        metric="auc",
        random_state=42,
        model_params={"n_estimators": 100, "max_depth": 5},
    ),
)
```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from __future__ import annotations
from typing import Any
ArrayLike = Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.svm import SVC, SVR
from .config import ModelConfig


def ml_model(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    config: ModelConfig | None = None,
    estimator: Any | None = None,
) -> float:
    """
    Fit a configured sklearn model and return the test-set score.

    Args:
        X_train (ArrayLike): Training feature matrix.
        y_train (ArrayLike): Training target vector.
        X_test (ArrayLike): Test feature matrix.
        y_test (ArrayLike): Test target vector.
        config (ModelConfig | None, optional): 
            Model and metric configuration. Defaults to None.
        estimator (Any | None, optional): 
            Optional custom sklearn-compatible estimator. Defaults to None.

    Returns:
        float: Test score using ``config.metric``.
    """
    config = config or ModelConfig()
    model = create_model(config=config, estimator=estimator)
    model.fit(X_train, y_train)
    return model.evaluate(X_test, y_test, metric=config.metric)



class SklearnModel:
    """
    Base class for wrapping sklearn-compatible ML estimators.
    """

    def __init__(self, 
                 estimator: Any, 
                 config: ModelConfig | None = None) -> None:
        """
        Initialize the sklearn model class.

        Args:
            estimator (Any): ML estimator that exposes fit and predict methods.
            config (ModelConfig | None, optional): 
                _description_. Defaults to None.
        """
        self.config = config or ModelConfig()
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            raise TypeError("estimator must have fit and predict methods")
        self.estimator = estimator

    def fit(self, 
            X: ArrayLike, 
            y: ArrayLike) -> "SklearnModel":
        """
        Fit the wrapped estimator and return ``self``.

        Args:
            X (ArrayLike): Feature matrix for training.
            y (ArrayLike): Target vector for training.

        Returns:
            SklearnModel: The fitted model.
        """
        self.estimator.fit(X, np.asarray(y).reshape(-1))
        return self

    def predict(self, 
                X: ArrayLike) -> ArrayLike:
        """
        Predict using the wrapped estimator's predict method.

        Args:
            X (ArrayLike): Feature matrix for prediction.

        Returns:
            ArrayLike: Predicted values.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict probabilities when the wrapped estimator supports them.

        Args:
            X (ArrayLike): Feature matrix for prediction.

        Returns:
            ArrayLike: Predicted probabilities.
        """
        if not hasattr(self.estimator, "predict_proba"):
            raise NotImplementedError(
                f"{self.estimator.__class__.__name__} does not provide "
                "predict_proba"
            )
        return self.estimator.predict_proba(X)

    def decision_scores(self, X: ArrayLike) -> ArrayLike:
        """
        Return one-dimensional continuous scores for ROC AUC.

        This follows sklearn conventions: class probabilities are preferred,
        decision-function scores are used next, and hard predictions are a
        fallback for estimators that expose neither.

        Args:
            X (ArrayLike): Feature matrix for prediction.

        Returns:
            ArrayLike: One-dimensional scores for each sample.
        """
        try:
            probabilities = np.asarray(self.predict_proba(X))
        except NotImplementedError:
            probabilities = None

        if probabilities is not None:
            if probabilities.ndim == 2:
                if probabilities.shape[1] != 2:
                    raise ValueError(
                        "AUC evaluation currently supports binary targets")
                return probabilities[:, 1]
            return probabilities.reshape(-1)

        if hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(X))
            if scores.ndim == 2:
                if scores.shape[1] != 2:
                    raise ValueError(
                        "AUC evaluation currently supports binary targets")
                scores = scores[:, 1]
            return scores.reshape(-1)

        return np.asarray(self.predict(X)).reshape(-1)

    def evaluate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        metric: str | None = None,
    ) -> float:
        """
        Evaluate the model with a sklearn metric.

        Args:
            X (ArrayLike): Test feature matrix.
            y (ArrayLike): Test target vector.
            metric (str | None, optional): Metric name. Defaults to None.

        Returns:
            float: Evaluation score.
        """
        metric = (metric or self.config.metric).lower()
        y_true = np.asarray(y).reshape(-1)

        if metric == "auc":
            return float(roc_auc_score(y_true, self.decision_scores(X)))

        y_pred = self.predict(X)
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "r2":
            return float(r2_score(y_true, y_pred))
        if metric == "mse":
            return float(mean_squared_error(y_true, y_pred))

        raise ValueError(f"Unsupported evaluation metric: {metric}")


def _random_forest_estimator(
        config: ModelConfig) -> Any:
    """
    Create a random forest estimator from ``ModelConfig``.

    Args:
        config (ModelConfig): Configuration for the random forest model.

    Returns:
        Any: The created random forest estimator.
    """

    params = dict(config.model_params)
    params.setdefault("random_state", config.random_state)
    params.setdefault("n_estimators", 100)

    if config.problem_type == "binary_classification":
        return RandomForestClassifier(**params)
    if config.problem_type == "regression":
        return RandomForestRegressor(**params)
    raise ValueError(f"Unsupported problem_type: {config.problem_type}")


def _svm_estimator(
        config: ModelConfig) -> Any:
    """
    Create an SVM estimator from ``ModelConfig``.

    Args:
        config (ModelConfig): Configuration for the SVM model.

    Returns:
        Any: The created SVM estimator.
    """
    params = dict(config.model_params)

    if config.problem_type == "binary_classification":
        params.setdefault("probability", config.metric == "auc")
        params.setdefault("random_state", config.random_state)
        return SVC(**params)
    if config.problem_type == "regression":
        return SVR(**params)
    raise ValueError(f"Unsupported problem_type: {config.problem_type}")


def create_model(
    config: ModelConfig | None = None,
    estimator: Any | None = None,
) -> SklearnModel:
    """
    Create a model wrapper from configuration or a custom estimator.

    Args:
        config (ModelConfig | None, optional): 
            Configuration for the model. Defaults to None.
        estimator (Any | None, optional): 
            Custom sklearn-compatible estimator. Defaults to None.

    Returns:
        SklearnModel: A wrapped sklearn model.
    """
    config = config or ModelConfig()

    if estimator is not None:
        return SklearnModel(estimator=estimator, config=config)

    if config.model == "random_forest":
        return SklearnModel(_random_forest_estimator(config), config=config)

    if config.model == "svm":
        return SklearnModel(_svm_estimator(config), config=config)

    if config.model == "custom":
        raise ValueError("Provide estimator=... when config.model='custom'")

    raise ValueError(f"Unsupported model: {config.model}")
