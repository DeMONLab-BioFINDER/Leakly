#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Machine learning pipeline.

Usage
-----
```python
from leakly import load_example_leakage_config, MLPipeline

leakage_config = load_example_leakage_config()

pipeline = MLPipeline(
    X, y, covariates=simulated.covariates,
    problem_type="binary_classification",
    config=leakage_config)

pipeline.fit()
auc = pipeline.evaluate(metric="auc")
print(f"Test AUC: {auc:.3f}")
```

Written by Lijun An and DeMON Lab under MIT license:
https://github.com/DeMONLab-BioFINDER/DeMONLabLicenses/blob/main/LICENSE
'''
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .config import (
    NO_LEAKAGE_PIPELINE_STEPS,
    PipelineConfig,
    create_default_config,
)
from .data import (
    data_split,
    fit_imputer,
    fit_normalizer,
    transform_imputer,
    transform_normalizer,
    validate_data,
)
from .feature_selection import feature_selection
from .models import create_model


STEP_NAMES = {
    "imputation",
    "normalization",
    "feature_selection",
    "data_split",
    "model",
}

class MLPipeline:
    """
    Machine learning pipeline that runs steps from the configuration.
    """
    def __init__(
        self,
        X: Any,
        y: Any,
        covariates: Any | None = None,
        *,
        problem_type: str = "binary_classification",
        config: PipelineConfig | Mapping[str, Any] | None = None,
        feature_names: list[str] | None = None,
        covariate_names: list[str] | None = None,
    ) -> None:
        """
        Initialize the MLPipeline with data, configuration, and pipeline steps.

        Args:
            X (Any): Feature matrix.
            y (Any): Target variable.
            covariates (Any | None, optional): 
                Covariate matrix. Defaults to None.
            problem_type (str, optional): 
                Type of the machine learning problem. 
                Defaults to "binary_classification".
            config (PipelineConfig | Mapping[str, Any] | None, optional): 
                Pipeline configuration. 
                Defaults to None.
            feature_names (list[str] | None, optional): 
                Names of the features. 
                Defaults to None.
            covariate_names (list[str] | None, optional): 
                Names of the covariates. 
                Defaults to None.
        """
        self.X = X
        self.y = y
        self.covariates = covariates
        self.problem_type = problem_type
        self.feature_names = feature_names
        self.covariate_names = covariate_names

        self.config = self._make_config(config)
        self.config.model.problem_type = problem_type
        if isinstance(config, Mapping):
            pipeline = config.get("pipeline") or NO_LEAKAGE_PIPELINE_STEPS
        else:
            pipeline = NO_LEAKAGE_PIPELINE_STEPS
        self.pipeline = self._make_pipeline(pipeline)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.covariates_train = None
        self.covariates_test = None

        self.imputer = None
        self.normalizer = None
        self.selected_feature_names = None
        self.selected_feature_indices = None
        self.model = None

        self.fitted_steps: list[tuple[str, Any]] = []
        self.has_split = False
        self.is_fitted = False

    def fit(self) -> "MLPipeline":
        """
        Fit the pipeline to the training data.

        Returns:
            MLPipeline: The fitted pipeline.
        """
        validate_data(self.X, self.y, self.covariates)
        current_X = np.asarray(self.X, dtype=float)
        current_y = np.asarray(self.y).reshape(-1)
        current_covariates = (
            None if self.covariates is None else np.asarray(
                self.covariates).copy()
        )
        current_feature_names = self.feature_names

        for step_name in self.pipeline:
            if step_name == "data_split":
                (
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    self.covariates_train,
                    self.covariates_test,
                ) = data_split(
                    current_X,
                    current_y,
                    current_covariates,
                    self.config.data_split,
                )
                self.has_split = True

            elif step_name == "imputation":
                fit_X = self.X_train if self.has_split else current_X
                self.imputer = fit_imputer(
                    fit_X, self.config.imputation)
                if self.has_split:
                    self.X_train = transform_imputer(
                        self.imputer, self.X_train)
                    self.X_test = transform_imputer(
                        self.imputer, self.X_test)
                else:
                    current_X = transform_imputer(
                        self.imputer, current_X)
                self.fitted_steps.append(("imputation", self.imputer))

            elif step_name == "normalization":
                fit_X = self.X_train if self.has_split else current_X
                self.normalizer = fit_normalizer(
                    fit_X, self.config.normalization)
                if self.has_split:
                    self.X_train = transform_normalizer(
                        self.normalizer, self.X_train)
                    self.X_test = transform_normalizer(
                        self.normalizer, self.X_test)
                else:
                    current_X = transform_normalizer(
                        self.normalizer, current_X)
                self.fitted_steps.append(("normalization", self.normalizer))

            elif step_name == "feature_selection":
                if self.has_split:
                    selected_names, selected_indices = feature_selection(
                        self.X_train,
                        self.y_train,
                        self.covariates_train,
                        self.config.feature_selection,
                        current_feature_names,
                        self.covariate_names,
                    )
                    self.X_train = np.asarray(
                        self.X_train)[:, selected_indices]
                    self.X_test = np.asarray(
                        self.X_test)[:, selected_indices]
                else:
                    selected_names, selected_indices = feature_selection(
                        current_X,
                        current_y,
                        current_covariates,
                        self.config.feature_selection,
                        current_feature_names,
                        self.covariate_names,
                    )
                    current_X = np.asarray(
                        current_X)[:, selected_indices]

                if not selected_indices:
                    raise ValueError("Feature selection returned no features")
                self.selected_feature_names = selected_names
                self.selected_feature_indices = selected_indices
                current_feature_names = selected_names
                self.fitted_steps.append(
                    ("feature_selection", selected_indices))

            elif step_name == "model":
                if not self.has_split:
                    raise ValueError("model step must appear after data_split")
                self.config.model.problem_type = self.problem_type
                self.model = create_model(self.config.model)
                self.model.fit(self.X_train, self.y_train)
                self.is_fitted = True

        return self

    def predict(self, 
                X: Any) -> Any:
        """
        Predict after applying fitted preprocessing and feature selection.

        Args:
            X (Any): The input data for prediction.

        Returns:
            Any: The predicted values.
        """
        self._require_fitted()
        return self.model.predict(self._transform_new_X(X))

    def predict_proba(self, X: Any) -> Any:
        """
        Predict probabilities after applying fitted preprocessing.

        Args:
            X (Any): The input data for prediction.

        Returns:
            Any: The predicted probabilities.
        """
        self._require_fitted()
        return self.model.predict_proba(self._transform_new_X(X))

    def evaluate(
        self,
        y_true: Any | None = None,
        y_pred: Any | None = None,
        X: Any | None = None,
        metric: str | None = None,
    ) -> float:
        """
        Evaluate provided predictions, provided X, or the stored test split.

        Args:
            y_true (Any | None, optional): The true labels. Defaults to None.
            y_pred (Any | None, optional): The predicted labels. Defaults to None.
            X (Any | None, optional): The input data for evaluation. Defaults to None.
            metric (str | None, optional): The evaluation metric. Defaults to None.

        Raises:
            ValueError: If the required arguments are not provided.

        Returns:
            float: The evaluation score.
        """
        self._require_fitted()
        metric = (metric or self.config.model.metric).lower()

        if y_pred is not None:
            if y_true is None:
                raise ValueError("y_true is required when y_pred is provided")
            return _score_predictions(y_true, y_pred, metric)

        if X is not None:
            if y_true is None:
                raise ValueError("y_true is required when X is provided")
            return self.model.evaluate(
                self._transform_new_X(X), y_true, metric)

        return self.model.evaluate(self.X_test, self.y_test, metric)

    def save(self, path: str | Path) -> None:
        """
        Save the fitted pipeline, model, and configuration to one joblib file.

        Args:
            path (str | Path): The path where the pipeline will be saved.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPipeline":
        """
        Load a pipeline saved by ``save``.

        Args:
            path (str | Path): The path to the saved pipeline.

        Returns:
            MLPipeline: MLPipeline instance loaded from the specified path.
        """
        return joblib.load(path)

    @staticmethod
    def _make_config(
        config: PipelineConfig | Mapping[str, Any] | None,
    ) -> PipelineConfig:
        """
        Create a pipeline configuration from the provided input.

        Args:
            config (PipelineConfig | Mapping[str, Any] | None): 
            The configuration input.

        Returns:
            PipelineConfig: The created pipeline configuration.
        """
        if config is None:
            return create_default_config()
        if isinstance(config, PipelineConfig):
            return config
        if isinstance(config, Mapping):
            values = dict(config)
            values.pop("pipeline", None)
            return PipelineConfig(**values)
        raise TypeError("config must be a PipelineConfig, mapping, or None")

    @staticmethod
    def _make_pipeline(pipeline: list[str]) -> list[str]:
        """
        Make a validated pipeline list from the provided input.

        Args:
            pipeline (list[str]): The pipeline steps to validate.

        Returns:
            list[str]: The validated pipeline steps.
        """
        steps = list(pipeline)
        for step_name in steps:
            if step_name not in STEP_NAMES:
                raise ValueError(f"Unsupported pipeline step: {step_name}")

        if "data_split" not in steps:
            raise ValueError("pipeline must include data_split")
        if "model" not in steps:
            raise ValueError("pipeline must include model")
        if steps[-1] != "model":
            raise ValueError("model step must be the final pipeline step")
        if steps.index("model") < steps.index("data_split"):
            raise ValueError("model step must appear after data_split")
        return steps

    def _transform_new_X(self, X: Any) -> np.ndarray:
        """
        Transform new input data using the fitted pipeline steps.

        Args:
            X (Any): The input data to transform.

        Returns:
            np.ndarray: The transformed input data.
        """
        transformed = np.asarray(X, dtype=float)
        for step_name, fitted_object in self.fitted_steps:
            if step_name == "imputation":
                transformed = transform_imputer(fitted_object, transformed)
            elif step_name == "normalization":
                transformed = transform_normalizer(fitted_object, transformed)
            elif step_name == "feature_selection":
                transformed = transformed[:, fitted_object]
        return transformed

    def _require_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("MLPipeline must be fitted before use")


def _score_predictions(
    y_true: Any,
    y_pred: Any,
    metric: str,
) -> float:
    """
    Score predictions using the specified metric.

    Args:
        y_true (Any): The true labels.
        y_pred (Any): The predicted labels.
        metric (str): The metric to use for scoring.

    Returns:
        float: The score.
    """
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred)

    if metric == "auc":
        if y_pred_array.ndim == 2:
            if y_pred_array.shape[1] != 2:
                raise ValueError(
                    "AUC evaluation currently supports binary targets")
            y_pred_array = y_pred_array[:, 1]
        return float(roc_auc_score(y_true_array, y_pred_array.reshape(-1)))
    if metric == "accuracy":
        return float(accuracy_score(y_true_array, y_pred_array.reshape(-1)))
    if metric == "r2":
        return float(r2_score(y_true_array, y_pred_array.reshape(-1)))
    if metric == "mse":
        return float(
            mean_squared_error(y_true_array, y_pred_array.reshape(-1)))
    raise ValueError(f"Unsupported evaluation metric: {metric}")
