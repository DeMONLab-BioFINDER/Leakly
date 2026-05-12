#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Leakage checker interfaces for Leakly.

The checker runs a user-defined ML pipeline after permuting a portion of the
target vector and records the held-out test score.
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ArrayLike = Any


class LeakageChecker:
    """
    Base leakage checker for permutation-based pipeline evaluation.
    """

    def __init__(
        self,
        ml_pipeline: Any,
        perc_permutation: float = 1.0,
        random_state: int | None = None,
        result_save_path: str | Path | None = None,
    ) -> None:
        """
        Store checker inputs.

        Parameters
        ----------
        ml_pipeline:
            User-defined pipeline exposing a target vector as ``.y`` and a
            ``run(y=None)`` method that can run with replacement labels.
        perc_permutation:
            Fraction of labels to permute.
        random_state:
            Optional random seed.
        result_save_path:
            Optional JSON result path.
        """
        self.ml_pipeline = ml_pipeline
        self.perc_permutation = perc_permutation
        self.random_state = random_state
        self.result_save_path = result_save_path

    def permute_y(self, y: ArrayLike) -> ArrayLike:
        """
        Permute a configured fraction of target labels.

        Parameters
        ----------
        y:
            Original target vector.

        Returns
        -------
        ArrayLike
            Target vector with a fraction of labels permuted.
        """
        if not 0.0 <= self.perc_permutation <= 1.0:
            raise ValueError("perc_permutation must be between 0 and 1")
        original = np.asarray(y)
        permuted = original.copy()
        n_samples = permuted.shape[0]
        n_permuted = int(round(n_samples * self.perc_permutation))
        if n_permuted <= 1:
            return permuted

        rng = np.random.default_rng(self.random_state)
        selected = rng.choice(n_samples, size=n_permuted, replace=False)
        permuted[selected] = rng.permutation(permuted[selected])
        return permuted

    def save_result(self, test_score: float) -> None:
        """
        Save one checker result.

        Parameters
        ----------
        test_score:
            Test set score from the permuted-label run.

        Returns
        -------
        None
        """
        if self.result_save_path is None:
            return
        path = Path(str(self.result_save_path).replace(
            "%random_state", str(self.random_state)))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "test_score": float(test_score),
            "perc_permutation": float(self.perc_permutation),
            "random_state": self.random_state,
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def run(self) -> float:
        """
        Run one leakage-check permutation.

        Returns
        -------
        float
            Test set score from the permuted-label run.
        """
        if not hasattr(self.ml_pipeline, "y"):
            raise AttributeError("ml_pipeline must expose the original target as .y")
        if not hasattr(self.ml_pipeline, "run"):
            raise AttributeError("ml_pipeline must expose a run(y=None) method")
        y_permuted = self.permute_y(self.ml_pipeline.y)
        test_score = float(self.ml_pipeline.run(y=y_permuted))
        self.save_result(test_score)
        return test_score


class LeakageCheckerOneRun(LeakageChecker):
    """
    Compatibility name for a single leakage-check run.
    """

    pass
