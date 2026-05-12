#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Summary plotting interfaces for Leakly.

Plotting implementations are deferred until the package structure is finalized.
'''
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .stats import permutation_test_auc, summarize_scores


class SummaryPlotter:
    """
    Plotter for permutation-score summaries.
    """

    def __init__(
        self,
        test_scores: Sequence[float],
        chance_level: float = 0.5,
        observed_score: float | None = None,
    ) -> None:
        """
        Store scores and plotting options.

        Parameters
        ----------
        test_scores:
            Scores from permuted-label runs.
        chance_level:
            Expected chance-level score.
        observed_score:
            Optional unpermuted pipeline score.
        """
        self.test_scores = list(test_scores)
        self.chance_level = chance_level
        self.observed_score = observed_score

    def summarize(self) -> dict[str, float]:
        """
        Summarize permutation scores.

        Returns
        -------
        dict[str, float]
            Summary statistics.
        """
        summary = summarize_scores(self.test_scores)
        summary["chance_level"] = float(self.chance_level)
        if self.observed_score is not None:
            summary["observed_score"] = float(self.observed_score)
            summary["p_value"] = self.permutation_pvalue()
        return summary

    def plot(self, save_path: str | Path | None = None) -> Any:
        """
        Plot the permutation-score distribution.

        Parameters
        ----------
        save_path:
            Optional output path for the figure.

        Returns
        -------
        Any
            Plot object or axis, depending on the plotting backend.
        """
        import matplotlib.pyplot as plt

        scores = np.asarray(self.test_scores, dtype=float)
        if scores.size == 0:
            raise ValueError("test_scores must contain at least one value")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores, bins="auto", alpha=0.75, edgecolor="black")
        ax.axvline(
            self.chance_level,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="chance",
        )
        if self.observed_score is not None:
            ax.axvline(
                self.observed_score,
                color="tab:red",
                linestyle="-",
                linewidth=1.5,
                label="observed",
            )
        ax.set_xlabel("Test set AUC")
        ax.set_ylabel("Permutation count")
        ax.set_title("Leakage check permutation scores")
        ax.legend()
        fig.tight_layout()
        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
        return ax

    def permutation_pvalue(self, alternative: str = "greater") -> float:
        """
        Compute a permutation-test p-value for the observed score.

        Parameters
        ----------
        alternative:
            Alternative hypothesis direction.

        Returns
        -------
        float
            Permutation-test p-value.
        """
        if self.observed_score is None:
            raise ValueError("observed_score is required for a permutation p-value")
        return permutation_test_auc(
            self.observed_score,
            self.test_scores,
            alternative=alternative,
        )
