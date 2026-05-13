#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Summary plotting interfaces for Leakly.
'''
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from .stats import permutation_test
        

PALETTE = {
    "histogram_teal": "#89BEC6",
    "histogram_edge": "#C7DFE3",
    "chance_line": "#C74440",
    "neutral_light": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}


class SummaryPlotter:
    """
    Plotter for permutation-score summaries.
    """

    def __init__(
        self,
        test_scores: Sequence[float],
        chance_level: float = 0.5,
        x_label: str = "AUC",
    ) -> None:
        """
        Store scores and plotting options.

        Parameters
        ----------
        test_scores:
            Scores from permuted-label runs.
        chance_level:
            Expected chance-level score.
        x_label:
            X-axis label for the plotted score.
        """
        self.test_scores = list(test_scores)
        self.chance_level = chance_level
        self.x_label = x_label

    def summarize(self) -> str:
        """
        Summarize permutation results for figure annotation.

        Returns:
            str: Multiline summary string.
        """
        scores = self._score_array()
        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
        p_value = permutation_test(
            self.chance_level,
            scores,
            alternative="less",
        )

        return "\n".join(
            [
                f"{scores.size:d} Permutations",
                f"Chance-level: {self.chance_level:.3f}",
                f"Mean: {mean_score:.3f}",
                f"Std: {std_score:.3f}",
                f"P-value: {p_value:.3g}",
            ]
        )

    def plot(self,
             save_path: str | Path | None = None,
             x_label: str | None = None) -> Any:
        """
        Plot the permutation-score distribution.

        Args:
            save_path (str | Path | None, optional):
                Optional output path for the figure.
                Defaults to None.
            x_label (str | None, optional):
                Optional x-axis label. Defaults to the constructor value.
        Returns:
            Any: Matplotlib axis containing the plot.
        """
        scores = self._score_array()
        summary_text = self.summarize()
        axis_label = self.x_label if x_label is None else x_label
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
        x_min, x_max = _x_axis_limits(mean_score, std_score)

        with plt.rc_context(_publication_rcparams()):
            fig, ax = plt.subplots(figsize=(6.8, 4.4))
            ax.hist(
                scores,
                bins="auto",
                color=PALETTE["histogram_teal"],
                edgecolor=PALETTE["histogram_edge"],
                linewidth=1.3,
                alpha=0.98,
                label="Permutation scores",
            )

            ax.axvline(
                self.chance_level,
                color=PALETTE["chance_line"],
                linestyle="-",
                linewidth=3.0,
                label="Chance level",
            )

            ax.text(
                0.03,
                0.95,
                summary_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                linespacing=1.35,
                color=PALETTE["neutral_black"],
            )

            ax.set_xlabel(axis_label)
            ax.set_ylabel("Permutation Counts")
            ax.set_xlim(x_min, x_max)
            _set_bounded_metric_ticks(ax, axis_label)
            ax.legend(loc="upper right", frameon=False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(PALETTE["neutral_black"])
            ax.spines["bottom"].set_color(PALETTE["neutral_black"])
            ax.spines["left"].set_linewidth(1.8)
            ax.spines["bottom"].set_linewidth(1.8)
            ax.tick_params(
                axis="both",
                colors=PALETTE["neutral_black"],
                width=1.6,
                length=7,
                labelsize=12,
            )
            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.margins(x=0.03)
            fig.tight_layout()

            if save_path is not None:
                output_path = Path(save_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return ax

    def _score_array(self) -> np.ndarray:
        scores = np.asarray(self.test_scores, dtype=float)
        if scores.size == 0:
            raise ValueError("test_scores must contain at least one value")
        if np.any(~np.isfinite(scores)):
            raise ValueError("test_scores must contain only finite values")
        if not np.isfinite(self.chance_level):
            raise ValueError("chance_level must be finite")
        return scores


def _publication_rcparams() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 16,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.8,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
    }


def _x_axis_limits(mean_score: float, std_score: float) -> tuple[float, float]:
    if std_score > 0.0:
        return mean_score - 6.0 * std_score, mean_score + 6.0 * std_score

    fallback_width = max(abs(mean_score) * 0.05, 0.05)
    return mean_score - fallback_width, mean_score + fallback_width


def _set_bounded_metric_ticks(ax: Any, x_label: str) -> None:
    bounded_labels = {"auc", "acc", "accuracy", "f1", "f1 score", "f1-score"}
    if x_label.strip().lower() not in bounded_labels:
        return

    x_min, x_max = ax.get_xlim()
    ticks = np.linspace(0.0, 1.0, 6)
    visible_ticks = [tick for tick in ticks if x_min <= tick <= x_max]
    if not visible_ticks:
        return

    ax.set_xticks(visible_ticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in visible_ticks])
