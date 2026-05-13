import numpy as np
import pandas as pd
import pytest

from leakly import FeatureSelectionConfig, feature_selection


def test_feature_selection_uses_explicit_selected_feature_names():
    X = np.arange(24, dtype=float).reshape(6, 4)
    feature_names = ["alpha", "beta", "gamma", "delta"]

    selected_names, selected_indices = feature_selection(
        X,
        [0, 1, 0, 1, 0, 1],
        config=FeatureSelectionConfig(
            selected_feature_names=["gamma", "alpha"],
        ),
        feature_names=feature_names,
    )

    assert selected_names == ["gamma", "alpha"]
    assert selected_indices == [2, 0]


def test_feature_selection_infers_pandas_dataframe_column_names():
    X = np.arange(24, dtype=float).reshape(6, 4)
    frame = pd.DataFrame(
        X,
        columns=["alpha", "beta", "gamma", "delta"],
    )

    selected_names, selected_indices = feature_selection(
        frame,
        [0, 1, 0, 1, 0, 1],
        config=FeatureSelectionConfig(
            selected_feature_names=["delta", "beta"],
        ),
    )

    assert selected_names == ["delta", "beta"]
    assert selected_indices == [3, 1]


def test_feature_selection_rejects_unknown_explicit_feature_name():
    X = np.arange(12, dtype=float).reshape(4, 3)

    with pytest.raises(ValueError, match="unknown features"):
        feature_selection(
            X,
            [0, 1, 0, 1],
            config=FeatureSelectionConfig(
                selected_feature_names=["missing"],
            ),
            feature_names=["a", "b", "c"],
        )


def test_feature_selection_rejects_duplicate_feature_names():
    X = np.arange(12, dtype=float).reshape(4, 3)

    with pytest.raises(ValueError, match="unique"):
        feature_selection(
            X,
            [0, 1, 0, 1],
            config=FeatureSelectionConfig(selected_feature_names=["a"]),
            feature_names=["a", "a", "c"],
        )


def test_linear_regression_daa_ranks_strong_signal_first():
    rng = np.random.default_rng(123)
    y = np.tile([0, 1], 40)
    X = rng.normal(size=(80, 3))
    X[:, 0] = y * 5.0 + rng.normal(scale=0.1, size=80)
    feature_names = ["signal", "noise_1", "noise_2"]

    selected_names, selected_indices = feature_selection(
        X,
        y,
        config=FeatureSelectionConfig(
            alpha=1.0,
            minimum_effect_size=0.0,
            top_ranks=1,
        ),
        feature_names=feature_names,
    )

    assert selected_names == ["signal"]
    assert selected_indices == [0]


def test_feature_selection_rejects_unsupported_method():
    X = np.arange(12, dtype=float).reshape(4, 3)

    with pytest.raises(ValueError, match="Unsupported feature selection method"):
        feature_selection(
            X,
            [0, 1, 0, 1],
            config=FeatureSelectionConfig(method="not_a_method"),
        )
