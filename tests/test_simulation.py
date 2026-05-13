import numpy as np
import pandas as pd
import pytest

from leakly import SimulationConfig, simulate_dataset


def test_simulate_dataset_shapes_names_and_signal_features():
    data = simulate_dataset(
        SimulationConfig(
            n_samples=50,
            n_features=10,
            n_covariates=3,
            effect_fraction=0.2,
            class_balance=0.4,
            random_state=123,
        )
    )

    assert data.X.shape == (50, 10)
    assert data.y.shape == (50,)
    assert data.covariates.shape == (50, 3)
    assert data.feature_names == [f"feature_{index}" for index in range(1, 11)]
    assert data.covariate_names == ["covariate_1", "covariate_2", "covariate_3"]
    assert data.signal_features == ["feature_1", "feature_2"]
    assert int(data.y.sum()) == 20


def test_simulate_dataset_is_reproducible_with_random_state():
    config = SimulationConfig(
        n_samples=30,
        n_features=6,
        n_covariates=2,
        random_state=7,
    )

    first = simulate_dataset(config)
    second = simulate_dataset(config)

    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    pd.testing.assert_frame_equal(first.covariates, second.covariates)


def test_simulate_dataset_default_config_is_reproducible():
    first = simulate_dataset(SimulationConfig())
    second = simulate_dataset(SimulationConfig())

    np.testing.assert_allclose(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    pd.testing.assert_frame_equal(first.covariates, second.covariates)


def test_simulate_dataset_accepts_custom_names_and_no_covariates():
    data = simulate_dataset(
        SimulationConfig(
            n_samples=12,
            n_features=3,
            n_covariates=0,
            feature_names=["age", "volume", "score"],
            effect_fraction=1 / 3,
            random_state=5,
        )
    )

    assert data.covariates is None
    assert data.covariate_names is None
    assert data.feature_names == ["age", "volume", "score"]
    assert data.signal_features == ["age"]


def test_simulate_dataset_preserves_categorical_covariate_dtype():
    data = simulate_dataset(
        SimulationConfig(
            n_samples=20,
            n_features=2,
            n_covariates=6,
            random_state=123,
        )
    )

    assert isinstance(data.covariates, pd.DataFrame)
    categorical_columns = data.covariates.select_dtypes(
        include=["category"]
    ).columns
    assert len(categorical_columns) >= 1


@pytest.mark.parametrize(
    "config, message",
    [
        (SimulationConfig(n_samples=1), "n_samples"),
        (SimulationConfig(n_features=0), "n_features"),
        (SimulationConfig(effect_fraction=-0.1), "effect_fraction"),
        (SimulationConfig(class_balance=1.0), "class_balance"),
        (SimulationConfig(n_covariates=-1), "n_covariates"),
    ],
)
def test_simulate_dataset_rejects_invalid_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        simulate_dataset(config)
