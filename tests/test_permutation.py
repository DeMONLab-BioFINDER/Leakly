import numpy as np
import pandas as pd
import pytest

from leakly import permute_label


def test_permute_label_is_reproducible_and_preserves_label_multiset():
    y = np.arange(20)

    first = permute_label(y, random_state=42)
    second = permute_label(y, random_state=42)

    assert isinstance(first, np.ndarray)
    assert first.shape == y.shape
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), y)
    assert not np.array_equal(first, y)


def test_permute_label_zero_fraction_keeps_values_and_list_type():
    y = ["case", "control", "case", "control"]

    result = permute_label(y, perc_permutation=0.0, random_state=10)

    assert isinstance(result, list)
    assert result == y


def test_permute_label_preserves_pandas_series_metadata():
    y = pd.Series([0, 1, 1, 0, 1], index=list("abcde"), name="label")

    result = permute_label(y, random_state=0)

    assert isinstance(result, pd.Series)
    assert result.index.tolist() == y.index.tolist()
    assert result.name == "label"
    assert sorted(result.tolist()) == sorted(y.tolist())


@pytest.mark.parametrize("percentage", [-0.1, 1.1, "bad"])
def test_permute_label_rejects_invalid_percentage(percentage):
    with pytest.raises(ValueError, match="perc_permutation"):
        permute_label([0, 1, 0], perc_permutation=percentage)


@pytest.mark.parametrize(
    "y",
    [
        [],
        1,
        np.zeros((2, 2)),
        np.zeros((1, 1, 1)),
    ],
)
def test_permute_label_rejects_invalid_label_shape(y):
    with pytest.raises(ValueError):
        permute_label(y)
