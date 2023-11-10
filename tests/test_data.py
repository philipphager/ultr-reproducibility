import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.data import pad, LabelEncoder, Digitize


@pytest.mark.parametrize(
    "x, max_n, expected",
    [
        (np.array([0, 1, 2, 3, 4]), 10, np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0])),
        (np.array([0, 1, 2, 3, 4]), 5, np.array([0, 1, 2, 3, 4])),
        (np.array([]), 5, np.array([0, 0, 0, 0, 0])),
        (np.array([0, 1, 2, 3, 4]), 3, np.array([0, 1, 2, 3, 4])),
    ],
)
def test_pad_1d(x, max_n, expected):
    actual = pad(x, max_n)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "x, max_n, expected",
    [
        (np.array([[0], [1]]), 5, np.array([[0], [1], [0], [0], [0]])),
        (np.array([[0, 1], [2, 3]]), 4, np.array([[0, 1], [2, 3], [0, 0], [0, 0]])),
    ],
)
def test_pad_2d(x, max_n, expected):
    actual = pad(x, max_n)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0, 5, 512, 978, 93481023981]), np.array([1, 2, 3, 4, 5])),
        (np.array([5, 10, 5, 10, 10]), np.array([1, 2, 1, 2, 2])),
    ],
)
def test_label_encoder(x, expected):
    encoder = LabelEncoder()
    actual = encoder(x)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "X, expected",
    [
        (
                [np.array([2]), np.array([9]), np.array([5])],
                [np.array([1]), np.array([2]), np.array([3])],
        ),
        (
                [np.array([5, 4, 3]), np.array([500, 400, 300]), np.array([5, 400, 3])],
                [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([1, 5, 3])],
        ),
    ],
)
def test_label_encoder_state(X, expected):
    encoder = LabelEncoder()
    actual = [encoder(x) for x in X]
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "low, high, buckets, x, expected",
    [
        (0, 0.3, 3, np.array([0, 0.001, 0.1, 0.19, 0.2]), np.array([1, 1, 2, 2, 3])),
        (0, 0.3, 3, np.array([-1, 0, 0.1, 0.2, 0.3]), np.array([0, 1, 2, 3, 4])),
        (-5, 5, 10, np.array([-5, -3, 0, 3, 5]), np.array([1, 3, 6, 9, 11])),
    ],
)
def test_digitize(low, high, buckets, x, expected):
    digitize = Digitize(low=low, high=high, buckets=buckets)
    actual = digitize(x)
    assert_array_equal(actual, expected)
