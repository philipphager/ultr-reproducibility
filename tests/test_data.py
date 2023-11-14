import numpy as np
import pytest
from numpy.testing import assert_array_equal

from src.data import pad, hash_labels, discretize


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
    "x, buckets",
    [
        (np.array([0, 5, 512, 978, 93481023981]), 10),
        (np.array([5, 10, 5, 10, 10]), 1),
        (np.random.randint(0, 1_000_000, (1000,)), 10),
    ],
)
def test_hash_labels(x, buckets):
    actual = hash_labels(x, buckets)

    assert len(set(actual)) <= buckets
    assert min(actual) >= 1
    assert max(actual) <= buckets


@pytest.mark.parametrize(
    "x, buckets, expected_collision_rate",
    [
        (np.random.randint(0, 1000, (10,)), 1_000, 0.01),
        (np.random.randint(0, 1000, (100,)), 1_000, 0.1),
        (np.random.randint(0, 1000, (100,)), 10_000, 0.05),
        (np.random.randint(0, 1_000_000, (1_000,)), 10_000, 0.1),
    ],
)
def test_hash_collisions(x, buckets, expected_collision_rate):
    actual = hash_labels(x, buckets)

    collision_rate = 1 - (len(set(actual)) / len(set(x)))
    assert collision_rate <= expected_collision_rate, collision_rate


@pytest.mark.parametrize(
    "low, high, buckets, x, expected",
    [
        (0, 0.3, 3, np.array([0, 0.001, 0.1, 0.19, 0.2]), np.array([1, 1, 2, 2, 3])),
        (0, 0.3, 3, np.array([-1, 0, 0.1, 0.2, 0.3]), np.array([0, 1, 2, 3, 4])),
        (-5, 5, 10, np.array([-5, -3, 0, 3, 5]), np.array([1, 3, 6, 9, 11])),
        (0, 0.3, 3, np.array([[0, 0.001, 0.1, 0.19]]), np.array([[1, 1, 2, 2]])),
    ],
)
def test_digitize(low, high, buckets, x, expected):
    actual = discretize(x, low=low, high=high, buckets=buckets)
    assert_array_equal(actual, expected)
