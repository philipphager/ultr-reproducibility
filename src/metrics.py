import rax
from jax import Array
from rax._src.types import ReduceFn

from src.util import reduce_per_query


def negative_log_likelihood(
    scores: Array,
    labels: Array,
    where: Array,
    reduce_fn: ReduceFn = reduce_per_query,
) -> Array:
    """
    Compute negative log likelihood.
    Expects scores to be click log-odds: log(c / (1 - c)).
    """
    return rax.pointwise_sigmoid_loss(
        scores=scores, labels=labels, where=where, reduce_fn=reduce_fn
    )
