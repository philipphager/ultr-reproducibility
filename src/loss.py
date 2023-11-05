import jax.numpy as jnp
import rax
from flax import linen as nn
from jax import Array
from jax._src.lax.lax import stop_gradient
from rax._src.types import LossFn


def regression_em(
    scores: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.pointwise_sigmoid_loss,
) -> Array:
    """
    Implementation of RegressionEM from Wang et al, 2018: https://research.google/pubs/pub46485/
    Numerically stable version as adopted from the Tensorflow Ranking library:
    https://github.com/tensorflow/ranking/blob/c46cede726fd453e0aaa6097871d23dc8e465bdc/tensorflow_ranking/python/losses_impl.py#L1324
    """

    exam_logits, rel_logits = scores

    exam_posterior = nn.sigmoid(exam_logits - nn.softplus(rel_logits))
    exam_posterior = jnp.where(labels, jnp.ones_like(exam_posterior), exam_posterior)
    exam_posterior = stop_gradient(exam_posterior)

    rel_posterior = nn.sigmoid(rel_logits - nn.softplus(exam_logits))
    rel_posterior = jnp.where(labels, jnp.ones_like(rel_posterior), rel_posterior)
    rel_posterior = stop_gradient(rel_posterior)

    exam_loss = loss_fn(
        exam_logits,
        exam_posterior,
        where=where,
    )

    rel_loss = loss_fn(
        rel_logits,
        rel_posterior,
        where=where,
    )

    return exam_loss + rel_loss


def dual_learning_algorithm(
    scores: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.softmax_loss,
    max_weight: float = 10,
) -> Array:
    """
    Implementation of the Dual Learning Algorithm from Ai et al, 2018: https://arxiv.org/pdf/1804.05938.pdf
    """

    examination, relevance = scores
    examination_weights = stop_gradient(
        get_normalized_weights(examination, where, max_weight)
    )
    relevance_weights = stop_gradient(
        get_normalized_weights(relevance, where, max_weight)
    )

    examination_loss = loss_fn(
        examination,
        labels,
        where=where,
        weights=relevance_weights,
    )
    relevance_loss = loss_fn(
        relevance,
        labels,
        where=where,
        weights=examination_weights,
    )

    return examination_loss + relevance_loss


def get_normalized_weights(
    scores: Array,
    where: Array,
    max_weight: float,
) -> Array:
    """
    Converts logits to normalized propensity weights by:
    1. Applying a softmax to all scores in a ranking (ignoring masked values)
    2. Normalizing the resulting probabilities by the first item in each ranking
    3. Inverting propensities to obtain weights: e.g., propensity 0.5 -> weight 2
    4. Clip the final weights to reduce variance
    """
    scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)
    probabilities = nn.softmax(scores, axis=-1)
    # Normalize propensities by the item in first position and convert propensities
    # to weights by computing weights as 1 / propensities:
    weights = probabilities[:, 0].reshape(-1, 1) / probabilities
    weights = jnp.where(where, weights, jnp.ones_like(scores))

    return weights.clip(max=max_weight)
