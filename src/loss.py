from typing import Optional, Callable

import jax.numpy as jnp
import rax
from flax import linen as nn
from jax import Array
from jax._src.lax.lax import stop_gradient
from rax._src.types import LossFn, LambdaweightFn


def regression_em(
    examination: Array,
    relevance: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.pointwise_sigmoid_loss,
    reduce_fn: Optional[Callable] = jnp.mean,
) -> Array:
    """
    Implementation of RegressionEM from Wang et al, 2018: https://research.google/pubs/pub46485/
    Numerically stable version as adopted from the Tensorflow Ranking library:
    https://github.com/tensorflow/ranking/blob/c46cede726fd453e0aaa6097871d23dc8e465bdc/tensorflow_ranking/python/losses_impl.py#L1324
    """
    examination_posterior = _get_posterior(examination, relevance, labels)
    relevance_posterior = _get_posterior(relevance, examination, labels)

    examination_loss = loss_fn(
        examination, examination_posterior, where=where, reduce_fn=reduce_fn
    )
    relevance_loss = loss_fn(
        relevance, relevance_posterior, where=where, reduce_fn=reduce_fn
    )

    return examination_loss + relevance_loss


def _get_posterior(x: Array, y: Array, labels: Array) -> Array:
    posterior = nn.sigmoid(x - nn.softplus(y))
    posterior = jnp.where(labels, jnp.ones_like(posterior), posterior)
    return stop_gradient(posterior)


def dual_learning_algorithm(
    examination: Array,
    relevance: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.softmax_loss,
    max_weight: float = 10,
    reduce_fn: Optional[Callable] = jnp.mean,
) -> Array:
    """
    Implementation of the Dual Learning Algorithm from Ai et al, 2018: https://arxiv.org/pdf/1804.05938.pdf
    """
    examination_weights = _get_normalized_weights(
        examination, where, max_weight, softmax=True
    )
    relevance_weights = _get_normalized_weights(
        relevance, where, max_weight, softmax=True
    )

    examination_loss = loss_fn(
        examination,
        labels,
        where=where,
        weights=relevance_weights,
        reduce_fn=reduce_fn,
    )
    relevance_loss = loss_fn(
        relevance,
        labels,
        where=where,
        weights=examination_weights,
        reduce_fn=reduce_fn,
    )

    return examination_loss + relevance_loss


def _get_normalized_weights(
    scores: Array,
    where: Array,
    max_weight: float,
    softmax: bool = False,
) -> Array:
    """
    Converts logits to normalized propensity weights by:
    1. [Optional] Applying a softmax to all scores in a ranking (ignoring masked values)
    2. Normalizing the resulting probabilities by the first item in each ranking
    3. Inverting propensities to obtain weights: e.g., propensity 0.5 -> weight 2
    4. Clip the final weights to reduce variance
    """
    scores = jnp.where(where, scores, -jnp.ones_like(scores) * jnp.inf)
    if softmax:
        probabilities = nn.softmax(scores, axis=-1)
    else:
        probabilities = scores
    # Normalize propensities by the item in first position and convert propensities
    # to weights by computing weights as 1 / propensities:
    weights = probabilities[:, 0].reshape(-1, 1) / probabilities
    weights = jnp.where(where, weights, jnp.ones_like(scores))
    weights = weights.clip(max=max_weight)

    return stop_gradient(weights)


def behavior_cloning(
    scores: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.pointwise_mse_loss,
    reduce_fn: Optional[Callable] = jnp.mean,
) -> Array:
    """
    Behavior cloning, i.e., replication of the logging policy, by learning to predict the observed position of the item at hand.
    """
    return loss_fn(
        scores,
        jnp.broadcast_to(
            jnp.power(1 / jnp.arange(1, labels.shape[1] + 1), 2), labels.shape
        ),
        where=where,
        reduce_fn=reduce_fn,
    )


def inverse_propensity_weighting(
    examination: Array,
    relevance: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.softmax_loss,
    max_weight: float = 10,
    reduce_fn: Optional[Callable] = jnp.mean,
):
    examination_weights = _get_normalized_weights(
        examination, where, max_weight, softmax=False
    )

    return loss_fn(
        relevance,
        labels,
        where=where,
        weights=examination_weights,
        reduce_fn=reduce_fn,
    )


def pairwise_debiasing(
    ratio_positive: Array,
    ratio_negative: Array,
    relevance: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.pairwise_logistic_loss,
    lambdaweight_fn: LambdaweightFn = rax.dcg_lambdaweight,
    max_weight: float = 10,
    l_norm=1,
    reduce_fn: Optional[Callable] = jnp.mean,
):
    """
    Implementation of the Pairwise Debiasing algorithm from Hu et al, 2019: https://dl.acm.org/doi/pdf/10.1145/3308558.3313447
    Propensity ratios are trained via gradient descent while the ranker is trained using LambdaRank (Burges et al., 2006)
    """
    weights = 1 / (ratio_positive * ratio_negative)
    examination_loss = loss_fn(
        stop_gradient(relevance), labels, where=where, weights=weights
    )
    examination_loss += jnp.power(jnp.linalg.norm(ratio_positive, l_norm), l_norm)
    examination_loss += jnp.power(jnp.linalg.norm(ratio_negative, l_norm), l_norm)

    positive_weight = _get_normalized_weights(
        ratio_positive, where, max_weight, softmax=False
    )
    negative_weight = _get_normalized_weights(
        ratio_negative, where, max_weight, softmax=False
    )

    def unbiased_lambdaweight_fn(scores, labels, where, segments, weights):
        weights = 1 / (positive_weight * negative_weight)
        return lambdaweight_fn(
            scores, labels, where=where, weights=weights, normalize=True
        )

    relevance_loss = loss_fn(
        relevance,
        labels,
        where=where,
        lambdaweight_fn=unbiased_lambdaweight_fn,
        reduce_fn=reduce_fn,
    )

    return relevance_loss + examination_loss
