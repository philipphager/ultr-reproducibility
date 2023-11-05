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
