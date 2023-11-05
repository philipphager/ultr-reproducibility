from typing import Optional

import jax.numpy as jnp
import rax
from flax import linen as nn
from jax import Array
from jax._src.lax.lax import stop_gradient
from rax._src import utils
from rax._src.types import ReduceFn, LossFn


def regression_em(
    scores: Array,
    labels: Array,
    where: Array,
    loss_fn: LossFn = rax.pointwise_sigmoid_loss,
    reduce_fn: Optional[ReduceFn] = jnp.mean,
) -> Array:
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
        reduce_fn=None,
    )

    rel_loss = loss_fn(
        rel_logits,
        rel_posterior,
        where=where,
        reduce_fn=None,
    )

    loss = exam_loss + rel_loss
    return utils.safe_reduce(loss, where=where, reduce_fn=reduce_fn)
