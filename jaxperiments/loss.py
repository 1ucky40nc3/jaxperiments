import jax
import jax.numpy as jnp

import optax


def softmax_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    num_classes: int,
    **kwargs
) -> jnp.ndarray:
    targets = jax.nn.one_hot(targets, num_classes)
    loss = optax.softmax_cross_entropy(logits, targets)
    return loss.mean()