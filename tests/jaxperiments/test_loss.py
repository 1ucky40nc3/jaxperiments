import jax
import jax.numpy as jnp

import jaxperiments as jp


def test_softmax_cross_entropy() -> None:
    seed = 42
    num_samples = 4
    num_classes = 8

    key = jax.random.PRNGKey(seed)
    logits = jax.random.uniform(key, (num_samples, num_classes))
    targets = jnp.ones((num_samples,), jnp.int32)

    sorted_logits = jnp.sort(logits, axis=-1)
    loss = jp.loss.softmax_cross_entropy(
        sorted_logits,
        targets,
        num_classes
    )
    
    assert loss.shape == ()