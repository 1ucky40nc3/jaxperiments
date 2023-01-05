from typing import (
    Any,
    Tuple,
    Union,
    Optional
)

from functools import partial

import ml_collections as mlc

import jax
import jax.numpy as jnp

import flax
from flax.training import train_state

import jaxperiments as jp


ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


class TrainState(train_state.TrainState):
    batch_stats: Optional[flax.core.frozen_dict.FrozenDict[str, Any]] = None


@partial(jax.jit, static_argnums=(1,), backend='cpu')
def init_fn(rng: jnp.ndarray, config: ConfigDict) -> TrainState:
    model = getattr(jp.models, config.model.name)(config.model.kwargs)
    samples = jnp.ones([1, *config.data.sample_shape])
    variables = model.init(rng, samples)
    tx = jp.optim.tx.make(config.optim)

    return TrainState.create(
        tx=tx,
        apply_fn=apply_fn,
        params=variables.get('params'),
        batch_stats=variables.get('batch_stats', {})
    )


@partial(
    jax.pmap, 
    axis_name='batch', 
    donate_argnums=(0,), 
    static_broadcasted_argnums=(3,)
)
def update_fn(
    state: TrainState, 
    samples: jnp.ndarray, 
    target: jnp.ndarray, 
    config: ConfigDict,
    rngs: Optional[jnp.ndarray] = None
) -> Tuple[TrainState, Tuple[jnp.ndarray, jnp.ndarray]]:
    def loss_fn(params: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        variables = {
            'params': params,
            'batch_stats': state.batch_stats
        }
        mutable = ['batch_stats']

        logits, mutvars = state.apply_fn(variables, samples, mutable=mutable, )
        loss = getattr(jp.loss, config.loss.name)(logits, target, **config.loss.kwargs)

        return loss, (logits, mutvars)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, mutvars)), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(
        grads=grads,
        batch_stats=mutvars['batch_stats']
    )

    return state, (loss, logits)


def inference_fn(
    state: TrainState, 
    samples: jnp.ndarray,
    config: ConfigDict,
    targets: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    def loss_fn(params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        variables = {
            'params': params,
            'batch_stats': state.batch_stats
        }
        logits = state.apply_fn(variables, samples, mutable=False, deterministic=True)
        loss = getattr(jp.loss, config.loss.name)(logits, labels, **config.loss.kwargs)

        return loss, logits

    loss, logits = loss_fn(state.params)
    return loss, logits