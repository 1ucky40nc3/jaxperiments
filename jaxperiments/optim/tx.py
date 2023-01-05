from typing import Union

from functools import partial

import ml_collections as mlc

import optax


ConfigDict = Union[mlc.ConfigDict, mlc.FrozenConfigDict]


def make(config: ConfigDict) -> optax.GradientTransformation:
    learning_rate = getattr(optax, config.schedule.name)(**config.schedule.kwargs)
    optimizer = getattr(optax, config.optimizer.name)(
        learning_rate=learning_rate, 
        **config.optimizer.kwargs
    )
    
    grad_clip_norm = optax.identity()
    if confg.grad_clip_norm:
        grad_clip_norm = optax.clip_by_global_norm(config.grad_clip_norm)

    weight_decay = optax.add_decayed_weights(config.weight_deday)

    return optax.chain(
        grad_clip_norm,
        optimizer,
        weight_decay,
        optax.scale(-1)
    )