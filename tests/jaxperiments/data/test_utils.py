from typing import (
    Tuple,
    Optional
)

import numpy as np
import einops
import jax
import tensorflow as tf

import jaxperiments as jp


def dummy_dataset(
    cardinality: Optional[int] = None,
    sample_shape: Tuple[int] = (2, 2, 1)
) -> tf.data.Dataset:
    def generator():
        for _ in range(cardinality):
            yield {'sample': tf.ones(sample_shape, dtype=tf.int32)}

    return tf.data.Dataset.from_generator(
        generator,
        output_signature={
            'sample': tf.TensorSpec(
                shape=sample_shape, 
                dtype=tf.int32
            )
        }
    )


def test_as_train_iter():
    cardinality = 3
    dataset = dummy_dataset(cardinality)

    per_device_batch_size = 2
    transforms = lambda x: x
    dataset = jp.datasets.utils.as_train_iter(
        dataset,
        batch_dims=per_device_batch_size,
        transforms=transforms
    )

    num_steps = 6
    for i, batch in zip(range(num_steps), dataset):
        pass

    assert i == num_steps - 1
    assert batch['sample'].shape[0] == jax.local_device_count()
    assert batch['sample'].shape[1] == per_device_batch_size


def test_as_test_iter():
    cardinality = 3
    sample_shape = (2, 2, 1)

    dataset = dummy_dataset(cardinality, sample_shape)

    per_device_batch_size = 2
    transforms = lambda x: x
    dataset = jp.datasets.utils.as_valid_iter(
        dataset,
        cardinality=cardinality,
        batch_dims=per_device_batch_size,
        transforms=transforms
    )

    num_steps = 6
    for i, batch in zip(range(num_steps), dataset):
        pass

    assert i == num_steps - 1
    assert batch['sample'].shape[0] == jax.local_device_count()
    assert batch['sample'].shape[1] == per_device_batch_size
    assert np.zeros(sample_shape) in batch['sample'].numpy()
    assert 'mask' in batch
    mask = einops.rearrange(batch['mask'], 'l b -> (l b)')
    sample = [tf.ones(sample_shape, tf.int32) * tf.cast(m, tf.int32) for m in mask]
    sample = einops.rearrange(tf.stack(sample), '(l b) ... -> l b ...', l=jax.local_device_count())
    assert tf.reduce_all(tf.math.equal(sample, batch['sample']))