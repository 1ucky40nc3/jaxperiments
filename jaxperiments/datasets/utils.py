from typing import (
    Any,
    Union,
    Callable,
    Optional,
    Sequence
)

import jax
import tensorflow as tf

from clu.deterministic_data import pad_dataset


def as_train_iter(
    dataset: tf.data.Dataset,
    batch_dims: Optional[Union[Sequence[int], int]] = None,
    filter_predicate: Optional[Callable[[Any], bool]] = None,
    cache_raw_dataset: bool = False,
    shuffle_buffer_size: Optional[int] = None,
    transforms: Optional[Callable[[Any], bool]] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    prefetch_buffer_size: int = 2
) -> tf.data.Dataset:
    if filter_predicate is not None:
        dataset = dataset.filter(filter_predicate)
    if cache_raw_dataset:
        dataset = dataset.cache()
    dataset = dataset.repeat(None)
    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(transforms, num_parallel_calls=num_parallel_calls)
    if batch_dims is not None:
        if isinstance(batch_dims, int):
            batch_dims = (jax.local_device_count(), batch_dims)
        for batch_size in reversed(batch_dims):
            dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(prefetch_buffer_size)


def as_valid_iter(
    dataset: tf.data.Dataset,
    batch_dims: Optional[Union[Sequence[int], int]] = None,
    cardinality: Optional[int] = None,
    transforms: Optional[Callable[[Any], bool]] = None,
    cache_raw_dataset: bool = False,
    cache_preprocessed_dataset: bool = False,
    prefetch_buffer_size: int = 1
) -> tf.data.Dataset:
    if cache_raw_dataset:
        dataset = dataset.cache()
    dataset = dataset.map(transforms, num_parallel_calls=tf.data.AUTOTUNE)
    # Apply infinite padding. This adds a 'mask' column and concatenates
    # `zero` tensors to the dataset until the length is a multiple of 
    # `batch_dims`. The mask can later be used to constrain
    # computations on original samples only. This avoids frequent recompilation.
    dataset = pad_dataset(dataset, batch_dims=batch_dims, cardinality=cardinality)
    if batch_dims is not None:
        if isinstance(batch_dims, int):
            batch_dims = (jax.local_device_count(), batch_dims)
        for batch_size in reversed(batch_dims):
            dataset = dataset.batch(batch_size, drop_remainder=False)
    if cache_preprocessed_dataset:
        dataset = dataset.cache()
    dataset = dataset.repeat()
    return dataset.prefetch(prefetch_buffer_size)