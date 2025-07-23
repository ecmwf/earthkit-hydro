import tensorflow as tf

from .array_backend import ArrayBackend


class TFBackend(ArrayBackend):
    def __init__(self):
        super().__init__(tf)

    def copy(self, x):
        return x

    def scatter_add(self, target, indices, updates):
        target_shape = tf.shape(target)
        batch_shape = target_shape[:-1]
        depth = target_shape[-1]
        num_indices = tf.shape(indices)[0]

        flat_batch_size = tf.reduce_prod(batch_shape)
        K = num_indices
        D = depth
        updates_flat = tf.reshape(updates, [-1])

        segment_ids = tf.tile(indices, [flat_batch_size])

        batch_ids = tf.repeat(tf.range(flat_batch_size, dtype=tf.int32), repeats=K)

        combined_segments = batch_ids * D + segment_ids

        scattered_flat = tf.math.unsorted_segment_sum(
            data=updates_flat,
            segment_ids=combined_segments,
            num_segments=flat_batch_size * D,
        )

        scattered = tf.reshape(scattered_flat, [flat_batch_size, D])

        result = tf.reshape(scattered, tf.concat([batch_shape, [depth]], axis=0))
        return target + result

    def take_along_axis(self, arr, indices, axis=-1):
        update_with = tf.gather(arr, indices, axis=axis)
        return update_with
