import tensorflow as tf

from .array_backend import ArrayBackend


class TFBackend(ArrayBackend):
    def __init__(self):
        super().__init__(tf)

    @property
    def name(self):
        return "tensorflow"

    def copy(self, x):
        return x

    def scatter_assign(self, target, indices, updates):
        target_shape = tf.shape(target)
        batch_dims = target_shape[:-1]
        num_batch = tf.reduce_prod(batch_dims)
        num_idx = tf.shape(indices)[0]

        flat_target = tf.reshape(target, (num_batch, -1))
        flat_values = tf.reshape(updates, (num_batch, num_idx))

        batch_range = tf.range(num_batch)[:, None]
        batch_ids = tf.tile(batch_range, [1, num_idx])
        scatter_idx = tf.stack(
            [batch_ids, tf.tile(tf.expand_dims(indices, 0), [num_batch, 1])], axis=-1
        )
        scatter_idx = tf.reshape(scatter_idx, (-1, 2))

        scatter_vals = tf.reshape(flat_values, (-1,))
        flat_result = tf.tensor_scatter_nd_update(
            flat_target, scatter_idx, scatter_vals
        )

        return tf.reshape(flat_result, target_shape)

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

    def gather(self, arr, indices, axis=-1):
        return tf.gather(arr, indices, axis=axis)

    def full_like(self, arr, value):
        return tf.fill(tf.shape(arr), value)

    def full(self, shape, value):
        return tf.fill(shape, value)

    @property
    def nan(self):
        return float("nan")

    def asarray(self, arr, dtype=None, device=None, copy=None):
        tensor = tf.convert_to_tensor(arr, dtype=dtype)

        if copy and device is None:
            tensor = tf.identity(tensor)

        if device is not None:
            with tf.device(device):
                tensor = tf.identity(tensor)
        return tensor
