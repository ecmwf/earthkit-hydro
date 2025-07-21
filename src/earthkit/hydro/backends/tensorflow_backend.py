import tensorflow as tf

from .array_backend import ArrayBackend


class TFBackend(ArrayBackend):
    def __init__(self):
        super().__init__(tf)

    def copy(self, x):
        return x

    def scatter_add(self, target, indices, updates):
        indices = tf.expand_dims(tf.convert_to_tensor(indices), axis=1)
        updates = tf.tensor_scatter_nd_add(tf.zeros_like(target), indices, updates)
        # updates = tf.math.unsorted_segment_sum(
        # data=update_with, segment_ids=did, num_segments=network.n_nodes
        # )
        return target + updates

    def take_along_axis(self, arr, indices, axis=-1):
        # indices_expanded = tf.reshape(indices, [1] * (tf.rank(arr) - 1) + [-1])
        # return tf.gather(arr, indices_expanded, batch_dims=tf.rank(arr) - 1, axis=-1)
        update_with = tf.gather(arr, indices)
        return update_with
