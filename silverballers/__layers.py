"""
@Author: Conghao Wong
@Date: 2022-06-20 21:50:44
@LastEditors: Conghao Wong
@LastEditTime: 2023-04-14 09:33:29
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class OuterLayer(tf.keras.layers.Layer):
    """
    Compute the outer product of two vectors.

    :param a_dim: the last dimension of the first input feature
    :param b_dim: the last dimension of the second input feature
    :param reshape: if `reshape == True`, output shape = `(..., a_dim * b_dim)`
        else output shape = `(..., a_dim, b_dim)`
    """

    def __init__(self, a_dim: int, b_dim: int,
                 reshape=False,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.M = a_dim
        self.N = b_dim
        self.reshape = reshape

    def call(self, tensorA: tf.Tensor, tensorB: tf.Tensor):
        """
        Compute the outer product of two vectors.

        :param tensorA: shape = (..., M)
        :param tensorB: shape = (..., N)
        :return outer: shape = (..., M, N) if `reshape` is `False`,
            else its output shape = (..., M*N)
        """

        _a = tf.expand_dims(tensorA, axis=-1)
        _b = tf.expand_dims(tensorB, axis=-2)

        _a = tf.repeat(_a, self.N, axis=-1)
        _b = tf.repeat(_b, self.M, axis=-2)

        outer = _a * _b

        if not self.reshape:
            return outer
        else:
            return tf.reshape(outer, list(outer.shape[:-2]) + [self.M*self.N])
