"""
@Author: Conghao Wong
@Date: 2023-06-15 15:28:10
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-15 17:00:02
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf


class Flatten(tf.keras.layers.Layer):

    def __init__(self, axes_num: int, *args, **kwargs):
        """
        Flatten the input on the given number of axes.
        It will flatten values in the last `axes_num` axes.
        For example, when `axes_num=2`, it outputs a tensor
        with shape `(a, b, c*d)` for the input tensor with
        shape `(a, b, c, d)`.
        """
        super().__init__(*args, **kwargs)

        self.trainable = False
        self.n = axes_num

    def call(self, inputs, *args, **kwargs):
        s = list(tf.shape(inputs))
        o = tf.reshape(inputs, s[:-self.n] + [-1])
        return o


class Padding(tf.keras.layers.Layer):

    def __init__(self, axis: int,
                 value: float = 0,
                 pos: str = 'end',
                 *args, **kwargs):
        """
        Padding the input Tensor.
        `pos` can be `'start'` or `'end'`.
        """
        super().__init__(*args, **kwargs)

        self.trainable = False
        self.axis = axis
        self.v = value

        if pos == 'start':
            self.pos = 1
        elif pos == 'end':
            self.pos = 0
        else:
            raise ValueError(pos)

    def call(self, inputs, padding: int, *args, **kwargs):
        ndim = inputs.ndim
        paddings = padding * \
            tf.one_hot(tf.math.mod(self.axis, ndim), ndim, dtype=tf.int32)
        zeros = tf.zeros([ndim], dtype=tf.int32)

        if self.pos == 1:
            paddings = tf.stack([paddings, zeros])
        else:
            paddings = tf.stack([zeros, paddings])

        return tf.pad(inputs, tf.transpose(paddings),
                      constant_values=self.v)
