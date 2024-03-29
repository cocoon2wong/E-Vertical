"""
@Author: Conghao Wong
@Date: 2022-11-21 10:15:13
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-15 15:07:32
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...base import BaseObject


class _BasePooling2D(tf.keras.layers.Layer, BaseObject):
    """
    The base pooling layer that supports both CPU and GPU.
    """

    pool_function: type[tf.keras.layers.MaxPooling2D] = None

    def __init__(self, pool_size=(2, 2), strides=None,
                 padding: str = 'valid',
                 data_format: str = None,
                 *args, **kwargs):

        tf.keras.layers.Layer.__init__(self, *args, **kwargs)
        BaseObject.__init__(self, name=self.name)

        self.gpu = is_gpu()
        self.transpose = False
        self.data_format = data_format

        # Pool layer with 'channels_first' runs only on gpus
        if (not self.gpu) and (self.data_format == 'channels_first'):
            self.log(f'Pooling layer with `data_format = "{self.data_format}"`' +
                     ' can not run on CPUs. It has been automatically changed to' +
                     ' `data_format = "channels_last"`.')
            self.data_format = 'channels_last'
            self.transpose = True

        self.pool_layer = self.pool_function(pool_size, strides,
                                             padding, self.data_format, **kwargs)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        """
        Run the 2D pooling operation.

        :param inputs: The input tensor, shape = `(..., channels, a, b)`
        """
        reshape = False
        if inputs.ndim >= 5:
            reshape = True
            s = list(tf.shape(inputs))
            inputs = tf.reshape(inputs, [-1] + s[-3:])

        # Pool layer with 'channels_first' runs only on gpus
        if self.transpose:
            # Reshape the input to (..., a, b, channels)
            i_reshape = tf.transpose(inputs, [0, 2, 3, 1])
            pooled = self.pool_layer(i_reshape)
            res = tf.transpose(pooled, [0, 3, 1, 2])
        else:
            res = self.pool_layer(inputs)

        if reshape:
            s1 = list(tf.shape(res))
            res = tf.reshape(res, s[:-3] + s1[1:])

        return res


class MaxPooling2D(_BasePooling2D):

    pool_function = tf.keras.layers.MaxPooling2D

    def __init__(self, pool_size=(2, 2), strides=None,
                 padding: str = 'valid',
                 data_format: str = None,
                 *args, **kwargs):

        super().__init__(pool_size, strides, padding,
                         data_format, *args, **kwargs)


def is_gpu():
    gpu_devices = tf.config.list_physical_devices('GPU')
    if len(gpu_devices):
        return True
    else:
        return False
