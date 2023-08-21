"""
@Author: Conghao Wong
@Date: 2023-05-09 20:30:01
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-09 20:30:04
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf

from .__base import _BaseTransformLayer


class FFTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs) -> tf.Tensor:
        """
        Run FFT on a batch of trajectories.

        :param inputs: batch inputs, \
            shape = `(batch, steps, channels)`
        :return fft: fft results (real and imag), \
            shape = `(batch, steps, 2*channels)`
        """

        ffts = []
        for index in range(0, inputs.shape[-1]):
            seq = tf.cast(tf.gather(inputs, index, axis=-1), tf.complex64)
            seq_fft = tf.signal.fft(seq)
            ffts.append(tf.expand_dims(seq_fft, -1))

        ffts = tf.concat(ffts, axis=-1)
        return tf.concat([tf.math.real(ffts), tf.math.imag(ffts)], axis=-1)


class FFT2DLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Run 2D FFT on a batch of trajectories.

        :param inputs: A batch of input trajectories, \
            shape = `(batch, steps, channels)`.
        :return fft: 2D fft results, including real and imag parts, \
            shape = `(batch, steps, 2*channels)`.
        """

        seq = tf.cast(inputs, tf.complex64)
        fft = tf.signal.fft2d(seq)

        return tf.concat([tf.math.real(fft), tf.math.imag(fft)], axis=-1)


class IFFTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):

        real = tf.gather(inputs, tf.range(
            0, self.channels), axis=-1)
        imag = tf.gather(inputs, tf.range(
            self.channels, 2*self.channels), axis=-1)

        ffts = []
        for index in range(0, real.shape[-1]):
            r = tf.gather(real, index, axis=-1)
            i = tf.gather(imag, index, axis=-1)
            ffts.append(
                tf.expand_dims(
                    tf.math.real(
                        tf.signal.ifft(
                            tf.complex(r, i)
                        )
                    ), axis=-1
                )
            )

        return tf.concat(ffts, axis=-1)


class IFFT2Dlayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, 2*self.channels]

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        real = inputs[..., :self.channels]
        imag = inputs[..., self.channels:]

        seq = tf.complex(real, imag)
        fft = tf.signal.ifft2d(seq)

        return tf.math.real(fft)
    