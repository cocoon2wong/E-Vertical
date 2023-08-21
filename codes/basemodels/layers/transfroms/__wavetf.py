"""
@Author: Conghao Wong
@Date: 2023-05-09 20:28:47
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-30 09:53:29
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf

from ...wavetf import WaveTFFactory
from .__base import _BaseTransformLayer


class Haar1D(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        if self.Oshape[0] % 2 == 1:
            raise ValueError('`steps` in haar wavelet must be an even')

        self.haar = WaveTFFactory.build(kernel_type='haar',
                                        dim=1,
                                        inverse=False)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):

        # (batch, steps, channels) -> (batch, steps//2, 2*channels)
        haar = self.haar(inputs)

        return haar


class InverseHaar1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        if self.Oshape[0] % 2 == 1:
            raise ValueError('`steps` in haar wavelet must be an even')

        self.haar = WaveTFFactory.build(kernel_type='haar',
                                        dim=1,
                                        inverse=True)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs) -> tf.Tensor:

        # (batch, steps//2, 2*channels) -> (batch, steps, channels)
        r = self.haar(inputs)

        return r


class DB2_1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.daub = WaveTFFactory.build(kernel_type='db2',
                                        dim=1,
                                        inverse=False)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):
        return self.daub(inputs)


class InverseDB2_1D(_BaseTransformLayer):

    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.daub = WaveTFFactory.build(kernel_type='db2',
                                        dim=1,
                                        inverse=True)

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return (self.steps//2, self.channels*2)

    def kernel_function(self, inputs: tf.Tensor, *args, **kwargs):
        return self.daub(inputs)
