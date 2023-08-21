"""
@Author: Conghao Wong
@Date: 2023-05-09 20:25:20
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-09 20:26:47
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf

from .__base import _BaseTransformLayer


class DCTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, self.channels]
    
    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs) -> tf.Tensor:
        
        dcts = []
        for index in range(tf.shape(inputs)[-1]):
            seq_dct = tf.signal.dct(inputs[..., index])
            dcts.append(seq_dct)

        return tf.stack(dcts, axis=-1)
    

class IDCTLayer(_BaseTransformLayer):
    def __init__(self, Oshape: tuple[int, int], *args, **kwargs):
        super().__init__(Oshape, *args, **kwargs)

        self.mode = 0

    def set_Tshape(self) -> Union[list[int], tuple[int, int]]:
        return [self.steps, self.channels]
    
    def kernel_function(self, inputs: tf.Tensor,
                        *args, **kwargs) -> tf.Tensor:
        
        idcts = []
        for index in range(tf.shape(inputs)[-1]):
            seq_idct = tf.signal.idct(inputs[..., index])
            idcts.append(seq_idct)

        return tf.stack(idcts, axis=-1)
    