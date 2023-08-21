"""
@Author: Conghao Wong
@Date: 2022-09-01 10:38:49
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-16 09:26:15
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import numpy as np
import tensorflow as tf

from ...base import BaseObject
from ...constant import ANN_TYPES
from ...dataset import Annotation


class BaseProcessLayer(tf.keras.layers.Layer, BaseObject):

    def __init__(self, anntype: str, ref,
                 *args, **kwargs):

        tf.keras.layers.Layer.__init__(self, *args, **kwargs)
        BaseObject.__init__(self, name=self.name)

        self.ref = ref
        self.anntype = anntype
        self.picker = Annotation(anntype)
        self.paras = None

        self.order = self.set_order(anntype)

    def call(self, inputs: list[tf.Tensor],
             preprocess: bool,
             update_paras=False,
             training=None, *args, **kwargs):
        """
        Run preprocess or postprocess on trajectories

        :param inputs: a list (tuple) of Tensors, where `inputs[0]` are \
            trajectories, whose shapes are `((batch,) (K,) steps, dim)`
        :param preprocess: set to `True` to run preprocess, or set to `False` \
            to run postprocess
        :param update_paras: choose whether to update process parameters
        """

        trajs = inputs[0]
        if preprocess:
            trajs_processed = self.preprocess(
                trajs, use_new_paras=update_paras)

        else:
            trajs_processed = self.postprocess(trajs)

        return update((trajs_processed,), inputs)

    def preprocess(self, trajs: tf.Tensor, use_new_paras=True) -> tf.Tensor:
        raise NotImplementedError('Please rewrite this method')

    def postprocess(self, trajs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError('Please rewrite this method')

    def update_paras(self, trajs: tf.Tensor) -> None:
        raise NotImplementedError('Please rewrite this method')

    def set_order(self, anntype: str):
        if anntype is None:
            return None

        if anntype == ANN_TYPES.CO_2D:
            order = [[0, 1]]
        elif anntype == ANN_TYPES.BB_2D:
            order = [[0, 1], [2, 3]]
        elif anntype == ANN_TYPES.BB_3D:
            order = [[0, 1, 2], [3, 4, 5]]
        elif anntype == ANN_TYPES.SKE_3D_17:
            order = np.arange(17*3).reshape([17, 3])
        else:
            raise NotImplementedError(anntype)

        return order


class ProcessModel(tf.keras.Model):

    def __init__(self, layers: list[BaseProcessLayer], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.players = layers

    def call(self, inputs: list[tf.Tensor],
             preprocess: bool,
             update_paras=True,
             training=None,
             *args, **kwargs) -> list[tf.Tensor]:

        if preprocess:
            layers = self.players

        else:
            layers = self.players[::-1]

        for p in layers:
            inputs = p(inputs, preprocess,
                       update_paras, training,
                       *args, **kwargs)
        return inputs


def update(new: Union[tuple, list],
           old: Union[tuple, list]) -> tuple:

    if type(old) == list:
        old = tuple(old)
    if type(new) == list:
        new = tuple(new)

    if len(new) < len(old):
        return new + old[len(new):]
    else:
        return new
