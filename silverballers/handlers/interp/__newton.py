"""
@Author: Conghao Wong
@Date: 2022-11-29 09:49:26
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-07 17:05:18
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.basemodels.layers import interpolation
from codes.managers import Structure

from ..__args import HandlerArgs
from .__baseInterpHandler import _BaseInterpHandlerModel


class NewtonHandlerModel(_BaseInterpHandlerModel):

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)
        self.interp_layer = interpolation.NewtonInterpolation()

    def interp(self, index: tf.Tensor, value: tf.Tensor, obs_traj: tf.Tensor) -> tf.Tensor:
        # Concat keypoints with the last observed point
        index = tf.concat([[-1], index], axis=0)
        obs_position = obs_traj[..., -1:, :]
        value = tf.concat([obs_position, value], axis=-2)

        # Calculate linear interpolation -> (batch, pred, 2)
        return self.interp_layer(index, value, ord=len(index)-1)
