"""
@Author: Conghao Wong
@Date: 2022-11-29 09:26:00
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-13 10:38:07
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from codes.constant import INPUT_TYPES
from codes.managers import Structure

from ..__args import HandlerArgs
from ..__baseHandler import BaseHandlerModel


class _BaseInterpHandlerModel(BaseHandlerModel):
    """
    The basic interpolation handler model.
    Subclass this class and rewrite the `interp` method to add 
    different interpolation layers.
    """

    is_interp_handler = True

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        self.args._set('T', 'none')
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.set_preprocess()

        self.accept_batchK_inputs = True
        self.interp_layer = None

        self.ext_traj_wise_outputs = {}
        self.ext_agent_wise_outputs = {}

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        # Unpack inputs
        trajs = inputs[0]

        if keypoints.ndim >= 4:     # (batch, K, steps, dim)
            K = keypoints.shape[-3]
            trajs = tf.repeat(trajs[..., tf.newaxis, :, :], K, axis=-3)

        return self.interp(keypoints_index, keypoints, obs_traj=trajs)

    def interp(self, index: tf.Tensor,
               value: tf.Tensor,
               obs_traj: tf.Tensor) -> tf.Tensor:

        raise NotImplementedError('Please rewrite this method.')
