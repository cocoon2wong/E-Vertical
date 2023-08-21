"""
@Author: Conghao Wong
@Date: 2022-06-22 09:35:52
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:24:02
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import tensorflow as tf

from codes.constant import INPUT_TYPES
from codes.managers import (AgentManager, MapParasManager, SecondaryBar,
                            Structure)
from codes.utils import POOLING_BEFORE_SAVING

from ..__args import SilverballersArgs
from ..base import BaseSubnetwork, BaseSubnetworkStructure
from .__args import HandlerArgs


class BaseHandlerModel(BaseSubnetwork):

    is_interp_handler = False

    def __init__(self, Args: HandlerArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, as_single_model, structure, *args, **kwargs)

        # For type hinting
        self.args: HandlerArgs
        self.structure: BaseHandlerStructure

        # Configs
        # GT in the inputs is only used when training
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.MAP,
                        INPUT_TYPES.MAP_PARAS,
                        INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Keypoints and their indices
        self.points = self.args.points
        self.key_points = self.args.key_points
        self.accept_batchK_inputs = False

        self.ext_traj_wise_outputs[1] = 'Interaction Scores'

        if POOLING_BEFORE_SAVING:
            self._upsampling = tf.keras.layers.UpSampling2D(
                size=[5, 5], data_format='channels_last')

    def call(self, inputs: list[tf.Tensor],
             keypoints: tf.Tensor,
             keypoints_index: tf.Tensor,
             training=None, mask=None):

        raise NotImplementedError

    def call_as_handler(self, inputs: list[tf.Tensor],
                        keypoints: tf.Tensor,
                        keypoints_index: tf.Tensor,
                        training=None, mask=None):
        """
        Call as the second stage handler model.
        Do NOT call this method when training.

        :param inputs: a list of trajs and context maps
        :param keypoints: predicted keypoints, shape is `(batch, K, n_k, 2)`
        :param keypoints_index: index of predicted keypoints, shape is `(n_k)`
        """

        if not self.accept_batchK_inputs:
            p_all = []
            for k in SecondaryBar(range(keypoints.shape[1]),
                                  manager=self.structure.manager,
                                  desc='Running Stage-2 Sub-Network...',
                                  update_main_bar=True):

                # Run stage-2 network on a batch of inputs
                pred = self(inputs=inputs,
                            keypoints=keypoints[:, k, :, :],
                            keypoints_index=keypoints_index)

                if type(pred) not in [list, tuple]:
                    pred = [pred]

                # A single output shape is (batch, pred, dim).
                p_all.append(pred[0])

            return tf.transpose(tf.stack(p_all), [1, 0, 2, 3])

        else:
            return self(inputs=inputs,
                        keypoints=keypoints,
                        keypoints_index=keypoints_index)

    def forward(self, inputs: list[tf.Tensor],
                training=None,
                *args, **kwargs):

        keypoints = [inputs[-1]]

        inputs_p = self.process(inputs, preprocess=True, training=training)
        keypoints_p = self.process(keypoints, preprocess=True,
                                   update_paras=False,
                                   training=training)

        # only when training the single model
        if self.as_single_model:
            gt_processed = keypoints_p[0]

            if self.key_points == 'null':
                index = np.arange(self.args.pred_frames-1)
                np.random.shuffle(index)
                index = tf.concat([np.sort(index[:self.points-1]),
                                   [self.args.pred_frames-1]], axis=0)
            else:
                index = self.key_indices_future

            points = tf.gather(gt_processed, index, axis=1)
            index = tf.cast(index, tf.float32)
            outputs = self(inputs_p,
                           keypoints=points,
                           keypoints_index=index,
                           training=True)

        # use as the second stage model
        else:
            outputs = self.call_as_handler(
                inputs_p,
                keypoints=keypoints_p[0],
                keypoints_index=tf.cast(self.key_indices_future, tf.float32),
                training=None)

        outputs_p = self.process(outputs, preprocess=False, training=training)
        pred_o = outputs_p[0]

        # Calculate scores
        if ((INPUT_TYPES.MAP in self.input_types)
                and (INPUT_TYPES.MAP_PARAS in self.input_types)):

            map_mgr = self.get_top_manager().get_member(
                AgentManager).get_member(MapParasManager)
            scores = map_mgr.score(trajs=outputs_p[0],
                                   maps=inputs[1],
                                   map_paras=inputs[2],
                                   centers=inputs[0][..., -1, :])

            # Pick trajectories
            # Only work when it play as the subnetwork
            if not self.as_single_model:
                run_args: SilverballersArgs = self.get_top_manager().args
                if (p := run_args.pick_trajectories) < 1.0:
                    pred_o = map_mgr.pick_trajectories(pred_o, scores, p)

            return (pred_o, scores) + outputs_p[1:]

        else:
            return outputs_p

    def print_info(self, **kwargs):
        info = {'Transform type': self.args.T,
                'Number of keypoints': self.args.points}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseHandlerStructure(BaseSubnetworkStructure):

    SUBNETWORK_INDEX = '2'
    ARG_TYPE = HandlerArgs
    MODEL_TYPE: type[BaseHandlerModel] = None

    def __init__(self, terminal_args: list[str],
                 manager: Structure = None,
                 as_single_model: bool = True):

        super().__init__(terminal_args, manager, as_single_model)

        # For type hinting
        self.args: HandlerArgs
        self.model: BaseHandlerModel

        # Configs, losses, and metrics
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)
        self.loss.set({self.loss.l2: 1.0})

        if self.args.key_points == 'null':
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0})
        else:
            self.metrics.set({self.metrics.ADE: 1.0,
                              self.metrics.FDE: 0.0,
                              self.avgKey: 0.0})
