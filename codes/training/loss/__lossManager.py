"""
@Author: Conghao Wong
@Date: 2022-10-12 11:13:46
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-16 09:11:12
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import Any, Union

import tensorflow as tf

from ...base import BaseManager
from ...dataset import AnnotationManager
from ...utils import get_loss_mask
from .__ade import ADE_2D
from .__iou import AIoU, FIoU


class LossManager(BaseManager):

    def __init__(self, manager: BaseManager, name='Loss Manager'):
        """
        Init a `LossManager` object.

        :param manager: The manager object, usually a `Structure` object.
        :param name: The name of the manager, which could appear in all dict
            keys in the final output `loss_dict`.
        """

        super().__init__(manager=manager, name=name)

        self.AIoU = AIoU
        self.FIoU = FIoU

        self.loss_list: list[Any] = []
        self.loss_weights: list[float] = []
        self.loss_paras: list[dict] = []

    @property
    def pickers(self) -> AnnotationManager:
        return self.manager.get_member(AnnotationManager)

    def set(self, loss_dict: Union[dict[Any, float],
                                   list[tuple[Any, tuple[float, dict]]]]):
        """
        Set loss functions and their weights.

        :param loss_dict: A dict of loss functions, where all dict keys
            are the callable loss function, and the dict values are the
            weights of the corresponding loss function.
            Accept other parameters of the loss function from a `dict`.
            For example,  `self.metrics.set([[self.metrics.FDE, [1.0, 
            {'index': 1, 'name': 'FDE@200ms'}]]])`.
            NOTE: The callable loss function must have the `**kwargs` in
            their definitions.
        """
        self.loss_list = []
        self.loss_weights = []
        self.loss_paras = []

        if type(loss_dict) is dict:
            items = loss_dict.items()
        elif type(loss_dict) in [list, tuple]:
            items = loss_dict
        else:
            raise TypeError(loss_dict)

        for k, vs in items:
            if type(vs) in [list, tuple]:
                v = vs[0]
                p = vs[1]
            else:
                v = vs
                p = {}

            self.loss_list.append(k)
            self.loss_weights.append(v)
            self.loss_paras.append(p)

    def call(self, outputs: list[tf.Tensor],
             labels: list[tf.Tensor],
             training=None,
             coefficient: float = 1.0,
             model_inputs: list[tf.Tensor] = None):
        """
        Call all loss functions recorded in the `loss_list`.

        :param outputs: A list of the model's output tensors. \
            `outputs[0]` should be the predicted trajectories.
        :param labels: A list of groundtruth tensors. \
            `labels[0]` should be the groundtruth trajectories.
        :param training: Choose whether to run as the training mode.
        :param coefficient: The scale parameter on the loss functions.

        :return summary: The weighted sum of all loss functions.
        :return loss_dict: A dict of values of all loss functions.
        """

        loss_dict = {}
        for loss_func, paras in zip(self.loss_list, self.loss_paras):
            name = loss_func.__name__
            if len(paras):
                if 'name' in paras.keys():
                    name = paras['name']
                else:
                    name += f'@{paras}'

            value = loss_func(outputs, labels,
                              coe=coefficient,
                              training=training,
                              model_inputs=model_inputs,
                              **paras)

            loss_dict[f'{name}({self.name})'] = value

        if (l := len(self.loss_weights)):
            if l != len(loss_dict):
                raise ValueError('Incorrect loss weights!')
            weights = self.loss_weights

        else:
            weights = tf.ones(len(loss_dict))

        summary = tf.matmul(tf.expand_dims(list(loss_dict.values()), 0),
                            tf.expand_dims(weights, 1))
        summary = tf.reshape(summary, ())
        return summary, loss_dict

    ####################################
    # Loss functions are defined below
    ####################################

    def l2(self, outputs: list[tf.Tensor],
           labels: list[tf.Tensor],
            model_inputs: list[tf.Tensor],
           coe: float = 1.0,
           *args, **kwargs):
        """
        l2 loss on the keypoints.
        Support M-dimensional trajectories.
        """
        mask = get_loss_mask(model_inputs[0], labels[0])
        return ADE_2D(outputs[0], labels[0], coe=coe, mask=mask)

    def ADE(self, outputs: list[tf.Tensor],
            labels: list[tf.Tensor],
            model_inputs: list[tf.Tensor],
            coe: float = 1.0,
            *args, **kwargs):
        """
        l2 (2D-point-wise) loss.
        Support M-dimensional trajectories.

        :param outputs: A list of tensors, where `outputs[0].shape` 
            is `(batch, K, pred, 2)` or `(batch, pred, 2)`.
        :param labels: Shape of `labels[0]` is `(batch, pred, 2)`.
        """
        pred = outputs[0]
        obs = model_inputs[0]

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        ade = []
        picker = self.pickers.target.get_coordinate_series
        for p, gt in zip(picker(pred), picker(labels[0])):
            mask = get_loss_mask(obs, gt)
            ade.append(ADE_2D(p, gt, coe, mask=mask))

        return tf.reduce_mean(ade)

    def avgCenter(self, outputs: list[tf.Tensor],
                  labels: list[tf.Tensor],
                  model_inputs: list[tf.Tensor],
                  coe: float = 1.0,
                  *args, **kwargs):
        """
        Average displacement error on the center of each prediction.
        """
        picker = self.pickers.target.get_center
        pred_center = picker(outputs[0])
        gt_center = picker(labels[0])
        mask = get_loss_mask(model_inputs[0], gt_center)
        return ADE_2D(pred_center, gt_center, coe, mask)

    def finalCenter(self, outputs: list[tf.Tensor],
                    labels: list[tf.Tensor],
                    model_inputs: list[tf.Tensor],
                    coe: float = 1.0,
                    *args, **kwargs):
        """
        Final displacement error on the center of each prediction.
        """
        picker = self.pickers.target.get_center
        pred_center = picker(outputs[0])
        gt_center = picker(labels[0])
        mask = get_loss_mask(model_inputs[0], gt_center)
        return ADE_2D(pred_center[..., -1:, :], gt_center[..., -1:, :], coe, mask)

    def FDE(self, outputs: list[tf.Tensor],
            labels: list[tf.Tensor],
            model_inputs: list[tf.Tensor],
            index: int = -1,
            coe: float = 1.0,
            *args, **kwargs):
        """
        l2 (2D-point-wise) loss on the last prediction point.
        Support M-dimensional trajectories.

        :param outputs: A list of tensors, where 
            `outputs[0].shape` is `(batch, K, pred, 2)`
            or `(batch, pred, 2)`.
        :param labels: Shape of `labels[0]` is `(batch, pred, 2)`.
        :param index: Index of the time step to calculate the FDE.
        """
        pred = outputs[0]

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        pred_final = pred[..., index, tf.newaxis, :]
        labels_final = labels[0][..., index, tf.newaxis, :]

        return self.ADE([pred_final], [labels_final],
                        model_inputs, coe)

    def print_info(self, **kwargs):
        funcs = [f.__name__ for f in self.loss_list]
        return super().print_info(Functions=funcs,
                                  Weights=self.loss_weights,
                                  **kwargs)
