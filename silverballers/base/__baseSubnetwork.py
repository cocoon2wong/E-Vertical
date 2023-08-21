"""
@Author: Conghao Wong
@Date: 2023-06-06 16:45:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-16 09:14:44
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from typing import Union

import tensorflow as tf

from codes.args import Args
from codes.basemodels import Model
from codes.managers import AgentManager, Model, Structure
from codes.training.loss import ADE_2D, get_loss_mask

from .__args import BaseSilverballersArgs


class BaseSubnetwork(Model):

    def __init__(self, Args: BaseSilverballersArgs,
                 as_single_model: bool = True,
                 structure: Structure = None,
                 *args, **kwargs):

        super().__init__(Args, structure, *args, **kwargs)

        self.args: BaseSilverballersArgs

        # Parameters
        self.as_single_model = as_single_model

        # Preprocess
        preprocess = {}
        for index, operation in enumerate(['move', 'scale', 'rotate']):
            if self.args.preprocess[index] == '1':
                preprocess[operation] = 'auto'

        self.set_preprocess(**preprocess)

        # Keypoints and their indices
        indices = [int(i) for i in self.args.key_points.split('_')]
        self.__ki = tf.cast(indices, tf.int32)

        self.n_key_past: int = tf.reduce_sum(
            tf.cast(self.__ki < 0, tf.int32))
        self.n_key_future: int = tf.reduce_sum(
            tf.cast(self.__ki >= 0, tf.int32))
        self.n_key = self.n_key_past + self.n_key_future

    @property
    def d(self) -> int:
        """
        Feature dimension used in most layers.
        """
        return self.args.feature_dim

    @property
    def d_id(self) -> int:
        """
        Dimension of the noise vectors.
        """
        return self.args.depth

    @property
    def dim(self) -> int:
        """
        Dimension of the predicted trajectory.
        For example, `dim = 4` for 2D bounding boxes.
        """
        return self.structure.annmanager.dim

    @property
    def key_indices_future(self) -> tf.Tensor:
        """
        Indices of the future keypoints.
        Data type is `tf.int32`.
        """
        return self.__ki[self.n_key_past:]

    @property
    def key_indices_past(self) -> tf.Tensor:
        """
        Indices of the past keypoints.
        Data type is `tf.int32`.
        It starts with `0`.
        """
        if self.n_key_past:
            return self.args.obs_frames + self.__ki[:self.n_key_past]
        else:
            return tf.cast([], tf.int32)

    @property
    def picker(self):
        """
        Trajectory picker (from the top manager object).
        """
        return self.get_top_manager().get_member(AgentManager).picker

    def print_info(self, **kwargs):
        info = {'Transform type': self.args.T,
                'Index of keypoints': self.key_indices_future,
                'Index of past keypoints': self.key_indices_past}

        kwargs.update(**info)
        return super().print_info(**kwargs)


class BaseSubnetworkStructure(Structure):

    SUBNETWORK_INDEX = 'Not Assigned'
    ARG_TYPE: type[BaseSilverballersArgs] = BaseSilverballersArgs
    MODEL_TYPE: type[BaseSubnetwork] = None

    def __init__(self, terminal_args: Union[list[str], Args],
                 manager: Structure = None,
                 as_single_model: bool = True):

        name = 'Train Manager'
        if not as_single_model:
            name += f' (Stage-{self.SUBNETWORK_INDEX} Sub-network)'

        if issubclass(type(terminal_args), Args):
            init_args = terminal_args
        else:
            init_args = self.ARG_TYPE(terminal_args,
                                      is_temporary=not as_single_model)

        super().__init__(args=init_args,
                         manager=manager,
                         name=name)

        # For type hinting
        self.args: BaseSilverballersArgs
        self.model: BaseSubnetwork

    def create_model(self, as_single_model=True) -> BaseSubnetwork:
        return self.MODEL_TYPE(self.args, as_single_model,
                               structure=self)

    def set_model_type(self, new_type: type[BaseSubnetwork]):
        self.MODEL_TYPE = new_type

    def substructure(self, structure_type: type[Structure],
                     args: list[str],
                     model_type: type[BaseSubnetwork],
                     create_args: dict = {},
                     load: str = None,
                     **kwargs):
        """
        Init a sub-structure (which contains its corresponding model).

        :param structure: class name of the training structure
        :param args: args to init the training structure
        :param model: class name of the model
        :param create_args: args to create the model, and they will be fed
            to the `structure.create_model` method
        :param load: path to load model weights
        :param **kwargs: a series of force-args that will be assigned to
            the structure's args
        """

        struct = structure_type(args, manager=self,
                                as_single_model=False)

        for key in kwargs.keys():
            struct.args._set(key, kwargs[key])

        struct.set_model_type(model_type)
        struct.model = struct.create_model(**create_args)

        if load:
            struct.model.load_weights_from_logDir(load)

        return struct

    #####################
    # New loss functions
    #####################
    def keyl2(self, outputs: list[tf.Tensor],
              labels: list[tf.Tensor],
              model_inputs: list[tf.Tensor],
              coe: float = 1.0,
              *args, **kwargs):
        """
        l2 loss on the future keypoints.
        Support M-dimensional trajectories.
        """
        indices = self.model.key_indices_future
        labels_pickled = tf.gather(labels[0], indices, axis=-2)
        mask = get_loss_mask(model_inputs[0], labels[0])
        return ADE_2D(outputs[0], labels_pickled,
                      coe=coe, mask=mask)

    def avgKey(self, outputs: list[tf.Tensor],
               labels: list[tf.Tensor],
               model_inputs: list[tf.Tensor],
               coe: float = 1.0,
               *args, **kwargs):
        """
        l2 (2D-point-wise) loss on the future keypoints.

        :param outputs: A list of tensors, where `outputs[0].shape`
            is `(batch, K, pred, 2)` or `(batch, pred, 2)`
            or `(batch, K, n_key, 2)` or `(batch, n_key, 2)`.
        :param labels: Shape of `labels[0]` is `(batch, pred, 2)`.
        """
        pred = outputs[0]
        indices = self.model.key_indices_future

        if pred.ndim == 3:
            pred = pred[:, tf.newaxis, :, :]

        if pred.shape[-2] != len(indices):
            pred = tf.gather(pred, indices, axis=-2)

        labels_key = tf.gather(labels[0], indices, axis=-2)

        return self.loss.ADE([pred], [labels_key], model_inputs, coe=coe)
