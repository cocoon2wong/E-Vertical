"""
@Author: Conghao Wong
@Date: 2022-08-30 09:52:17
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 15:16:14
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from typing import TypeVar, Union

import numpy as np
import tensorflow as tf

from ...base import BaseManager
from ...utils import ANNOTATION_CONFIGS_FILE, load_from_plist

T = TypeVar('T')


class Annotation():

    annotation_configs: dict = None

    def __init__(self, name: str) -> None:

        if self.annotation_configs is None:
            Annotation.load_dict()

        if not name in self.annotation_configs.keys():
            raise KeyError(f'Annotation type `{name}` is not supported!')

        self.__name = name
        self.info = self.annotation_configs[name]

    @classmethod
    def load_dict(cls):
        cls.annotation_configs = load_from_plist(ANNOTATION_CONFIGS_FILE)

    @property
    def dim(self) -> int:
        """
        Dimension of the annotation.
        """
        return self.info['dim']

    @property
    def base_dim(self) -> int:
        """
        Base dimension of the annotation.
        If the annotation consists of a series of points with the same
        dimension, then `base_dim` is the dimension of these points.
        Otherwise, we set `base_dim = -1`.
        For example, for a 2D bounding box `(xl, xr, yl, yr)`, we have 
        `base_dim = 2`.
        """
        return self.info['base_dim']

    @property
    def base_len(self) -> int:
        """
        Data length of the series of points with the same dimension.
        If the annotation dose not consist of a series of points with
        the same dimension, then we set `base_len = -1`.
        For example, for a 2D bounding box with a rotation value
        `(xl, xr, yl, yr, r)`, we have `base_len = 4`.
        """
        return self.info['base_len']

    @property
    def name(self) -> str:
        """
        Name of the annotation type.
        """
        return self.__name

    def get_center(self, inputs: T) -> T:
        """
        Get the center points of the input trajectories.

        :param inputs: Input trajectories. Accept `np.ndarray` or `tf.Tensor`.
        """

        coordinates = self.get_coordinate_series(inputs)
        if isNumpy(inputs):
            center = np.mean(coordinates, axis=0)
        elif isTensor(inputs):
            center = tf.reduce_mean(tf.stack(coordinates), axis=0)
        else:
            raise TypeError(f'Wrong trajectory type `{type(inputs)}`.')

        return center

    def get_coordinate_series(self, inputs: T) -> T:
        """
        Reshape the **predicted trajectories** into a series of single
        coordinates. For example, when inputs have the annotation type
        `2Dboundingbox`, then this function will reshape it into two
        slices of single 2D coordinates with shapes `(..., steps, 2)`,
        and then return a list containing them.

        :param inputs: Input trajectories. Accept `np.ndarray` or `tf.Tensor`.
        """
        if -1 in [self.base_dim, self.base_len]:
            raise ValueError(f'Can not get a series of coordinates from \
                             trajectories with type `{self.name}`.')

        results = []
        for p_index in range(self.base_len//self.base_dim):
            results.append(inputs[..., p_index*self.base_dim:
                                  (p_index+1)*self.base_dim])

        return results


class AnnotationManager(BaseManager):
    """
    Annotation Manager
    ---
    A manager to control all annotations and their transformations
    in dataset files and prediction models. The `AnnotationManager`
    object is managed by the `Structure` object directly.
    """

    def __init__(self, manager: BaseManager,
                 dataset_type: str,
                 name: str = 'Annotation Manager'):

        super().__init__(manager=manager, name=name)

        self.annotations: dict[str, Annotation] = {}

        self.Tsource = dataset_type
        self.Ttarget = self.args.anntype

        if self.args.force_anntype != 'null':
            self.Ttarget = self.args.force_anntype

        self.source = self.get_annotation(self.Tsource)
        self.target = self.get_annotation(self.Ttarget)

    @property
    def dim(self) -> int:
        """
        Dimension of the target prediction trajectory.
        """
        return self.target.dim

    def add_annotation(self, name: str) -> None:
        self.annotations[name] = Annotation(name)

    def get_annotation(self, name: str) -> Annotation:
        if name not in self.annotations.keys():
            self.add_annotation(name)
        return self.annotations[name]

    def get(self, inputs: Union[tf.Tensor, np.ndarray]):
        """
        Get data with target annotation forms from original dataset files.
        """
        if self.Tsource == self.Ttarget:
            return inputs

        flag = True

        if ((self.source.base_dim != self.target.base_dim) or
                (self.source.base_len < self.target.base_len)):
            flag = False

        if ((self.source.base_len % self.target.base_len != 0) or
                (self.target.dim - self.target.base_len > 0)):
            flag = False

        if not flag:
            self.log(f'Can not tranfer annotations with different ' +
                     f'base-dimensions `{self.source.name}` and ' +
                     f'{self.target.name}!',
                     level='error', raiseError=ValueError)

        # align coordinates
        outputs = inputs[..., :self.source.base_len]

        if ((self.source.base_len % self.target.base_len == 0) and
                (self.source.base_len != self.target.base_len)):
            outputs = self.source.get_center(outputs)

        return outputs

    def print_info(self, **kwargs):
        info = {'Dataset annotation type': self.Tsource,
                'Model prediction type': self.Ttarget}

        kwargs.update(**info)
        return super().print_info(**kwargs)


def isNumpy(value):
    if issubclass(type(value), np.ndarray):
        return True
    else:
        return False


def isTensor(value):
    if issubclass(type(value), tf.Tensor):
        return True
    else:
        return False
