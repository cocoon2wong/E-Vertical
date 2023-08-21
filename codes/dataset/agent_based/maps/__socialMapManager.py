"""
@Author: Conghao Wong
@Date: 2023-05-22 16:26:26
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:33:10
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ....base import BaseManager, SecondaryBar
from ....constant import INPUT_TYPES
from ....utils import AVOID_SIZE, INTEREST_SIZE
from ...__base import BaseInputManager
from ...__splitManager import Clip
from ..__inputObject import Agent
from .__trajMapManager import TrajMapManager
from .__utils import add, cut, pooling2D


class SocialMapManager(TrajMapManager):
    """
    Social Map Manager
    ---
    The social map is a map that builds from all neighbor agents'
    observed trajectories. It indicates their potential social
    interactions in the prediction period. The value of the social map
    is in the range `[0, 1]`. A higher value indicates that
    the area may not suitable for walking under different kinds of
    social interactions.
    """

    TEMP_FILES = {'FILE': 'socialMap.npy'}

    MAP_NAME = 'Social Map'
    INPUT_TYPE = INPUT_TYPES.MAP

    def __init__(self, manager: BaseManager,
                 pool_maps=False,
                 name='Social Map Manager'):

        super().__init__(manager, pool_maps, name)

        if pool_maps:
            self.TEMP_FILES['FILE_WITH_POOLING'] = 'socialMap_pooling.npy'

    def run(self, clip: Clip,
            agents: list[Agent],
            regulation=True,
            max_neighbor=15,
            *args, **kwargs) -> list:

        return BaseInputManager.run(self, clip=clip, agents=agents,
                                    regulation=regulation,
                                    max_neighbor=max_neighbor,
                                    *args, **kwargs)

    def save(self, agents: list[Agent],
             source: np.ndarray = None,
             regulation=True,
             max_neighbor=15,
             *args, **kwargs):
        """
        Build and save local social maps for all agents.

        - Required files: No required files.
        - Saved files: `FILE`.

        :param agents: The target `Agent` objects to calculate the map.
        :param source: The source map, default are zeros.
        :param regulation: Controls if scale the map into [0, 1].
        :param max_neighbor: The maximum number of neighbors to calculate
            the social map. Set it to a smaller value to speed up the building
            on datasets that contain more agents.
        """
        maps = []
        for agent in SecondaryBar(agents,
                                  manager=self.manager,
                                  desc=f'Building {self.MAP_NAME}...'):

            # build the global social map
            if source is None:
                source = self.map_mgr.void_map

            source = source.copy()

            # Get center points
            traj = self.C(agent.traj)
            pred_linear = self.C(agent.pred_linear)
            pred_linear_neighbor = self.C(agent.pred_linear_neighbor)

            # Destination
            source = add(target_map=source,
                         grid_trajs=self.real2grid(pred_linear),
                         amplitude=[-2],
                         radius=INTEREST_SIZE)

            # Interplay
            traj_neighbors = pred_linear_neighbor
            amp_neighbors = []

            vec_target = pred_linear[-1] - pred_linear[0]
            len_target = calculate_length(vec_target)

            vec_neighbor = traj_neighbors[:, -1] - traj_neighbors[:, 0]

            if len_target >= 0.05:
                cosine = activation(
                    calculate_cosine(vec_target[np.newaxis, :], vec_neighbor),
                    a=1.0,
                    b=0.2)
                velocity = (calculate_length(vec_neighbor) /
                            calculate_length(vec_target[np.newaxis, :]))

            else:
                cosine = np.ones(len(traj_neighbors))
                velocity = 2

            amp_neighbors = - cosine * velocity

            amps = amp_neighbors.tolist()
            trajs = traj_neighbors.tolist()

            if len(trajs) > max_neighbor + 1:
                trajs = np.array(trajs)
                dis = calculate_length(trajs[:1, 0, :] - trajs[:, 0, :])
                index = np.argsort(dis)
                trajs = trajs[index[:max_neighbor+1]]

            source = add(target_map=source,
                         grid_trajs=self.real2grid(trajs),
                         amplitude=amps,
                         radius=AVOID_SIZE)

            if regulation:
                if (np.max(source) - np.min(source)) <= 0.01:
                    source = 0.5 * np.ones_like(source)
                else:
                    source = (source - np.min(source)) / \
                        (np.max(source) - np.min(source))

            # Get the local social map from the global map
            # center point: the last observed point
            center_real = traj[-1:, :]
            center_pixel = self.real2grid(center_real)
            local_map = cut(source, center_pixel, self.HALF_SIZE)[0]
            maps.append(local_map)

        # Save maps
        np.save(self.temp_files['FILE'], maps)

        # Pool the maps
        maps_pooling = pooling2D(np.array(maps))
        np.save(self.temp_files['FILE_WITH_POOLING'], maps_pooling)


def calculate_cosine(vec1: np.ndarray,
                     vec2: np.ndarray):

    length1 = np.linalg.norm(vec1, axis=-1)
    length2 = np.linalg.norm(vec2, axis=-1)

    return (np.sum(vec1 * vec2, axis=-1) + 0.0001) / ((length1 * length2) + 0.0001)


def calculate_length(vec1):
    return np.linalg.norm(vec1, axis=-1)


def activation(x: np.ndarray, a=1, b=1):
    return np.less_equal(x, 0) * a * x + np.greater(x, 0) * b * x
