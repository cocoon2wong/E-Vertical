"""
@Author: Conghao Wong
@Date: 2023-05-22 16:26:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:33:42
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""


import cv2
import numpy as np

from ....base import BaseManager
from ....constant import INPUT_TYPES
from ....utils import SEG_IMG
from ...__base import BaseInputManager
from ...__splitManager import Clip, SplitManager
from ..__inputObject import Agent
from .__trajMapManager import TrajMapManager


class TrajMapManager_seg(TrajMapManager):
    """
    Trajectory Map Manager (with Segmentation Maps)
    ---
    The trajectory map is a map that builds from all agents'
    observed trajectories. It indicates all possible walkable
    areas around the target agent. The value of the trajectory map
    is in the range `[0, 1]`. A higher value indicates that
    the area may not walkable.
    """

    TEMP_FILES = {'FILE': 'trajMap_seg.npy',
                  'GLOBAL_FILE': 'trajMap_seg.png'}

    MAP_NAME = 'Trajectory Map (Segmentation)'
    INPUT_TYPE = INPUT_TYPES.MAP

    def __init__(self, manager: BaseManager,
                 pool_maps=False,
                 name='Trajectory Map Manager (Segmentation)'):

        super().__init__(manager, pool_maps, name)

        if pool_maps:
            self.TEMP_FILES['FILE_WITH_POOLING'] = 'trajMap_seg_pooling.npy'

    def run(self, clip: Clip, trajs: np.ndarray,
            agents: list[Agent], *args, **kwargs) -> list:

        if not self.map_mgr.use_seg_map:
            return 0
        else:
            return BaseInputManager.run(self, clip=clip, trajs=trajs,
                                        agents=agents, *args, **kwargs)

    def build_and_save_global_map(self, trajs: np.ndarray,
                                  source: np.ndarray = None):
        """
        Build and save the global trajectory map.

        - Saved files: `GLOBAL_FILE`, `GLOBAL_CONFIG_FILE`.
        """

        # Load the segmentation image
        f_seg = cv2.imread(self.working_clip.other_files[SEG_IMG])[..., 0]

        a, b = f_seg.shape[:2]
        x_a, x_b = np.meshgrid(np.arange(b), np.arange(a))
        pixel_pos = np.concatenate([x_a[..., np.newaxis],
                                    x_b[..., np.newaxis]], axis=-1)
        real_pos = self.pixel2real(pixel_pos)
        grid_pos = self.real2grid(real_pos)

        pixel_pos_f = pixel_pos.reshape([-1, 2])
        grid_pos_f = grid_pos.reshape([-1, 2])

        source = self.map_mgr.void_map
        source = 255 * np.ones_like(source)
        for p, g in zip(pixel_pos_f, grid_pos_f):
            source[g[0], g[1]] = f_seg[p[1], p[0]]

        # save global trajectory map
        cv2.imwrite(self.temp_files['GLOBAL_FILE'], source)

        return source/255

    def pixel2real(self, pixel: np.ndarray):
        weights = self.working_clip.matrix
        w = [weights[0], weights[2]]
        b = [weights[1], weights[3]]

        order = self.working_clip.order
        scale = self.working_clip.get_manager(SplitManager).scale

        x = (pixel[..., order[1]] - b[0]) / w[0]
        y = (pixel[..., order[0]] - b[1]) / w[1]

        return np.stack([x, y], axis=-1)/scale
