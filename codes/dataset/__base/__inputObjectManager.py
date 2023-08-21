"""
@Author: Conghao Wong
@Date: 2023-06-12 19:15:43
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 19:52:01
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

from ...base import BaseManager, SecondaryBar
from ...utils import INIT_POSITION, dir_check
from ..__splitManager import Clip, SplitManager
from .__inputManager import BaseInputManager
from .__inputObject import BaseInputObject


class BaseInputObjectManager(BaseInputManager):

    TEMP_FILE = 'data.npz'

    def __init__(self, manager: BaseManager, name: str = None):
        super().__init__(manager, name)

    # For type hinting
    def run(self, clip: Clip, root_dir: str = None, 
            *args, **kwargs) -> list[BaseInputObject]:
        return super().run(clip, root_dir, *args, **kwargs)
    
    def save(self, **kwargs) -> None:
        """
        Load trajectory data from the annotation text file.
        The data format of the `ann.txt`:
        It is a matrix with the shape = `(N, M)`, where
        - `N` is the number of records in the file;
        - `M` is the length of each record.

        A record may contain several items, where
        - `item[0]`: frame name (or called the frame id);
        - `item[1]`: agent name (or called the agent id);
        - `item[2:M-1]`: dataset records, like coordinates, 
            bounding boxes, and other types of trajectory series.
        - `item[M-1]`: type of the agent
        """

        dataset: SplitManager = self.working_clip.manager
        anndim = dataset.dimension

        data = np.genfromtxt(self.working_clip.annpath, str, delimiter=',')

        agent_dict = {}
        agent_names, name_index = np.unique(agent_order := data.T[1],
                                            return_index=True)
        agent_types = data.T[anndim+2][name_index]
        names_and_types = []

        try:
            agent_ids = [int(n.split('_')[0]) for n in agent_names]
            agent_order = np.argsort(agent_ids)
        except:
            agent_order = np.arange(len(agent_names))

        for agent_index in agent_order:
            _name = agent_names[agent_index]
            _type = agent_types[agent_index]
            names_and_types.append((_name, _type))

            index = np.where(data.T[1] == _name)[0]
            _dat = np.delete(data[index], 1, axis=1)
            agent_dict[_name] = _dat[:, :anndim+1].astype(np.float64)

        frame_ids = list(set(data.T[0].astype(np.int32)))
        frame_ids.sort()

        # start making temp files
        agent_names = [n[0] for n in names_and_types]
        p = len(agent_names)
        f = len(frame_ids)

        # agent_name -> agent_index
        name_dict: dict[str, int] = dict(zip(agent_names, np.arange(p)))

        # frame_id -> frame_index
        frame_dict: dict[int, int] = dict(zip(frame_ids, np.arange(f)))

        # init the matrix
        dim = agent_dict[agent_names[0]].shape[-1] - 1
        matrix = INIT_POSITION * np.ones([f, p, dim])

        for name, index in SecondaryBar(name_dict.items(),
                                        manager=self.manager,
                                        desc='Processing dataset...'):

            frame_id = agent_dict[name].T[0].astype(np.int32)
            frame_index = [frame_dict[fi] for fi in frame_id]
            matrix[frame_index, index, :] = agent_dict[name][:, 1:]

        neighbor_indexes = np.array([
            np.where(np.not_equal(data, INIT_POSITION))[0]
            for data in matrix[:, :, 0]], dtype=object)

        dir_check(os.path.dirname(self.temp_file))
        np.savez(self.temp_file,
                 neighbor_indexes=neighbor_indexes,
                 matrix=matrix,
                 frame_ids=frame_ids,
                 person_ids=names_and_types)
    
    def load(self, **kwargs) -> list[BaseInputObject]:
        raise NotImplementedError