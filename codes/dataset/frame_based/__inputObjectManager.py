"""
@Author: Conghao Wong
@Date: 2023-06-12 10:33:29
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-16 10:39:06
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ...base import BaseManager, SecondaryBar
from ...utils import INIT_POSITION
from ..__base import BaseInputObjectManager
from .__inputObject import Frame


class FrameManager(BaseInputObjectManager):

    def __init__(self, manager: BaseManager,
                 name='Frame Manager'):

        super().__init__(manager, name)

    def load(self, **kwargs) -> list[Frame]:
        # load from saved files
        dat = np.load(self.temp_file, allow_pickle=True)
        matrix = dat['matrix']
        neighbor_indices = dat['neighbor_indexes']
        frame_ids = dat['frame_ids']
        names_and_types = dat['person_ids']

        frame_count = matrix.shape[0]
        sample_rate, frame_rate = self.working_clip.paras
        frame_step = int(self.args.interval / (sample_rate / frame_rate))

        train_samples = []

        gone_agents = []

        for p in SecondaryBar(
                range(frame_step * self.args.obs_frames,
                      frame_count,
                      int(np.ceil(self.args.step * frame_step))),
                manager=self.manager,
                desc='Process frames...'):

            # Calculate frame indices
            obs = p - frame_step * self.args.obs_frames
            end = p + frame_step * self.args.pred_frames

            if end > frame_count:
                break

            # Only considers agents apperaed during observation period
            appeared_agents = neighbor_indices[obs: p: frame_step]
            current_agents = appeared_agents[-1]
            last_agents = appeared_agents[-2]

            gone_agents += [i for i in last_agents
                            if i not in current_agents]
            gone_agents = list(set(gone_agents))

            appeared_agents = np.unique(np.concatenate(appeared_agents))
            appeared_agents = np.array([i for i in appeared_agents
                                        if i not in gone_agents])

            if not len(appeared_agents):
                continue

            traj_matrix = matrix[obs: end: frame_step, appeared_agents]
            traj_matrix = np.transpose(traj_matrix, [1, 0, 2])

            train_samples.append(Frame().init_data(
                id=frame_ids[p],
                agent_ids=names_and_types[appeared_agents][:, 0],
                agent_types=names_and_types[appeared_agents][:, 1],
                traj=traj_matrix,
                frames=frame_ids[obs: end: frame_step],
                start_frame=obs,
                obs_frame=p,
                end_frame=end,
                init_position=INIT_POSITION,
                frame_step=frame_step,
                linear_predict=True,
            ))

        return train_samples
