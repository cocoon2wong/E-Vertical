"""
@Author: Conghao Wong
@Date: 2023-05-19 14:38:26
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-16 10:38:17
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import numpy as np

from ...base import BaseManager, SecondaryBar
from ...utils import INIT_POSITION
from ..__base import BaseInputObjectManager
from .__inputObject import Agent, Trajectory


class TrajectoryManager(BaseInputObjectManager):

    def __init__(self, manager: BaseManager,
                 name='Trajectory Manager'):

        super().__init__(manager, name)

    def load(self, **kwargs) -> list[Agent]:
        # load from saved files
        dat = np.load(self.temp_file, allow_pickle=True)
        matrix = dat['matrix']
        neighbor_indexes = dat['neighbor_indexes']
        frame_ids = dat['frame_ids']
        names_and_types = dat['person_ids']

        agent_count = matrix.shape[1]
        frame_number = matrix.shape[0]

        trajs = [Trajectory(agent_id=names_and_types[agent_index][0],
                            agent_type=names_and_types[agent_index][1],
                            trajectory=matrix[:, agent_index, :],
                            neighbors=neighbor_indexes,
                            frames=frame_ids,
                            init_position=INIT_POSITION,
                            ) for agent_index in range(agent_count)]

        sample_rate, frame_rate = self.working_clip.paras
        frame_step = int(self.args.interval / (sample_rate / frame_rate))
        train_samples = []

        for agent_index in SecondaryBar(range(agent_count),
                                        manager=self.manager,
                                        desc='Process dataset files...'):

            trajectory = trajs[agent_index]
            start_frame = trajectory.start_frame
            end_frame = trajectory.end_frame

            for p in range(start_frame, end_frame, 
                           int(np.ceil(self.args.step * frame_step))):
                # Normal mode
                if self.args.pred_frames > 0:
                    if p + (self.args.obs_frames + self.args.pred_frames) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = p + (self.args.obs_frames +
                               self.args.pred_frames) * frame_step

                # Infinity mode, only works for destination models
                elif self.args.pred_frames == -1:
                    if p + (self.args.obs_frames + 1) * frame_step > end_frame:
                        break

                    obs = p + self.args.obs_frames * frame_step
                    end = end_frame

                else:
                    self.log('`pred_frames` should be a positive integer or -1, ' +
                             f'got `{self.args.pred_frames}`',
                             level='error', raiseError=ValueError)

                train_samples.append(trajectory.sample(start_frame=p,
                                                       obs_frame=obs,
                                                       end_frame=end,
                                                       matrix=matrix,
                                                       frame_step=frame_step,
                                                       add_noise=False))

        return train_samples
