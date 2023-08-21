"""
@Author: Conghao Wong
@Date: 2023-06-12 18:44:58
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-16 10:31:18
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np

from ..__splitManager import Clip
from .__inputManager import BaseInputManager, BaseManager
from .__inputObject import BaseInputObject
from .__inputObjectManager import BaseInputObjectManager


class BaseFilesManager(BaseInputManager):
    """
    BaseFilesManager
    ---
    A manager to save processed dataset files (types are `BaseInputObject`).

    - Load items: A list of `BaseInputObject` to save;
    - Run items: Load files and save them into `npz` files.
        If the saved file exists, it will load these files and make
        the corresponding `BaseInputObject` objects.
    """

    FILE_PREFIX = None
    DATA_MGR: type[BaseInputObjectManager] = None
    DATA_TYPE: type[BaseInputObject] = None

    def __init__(self, manager: BaseManager,
                 name='Agent Files Manager'):

        super().__init__(manager, name)

    def get_temp_file_path(self, clip: Clip) -> str:
        base_dir = clip.temp_dir
        if (self.args.obs_frames, self.args.pred_frames) == (8, 12):
            f_name = self.FILE_PREFIX
        else:
            f_name = (f'{self.FILE_PREFIX}_' +
                      f'{self.args.obs_frames}to{self.args.pred_frames}')

        endstring = '' if self.args.step == 4 else str(self.args.step)
        if endstring.endswith('.0'):
            endstring = endstring[:-2]
        f_name = f_name + endstring + '.npz'
        return os.path.join(base_dir, f_name)

    # For type hinting
    def run(self, clip: Clip, agents: list[DATA_TYPE] = None,
            *args, **kwargs) -> list[DATA_TYPE]:

        return super().run(clip=clip, agents=agents, *args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        agents = self.manager.get_member(
            self.DATA_MGR).run(self.working_clip)

        save_dict = {}
        for index, agent in enumerate(agents):
            save_dict[str(index)] = agent.zip_data()

        np.savez(self.temp_file, **save_dict)

    def load(self, *args, **kwargs) -> list:
        saved: dict = np.load(self.temp_file, allow_pickle=True)

        if not len(saved):
            self.log(f'Please delete file `{self.temp_file}` and re-run the program.',
                     level='error', raiseError=FileNotFoundError)

        if (v := saved['0'].tolist()['__version__']) < (
                v1 := self.DATA_TYPE.__version__):
            self.log((f'Saved {self.FILE_PREFIX} managers\' version is {v}, ' +
                      f'which is lower than current {v1}. Please delete' +
                      ' them and re-run this program, or there could' +
                      ' happen something wrong.'),
                     level='error')

        return [self.DATA_TYPE().load_data(v.tolist()) for v in saved.values()]
