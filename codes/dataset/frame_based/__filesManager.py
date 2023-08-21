"""
@Author: Conghao Wong
@Date: 2023-06-12 14:44:44
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:15:45
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...base import BaseManager
from ..__base import BaseFilesManager
from ..__splitManager import Clip
from .__inputObject import Frame
from .__inputObjectManager import FrameManager


class FrameFilesManager(BaseFilesManager):

    FILE_PREFIX = 'frame'
    DATA_MGR = FrameManager
    DATA_TYPE = Frame

    def __init__(self, manager: BaseManager,
                 name='Frame Files Manager'):

        super().__init__(manager, name)

    # For type hinting
    def run(self, clip: Clip, agents: list[Frame] = None, 
            *args, **kwargs) -> list[Frame]:
        return super().run(clip, agents, *args, **kwargs)
