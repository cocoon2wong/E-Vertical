"""
@Author: Conghao Wong
@Date: 2023-05-19 16:05:54
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:13:47
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from ...base import BaseManager
from ..__base import BaseFilesManager
from ..__splitManager import Clip
from .__inputObject import Agent
from .__inputObjectManager import TrajectoryManager


class AgentFilesManager(BaseFilesManager):
    """
    AgentFilesManager
    ---
    A manager to save processed agent files.

    - Load items: A list of agents (type is `list[Agent]`) to save;
    - Run items: Load agents and save them into `npz` files.
        If the saved file exists, it will load these files into agents.
    """

    FILE_PREFIX = 'agent'
    DATA_MGR = TrajectoryManager
    DATA_TYPE = Agent

    def __init__(self, manager: BaseManager,
                 name='Agent Files Manager'):

        super().__init__(manager, name)

    # For type hinting
    def run(self, clip: Clip, agents: list[Agent] = None,
            *args, **kwargs) -> list[Agent]:

        return super().run(clip=clip, agents=agents, *args, **kwargs)
