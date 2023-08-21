"""
@Author: Conghao Wong
@Date: 2023-06-07 11:08:13
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-07 15:52:09
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY

from ..base import BaseSilverballersArgs


class AgentArgs(BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def depth(self) -> int:
        """
        Depth of the random noise vector.
        """
        return self._arg('depth', 16, argtype=STATIC)

    @property
    def deterministic(self) -> int:
        """
        Controls if predict trajectories in the deterministic way.
        """
        return self._arg('deterministic', 0, argtype=STATIC)

    @property
    def loss(self) -> str:
        """
        Loss used to train agent models.
        Canbe `'avgkey'` or `'keyl2'`.
        """
        return self._arg('loss', 'keyl2', argtype=TEMPORARY)
