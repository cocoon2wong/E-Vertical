"""
@Author: Conghao Wong
@Date: 2022-06-20 21:41:10
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-07 11:11:09
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY, Args

from .base import BaseSilverballersArgs


class SilverballersArgs(BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

    @property
    def loada(self) -> str:
        """
        Path to load the first-stage agent model.
        """
        return self._arg('loada', 'null', argtype=TEMPORARY, short_name='la')

    @property
    def loadb(self) -> str:
        """
        Path to load the second-stage handler model.
        """
        return self._arg('loadb', 'null', argtype=TEMPORARY, short_name='lb')

    @property
    def pick_trajectories(self) -> float:
        """
        Calculates the sum of the context map values of the predicted trajectories
        and picks the top n (percentage) best predictions. This parameter is only
        valid when the model's input contains `MAPS` and `MAP_PARAS`.
        """
        return self._arg('pick_trajectories', 1.0, argtype=TEMPORARY, short_name='p')
