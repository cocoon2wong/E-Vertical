"""
@Author: Conghao Wong
@Date: 2023-06-07 11:09:55
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-07 11:09:58
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from codes.args import DYNAMIC, STATIC, TEMPORARY

from ..base import BaseSilverballersArgs


class HandlerArgs(BaseSilverballersArgs):

    def __init__(self, terminal_args: list[str] = None,
                 is_temporary=False) -> None:

        super().__init__(terminal_args, is_temporary)

        self._set_default('key_points', 'null', overwrite=False)

    @property
    def points(self) -> int:
        """
        The number of keypoints accepted in the handler model.
        """
        return self._arg('points', 1, argtype=STATIC)
    