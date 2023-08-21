"""
@Author: Conghao Wong
@Date: 2023-06-12 15:11:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 19:15:59
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import copy

from .__picker import AnnotationManager


class BaseInputObject():
    """
    BaseInputObject
    ---

    The basic class to load dataset files directly.
    """

    __version__ = 0.0
    _save_items = []

    def __init__(self) -> None:
        pass

    def copy(self):
        return copy.deepcopy(self)

    @property
    def pickers(self) -> AnnotationManager:
        return self.manager.pickers

    def init_data(self):
        raise NotImplementedError

    def zip_data(self) -> dict[str, object]:
        zipped = {}
        for item in self._save_items:
            zipped[item] = getattr(self, item)
        return zipped

    def load_data(self, zipped_data: dict[str, object]):
        for item in self._save_items:
            if not item in zipped_data.keys():
                continue
            else:
                setattr(self, item, zipped_data[item])
        return self
