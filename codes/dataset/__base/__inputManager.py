"""
@Author: Conghao Wong
@Date: 2023-05-19 09:51:56
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:07:01
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import os
from typing import Any

from ...base import BaseManager
from ..__splitManager import Clip
from .__picker import Annotation


class BaseInputManager(BaseManager):
    """
    BaseInputManager
    ---

    Basic class for all `InputManagers`.
    It should be managed by the `AgentManager` object.
    """

    TEMP_FILE: str = None
    TEMP_FILES: dict[str, str] = None

    ROOT_DIR: str = None
    INPUT_TYPE: str = None

    def __init__(self, manager: BaseManager, name: str = None):
        super().__init__(manager=manager, name=name)

        self.__clip: Clip = None

    @property
    def picker(self) -> Annotation:
        return self.manager.picker

    def get_temp_dir(self, clip: Clip) -> str:
        if not (r := self.ROOT_DIR):
            return clip.temp_dir
        else:
            return os.path.join(clip.temp_dir, r)

    def get_temp_file_path(self, clip: Clip) -> str:
        if not self.TEMP_FILE:
            return None

        temp_dir = self.get_temp_dir(clip)
        return os.path.join(temp_dir, self.TEMP_FILE)

    def get_temp_files_paths(self, clip: Clip) -> dict[str, str]:
        if not self.TEMP_FILES:
            return None

        dic = {}
        temp_dir = self.get_temp_dir(clip)
        for key, value in self.TEMP_FILES.items():
            dic[key] = os.path.join(temp_dir, value)
        return dic

    def run(self, clip: Clip,
            root_dir: str = None,
            *args, **kwargs) -> list:
        """
        Run all dataset-related operations within this manager,
        including load, preprocess, read or write files, and
        then make train samples on the given clip.

        NOTE: All custom parameters defined in subclasses should
        be given with `argname=argvalue`.
        """
        # Reset args
        self.clean()
        self.__clip = clip
        self.ROOT_DIR = root_dir

        self.init_clip(clip)

        if not self.temp_file_exists:
            self.save(*args, **kwargs)

        return self.load(*args, **kwargs)

    @property
    def working_clip(self) -> Clip:
        if not self.__clip:
            raise ValueError(self.__clip)

        return self.__clip

    @property
    def temp_dir(self) -> str:
        return self.get_temp_dir(self.working_clip)

    @property
    def temp_file(self) -> str:
        return self.get_temp_file_path(self.working_clip)

    @property
    def temp_files(self) -> dict[str, str]:
        return self.get_temp_files_paths(self.working_clip)

    @property
    def temp_file_exists(self) -> bool:
        files = []
        if (t := self.temp_file):
            files.append(t)
        elif (t := self.temp_files):
            files += list(t.values())
        else:
            raise ValueError('Wrong temp file settings!')

        exists = True
        for f in files:
            if not os.path.exists(f):
                exists = False
                break

        return exists

    def clean(self):
        self.__clip = None
        self.ROOT_DIR = None

    def init_clip(self, clip: Clip):
        pass

    def save(self, *args, **kwargs) -> Any:
        """
        Process original dataset files and them save the processed
        temp files.
        """
        raise NotImplementedError

    def load(self, *args, **kwargs) -> list:
        """
        Load the processed data to a list of values to train or test.
        """
        raise NotImplementedError
