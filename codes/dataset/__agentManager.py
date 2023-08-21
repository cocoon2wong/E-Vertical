"""
@Author: Conghao Wong
@Date: 2022-08-03 10:50:46
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-12 20:19:57
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..base import BaseManager
from ..basemodels.layers.transfroms import (_BaseTransformLayer,
                                            get_transform_layers)
from ..constant import INPUT_TYPES
from ..utils import POOLING_BEFORE_SAVING, dir_check
from .__base import Annotation, AnnotationManager, BaseInputManager
from .__splitManager import SplitManager
from .agent_based import Agent, AgentFilesManager, TrajectoryManager, maps
from .frame_based import FrameFilesManager, FrameManager


class AgentManager(BaseManager):
    """
    AgentManager
    ---
    Structure to manage several training and test `Agent` objects.
    The `AgentManager` object is managed by the `Structure` object.

    Member Managers
    ---
    - Dataset split's manager: type is `SplitManager`;
    - Trajectory map manager (optional, dynamic): type is `TrajMapManager`;
    - Social map manager (optional, dynamic): type is `SocialMapManager`.

    Public Methods
    ---
    ```python
    # concat agents to this `AgentManager`
    (method) append: (self: Self@AgentManager, target: Any) -> None

    # set inputs and outputs
    (method) set: (self: Self@AgentManager, dimension: int, 
                   inputs_type: list[str],
                   labels_type: list[str]) -> None

    # get inputs
    (method) get_inputs: (self: Self@AgentManager) -> list[Tensor]

    # get labels
    (method) get_labels: (self: Self@AgentManager) -> list[Tensor]

    # make inputs and labels into a dataset object
    (method) make_dataset: (self: Self@AgentManager, 
                            shuffle: bool = False) -> DatasetV2

    # save all agents' data
    (method) save: (self: Self@AgentManager, save_dir: str) -> None

    # load from saved agents' data
    (method) load: (cls: Type[Self@AgentManager], path: str) -> AgentManager
    ```

    Context Map Methods
    ---
    ```python
    # init map managers that manage to make context maps
    (method) init_map_managers: (self: Self@AgentManager, 
                                 map_type: str, 
                                 base_path: str) -> None

    #  load context maps to `Agent` objects
    (method) load_maps: (self: Self@AgentManager) -> None
    ```
    """

    def __init__(self, manager: BaseManager, name='Agent Manager'):
        super().__init__(manager=manager, name=name)

        # Dataset split and basic inputs
        self.split_manager = SplitManager(manager=self,
                                          dataset=self.args.dataset,
                                          split=self.args.split)

        if (t := self.args.model_type) == 'agent-based':
            self.traj_manager = TrajectoryManager(self)
            self.file_manager = AgentFilesManager(self)
        elif t == 'frame-based':
            self.frame_manager = FrameManager(self)
            self.file_manager = FrameFilesManager(self)
        else:
            self.log(f'Wrong model type `{t}`!',
                     level='error', raiseError=ValueError)

        # file root paths
        self.base_path: str = None
        self.npz_path: str = None
        self.maps_dir: str = None

        # Settings and variations
        self._agents: list[Agent] = []
        self.model_inputs: list[str] = None
        self.model_labels: list[str] = None
        self.processed_clips: dict[str, list[str]] = {'train': [], 'test': []}

        # Managers for extra model inputs
        self.ext_mgrs: list[BaseInputManager] = []
        self.ext_types: list[str] = []
        self.ext_inputs: dict[str, dict[str, list]] = {}

        # Transform layer
        self.t_layers: dict[str, _BaseTransformLayer] = {}

    @property
    def agents(self) -> list[Agent]:
        return self._agents

    @agents.setter
    def agents(self, value: list[Agent]) -> list[Agent]:
        self._agents = self.update_agents(value)

    @property
    def picker(self) -> Annotation:
        return self.pickers.annotations[self.args.anntype]

    @property
    def pickers(self) -> AnnotationManager:
        return self.manager.get_member(AnnotationManager)

    def set_path(self, npz_path: str):
        self.npz_path = npz_path
        self.base_path = npz_path.split('.np')[0]
        self.maps_dir = self.base_path + '_maps'

    def update_agents(self, agents: list[Agent]):
        for a in agents:
            a.manager = self
        return agents

    def append(self, target: list[Agent]):
        self._agents += self.update_agents(target)

    def set_types(self, inputs_type: list[str], labels_type: list[str]):
        """
        Set the type of model inputs and outputs.
        Accept all types in `INPUT_TYPES`.
        """
        if (t := INPUT_TYPES.MAP) in inputs_type:
            p = POOLING_BEFORE_SAVING
            self.ext_types.append(t)
            self.ext_mgrs.append(maps.MapParasManager(self))
            self.ext_mgrs.append(maps.TrajMapManager(self, p))
            self.ext_mgrs.append(maps.TrajMapManager_seg(self, p))
            self.ext_mgrs.append(maps.SocialMapManager(self, p))

        if (t := INPUT_TYPES.MAP_PARAS) in inputs_type:
            self.ext_types.append(t)

        self.model_inputs = inputs_type
        self.model_labels = labels_type

    def gather_inputs(self) -> list[tf.Tensor]:
        """
        Get all model inputs from agents.
        """
        return [self._gather(T) for T in self.model_inputs]

    def gather_labels(self) -> list[tf.Tensor]:
        """
        Get all model labels from agents.
        """
        return [self._gather(T) for T in self.model_labels]

    def make_dataset(self, shuffle=False) -> tf.data.Dataset:
        """
        Get inputs from all agents and make the `tf.data.Dataset`
        object. Note that the dataset contains both model inputs
        and labels.
        """
        data = tuple(self.gather_inputs() + self.gather_labels())
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(
                len(dataset),
                reshuffle_each_iteration=True
            )

        return dataset

    def make(self, clips: list[str], mode: str) -> tf.data.Dataset:
        """
        Load train samples and make the `tf.data.Dataset` object to train.

        :param clips: Clips to load.
        :param mode: The load mode, can be `'test'` or `'train'`.
        :return dataset: The loaded `tf.data.Dataset` object.
        """
        if type(clips) is str:
            clips = [clips]

        # shuffle agents and video clips when training
        if mode == 'train':
            shuffle = True
            random.shuffle(clips)
        else:
            shuffle = False

        # load agent data in each video clip
        for clip_name in self.timebar(clips):
            # get clip info
            clip = self.split_manager.clips_dict[clip_name]

            # update time bar
            s = f'Prepare data of {mode} agents in `{clip.clip_name}`...'
            self.update_timebar(s, pos='start')

            # Get new agents
            agents = self.file_manager.run(clip)
            self.append(agents)

            # Load extra model inputs
            self.ext_inputs[clip_name] = {}
            for mgr in self.ext_mgrs:
                key = mgr.INPUT_TYPE
                dir_path = f'{self.file_manager.get_temp_file_path(clip)}.{key}'
                dir_name = dir_check(dir_path).split('/')[-1]
                value = mgr.run(clip=clip,
                                root_dir=dir_name,
                                agents=agents,
                                trajs=self._gather_obs_trajs(agents))

                if not key in self.ext_inputs[clip_name].keys():
                    self.ext_inputs[clip_name][key] = value
                else:
                    self.ext_inputs[clip_name][key] += value

        self.processed_clips[mode] += clips
        return self.make_dataset(shuffle=shuffle)

    def _gather_obs_trajs(self, agents: list[Agent] = None) -> np.ndarray:
        if not agents:
            agents = self.agents
        return np.array([a.traj for a in agents])

    def _gather(self, type_name: str) -> tf.Tensor:
        """
        Get model inputs or labels from a list of `Agent`-like objects.

        :param type_name: Types of all inputs, accept all type names \
            in `INPUT_TYPES`.
        :return inputs: A tensor of stacked inputs.
        """
        t = type_name
        if t in self.ext_types:
            res = None
            for _, _res in self.ext_inputs.items():
                if res is None:
                    res = _res[t]
                else:
                    res = tf.concat([res, _res[t]], axis=0)
            return tf.cast(res, tf.float32)

        if t == INPUT_TYPES.OBSERVED_TRAJ:
            return _get_obs_traj(self.agents)

        elif t == INPUT_TYPES.DESTINATION_TRAJ:
            return _get_dest_traj(self.agents)

        elif t == INPUT_TYPES.GROUNDTRUTH_TRAJ:
            return _get_gt_traj(self.agents)

        elif t == INPUT_TYPES.GROUNDTRUTH_SPECTRUM:
            if t not in self.t_layers.keys():
                t_type, _ = get_transform_layers(self.args.T)
                self.t_layers[t] = t_type((self.args.pred_frames,
                                           self.picker.dim))

            t_layer = self.t_layers[t]
            return t_layer(_get_gt_traj(self.agents, text='groundtruth spectrums'))

        elif t == INPUT_TYPES.ALL_SPECTRUM:
            if t not in self.t_layers.keys():
                t_type, _ = get_transform_layers(self.args.T)
                steps = self.args.obs_frames + self.args.pred_frames
                self.t_layers[t] = t_type((steps, self.picker.dim))

            trajs = []
            for agent in tqdm(self.agents, 'Prepare trajectory spectrums (all)...'):
                trajs.append(np.concatenate(
                    [agent.traj, agent.groundtruth], axis=-2))

            t_layer = self.t_layers[t]
            return t_layer(tf.cast(trajs, tf.float32))

        else:
            raise ValueError(type_name)

    def print_info(self, **kwargs):
        t_info = {}
        for mode in ['train', 'test']:
            if len(t := self.processed_clips[mode]):
                t_info.update({'T' + f'{mode} agents come from'[1:]: t})

        return super().print_info(**t_info, **kwargs)


def _get_obs_traj(input_agents: list[Agent]) -> tf.Tensor:
    """
    Get observed trajectories from agents.

    :param input_agents: A list of input agents, type = `list[Agent]`.
    :return inputs: A Tensor of observed trajectories.
    """
    inputs = []
    for agent in tqdm(input_agents, 'Prepare trajectories...'):
        inputs.append(agent.traj)
    return tf.cast(inputs, tf.float32)


def _get_gt_traj(input_agents: list[Agent],
                 destination=False,
                 text='groundtruth') -> tf.Tensor:
    """
    Get groundtruth trajectories from agents.

    :param input_agents: A list of input agents, type = `list[Agent]`.
    :return inputs: A Tensor of gt trajectories.
    """
    inputs = []
    for agent in tqdm(input_agents, f'Prepare {text}...'):
        if destination:
            inputs.append(np.expand_dims(agent.groundtruth[-1], 0))
        else:
            inputs.append(agent.groundtruth)

    return tf.cast(inputs, tf.float32)


def _get_dest_traj(input_agents: list[Agent]) -> tf.Tensor:
    return _get_gt_traj(input_agents, destination=True, text='destinations')
