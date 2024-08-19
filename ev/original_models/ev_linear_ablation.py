"""
@Author: Conghao Wong
@Date: 2024-07-26 10:36:17
@LastEditors: Conghao Wong
@LastEditTime: 2024-07-26 17:01:47
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure


class Linear1Model(Model):

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.fc_x = layers.Dense(3*self.args.obs_frames, 2)
        self.fc_y = layers.Dense(3*self.args.obs_frames, 2)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):

        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        x = obs[..., 0:1]
        y = obs[..., 1:2]

        _ones = torch.ones_like(x)
        Ax = torch.concat([_ones, x, _ones], dim=-1)
        Ay = torch.concat([_ones, y, _ones], dim=-1)

        Ax = torch.flatten(Ax, start_dim=-2, end_dim=-1)
        Ay = torch.flatten(Ay, start_dim=-2, end_dim=-1)

        w_x = self.fc_x(Ax)
        w_y = self.fc_y(Ay)

        k = torch.arange(1, self.args.pred_frames + 1).to(x.device)
        k = k[None]
        pred_x = x[..., -1:, 0] + w_x[..., 1:] + k * w_x[..., :1]
        pred_y = y[..., -1:, 0] + w_y[..., 1:] + k * w_y[..., :1]

        return torch.stack([pred_x, pred_y], dim=-1)


class Linear2Model(Model):

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.fc_x = layers.Dense(3*self.args.obs_frames, 2)
        self.fc_y = layers.Dense(3*self.args.obs_frames, 2)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        x = obs[..., 0:1]
        y = obs[..., 1:2]

        _ones = torch.ones_like(x)
        Ax = torch.concat([_ones, x, x*y], dim=-1)
        Ay = torch.concat([_ones, y, x*y], dim=-1)

        Ax = torch.flatten(Ax, start_dim=-2, end_dim=-1)
        Ay = torch.flatten(Ay, start_dim=-2, end_dim=-1)

        w_x = self.fc_x(Ax)
        w_y = self.fc_y(Ay)

        k = torch.arange(1, self.args.pred_frames + 1).to(x.device)
        k = k[None]
        pred_x = x[..., -1:, 0] + w_x[..., 1:] + k * w_x[..., :1]
        pred_y = y[..., -1:, 0] + w_y[..., 1:] + k * w_y[..., :1]

        return torch.stack([pred_x, pred_y], dim=-1)


class Linear1(Structure):
    MODEL_TYPE = Linear1Model


class Linear2(Structure):
    MODEL_TYPE = Linear2Model
