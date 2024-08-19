"""
@Author: Conghao Wong
@Date: 2024-07-26 09:49:29
@LastEditors: Conghao Wong
@LastEditTime: 2024-07-30 16:58:07
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import ANN_TYPES, INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.training import Structure

from .__args import VArgs


class VASModel(Model):

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.v_args = self.args.register_subargs(VArgs, 'v_args')

        # Assign input and label types
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ)
        self.set_labels(INPUT_TYPES.GROUNDTRUTH_TRAJ)

        # Layers
        tlayer, itlayer = layers.get_transform_layers(self.v_args.T)

        # Transform layers
        if self.args.anntype in [ANN_TYPES.CO_2D,
                                 ANN_TYPES.CO_3D]:
            dim = 1
        elif self.args.anntype in [ANN_TYPES.BB_2D]:
            dim = 2
        elif self.args.anntype in [ANN_TYPES.BB_3D,
                                   ANN_TYPES.SKE_3D_17]:
            dim = 3
        else:
            self.log(f'Prediction type `{self.args.anntype}` not supported!',
                     level='error', raiseError=ValueError)

        self.t1 = tlayer((self.args.obs_frames, dim))
        self.it1 = itlayer((len(self.output_pred_steps), dim))

        # steps and shapes after applying transforms
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # -----------------------------------
        # Networks for a single dimension
        # -----------------------------------
        # Trajectory encoding
        self.te_x = layers.TrajEncoding(self.dim, self.d//2,
                                        torch.nn.Tanh,
                                        transform_layer=self.t1)
        # Noise encoding
        self.ie_x = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)

        # Transformer is used as a feature extractor
        self.T_x = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_de,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        # Trainable adj matrix and gcn layer
        # See our previous work "MSN: Multi-Style Network for Trajectory Prediction" for detail
        # It is used to generate multiple predictions within one model implementation
        self.ms_fc_x = layers.Dense(self.d, self.v_args.Kc, torch.nn.Tanh)
        self.ms_conv_x = layers.GraphConv(self.d, self.d)

        # Decoder layers
        self.decoder_fc1_x = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.decoder_fc2_x = layers.Dense(self.d,
                                          self.Tsteps_de * self.Tchannels_de)

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, 2)
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        if self.args.anntype in [ANN_TYPES.BB_2D,
                                 ANN_TYPES.BB_3D,
                                 ANN_TYPES.SKE_3D_17]:
            obs = torch.stack(self.picker.get_coordinate_series(obs), dim=-3)
        else:
            obs = torch.transpose(obs, -2, -1)[..., None]

        # Feature embedding and encoding -> (batch, obs, d/2)
        f_traj_x = self.te_x(obs)

        # Sampling random noise vectors
        all_predictions_x = []
        repeats = self.args.K_train if training else self.args.K

        traj_targets_x = self.t1(obs)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f_traj_x.shape[:-1]) + [self.d_id])
            f_z = self.ie_x(z.to(obs.device))

            # Transformer inputs -> (batch, steps, d)
            f_final_x = torch.concat([f_traj_x, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran_x, _ = self.T_x(inputs=f_final_x,
                                   targets=traj_targets_x,
                                   training=training)

            # Multiple generations -> (batch, Kc, d)
            adj_x = self.ms_fc_x(f_final_x)               # (batch, steps, Kc)
            adj_x = torch.transpose(adj_x, -1, -2)
            f_multi = self.ms_conv_x(f_tran_x, adj_x)     # (batch, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y_x = self.decoder_fc1_x(f_multi)
            y_x = self.decoder_fc2_x(y_x)
            y_x = torch.reshape(y_x, list(y_x.shape[:-1]) +
                                [self.Tsteps_de, self.Tchannels_de])

            y_x = self.it1(y_x)
            all_predictions_x.append(y_x)

        pred_x = torch.concat(all_predictions_x, dim=-3)    # K

        # Concat separate predictions
        pred_x = torch.transpose(pred_x, -2, -4)
        pred_x = torch.transpose(pred_x, -3, -4)
        pred_x = torch.flatten(pred_x, -2, -1)
        return pred_x


class VAS(Structure):
    MODEL_TYPE = VASModel
