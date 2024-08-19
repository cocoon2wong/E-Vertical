"""
@Author: Conghao Wong
@Date: 2023-09-06 20:45:28
@LastEditors: Conghao Wong
@LastEditTime: 2024-07-30 17:02:24
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import VArgs
from .ev import EV, EVModel
from .ev_linear_ablation import Linear1, Linear1Model, Linear2, Linear2Model
from .ev_separate import EVS, EVSModel
from .msn import MSNAlpha, MSNAlphaModel
from .trans import MinimalV, MinimalVModel
from .v import VA, VB, VAModel, VBModel
from .v_separate import VAS, VASModel

qpid.register_args(VArgs, 'V^2-Net Args')
qpid.register(
    # MSN
    msna=[MSNAlpha, MSNAlphaModel],

    # V^2-Net
    va=[VA, VAModel],
    vas=[VAS, VASModel],
    agent=[VA, VAModel],
    vb=[VB, VBModel],

    # E-V^2-Net
    eva=[EV, EVModel],
    evas=[EVS, EVSModel],
    agent47C=[EV, EVModel],

    # Other models
    trans=[MinimalV, MinimalVModel],
    mv=[MinimalV, MinimalVModel],

    # Ablation variations for EV
    linear1=[Linear1, Linear1Model],
    linear2=[Linear2, Linear2Model],
)
