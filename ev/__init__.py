"""
@Author: Conghao Wong
@Date: 2023-08-08 15:52:46
@LastEditors: Conghao Wong
@LastEditTime: 2024-08-06 11:49:05
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import qpid as qpid

from . import original_models

qpid.add_arg_alias(alias=['--sc', '-sc', '--ev', '-ev'],
                   command=['--model', 'MKII', '--loads'],
                   pattern='{},speed')
