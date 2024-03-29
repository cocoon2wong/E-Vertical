"""
@Author: Conghao Wong
@Date: 2022-06-20 15:28:14
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-07 15:34:48
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys

import codes as C
import silverballers
from scripts.update_readme import print_help_info
from scripts.utils import get_value


def main(args: list[str]):
    h_value = None

    if '--help' in args:
        h_value = get_value('--help', args, default='all_args')
    elif '-h' in args:
        h_value = get_value('-h', args, default='all_args')

    if h_value:
        print_help_info(h_value)
        exit()

    min_args = C.args.Args(terminal_args=args,
                           is_temporary=True)

    model = min_args.model
    if model == 'linear':
        s = C.models.Linear
    else:
        s = silverballers.get_structure(model)

    t = s(terminal_args=args)
    t.train_or_test()

    # It is used to debug
    # t.print_info_all()

    return t

if __name__ == '__main__':
    main(sys.argv)
