"""
@Author: Conghao Wong
@Date: 2023-04-25 11:19:37
@LastEditors: Conghao Wong
@LastEditTime: 2023-04-25 13:35:07
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

import plistlib


def write_plist(value: dict, path: str):
    with open(path, 'wb+') as f:
        plistlib.dump(value, f)


def main():
    d = {
        'coordinate': dict(
            dim=2,
            base_dim=2,
            base_len=2,
        ),

        'boundingbox': dict(
            dim=4,
            base_dim=2,
            base_len=4,
        ),

        'boundingbox-rotate': dict(
            dim=5,
            base_dim=2,
            base_len=4,
        ),

        '3Dcoordinate': dict(
            dim=3,
            base_dim=3,
            base_len=3,
        ),

        '3Dboundingbox': dict(
            dim=6,
            base_dim=3,
            base_len=6,
        ),

        '3Dboundingbox-rotate': dict(
            dim=10,
            base_dim=3,
            base_len=6,
        ),

        '3Dskeleton-17': dict(
            dim=17*3,
            base_dim=3,
            base_len=17*3,
        ),
    }

    write_plist(d, './qpid/dataset/__base/annSettings.plist')


if __name__ == '__main__':
    main()
