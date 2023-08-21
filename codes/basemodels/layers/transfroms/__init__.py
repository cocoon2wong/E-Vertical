"""
@Author: Conghao Wong
@Date: 2023-05-09 20:24:48
@LastEditors: Conghao Wong
@LastEditTime: 2023-05-09 20:29:44
@Description: file content
@Github: https://cocoon2wong.github.io
@Copyright 2023 Conghao Wong, All Rights Reserved.
"""

from .__base import NoneTransformLayer, _BaseTransformLayer
from .__dct import DCTLayer, IDCTLayer
from .__fft import FFT2DLayer, FFTLayer, IFFT2Dlayer, IFFTLayer
from .__wavetf import DB2_1D, Haar1D, InverseDB2_1D, InverseHaar1D


def get_transform_layers(Tname: str) -> \
        tuple[type[_BaseTransformLayer],
              type[_BaseTransformLayer]]:
    """
    Set transformation layers used when encoding or 
    decoding trajectories.

    :param Tname: name of the transform, canbe
        - `'none'`
        - `'fft'`
        - `'fft2d'`
        - `'haar'`
        - `'db2'`
    """

    if Tname == 'none':
        Tlayer = NoneTransformLayer
        ITlayer = NoneTransformLayer

    elif Tname == 'fft':
        Tlayer = FFTLayer
        ITlayer = IFFTLayer

    elif Tname == 'fft2d':
        Tlayer = FFT2DLayer
        ITlayer = IFFT2Dlayer

    elif Tname == 'haar':
        Tlayer = Haar1D
        ITlayer = InverseHaar1D

    elif Tname == 'db2':
        Tlayer = DB2_1D
        ITlayer = InverseDB2_1D

    elif Tname == 'dct':
        Tlayer = DCTLayer
        ITlayer = IDCTLayer

    else:
        raise ValueError('Transform name not found.')

    return Tlayer, ITlayer
