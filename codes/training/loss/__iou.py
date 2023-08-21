"""
@Author: Conghao Wong
@Date: 2022-10-12 10:50:35
@LastEditors: Conghao Wong
@LastEditTime: 2023-06-13 17:54:31
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import tensorflow as tf

from ...utils import get_loss_mask


def AIoU(outputs: list[tf.Tensor],
         labels: list[tf.Tensor],
         model_inputs: list[tf.Tensor],
         coe: float = 1.0,
         *args, **kwargs) -> tf.Tensor:
    """
    Calculate the average IoU on predicted bounding boxes among the `time` axis.
    It is only used for models with `anntype == 'boundingbox'`.
    Each dimension of the predictions should be `(xl, yl, xr, yr)`.
    """
    pred = outputs[0]
    GT = labels[0]

    # (batch, steps, dim)
    if pred.ndim == 3:
        pred = pred[:, tf.newaxis, :, :]

    mask = get_loss_mask(model_inputs[0], GT)
    count = tf.reduce_sum(mask)

    K = pred.shape[-3]
    GT = tf.repeat(GT[..., tf.newaxis, :, :], K, axis=-3)

    # (batch, K, steps)
    dim = pred.shape[-1]
    if dim == 4:
        func = __IoU_single_2Dbox
    elif dim == 6:
        func = __IoU_single_3Dbox
    else:
        raise ValueError(dim)

    iou = func(pred, GT)
    iou = tf.reduce_mean(iou, axis=-1)
    iou = tf.reduce_max(iou, axis=-1)
    iou = tf.maximum(iou, 0.0)
    return tf.reduce_sum(iou * mask)/count


def FIoU(outputs: list[tf.Tensor],
         labels: list[tf.Tensor],
         model_inputs: list[tf.Tensor],
         coe: float = 1.0,
         index: int = -1,
         length: int = 1,
         *args, **kwargs) -> tf.Tensor:
    """
    Calculate the IoU on the final prediction time step.
    It is only used for models with `anntype == 'boundingbox'`.
    Each dimension of the predictions should be `(xl, yl, xr, yr)`.
    """
    pred = outputs[0]
    GT = labels[0]

    steps = pred.shape[-2]
    index = tf.math.mod(index, steps)
    return AIoU([pred[..., index:index+length, :]],
                [GT[..., index:index+length, :]],
                model_inputs=model_inputs)


def __IoU_single_3Dbox(box1: tf.Tensor, box2: tf.Tensor) -> tf.Tensor:
    """
    Calculate IoU on pred and GT.
    Boxes must have the same shape.

    - box1:

    |  0  |  1  |  2  |  3  |  4  |  5  |
    |-----|-----|-----|-----|-----|-----|
    | xl1 | yl1 | zl1 | xr1 | yr1 | zr1 |

    - box2:

    |  0  |  1  |  2  |  3  |  4  |  5  |
    |-----|-----|-----|-----|-----|-----|
    | xl2 | yl2 | zl2 | xr2 | yr2 | zr2 |

    :param box1: Shape = (..., 6).
    :param box2: Shape = (..., 6).
    """

    len1_x, len2_x, len_inter_x = \
        __get_len_1Dbox(tf.gather(box1, [0, 3], axis=-1),
                        tf.gather(box2, [0, 3], axis=-1))

    len1_y, len2_y, len_inter_y = \
        __get_len_1Dbox(tf.gather(box1, [1, 4], axis=-1),
                        tf.gather(box2, [1, 4], axis=-1))

    len1_z, len2_z, len_inter_z = \
        __get_len_1Dbox(tf.gather(box1, [2, 5], axis=-1),
                        tf.gather(box2, [2, 5], axis=-1))

    v_inter = len_inter_x * len_inter_y * len_inter_z
    v_box1 = len1_x * len1_y * len1_z
    v_box2 = len2_x * len2_y * len2_z
    iou = v_inter / (v_box1 + v_box2 - v_inter)
    return iou


def __IoU_single_2Dbox(box1: tf.Tensor, box2: tf.Tensor) -> tf.Tensor:
    """
    Calculate IoU on pred and GT.
    Boxes must have the same shape.

    - box1:

    |  0  |  1  |  2  |  3  |
    |-----|-----|-----|-----|
    | xl1 | yl1 | xr1 | yr1 |

    - box2:

    |  0  |  1  |  2  |  3  |
    |-----|-----|-----|-----|
    | xl2 | yl2 | xr2 | yr2 |

    :param box1: Shape = (..., 4).
    :param box2: Shape = (..., 4).
    """

    len1_x, len2_x, len_inter_x = \
        __get_len_1Dbox(tf.gather(box1, [0, 2], axis=-1),
                        tf.gather(box2, [0, 2], axis=-1))

    len1_y, len2_y, len_inter_y = \
        __get_len_1Dbox(tf.gather(box1, [1, 3], axis=-1),
                        tf.gather(box2, [1, 3], axis=-1))

    s_inter = len_inter_x * len_inter_y
    s_all = len1_x * len1_y + len2_x * len2_y
    iou = s_inter / (s_all - s_inter)
    return iou


def __get_len_1Dbox(box1: tf.Tensor, box2: tf.Tensor):
    """
    The shape of each box should be `(..., 2)`.
    Boxes should have the same shape.
    """
    len1 = tf.abs(box1[..., 0] - box1[..., 1])
    len2 = tf.abs(box2[..., 0] - box2[..., 1])

    len_all = [len1, len2]
    for i in [0, 1]:
        for j in [0, 1]:
            len_all.append(box1[..., i] - box2[..., j])

    len_max = tf.reduce_max(tf.abs(len_all), axis=0)
    len_inter = tf.maximum(len1 + len2 - len_max, 0.0)

    return len1, len2, len_inter
