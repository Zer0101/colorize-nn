import numpy as np
import tensorflow as tf
import argparse
from app.lib.normalize import ConvolutionBatchNormalizer


def concat_images(first_image, second_image):
    """
        Make concatenation side-by-side of two images in naraay format
    :param first_image: narray of the image
    :param second_image: narray of the image
    :return: narray with new image
    """
    first_image_height, first_image_width = first_image.shape[:2]
    second_image_height, second_image_width = second_image.shape[:2]

    max_height = np.max([first_image_height, second_image_height])
    total_width = first_image_width + second_image_width

    new_image = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_image[:first_image_height, :first_image_width] = first_image
    new_image[:second_image_height, first_image_width:first_image_width + second_image_width] = second_image

    return new_image


def batch_normalize(x, depth, phase_train):
    with tf.variable_scope('batch_normalize'):
        ema = tf.train.ExponentialMovingAverage(decay=0.99999)
        normalizer = ConvolutionBatchNormalizer(depth, 0.001, ema, True)
        normalizer.get_assigner()
        x = normalizer.normalize(x, train=phase_train)
    return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')