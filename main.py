from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils.util

import argparse
import sys
import os

import tensorflow as tf

from app.app import Application

FLAGS = None


def main(_):
    try:
        directory = os.path.dirname(os.path.abspath(__file__))
        directory = directory.replace("\\", '/') + '/'
        Application.run(work_type=FLAGS.type, config=FLAGS, work_dir=directory)
    except ValueError as error:
        print(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', '-m', type=str, help='UID of the trained NN model')
    parser.add_argument('--type', type=str, default='train',
                        help='Mode of work for this application (train or colorize)')
    """
        CLI arguments for training the model
    """
    parser.add_argument('--continue', '-c', type=distutils.util.strtobool, default='false',
                        help='Change this value to continue train existing model')
    parser.add_argument('--model_learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--model_learning_rate_step', type=float, default=1000, help='Learning rate decrease step')
    parser.add_argument('--model_learning_rate_decay', type=float, default=0.001, help='Learning rate decay value')
    parser.add_argument('--model_epochs', type=int, default=10,
                        help='Number of training epochs. Different values have impact on NN model')
    parser.add_argument('--model_save', type=distutils.util.strtobool, default='true',
                        help='Indicates if we need to save trained model')
    parser.add_argument('--model_save_pass', type=int, default=100,
                        help='Numbers of steps to save model in process of work')
    parser.add_argument('--model_save_path', type=str, help='Path where will be saved trained model')
    parser.add_argument('--model_log_level', type=int, default=1, help='Enable/disable logging of model train process')
    parser.add_argument('--model_log_dir', type=str, help='Path to log directory')
    parser.add_argument('--images_batch_size', type=int, default=1,
                        help='Number of images in batch. Can affect performance')
    parser.add_argument('--images_input', type=str, help='Path to directory with training samples')
    parser.add_argument('--images_output_enable', default='false', type=distutils.util.strtobool,
                        help='Enable/disable image saving in process of training')
    parser.add_argument('--images_output_step', type=int, default=10, help='Indicates step for images saving')
    parser.add_argument('--images_output_format', type=str, default='.jpg', help='Indicates step for images saving')
    parser.add_argument('--images_output', type=str, help='Path to directory with training samples output')

    """
        CLI arguments for validation & colorizing
    """
    parser.add_argument('--multiple', type=bool, default=False, help='Process batch of images or single image')
    parser.add_argument('--prefix', type=str, help='Add prefix to processed images')
    parser.add_argument('--samples', type=str, help='Path to directory with images')
    parser.add_argument('--output', type=str, help='Path to directory for images output')
    parser.add_argument('--format', type=str, default='.jpg', help='Path to directory for images output')
    parser.add_argument('--model_dir', type=str, default='assets/models', help='Path to directory with models')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
