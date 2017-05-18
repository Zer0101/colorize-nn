import tensorflow as tf
from app.utils import batch_normalize
from app.models.model import Model


class Colorize (Model):
    def __init__(self, tensors, phase):
        # Bx28x28x512 -> batch norm -> 1x1 convolution = Bx28x28x256
        conv1 = batch_normalize(x=tensors["conv4_3"], depth=512, phase_train=phase)
        conv1 = tf.nn.conv2d(conv1, tensors["weights"]["wc1"], [1, 1, 1, 1], 'SAME')
        conv1 = tf.nn.relu(conv1)
        # upscale to 56x56x256
        conv1 = tf.image.resize_bicubic(conv1, (56, 56))
        conv1 = tf.add(conv1, batch_normalize(x=tensors["conv3_3"], depth=256, phase_train=phase))

        # Bx56x56x256-> 3x3 convolution = Bx56x56x128
        conv2 = tf.nn.conv2d(conv1, tensors["weights"]['wc2'], [1, 1, 1, 1], 'SAME')
        conv2 = batch_normalize(conv2, tensors["weights"]['wc2'].get_shape()[3], phase_train=phase)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.maximum(0.01 * conv2, conv2)
        # upscale to 112x112x128
        conv2 = tf.image.resize_bicubic(conv2, (112, 112))
        conv2 = tf.add(conv2, batch_normalize(x=tensors["conv2_2"], depth=128, phase_train=phase))

        # Bx112x112x128 -> 3x3 convolution = Bx112x112x64
        conv3 = tf.nn.conv2d(conv2, tensors["weights"]['wc3'], [1, 1, 1, 1], 'SAME')
        conv3 = batch_normalize(conv3, tensors["weights"]['wc3'].get_shape()[3], phase_train=phase)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.maximum(0.01 * conv3, conv3)
        # upscale to Bx224x224x64
        conv3 = tf.image.resize_bicubic(conv3, (224, 224))
        conv3 = tf.add(conv3, batch_normalize(x=tensors["conv1_2"], depth=64, phase_train=phase))

        # Bx224x224x64 -> 3x3 convolution = Bx224x224x3
        conv4 = tf.nn.conv2d(conv3, tensors["weights"]['wc4'], [1, 1, 1, 1], 'SAME')
        conv4 = batch_normalize(conv4, tensors["weights"]['wc4'].get_shape()[3], phase_train=phase)
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.maximum(0.01 * conv4, conv4)
        conv4 = tf.add(conv4, batch_normalize(x=tensors["grayscale"], depth=3, phase_train=phase))

        # Bx224x224x3 -> 3x3 convolution = Bx224x224x3
        conv5 = tf.nn.conv2d(conv4, tensors["weights"]['wc5'], [1, 1, 1, 1], 'SAME')
        conv5 = batch_normalize(conv5, tensors["weights"]['wc5'].get_shape()[3], phase_train=phase)
        conv5 = tf.nn.relu(conv5)
        conv5 = tf.maximum(0.01 * conv5, conv5)

        # Bx224x224x3 -> 3x3 convolution = Bx224x224x2
        conv6 = tf.nn.conv2d(conv5, tensors["weights"]['wc6'], [1, 1, 1, 1], 'SAME')
        conv6 = batch_normalize(conv6, tensors["weights"]['wc6'].get_shape()[3], phase_train=phase)
        conv6 = tf.sigmoid(conv6)

        self.layers = {"conv_1": conv1, "conv_2": conv2, "conv_3": conv3, "conv_4": conv4, "conv_5": conv5,
                       "conv_6": conv6}

    def get_last_layer(self):
        return self.layers["conv_6"]
