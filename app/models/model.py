from app.lib.normalize import ConvolutionBatchNormalizer
import tensorflow as tf


class Model:
    def batch_normalize(self, x, depth, phase_train):
        ema = tf.train.ExponentialMovingAverage(decay=0.99999)
        normalizer = ConvolutionBatchNormalizer(depth, 0.001, ema, True)
        normalizer.get_assigner()
        x = normalizer.normalize(x, train=phase_train)

        return x
