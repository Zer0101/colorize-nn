import tensorflow as tf


class ImageTransformer:
    @staticmethod
    def rgb_transform_filter():
        transform_filter = tf.constant(
            [[[
                [0.299, -0.169, 0.499],
                [0.587, -0.331, -0.418],
                [0.114, 0.499, -0.0813]
            ]]], name="rgb_to_yuv_transform_filter"
        )

        return transform_filter

    @staticmethod
    def grb_bias():
        bias = tf.constant([0., 0.5, 0.5], name="rgb_to_yuv_bias")

        return bias

    @staticmethod
    def yuv_transform_filter():
        transform_filter = tf.constant(
            [[[
                [1., 1., 1.],
                [0., -0.34413999, 1.77199996],
                [1.40199995, -0.71414, 0.]
            ]]], name="yuv_to_rgb_transform_filter"
        )

        return transform_filter

    @staticmethod
    def yuv_bias():
        bias = tf.constant([-179.45599365, 135.45983887, -226.81599426], name="yuv_to_rgb_bias")

        return bias

    def from_rgb_to_yuv(self, rgb):
        # Get RGB to YUV transform matrix
        # transform_filter = tf.get_variable("rgb_to_yuv_transform_filter", initializer=self.rgb_transform_filter())
        transform_filter = tf.constant(
            [[[
                [0.299, -0.169, 0.499],
                [0.587, -0.331, -0.418],
                [0.114, 0.499, -0.0813]
            ]]], name="rgb_to_yuv_transform_filter"
        )
        # Get transformation bias
        # bias = tf.get_variable("rgb_to_yuv_bias", initializer=self.grb_bias())
        bias = tf.constant([0., 0.5, 0.5], name="rgb_to_yuv_bias")
        yuv = tf.nn.conv2d(rgb, transform_filter, [1, 1, 1, 1], 'SAME')
        yuv = tf.nn.bias_add(yuv, bias)

        return yuv

    def from_yuv_to_rgb(self, yuv):
        # Get YUV to RGB transform matrix
        # transform_filter = tf.get_variable("yuv_to_rgb_transform_filter", initializer=self.yuv_transform_filter())
        transform_filter = tf.constant(
            [[[
                [1., 1., 1.],
                [0., -0.34413999, 1.77199996],
                [1.40199995, -0.71414, 0.]
            ]]], name="yuv_to_rgb_transform_filter"
        )
        # Get transformation bias
        # bias = tf.get_variable("yuv_to_rgb_bias", initializer=self.yuv_bias())
        bias = tf.constant([-179.45599365, 135.45983887, -226.81599426], name="yuv_to_rgb_bias")

        yuv = tf.multiply(yuv, 255)
        rgb = tf.nn.conv2d(yuv, transform_filter, [1, 1, 1, 1], 'SAME')
        rgb = tf.nn.bias_add(rgb, bias)
        rgb = tf.maximum(rgb, tf.zeros(rgb.get_shape(), dtype=tf.float32))
        rgb = tf.minimum(rgb, tf.multiply(tf.ones(rgb.get_shape(), dtype=tf.float32), 255))
        rgb = tf.div(rgb, 255)

        return rgb

    def from_rgb_to_grayscale(self, image):
        return tf.image.rgb_to_grayscale(image)

    def from_grayscale_to_rgb(self, image):
        return tf.image.grayscale_to_rgb(image)
