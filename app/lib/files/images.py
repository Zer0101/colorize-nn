import tensorflow as tf
from app.lib.files.files import Files


class Images(Files):
    def load(self, queue, random_crop=True, randomize=False):
        key, file = self.io.read(queue)
        uint8_image = tf.image.decode_jpeg(file, channels=3)
        if random_crop:
            uint8_image = tf.random_crop(uint8_image, (224, 224, 3))
        if randomize:
            uint8_image = tf.image.random_flip_left_right(uint8_image)
            uint8_image = tf.image.random_flip_up_down(uint8_image, seed=None)
        float_image = tf.div(tf.cast(uint8_image, tf.float32), 255)

        return float_image
