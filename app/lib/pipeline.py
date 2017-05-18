import tensorflow as tf
from app.lib.files.images import Images


class Pipeline:
    def __init__(self):
        self.reader = Images()

    def load(self, paths, batch_size=1, epochs=None, min_after_dequeue=100):
        if paths is None:
            pass
        queue = tf.train.string_input_producer(paths, num_epochs=epochs, shuffle=False)
        files = self.reader.load(queue, randomize=False)
        capacity = min_after_dequeue + 3 * batch_size
        batch = tf.train.shuffle_batch([files], batch_size=batch_size, capacity=capacity,
                                       min_after_dequeue=min_after_dequeue)

        return batch
