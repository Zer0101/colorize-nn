import tensorflow as tf
import glob


class Files:
    def __init__(self):
        self.io = tf.WholeFileReader()

    def glob(self, paths, image_format='.jpg'):
        # Get names of training cases - sample files
        filenames = sorted(glob.glob(paths + '/*' + image_format))

        if not filenames:
            raise FileNotFoundError("Files not found!")

        return filenames
