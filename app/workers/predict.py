"""
    :deprecated
"""

import os
import sys

import tensorflow as tf
from matplotlib import pyplot as plt

from app.lib.color.image import ImageTransformer
from app.lib.files.files import Files
from app.lib.pipeline import Pipeline
from app.models.standard import Colorize


class Predict:
    @staticmethod
    def run(self, work_configs, work_dir):
        train_configs = None
        configs = work_configs.configs
        model_id = configs['id']
        model_dir = configs['model_dir']
        model_path = model_dir + '/' + model_id + '/'
        save_path = configs['output']
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print("Images will be saved to \"" + save_path + "\" ")

        save_model = False
        if save_model is not None:
            # saver = tf.train.Saver()
            save_path = ''

        # This value indicates if images must be saved
        save_images = True
        if save_images is not None:
            # saver = tf.train.Saver()
            image_save_path = ''
        log_save_path = ''

        batch_size = tf.constant(1, name='batch_size')
        # Initialize number of epochs - very important and sensitive value
        epochs = tf.constant(1, name='global_epochs')
        # Create global step value. It will change automatically with in tensorflow context
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Activate learning rate with decay - NN with batch normalization MUST have decaying leaning rate
        learning_rate = tf.train.exponential_decay(0.01, 1000, 0.001, global_step, staircase=True)
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        uv = tf.placeholder(tf.uint8, name='uv')

        # Prepare input data reader and image color transformer
        image_transformer = ImageTransformer()
        image_pipeline = Pipeline()
        file_reader = Files()

        files = []
        if configs['multiple'] is not None:
            try:
                files = file_reader.glob(work_dir + configs['samples'])
            except FileNotFoundError:
                print("In directory must be samples for colorization")
                exit()
            except NotADirectoryError:
                print("Invalid directory specified")
                exit()
        else:
            files = [configs['samples']]
        #
        # grayscale_image_rgb = color_image_rgb = grayscale_image = image_pipeline.load(files, 1, 1)
        # grayscale_image_yuv = color_image_yuv = image_transformer.from_rgb_to_yuv(color_image_rgb)
        # grayscale_image_yuv = image_transformer.from_rgb_to_yuv(grayscale_image)
        # grayscale = grayscale_image

        color_image_rgb = image_pipeline.load(files, 1, 1, 1)
        color_image_yuv = image_transformer.from_rgb_to_yuv(color_image_rgb)
        grayscale_image = image_transformer.from_rgb_to_grayscale(color_image_rgb)
        grayscale_image_rgb = image_transformer.from_grayscale_to_rgb(grayscale_image)
        grayscale_image_yuv = image_transformer.from_rgb_to_yuv(grayscale_image_rgb)
        grayscale = tf.concat([grayscale_image, grayscale_image, grayscale_image], 3, 'grayscale_image_tensor')
        # grayscale = tf.concat([grayscale_image, grayscale_image, grayscale_image], 3, 'grayscale_image_tensor')

        graph_def = tf.GraphDef()
        # We load trained classification model to use it intermediary convolution layers
        # This pass will is only for first training
        # Every training continuation will use model generated from VGG-16
        try:
            with open("vgg/tensorflow-vgg16/vgg16.tfmodel", mode='rb') as file:
                print('Loaded VGG-16 model')
                file_content = file.read()
                graph_def.ParseFromString(file_content)
                file.close()
        except FileNotFoundError as e:
            print('Cannot find VGG-16 model. Training is stopped')
            exit()
        finally:
            sys.stdout.flush()

        tf.import_graph_def(graph_def, input_map={"images": grayscale})

        graph = tf.get_default_graph()
        with tf.variable_scope('vgg'):
            conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
            conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
            conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
            conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")

        with tf.variable_scope('Colorize'):
            weights = {
                # 1x1 conv, 512 inputs, 256 outputs
                'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
                # 3x3 conv, 512 inputs, 128 outputs
                'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
                # 3x3 conv, 256 inputs, 64 outputs
                'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
                # 3x3 conv, 128 inputs, 3 outputs
                'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
                # 3x3 conv, 6 inputs, 3 outputs
                'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
                # 3x3 conv, 3 inputs, 2 outputs
                'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
            }
            tensors = {
                "conv1_2": conv1_2,
                "conv2_2": conv2_2,
                "conv3_3": conv3_3,
                "conv4_3": conv4_3,
                "grayscale": grayscale,
                "weights": weights
            }

        model = Colorize(tensors=tensors, phase=phase_train)
        # Get weights from model
        last_layer = model.get_last_layer()
        # Transform them to RGB
        last_layer_yuv = tf.concat(values=[tf.split(axis=3, num_or_size_splits=3,
                                                    value=grayscale_image_yuv)[0], last_layer], axis=3)
        # transform yuv back to RGB
        last_layer_rgb = image_transformer.from_yuv_to_rgb(last_layer_yuv)

        # Calculate the loss
        loss = tf.square(tf.subtract(last_layer, tf.concat(
            [tf.split(axis=3, num_or_size_splits=3, value=color_image_yuv)[1],
             tf.split(axis=3, num_or_size_splits=3, value=color_image_yuv)[2]], 3)))
        if uv == 1:
            loss = tf.split(axis=3, num_or_size_splits=2, value=loss)[0]
        elif uv == 2:
            loss = tf.split(axis=3, num_or_size_splits=2, value=loss)[1]
        else:
            loss = (tf.split(axis=3, num_or_size_splits=2, value=loss)[0] + tf.split(axis=3, num_or_size_splits=2,
                                                                                     value=loss)[1]) / 2
        # Run the optimizer
        opt = None
        if phase_train is not None:
            optimizer = tf.train.GradientDescentOptimizer(0.0001)
            opt = optimizer.minimize(loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)

        # Summaries
        tf.summary.histogram(name="weights1", values=weights["wc1"])
        tf.summary.histogram(name="weights2", values=weights["wc2"])
        tf.summary.histogram(name="weights3", values=weights["wc3"])
        tf.summary.histogram(name="weights4", values=weights["wc4"])
        tf.summary.histogram(name="weights5", values=weights["wc5"])
        tf.summary.histogram(name="weights6", values=weights["wc6"])
        tf.summary.histogram(name="instant_loss", values=tf.reduce_mean(loss))
        tf.summary.image(name="colorimage", tensor=color_image_rgb, max_outputs=1)
        tf.summary.image(name="pred_rgb", tensor=last_layer_rgb, max_outputs=1)
        tf.summary.image(name="grayscale", tensor=grayscale_image_rgb, max_outputs=1)

        # Create a session for running operations in the Graph.
        session = tf.InteractiveSession()
        # Create the graph, etc.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Saver
        saver = tf.train.Saver()
        # Initialize the variables.
        session.run(init_op)

        ckpt = tf.train.get_checkpoint_state(model_path)
        # print
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print("Load model finished!")
        else:
            print("Failed to restore model")
            exit()

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        # Merge summary
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('', session.graph)

        # Start a number of threads, passing the coordinator to each of them.
        with coord.stop_on_exception():
            while not coord.should_stop():
                training_opt = session.run(opt, feed_dict={phase_train: False})

                step = session.run(global_step)
                last_layer_, last_layer_rgb_, color_image_rgb_, grayscale_image_rgb_, cost, merged_ = session.run(
                    [last_layer, last_layer_rgb, color_image_rgb, grayscale_image_rgb, loss, merged],
                    feed_dict={phase_train: False, uv: 3})
                # colorized = session.run([grayscale], feed_dict={phase_train: False, uv: 3})
                # print(step)
                # colorized = session.run(grayscale_image, feed_dict={phase_train: False})
                name = save_path + configs['prefix'] + "_" + step + configs['format']
                plt.imsave(name, last_layer_rgb_[0])
                print("Image saved in: %s" % name)

        # Wait for threads to finish.
        coord.join(threads)
        session.close()
        exit()
        # This value indicates if model must be saved
