import os
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from app.lib.pipeline import Pipeline
from app.lib.color.image import ImageTransformer
from app.lib.files.files import Files
from app.models.standard import Colorize
from app.utils import concat_images


class Train:
    @staticmethod
    def run(train_configs, work_dir, predict=False):
        configs = train_configs.configs
        model_id = configs['id']
        model_dir = configs['dir'] + '/' + model_id
        # This value indicates if model must be saved

        save_model = configs['save']['enable']
        if save_model is not None and save_model:
            # saver = tf.train.Saver()
            save_path = configs['save']['path'] + model_id + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print("Model will be saved to \"" + save_path + "\" ")

        # This value indicates if images must be saved
        save_images = configs['images']['output']['enable']
        if save_images is not None and save_images:
            # saver = tf.train.Saver()
            image_save_path = configs['images']['output']['path'] + model_id + '/'
            if not os.path.exists(image_save_path):
                os.makedirs(image_save_path)
            print("Images will be saved to \"" + image_save_path + "\" ")

        log_save_path = configs['log']['dir'] + model_id + '/'
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)
        print("Logs will be saved to \"" + log_save_path + "\" ")

        batch_size = tf.constant(configs['images']['batch_size'], name='batch_size')
        # Initialize number of epochs - very important and sensitive value
        epochs = tf.constant(configs['epochs'], name='global_epochs')
        # Create global step value. It will change automatically with in tensorflow context
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Activate learning rate with decay - NN with batch normalization MUST have decaying leaning rate
        learning_rate = tf.train.exponential_decay(configs['learning_rate']['value'], global_step,
                                                   configs['learning_rate']['step'],
                                                   configs['learning_rate']['decay'], staircase=True)
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        uv = tf.placeholder(tf.uint8, name='uv')

        # Prepare input data reader and image color transformer
        image_transformer = ImageTransformer()
        image_pipeline = Pipeline()
        file_reader = Files()

        """
           Next step - work with images
           -- Creating images pipeline
           -- Creating grayscale images
        """
        color_image_rgb = image_pipeline.load(file_reader.glob(work_dir + configs['images']['input']['path']),
                                              configs['images']['batch_size'], epochs=configs['epochs'])
        color_image_yuv = image_transformer.from_rgb_to_yuv(color_image_rgb)
        grayscale_image = image_transformer.from_rgb_to_grayscale(color_image_rgb)
        grayscale_image_rgb = image_transformer.from_grayscale_to_rgb(grayscale_image)
        grayscale_image_yuv = image_transformer.from_rgb_to_yuv(grayscale_image_rgb)
        grayscale = tf.concat([grayscale_image, grayscale_image, grayscale_image], 3, 'grayscale_image_tensor')

        """
            Initializing tensor with weights
        """
        weights = {
            # 1x1 convolution, 512 inputs, 256 outputs
            'wc1': tf.Variable(tf.truncated_normal([1, 1, 512, 256], stddev=0.01)),
            # 3x3 convolution, 512 inputs, 128 outputs
            'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.01)),
            # 3x3 convolution, 256 inputs, 64 outputs
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.01)),
            # 3x3 convolution, 128 inputs, 3 outputs
            'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.01)),
            # 3x3 convolution, 6 inputs, 3 outputs
            'wc5': tf.Variable(tf.truncated_normal([3, 3, 3, 3], stddev=0.01)),
            # 3x3 convolution, 3 inputs, 2 outputs
            'wc6': tf.Variable(tf.truncated_normal([3, 3, 3, 2], stddev=0.01)),
        }

        # Create a session for running operations in the Graph.
        session = tf.InteractiveSession()

        graph_def = tf.GraphDef()
        # We load trained classification model to use it intermediary convolution layers
        # This pass will is only for first training
        # Every training continuation will use model generated from VGG-16
        try:
            with open(configs['vgg'], mode='rb') as file:
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

        # Create the graph, etc.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Saver
        saver = tf.train.Saver()
        # Initialize the variables.

        session.run(init_op)

        # Merge summary
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_save_path, session.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        # If continue is set we reinit from checkpoint
        if configs['continue']:
            """
                Continue to train model
            """
            if model_dir is None:
                print("Cannot find the model. Please specify path to model")
                exit()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            # print
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
                print("Load model finished!")
            else:
                print("Failed to restore model")
                exit()

        # Start a number of threads, passing the coordinator to each of them.
        with coord.stop_on_exception():
            while not coord.should_stop():
                if predict is not None and not predict:
                    print('Running training...')
                    # Run training steps
                    training_opt = session.run(opt, feed_dict={phase_train: True, uv: 1})
                    training_opt = session.run(opt, feed_dict={phase_train: True, uv: 2})

                step = session.run(global_step)
                if step % 1 == 0:
                    last_layer_, last_layer_rgb_, color_image_rgb_, grayscale_image_rgb_, cost, merged_ = session.run(
                        [last_layer, last_layer_rgb, color_image_rgb, grayscale_image_rgb, loss, merged],
                        feed_dict={phase_train: False, uv: 3})

                    if predict is not None and not predict:
                        print("Running step: %d" % step)
                        print("Cost: %f" % np.mean(cost))

                    if save_images is not None and save_images:
                        if configs['images']['output']['step'] is not None \
                                and step % configs['images']['output']['step'] == 0:
                            print("Saving images...")
                            images_format = configs['images']['output']['format']
                            summary_image = concat_images(grayscale_image_rgb_[0], last_layer_rgb_[0])
                            summary_image = concat_images(summary_image, color_image_rgb_[0])

                            step_prefix = str(step)
                            image_name_prefix = image_save_path + step_prefix

                            plt.imsave(image_name_prefix + "_summary" + images_format, summary_image)
                            plt.imsave(image_name_prefix + "_grayscale" + images_format,
                                       grayscale_image_rgb_[0])
                            plt.imsave(image_name_prefix + "_color" + images_format, color_image_rgb_[0])
                            plt.imsave(image_name_prefix + "_colorized" + images_format,
                                       last_layer_rgb_[0])

                            print("Saved image at run: %d" % step)
                            sys.stdout.flush()

                    sys.stdout.flush()
                    writer.add_summary(merged_, step)
                    writer.flush()
                if save_model is not None and save_model:
                    if configs['save']['step'] is not None and step % configs['save']['step'] == 0:
                        print("Saving model...")
                        saver.save(session, save_path + 'model.ckpt')
                        print("Model saved in file: %s" % save_path + 'model.ckpt')
                        sys.stdout.flush()

        # Wait for threads to finish.
        coord.join(threads)
        session.close()
