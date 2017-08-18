# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This script is used to train and save TensorFlow models.
You should run downloadCaptchas.py, processImg.py and build_tfrecords.py in order beforehand.
Reference:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

import tensorflow as tf

import mnist
from PIL import Image
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = []
VALIDATION_FILE = []

Height = 47
Width = 100


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.

    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
      Note that an tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    filename = []
    if train:
        for name in TRAIN_FILE:
            filename.append(os.path.join(FLAGS.train_dir, name))
    else:
        for name in VALIDATION_FILE:
            filename.append(os.path.join(FLAGS.train_dir, name))
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filename, num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000, allow_smaller_final_batch=False)
        return images, sparse_labels


def read_single_image(filename):
    image = Image.open(filename)
    image_array = np.asarray(image, np.uint8)
    image = tf.decode_raw(image_array.tobytes(), tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image


def run_training():
    """Train for a number of steps."""
    images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)
    vimages, vlabels = inputs(train=False, batch_size=FLAGS.batch_size,
                              num_epochs=FLAGS.num_epochs)

    my_image = read_single_image('D:\\xmuBKXK_captcha\\valData\\8\\00ce70de-8023-11e7-80f3-000c29187544.jpg')

    # simple model
    # w = tf.get_variable(name='w1', shape=[Height * Width, 10])
    w = tf.Variable(tf.zeros([Height * Width, 10]), name='w1')
    y_pred = tf.matmul(images, w)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y_pred)
    my_y = tf.matmul([my_image], w)

    y_vpred = tf.matmul(vimages, w)
    correct_prediction = tf.equal(tf.argmax(y_vpred, 1), tf.cast(vlabels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.double))
    # for monitoring
    loss_mean = tf.reduce_mean(loss)
    train_op = tf.train.AdamOptimizer().minimize(loss)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    saver = tf.train.Saver()

    # Create a session for running operations in the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        step = 0
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss_mean])

            duration = time.time() - start_time
            # Print an overview fairly often.
            if step % 100 == 0:
                print('Step %d: loss = %.2f  (%.3f sec)' % (step, loss_value,
                                                            duration))
                print('Validation accuracy = %.6f' % sess.run(accuracy))
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.

    coord.join(threads)
    # save trained model
    if not os.path.exists(FLAGS.models_dir):
        os.makedirs(FLAGS.models_dir)
    save_path = saver.save(sess, FLAGS.models_dir + '/model.ckpt')

    print('Models saved in file: ', save_path)

    sess.close()


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
        help='Number of epochs to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./TFrecord',
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='./models',
        help='Directory to save trained model.'
    )

    for i in range(0, 1024):
        TRAIN_FILE.append('train-{:05d}-of-01024'.format(i))
    for i in range(0, 128):
        VALIDATION_FILE.append('validation-{:05d}-of-00128'.format(i))
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
