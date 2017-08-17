import tensorflow as tf
from PIL import Image


def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'image/height': tf.FixedLenFeature([], tf.int64),
                                                    'image/width': tf.FixedLenFeature([], tf.int64),
                                                    'image/filename': tf.FixedLenFeature([], tf.string),
                                                    'image/class/text': tf.FixedLenFeature([], tf.string),
                                                    'image/encoded': tf.FixedLenFeature([], tf.string)
                                                }, name='features')
    height = tfrecord_features['image/height']
    width = tfrecord_features['image/width']
    filename = tfrecord_features['image/filename']
    label = tfrecord_features['image/class/text']
    image = tf.decode_raw(tfrecord_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [100, 47])
    return height, width, filename, label, image


def read_tfrecord(filename):
    height, width, filename, label, image = read_from_tfrecord([filename])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        height, width, filename, label, image = sess.run([height, width, filename, label, image])
        coord.request_stop()
        coord.join(threads)

    print(height)
    print(width)
    print(label)
    print(filename)
    Image.fromarray(image).show()


def main():
    read_tfrecord('./TFrecord1/validation-00000-of-00002')


if __name__ == '__main__':
    main()
