import tensorflow as tf
import numpy as np
import processImg
from PIL import Image

Height = 47
Width = 100
IMAGE_PIXELS = Height * Width


def main():
    image = Image.open('./trainData/rawData/0002_728cf82c-8009-11e7-80c1-000c29187544.jpg')
    image.show()
    print(ocrCaptcha(image))


def ocrCaptcha(image):
    """
    recognize a captcha from http://bkxk.xmu.edu.cn/xsxk/login.html
    :param image: image data of the captcha
    :return: a string with four character
    """
    images, _ = processImg.processImg(image)
    result = []
    for img in images:
        result.append(ocrSingleCaptcha(img))
    return str(result)


def ocrSingleCaptcha(img):
    """
    recognize the image with a single character
    :param img: image with only one character, returned by processImg()
    :return: recognition result
    """
    tf.reset_default_graph()
    image_array = np.asarray(img, np.uint8)
    image = tf.decode_raw(image_array.tobytes(), tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # w = tf.get_variable(name='w1', shape=[Height * Width, 10])
    w = tf.Variable(tf.zeros([Height*Width, 10]), name='w1')
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    with sess:
        saver.restore(sess, './model.ckpt')
        y_pred = tf.matmul([image], w)
        prediction = tf.arg_max(y_pred, 1)
        return sess.run(prediction)


if __name__ == '__main__':
    main()
