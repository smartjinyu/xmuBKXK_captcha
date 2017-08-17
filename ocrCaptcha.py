import tensorflow as tf
import numpy as np
import processImg
from PIL import Image
IMAGE_PIXELS = 47 * 100


def main():
    image = Image.open('./trainData/rawData/0002_728cf82c-8009-11e7-80c1-000c29187544.jpg')
    print(ocrCaptcha(image))



def ocrCaptcha(image):
    """
    recognize a captcha from http://bkxk.xmu.edu.cn/xsxk/login.html
    :param image: image data of the captcha
    :return: a string with four character
    """
    images,_ = processImg.processImg(image)
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
    image_array = np.asarray(img, np.uint8)
    image = tf.decode_raw(image_array.tobytes(), tf.uint8)
    image.set_shape([IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess:
        saver = tf.train.import_meta_graph('./model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        w = graph.get_tensor_by_name('w1')
        y_pred = tf.matmul(image, w)
        prediction = tf.arg_max(y_pred, 1)
        return sess.run(prediction)


if __name__ == '__main__':
    main()
