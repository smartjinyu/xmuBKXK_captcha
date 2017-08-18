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
    result = ocrSingleCaptcha(images)
    return result


def ocrSingleCaptcha(images):
    """
    recognize the image after processing
    :param images: list of four images returned by processImg()
    :return: recognition result
    """
    image_data = []
    tf.reset_default_graph()

    for img in images:
        image_array = np.asarray(img, np.uint8)
        image = tf.decode_raw(image_array.tobytes(), tf.uint8)
        image.set_shape([IMAGE_PIXELS])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image_data.append(image)
    # w = tf.get_variable(name='w1', shape=[Height * Width, 10])
    w = tf.Variable(tf.zeros([Height * Width, 10]), name='w1')
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    with sess:
        saver.restore(sess, './model.ckpt')
        y_pred = tf.matmul(image_data, w)
        pred_array = np.asarray(sess.run(y_pred))
        result = []
        for item in pred_array:
            item[0] = -20  # It is impossible for the result to be 1
            index = np.argmax(item, 0)
            if index == 1:
                index = 0  # index 1 represents result 1 in truth
            result.append(index)
        return ''.join(str(e) for e in result)


if __name__ == '__main__':
    main()