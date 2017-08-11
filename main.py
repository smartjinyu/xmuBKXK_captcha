#!/usr/bin/python

from PIL import Image
from tesserocr import PyTessBaseAPI
import processImg
import requests
import io
import uuid
import os

__author__ = 'smartjinyu'

savingDir = './trainData'


def main():
    if not os.path.exists(savingDir):
        createDir()

    count = 0
    for i in range(1, 10001):
        if login():
            count = count + 1
        print('{} / {} = {}%'.format(count, i, count / i))
    print(count)


def login():
    # try to login to http://bkxk.xmu.edu.cn/xsxk
    id = 12345  # raw_input('Please input your student id:')
    pwd = 'testtest'  # getpass.getpass('Please input the corresponding password:')
    loginUrl = 'http://bkxk.xmu.edu.cn/xsxk/login.html'
    # localInfoUrl = 'http://bkxk.xmu.edu.cn/xsxk/localInfo.html'
    try:
        session = requests.session()
        captcha_img = session.get('http://bkxk.xmu.edu.cn/xsxk/getCheckCode')
        rawImage = Image.open(io.BytesIO(captcha_img.content))
        imgs = processImg.processImg(rawImage)
        resultCaptcha = []
        with PyTessBaseAPI() as api:
            for img in imgs:
                # img.show()
                api.SetImage(img)
                api.SetVariable("tessedit_char_whitelist", "023456789")  # seems that no 1 in captcha
                api.SetPageSegMode(10)
                resultCaptcha.append(api.GetUTF8Text())

        captcha = ''.join(str(x) for x in resultCaptcha).replace('\n', '')
        print(captcha)

        loginData = {
            'username': id,
            'password': pwd,
            'checkCode': captcha,
        }
        html = session.post(loginUrl, loginData)
        if u'进入学生选课系统' in html.text:
            # print('Login successfully!')
            savePositive(imgs, captcha)
            return True
        elif u'用户名或密码错误' in html.text:
            # print('Wrong id or password!')
            savePositive(imgs, captcha)
            return True
        else:
            # print('Wrong captcha!')
            saveNegative(rawImage, captcha)
            return False
    except KeyboardInterrupt:
        exit(-1)
    except:
        print("Something went wrong in this iteration")
        return False



def createDir():
    '''
    create directory to save training set
    :return:
    '''
    for i in range(0, 10):
        os.makedirs(savingDir + '/' + str(i))
    os.makedirs(savingDir + '/failures')


def savePositive(imgs, captcha):
    '''
    save right captcha recognized by Tesseract
    :param imgs: a list of four images, with only one digit in each img
    :param captcha: str of captcha
    :return:
    '''
    for i in range(0, 4):
        img = imgs[i]
        filename = savingDir + '/' + captcha[i] + '/' + str(uuid.uuid1()) + '.jpg'
        img.save(filename, 'JPEG')


def saveNegative(img, captcha):
    '''
    save wrong captcha recognized by Tesseract
    :param img: raw image of captcha
    :param captcha: result given by Tesseract
    :return:
    '''
    filename = savingDir + '/failures/' + captcha.replace(' ', 'o') + '.jpg'
    img.save(filename, 'JPEG')


if __name__ == '__main__':
    main()
