#!/usr/bin/python

__author__ = 'smartjinyu'
from PIL import Image
from tesserocr import PyTessBaseAPI
import processImg
import requests
import io


def main():
    count = 0
    for i in range(0, 99):
        if login():
            count = count + 1
    print(count)


def login():
    id = 12345  # raw_input('Please input your student id:')
    pwd = 'testtest'  # getpass.getpass('Please input the corresponding password:')
    loginUrl = 'http://bkxk.xmu.edu.cn/xsxk/login.html'
    localInfoUrl = 'http://bkxk.xmu.edu.cn/xsxk/localInfo.html'
    session = requests.session()
    captcha_img = session.get('http://bkxk.xmu.edu.cn/xsxk/getCheckCode')

    rawImage = Image.open(io.BytesIO(captcha_img.content))
    imgs = processImg.processImg(rawImage)
    resultCaptcha = []
    with PyTessBaseAPI() as api:
        for img in imgs:
            # img.show()
            api.SetImage(img)
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetPageSegMode(10)
            resultCaptcha.append(api.GetUTF8Text())

    captcha = ''.join(str(x) for x in resultCaptcha).replace('\n', '')
    print(captcha)
    if len(captcha) != 4:
        return False

    loginData = {
        'username': id,
        'password': pwd,
        'checkCode': captcha,
    }
    html = session.post(loginUrl, loginData)
    if u'进入学生选课系统' in html.text:
        print('Login successfully!')
        return True
    elif u'用户名或密码错误' in html.text:
        print('Wrong id or password!')
        return True
    elif u'验证码错误' in html.text:
        print('Wrong captcha!')
        return False


if __name__ == '__main__':
    main()
