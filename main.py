#!/usr/bin/python

__author__ = 'smartjinyu'
from PIL import Image
from tesserocr import PyTessBaseAPI
import processImg

def main():
    filePath = "./captchas/8979.jpg"
    rawImage = Image.open(filePath)
    imgs = processImg.processImg(rawImage)
    resultCaptcha = []
    with PyTessBaseAPI() as api:
        for img in imgs:
            api.SetImage(img)
            api.SetVariable("tessedit_char_whitelist", "0123456789")
            api.SetPageSegMode(10)
            resultCaptcha.append(api.GetUTF8Text())

if __name__ == '__main__':
    main()