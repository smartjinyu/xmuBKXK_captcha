#!/usr/bin/python

from PIL import Image
import numpy
import os
import random

__author__ = 'smartjinyu'


def convertRawImg():
    inputPath = './trainData/rawData'
    outputPath_train = './trainData'
    outputPath_validate = './valData'

    if not os.path.exists(outputPath_train + '/processed'):
        os.makedirs(outputPath_train + '/processed')
    if not os.path.exists(outputPath_validate + '/processed'):
        os.makedirs(outputPath_validate + '/processed')

    for i in range(0, 10):
        if not os.path.exists(outputPath_train + '/' + str(i)):
            os.makedirs(outputPath_train + '/' + str(i))
        if not os.path.exists(outputPath_validate + '/' + str(i)):
            os.makedirs(outputPath_validate + '/' + str(i))

    fileList = os.listdir(inputPath)
    random.shuffle(fileList)
    length = len(fileList)
    j = 0
    for filename in fileList:
        # filename = '0008_b823fc04-81b0-11e7-b502-000c29187544.jpg'
        rawImg = Image.open(inputPath + '/' + filename)
        print('Processing ' + inputPath + '/' + filename)
        imgs, processed = processImg(rawImg)
        # processed.show()
        # input("Input to continue...")
        if j < round(length * 0.8):
            outputPath = outputPath_train
        else:
            outputPath = outputPath_validate
        outputFilename = outputPath + '/processed/' + filename
        processed.save(outputFilename, 'JPEG')
        names = filename.split('_')
        for i in range(0, 4):
            imgs[i].save(outputPath + '/' + names[0][i] + '/' + names[1], 'JPEG')
        j = j + 1
    return


def processImg(rawImg):
    """
    process the raw image, eliminate the interfering line, separate into four images, with only one digit in eah
    :param rawImg: the captcha to process
    :return: list of four images,first four with only one digit in each image; and the full processed image
    """
    BlackWhiteImage = rawImg.convert('1')
    imArray = numpy.array(BlackWhiteImage)[:, 5:193]  # discard the start and end columns
    # print(imArray.shape)

    # compute start and end points of two lines in first and last column
    indexFirstColumn = []
    indexLastColumn = []
    i = 0
    while i < imArray.shape[0]:
        if imArray[i, 0] == 0:
            indexFirstColumn.append(i + 1)
            if len(indexFirstColumn) == 2:
                break
            else:
                i = i + 5
        i = i + 1
    i = 0
    while i < imArray.shape[0]:
        if imArray[i, 187] == 0:
            indexLastColumn.append(i + 1)
            if len(indexLastColumn) == 2:
                break
            else:
                i = i + 5
        i = i + 1

    # avoid two lines intersect at the start or the end
    if len(indexFirstColumn) == 1:
        indexFirstColumn.append(indexFirstColumn[0] + 2)
    if len(indexLastColumn) == 1:
        indexLastColumn.append(indexLastColumn[0] + 2)

    # print(indexFirstColumn)
    # print(indexLastColumn)

    # check whether indexFirstColumn[0] and indexLastColumn[0] are in the same line
    k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 188.0
    count = 0
    for x in range(0, 188, 10):
        y = round(k0 * x + indexFirstColumn[0])
        if imArray[y, x] == 0:
            count = count + 1

    # print(count)
    if count < 14:  # typically if they are in the same line, count >= 18
        temp = indexLastColumn[1]
        indexLastColumn[1] = indexLastColumn[0]
        indexLastColumn[0] = temp
        k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 188.0
    k1 = (indexLastColumn[1] - indexFirstColumn[1]) / 188.0

    # eliminate interfering lines
    lowerBound = 2.7
    upperBound = 3.9
    # points in [y-lowerBond,y+upperBound] will be set to True (if no digit pixel)
    for x in range(0, 188):
        y0 = k0 * x + indexFirstColumn[0]
        lower = max(round(y0 - lowerBound), 0)
        upper = min(round(y0 + upperBound), 99)  # avoid array index out of bound
        # imArray[round(y0), x] = True
        if (imArray[lower, x] != 0) and (imArray[upper, x] != 0):
            for j in range(lower, upper + 1):
                imArray[j, x] = True

        y1 = k1 * x + indexFirstColumn[1]
        lower = max(round(y1 - lowerBound), 0)
        upper = min(round(y1 + upperBound), 99)
        # imArray[round(y1), x] = True
        if (imArray[lower, x] != 0) and (imArray[upper, x] != 0):
            for j in range(lower, upper + 1):
                imArray[j, x] = True

    # result = tesserocr.image_to_text(im)
    imgs = [Image.fromarray(numpy.uint8(imArray[:, 0:47] * 255))]
    imgs.append(Image.fromarray(numpy.uint8(imArray[:, 47:94] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray[:, 94:141] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray[:, 141:188] * 255)))
    processed = Image.fromarray(numpy.uint8(imArray * 255))
    return imgs, processed


if __name__ == '__main__':
    convertRawImg()
