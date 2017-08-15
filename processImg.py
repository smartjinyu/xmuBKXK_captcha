#!/usr/bin/python

from PIL import Image
import numpy
import os

__author__ = 'smartjinyu'


def convertRawImg():
    inputPath = '../trainData/rawData'
    outputPath = '../trainData/processed'
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    fileList = os.listdir(inputPath)
    '''
    for filename in fileList:
        rawImg = Image.open(inputPath + '/' + filename)
        print('Processing ' + inputPath + '/' + filename)
        processed = processImg(rawImg)[4]
        outputFilename = outputPath + '/' + filename
        processed.save(outputFilename, 'JPEG')
        '''
    rawImg = Image.open(inputPath + '/' + fileList[3])
    imgs = processImg(rawImg)
    imgs[0].show()
    imgs[1].show()
    imgs[2].show()
    imgs[3].show()

    return


def processImg(rawImg):
    """
    process the raw image, eliminate the interfering line, separate into four images, with only one digit in eah
    :param rawImg: the captcha to process
    :return: list of five images,first four with only one digit in each image, and the last one is the full processed image
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
                i = i + 4
        i = i + 1
    i = 0
    while i < imArray.shape[0]:
        if imArray[i, 187] == 0:
            indexLastColumn.append(i + 1)
            if len(indexLastColumn) == 2:
                break
            else:
                i = i + 4
        i = i + 1

    # avoid two lines intersect at the start or the end
    if len(indexFirstColumn) == 1:
        indexFirstColumn.append(indexFirstColumn[0] + 1)
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
    if count < 17:  # typically if they are in the same line, count >= 18
        temp = indexLastColumn[1]
        indexLastColumn[1] = indexLastColumn[0]
        indexLastColumn[0] = temp
        k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 188.0
    k1 = (indexLastColumn[1] - indexFirstColumn[1]) / 188.0

    # eliminate interfering lines
    lowerBound = 2.5
    upperBound = 3.6
    # points in [y-lowerBond,y+upperBound] will be set to True (if no digit pixel)
    for x in range(0, 188):
        y0 = k0 * x + indexFirstColumn[0]
        lower = max(round(y0 - lowerBound), 0)
        upper = min(round(y0 + upperBound), 99)  # avoid array index out of bound
        if (imArray[lower, x] != 0) and (imArray[upper, x] != 0):
            for j in range(lower, upper):
                imArray[j, x] = True

        y1 = k1 * x + indexFirstColumn[1]
        lower = max(round(y1 - lowerBound), 0)
        upper = min(round(y1 + upperBound), 99)
        if (imArray[lower, x] != 0) and (imArray[upper, x] != 0):
            for j in range(lower, upper):
                imArray[j, x] = True

    # result = tesserocr.image_to_text(im)
    imgs = [Image.fromarray(numpy.uint8(imArray[:, 0:47] * 255))]
    imgs.append(Image.fromarray(numpy.uint8(imArray[:, 47:94] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray[:, 94:141] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray[:, 141:188] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray * 255)))
    return imgs


if __name__ == '__main__':
    convertRawImg()
