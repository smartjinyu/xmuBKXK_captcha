#!/usr/bin/python
# -*- coding=utf-8 -*-

__author__ = 'smartjinyu'
from PIL import Image
import numpy

filePath = "C:\\Users\\smart\\Desktop\\captcha\\0547.jpg"


def main():
    rawImage = Image.open(filePath)
    BlackWhiteImage = rawImage.convert('1')
    imArray = numpy.array(BlackWhiteImage)[:, 4:195]  # discard the start and end columns
    print(imArray.shape[1])
    print(imArray[:, 146])

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
        if imArray[i, 190] == 0:
            indexLastColumn.append(i + 1)
            if len(indexLastColumn) == 2:
                break
            else:
                i = i + 4
        i = i + 1

    print(indexFirstColumn)
    print(indexLastColumn)

    # check whether indexFirstColumn[0] and indexLastColumn[0] are in the same line
    k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 190.0
    count = 0
    for x in range(0, 190, 10):
        y = round(k0 * x + indexFirstColumn[0])
        if imArray[y, x] == 0:
            count = count + 1

    if count < 15:  # typically if they are in the same line, count >= 18
        temp = indexLastColumn[1]
        indexLastColumn[1] = indexLastColumn[0]
        indexLastColumn[0] = temp
        k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 190.0
    k1 = (indexLastColumn[1] - indexFirstColumn[1]) / 190.0

    for x in range(0, 190):
        y0 = round(k0 * x + indexFirstColumn[0])
        if 3 <= y0 <= 95 and ((imArray[y0 - 3, x] != 0) and (imArray[y0 + 4, x] != 0)):
            for j in range(y0 - 3, y0 + 4):
                imArray[j, x] = True

        y1 = round(k1 * x + indexFirstColumn[1])
        if 3 <= y1 <= 95 and ((imArray[y1 - 3, x] != 0) and (imArray[y1 + 4, x] != 0)):
            for j in range(y1 - 3, y1 + 4):
                imArray[j, x] = True

    im = Image.fromarray(numpy.uint(imArray * 255))
    im.show()


if __name__ == '__main__':
    main()
