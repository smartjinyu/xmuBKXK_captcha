#!/usr/bin/python

__author__ = 'smartjinyu'
from PIL import Image
import numpy



def processImg(rawImg):
    BlackWhiteImage = rawImg.convert('1')
    imArray = numpy.array(BlackWhiteImage)[:, 4:195]  # discard the start and end columns
    print(imArray.shape)

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

    if len(indexFirstColumn) == 1:
        indexFirstColumn.append(indexFirstColumn[0]+1)
    if len(indexLastColumn) == 1:
        indexLastColumn.append(indexLastColumn[0]+2)

    print(indexFirstColumn)
    print(indexLastColumn)

    # check whether indexFirstColumn[0] and indexLastColumn[0] are in the same line
    k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 190.0
    count = 0
    for x in range(0, 190, 10):
        y = round(k0 * x + indexFirstColumn[0])
        if imArray[y, x] == 0:
            count = count + 1

    if count < 16:  # typically if they are in the same line, count >= 18
        temp = indexLastColumn[1]
        indexLastColumn[1] = indexLastColumn[0]
        indexLastColumn[0] = temp
        k0 = (indexLastColumn[0] - indexFirstColumn[0]) / 190.0
    k1 = (indexLastColumn[1] - indexFirstColumn[1]) / 190.0

    # eliminate interfering lines
    lowerBound = 2.5
    upperBound = 3.6
    # points in [y-lowerBond,y+upperBound] will be set to True (if no digit pixel)
    for x in range(0, 190):
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

    im = Image.fromarray(numpy.uint8(imArray * 255))
    # result = tesserocr.image_to_text(im)
    imgs = [Image.fromarray(numpy.uint8(imArray[:,0:50] * 255))]
    imgs.append(Image.fromarray(numpy.uint8(imArray[:,51:96] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray[:,97:146] * 255)))
    imgs.append(Image.fromarray(numpy.uint8(imArray[:,147:190] * 255)))
    return imgs



