#!/usr/bin/python
"""
This script is used to calculate the accuracy of Tesseract
You need to run downloadCaptchas.py first
"""

import os

__author__ = 'smartjinyu'

savingDir = '../trainData'


def calcAccuracy():
    total = 0
    for root, dir, files in os.walk(savingDir):
        total += len(files)
    rawData = len([name for name in os.listdir(savingDir + '/rawData')]) + len(
        [name for name in os.listdir(savingDir + '/processed')])

    negative = len([name for name in os.listdir(savingDir + '/failures')])
    positive = (total - rawData - negative) / 4
    accuracy = positive / (positive + negative)
    print('Accuracy = {} / {} = {:.7f}'.format(positive, positive + negative, accuracy))
    return accuracy


if __name__ == '__main__':
    print('This script is used to calculate the accuracy of Tesseract')
    print('You need to run downloadCaptchas.py first')
    calcAccuracy()
