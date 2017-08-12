#!/usr/bin/python
import os

__author__ = 'smartjinyu'

savingDir = '../trainData'


def calcAccuracy():
    total = 0
    for root, dir, files in os.walk(savingDir):
        total += len(files)
    rawData = len([name for name in os.listdir(savingDir + '/rawData')])
    negative = len([name for name in os.listdir(savingDir + '/failures')])
    positive = (total - rawData - negative) / 4
    accuracy = positive / (positive + negative)
    print('Accuracy = {} / {} = {:.6f}'.format(positive, positive + negative, accuracy))
    return accuracy


if __name__ == '__main__':
    calcAccuracy()
