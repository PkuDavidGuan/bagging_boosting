import os
import numpy as np
from sklearn.model_selection import train_test_split


def parse(line):
    items = line.strip().split(',')
    levelDict = {'a': 0, 'b': 1, 'c': 2, 'd': 4}
    shiftDict = {'W': 0, 'N': 1}
    x = items[:-1]
    for ids in [0, 1, 7]:
        x[ids] = levelDict[x[ids]]
    x[2] = shiftDict[x[2]]
    x = [float(i) for i in x]
    y = items[-1]
    return x, y


def Seismic(data_path):
    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
    raw_x = []
    raw_y = []
    rawFile = open(os.path.join(data_path, 'seismic-bumps.arff'), 'r')
    for l in rawFile:
        tmpx, tmpy = parse(l)
        raw_x.append(tmpx)
        raw_y.append(tmpy)
    rawFile.close()
    return train_test_split(raw_x, raw_y, train_size=0.8, random_state=2018)
    # x_train, x_rest, y_train, y_rest = train_test_split(
    #     raw_x, raw_y, train_size=0.7, random_state=2018)
    # x_valid, y_valid, x_test, y_test = [], [], [], []
    # for idx in range(len(x_rest)):
    #     if idx % 2 == 0:
    #         x_valid.append(x_rest[idx])
    #         y_valid.append(y_rest[idx])
    #     else:
    #         x_test.append(x_rest[idx])
    #         y_test.append(y_rest[idx])
    # print("Train instances: {}\n".format(len(y_train)))
    # print("Valid instances: {}\n".format(len(y_valid)))
    # print("Test instances: {}\n".format(len(y_test)))
    # return x_train, x_valid, x_test, y_train, y_valid, y_test
