import os
import numpy as np
from sklearn.model_selection import train_test_split


def parse(line):
    features = [float(f) for f in line.strip().split(',')]
    return features[:-1], features[-1]


def Concrete(data_path):
    raw_x = []
    raw_y = []
    rawFile = open(os.path.join(data_path, 'Concrete_Data.csv'), 'r')
    for l in rawFile:
        tmpx, tmpy = parse(l)
        raw_x.append(tmpx)
        raw_y.append(tmpy)
    rawFile.close()
    return train_test_split(raw_x, raw_y, train_size=0.8, random_state=2018)
