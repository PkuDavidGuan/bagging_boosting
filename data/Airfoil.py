import os
import numpy as np
from sklearn.model_selection import train_test_split


def parse(line):
    features = [float(f) for f in line.strip().split('\t')]
    return features[:-1], features[-1]


def Airfoil(data_path):
    raw_x = []
    raw_y = []
    rawFile = open(os.path.join(data_path, 'airfoil_self_noise.txt'), 'r')
    for l in rawFile:
        tmpx, tmpy = parse(l)
        raw_x.append(tmpx)
        raw_y.append(tmpy)
    rawFile.close()
    return train_test_split(raw_x, raw_y, train_size=0.8, random_state=2018)
