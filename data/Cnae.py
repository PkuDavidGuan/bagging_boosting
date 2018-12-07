import os.path as osp
from sklearn.model_selection import train_test_split


def parse(line):
  features = [int(f) for f in line.split(',')]
  return features[1:], features[0]


def Cnae(data_path, with_val=False):
  x_array, y_array = [], []
  with open(osp.join(data_path,'CNAE-9.data'), 'r') as infile:
    while True:
      line = infile.readline().strip()
      if not line:
        break
      x, y = parse(line)
      x_array.append(x)
      y_array.append(y)
  if not with_val:
    return train_test_split(x_array, y_array, train_size=900)
  else:
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, train_size=900)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=720)
    return x_train, x_val, x_test, y_train, y_val, y_test
