'''
Adaboost MH
'''
import argparse
import os.path as osp

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from pandas import DataFrame

from ..data import create
from ..utils.utils import draw_line_chart, save_results


def SC_termimate(i, epoch_num):
  '''
  Determining sub-committee termination indexes
  :param i: # of the sub-committee
  :param epoch_num: # of the epochs
  :return: specifying the iteration at which each subcommittee i ≥ 1 should terminate.
  '''
  n = int(np.sqrt(epoch_num))
  if i < n:
    return np.ceil(i * epoch_num / n).astype(np.int64)
  else:
    return epoch_num


def possion_sample(length):
  sample = -np.log(np.random.random(length))
  sample = sample / np.sum(sample)
  return sample


def vote(x_test, clfs, beta):
  test_set_length = len(x_test)
  epoch_num = len(clfs)

  results_dic = [{} for i in range(test_set_length)]
  for t in range(epoch_num):
    result = clfs[t].predict(x_test)
    weight = np.log(1 / beta[t])

    for i in range(test_set_length):
      if result[i] not in results_dic[i]:
        results_dic[i][result[i]] = weight
      else:
        results_dic[i][result[i]] += weight
  y_predicted = []
  for i in range(test_set_length):
    dic = results_dic[i]
    y_predicted.append(sorted(dic, key=lambda k: dic[k])[-1])

  y_predicted = np.array(y_predicted)
  return y_predicted


def get_accuracy(predicted, gt):
  errors = predicted != gt
  error_rate = np.sum(errors) / len(gt)
  print('Accuracy: {}'.format(1 - error_rate))
  return 1 - error_rate


def multi_boosting(args):
  epoch_num = args.epoch_num

  x_train, x_test, y_train, y_test = create('cnae', args.data_dir)
  train_set_length = len(x_train)
  test_set_length = len(x_test)
  weights = np.ones(train_set_length, dtype=np.float64)
  k = 1
  beta = np.zeros(epoch_num, dtype=np.float64)
  clfs = []
  log = []

  for t in range(epoch_num):
    if SC_termimate(k, epoch_num) == t:
      weights = possion_sample(train_set_length)
      k += 1
    while True:
      clf = DecisionTreeClassifier(max_depth=20)
      clf.fit(x_train, y_train, sample_weight=weights)
      error_rate = 1 - clf.score(x_train, y_train, sample_weight=weights)
      if error_rate <= 0.5:
        break

    clfs.append(clf)

    if error_rate == 0:
      beta[t] = 1e-10
      weights = possion_sample(train_set_length)
      k += 1
    else:
      beta[t] = error_rate / (1 - error_rate)
      errors = clf.predict(x_train) != y_train
      for i in range(train_set_length):
        weights[i] = weights[i] / 2 / error_rate if errors[i] else weights[i] / 2 / (1 - error_rate)
        if weights[i] < 1e-8:
          weights[i] = 1e-8

    if args.VIS:
      y_predicted = vote(x_test, clfs, beta)
      acc = get_accuracy(y_predicted, y_test)
      log.append([t + 1, acc])

  y_predicted = vote(x_test, clfs, beta)
  acc = get_accuracy(y_predicted, y_test)

  if args.VIS:
    draw_line_chart(DataFrame(log, columns=['the number of epochs', 'accuracy']), 'the number of epochs', 'accuracy',
                    'MultiBoosting_seismic')
  save_results(y_predicted, osp.join(args.results_dir, 'multiboosting_cnae.txt'))
  return y_predicted


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch_num', type=int, default=10)
  parser.add_argument('--data_dir', type=str, default='/Users/DavidGuan/Desktop/机器学习/homework3/data/yeast/')
  parser.add_argument('--results_dir', type=str, default='/Users/DavidGuan/Desktop/机器学习/homework3/results/')
  parser.add_argument('--VIS', action='store_true')
  multi_boosting(parser.parse_args())