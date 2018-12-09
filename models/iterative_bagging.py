'''
Iterative_bagging
'''
import argparse
import os.path as osp
import copy

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from pandas import DataFrame
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import random

from ..data import create
from ..utils.utils import draw_line_chart, save_results

def get_RMSE(predicted, gt):
  return np.sqrt(mean_squared_error(predicted,gt))
  

def iterative_bagging(args):
  epoch_num = args.epoch_num
  predictor_num = args.predictor_num
  x_train, x_test, y_train, y_test = create('concrete',args.data_dir) # initial x,y for train and test
  x_train = np.array(x_train)
  x_test = np.array(x_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  train_num = np.shape(x_train)[0]
  test_num = np.shape(x_test)[0]
  train_residual = copy.deepcopy(y_train) # y_n^j, the residual to fit in j-th iteration
  test_predict  = np.array([0]*test_num) # the predict result  after j-th iteration
  log = []
  for t in range(epoch_num):
    # print(train_residual[0])
    train_predict_dict = defaultdict(list)
    for i in range(predictor_num):
      tmpIds = []
      id_get = {}
      valid_id = []
      for j in range(train_num):
        tmpId = random.randint(0,train_num-1)
        tmpIds.append(tmpId)
        id_get[tmpId] = True
      for j in range(train_num):
        if j not in id_get:
          valid_id.append(j) # samples that not included in tmp trainset
      tmp_x_train = x_train[tmpIds]
      tmp_train_residual = train_residual[tmpIds]
      clf = DecisionTreeRegressor(max_depth=2)
      clf.fit(tmp_x_train, tmp_train_residual)
      train_predict = clf.predict(x_train)
      for j in valid_id:
        train_predict_dict[j].append(train_predict[j])
      test_predict = test_predict + clf.predict(x_test)*(1.0/predictor_num) # update the predict for sample j in testSet
    for i in train_predict_dict:
      train_residual[i] -= np.mean(train_predict_dict[i]) # update the residual of sample i in trainSet
    tRMSE  = get_RMSE(test_predict,y_test)
    print('{}\t{}'.format(t+1,tRMSE))
    sum_square_residual = np.mean(train_residual*train_residual)   
    if args.VIS:
      log.append([t + 1, tRMSE])
    if t==0:
      min_sum_square_residual = sum_square_residual
    else:
      if sum_square_residual > 1.1*min_sum_square_residual:
        break
    if sum_square_residual < min_sum_square_residual:
      min_sum_square_residual = sum_square_residual
  if args.VIS:
    draw_line_chart(DataFrame(log, columns=['the number of epochs', 'RMSE']), 'the number of epochs', 'RMSE',
                    'iterative_bagging_concrete')
  save_results(test_predict, osp.join(args.results_dir, 'iterative_bagging_concrete.txt'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch_num', type=int, default=50)
  parser.add_argument('--predictor_num', type=int, default=30)
  parser.add_argument('--data_dir', type=str, default='/Users/gaojingyue/Desktop/bagging_boosting/data/concrete')
  parser.add_argument('--results_dir', type=str, default='/Users/gaojingyue/Desktop/bagging_boosting/results')
  parser.add_argument('--VIS', action='store_true',default=False)
  iterative_bagging(parser.parse_args())