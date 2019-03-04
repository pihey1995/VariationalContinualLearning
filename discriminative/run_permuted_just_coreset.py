import numpy as np
import tensorflow as tf
import gzip
import pickle as cp
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy
import os

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):

        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = cp._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = np.arange(self.X_train.shape[1])
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = self.Y_train#np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = self.Y_test#np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [100, 100]
batch_size = 256
no_epochs = 100
single_head = True
num_tasks = 10

np.random.seed(0)


for coreset_size in [200,400,1000,2500,5000]:
    data_gen = PermutedMnistGenerator(num_tasks)
    vcl_result = vcl.run_vcl_vanilla(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, single_head)
    np.save("./results/only-coreset-{}".format(coreset_size), vcl_result)
    print(vcl_result)


