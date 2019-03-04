import numpy as np
import gzip
import pickle as cp
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy

class SplitMnistGenerator():
    def __init__(self):
        with gzip.open('data/mnist.pkl.gz', 'rb') as file:
            u = cp._Unpickler(file)
            u.encoding = 'latin1'
            p = u.load()
            train_set, valid_set, test_set = p

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0],1 )), np.zeros((train_1_id.shape[0],1 )))).squeeze(-1)
            #next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0],1 )), np.zeros((test_1_id.shape[0], 1)))).squeeze(-1)
            #next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

hidden_size = [256, 256]
batch_size = None
no_epochs = 120
single_head = False

# Run vanilla VCL

coreset_size = 0
data_gen = SplitMnistGenerator()
vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(vcl_result)
np.save("./results/VCL-split{}".format(""), vcl_result)
# Run random coreset VCL
np.random.seed(0)
coreset_size = 40
data_gen = SplitMnistGenerator()
rand_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, single_head)
print(rand_vcl_result)
np.save("./results/randVCL-split{}".format(""), rand_vcl_result)
# Run k-center coreset VCL
np.random.seed(1)

data_gen = SplitMnistGenerator()
kcen_vcl_result = vcl.run_vcl(hidden_size, no_epochs, data_gen, 
    coreset.k_center, coreset_size, batch_size, single_head)
print(kcen_vcl_result)
np.save("./results/kcenVCL-split{}".format(""), kcen_vcl_result)
# Plot average accuracy
vcl_avg = np.nanmean(vcl_result, 1)
rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
utils.plot('results/split.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)
