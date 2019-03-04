import numpy as np
import tensorflow as tf
import utils
from multihead_models import Vanilla_NN, MFVI_NN
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchviz import make_dot, make_dot_from_trace

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Train network with maximum likelihood to initialize first model
        if task_id == 0:
            print_graph = False

            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            #mf_weights = [[torch.ones((784, 100)).to(device),torch.ones((100, 100)).to(device)],[torch.ones((100,)).to(device),torch.ones((100,)).to(device)],[torch.ones((100, 10)).to(device)],[torch.ones((10,)).to(device)]]
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=mf_weights)



        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
        # Train on non-coreset data
        if print_graph:
            params = dict()
            for i in range(len(mf_model.W_m)):
                params["W_m{}".format(i)] = mf_model.W_m[i]
                params["W_v{}".format(i)] = mf_model.W_v[i]
                params["b_m{}".format(i)] = mf_model.b_m[i]
                params["b_v{}".format(i)] = mf_model.b_v[i]
                params["prior_W_m".format(i)] = mf_model.prior_W_m[i]
                params["prior_W_v".format(i)] = mf_model.prior_W_v[i]
                params["prior_b_m".format(i)] = mf_model.prior_b_m[i]
                params["prior_b_v".format(i)] = mf_model.prior_b_v[i]

            for i in range(len(mf_model.W_last_m)):
                 params["W_last_m".format(i)] = mf_model.W_last_m[i]
                 params["W_last_v".format(i)] = mf_model.W_last_v[i]
                 params["b_last_m".format(i)] = mf_model.b_last_m[i]
                 params["b_last_v".format(i)] = mf_model.b_last_v[i]
                 params["prior_W_last_m".format(i)] = mf_model.prior_W_last_m[i]
                 params["prior_W_last_v".format(i)] = mf_model.prior_W_last_v[i]
                 params["prior_b_last_m".format(i)] = mf_model.prior_b_last_m[i]
                 params["prior_b_last_v".format(i)] = mf_model.prior_b_last_v[i]

            dot = make_dot(mf_model._KL_term(), params = params)#(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), task_id), params=params)
            dot.view()
            print_graph = False

        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        mf_model.save_weights()

        # Incorporate coreset data and make prediction
        mf_model.update_prior()
        acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size)
        all_acc = utils.concatenate_results(acc, all_acc)

        mf_model.load_weights()
        mf_model.clean_copy_weights()
        ##Create head + update prior
        mf_model.new_task()

    return all_acc

def run_vcl_vanilla(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])


    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size
        if task_id == 0:
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=None)

        # Train network with maximum likelihood to initialize first model
        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)


        mf_model.save_weights()
        # Incorporate coreset data and make prediction
        acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size, just_vanilla =True)
        all_acc = utils.concatenate_results(acc, all_acc)
        mf_model.load_weights()
        mf_model.clean_copy_weights()

    return all_acc

