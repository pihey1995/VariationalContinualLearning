import numpy as np
import discriminative.utils.test  as test
from discriminative.utils.multihead_models import Vanilla_NN, MFVI_NN
import torch
import discriminative.utils.GAN as GAN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
try:
    from torchviz import make_dot, make_dot_from_trace
except ImportError:
    print("Torchviz was not found.")

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, gan_bol = False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    gans = []
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
            print_graph_bol = False #set to True if you want to see the graph
            ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0])
            ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
            mf_weights = ml_model.get_weights()
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=mf_weights)

        if not gan_bol:
            if coreset_size > 0:
                x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)
            gans = None
        if print_graph_bol:
            #Just if you want to see the computational graph
            output_tensor = mf_model._KL_term() #mf_model.get_loss(torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device), task_id), params=params)
            print_graph(mf_model, output_tensor)
            print_graph_bol = False

        if gan_bol:
            gan_i = GAN.VGR(task_id)
            gan_i.train(x_train, y_train)
            gans.append(gan_i)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)

        mf_model.update_prior()
        # Save weights before test (and last-minute training on coreset
        mf_model.save_weights()

        acc = test.get_scores(mf_model, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, False,gans)
        all_acc = test.concatenate_results(acc, all_acc)

        mf_model.load_weights()
        mf_model.clean_copy_weights()


        if not single_head:
            mf_model.create_head()

    return all_acc

def run_coreset_only(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    all_acc = np.array([])

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        if task_id == 0:
            mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], single_head = single_head, prev_means=None)

        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)


        mf_model.save_weights()

        acc = test.get_scores(mf_model, x_testsets, y_testsets, no_epochs, single_head, x_coresets, y_coresets, batch_size, just_vanilla =False)

        all_acc = test.concatenate_results(acc, all_acc)

        mf_model.load_weights()
        mf_model.clean_copy_weights()

        if not single_head:
            mf_model.create_head()

    return all_acc

def print_graph(model, output):
    params = dict()
    for i in range(len(model.W_m)):
        params["W_m{}".format(i)] = model.W_m[i]
        params["W_v{}".format(i)] = model.W_v[i]
        params["b_m{}".format(i)] = model.b_m[i]
        params["b_v{}".format(i)] = model.b_v[i]
        params["prior_W_m".format(i)] = model.prior_W_m[i]
        params["prior_W_v".format(i)] = model.prior_W_v[i]
        params["prior_b_m".format(i)] = model.prior_b_m[i]
        params["prior_b_v".format(i)] = model.prior_b_v[i]

    for i in range(len(model.W_last_m)):
         params["W_last_m".format(i)] = model.W_last_m[i]
         params["W_last_v".format(i)] = model.W_last_v[i]
         params["b_last_m".format(i)] = model.b_last_m[i]
         params["b_last_v".format(i)] = model.b_last_v[i]
         params["prior_W_last_m".format(i)] = model.prior_W_last_m[i]
         params["prior_W_last_v".format(i)] = model.prior_W_last_v[i]
         params["prior_b_last_m".format(i)] = model.prior_b_last_m[i]
         params["prior_b_last_v".format(i)] = model.prior_b_last_v[i]
    dot = make_dot(output, params=params)
    dot.view()

    return