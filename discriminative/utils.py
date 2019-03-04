import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from multihead_models import MFVI_NN
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.hstack((merged_y, y_coresets[i]))
    return merged_x, merged_y

def get_scores(model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size=None, just_vanilla = False):

    acc = []
    if single_head:
        if len(x_coresets) > 0:
            #model.load_weights()
            x_train, y_train = merge_coresets(x_coresets, y_coresets)
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            x_train = torch.Tensor(x_train)
            y_train = torch.Tensor(y_train)
            model.train(x_train, y_train, 0, no_epochs, bsize)

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets) > 0:
                model.load_weights()
                x_train, y_train = x_coresets[i], y_coresets[i]
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                x_train = torch.Tensor(x_train)
                y_train = torch.Tensor(y_train)
                model.train(x_train, y_train, i, no_epochs, bsize)

        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        N = x_test.shape[0]
        bsize = N if (batch_size is None) else batch_size
        cur_acc = 0
        total_batch = int(np.ceil(N * 1.0 / bsize))
        # Loop over all batches
        for i in range(total_batch):
            start_ind = i*bsize
            end_ind = np.min([(i+1)*bsize, N])
            batch_x_test = torch.Tensor(x_test[start_ind:end_ind, :]).to(device = device)
            batch_y_test = torch.Tensor(y_test[start_ind:end_ind]).type(torch.LongTensor).to(device = device)
            pred = model.prediction_prob(batch_x_test, head)
            if not just_vanilla:
                pred_mean = pred.mean(0)
            else:
                pred_mean = pred
            pred_y = torch.argmax(pred_mean, dim=1)
            cur_acc += end_ind - start_ind-(pred_y - batch_y_test).nonzero().shape[0]

        cur_acc = float(cur_acc)
        cur_acc /= N
        acc.append(cur_acc)
        print("Accuracy is {}".format(cur_acc))
    return acc

def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score

def plot3(filename, vcl, rand_vcl, kcen_vcl):

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()
    plt.show()
    #fig.savefig(filename)
    #plt.close()
