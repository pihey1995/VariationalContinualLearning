import numpy as np
import matplotlib
matplotlib.use('agg')
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.hstack((merged_y, y_coresets[i]))
    return merged_x, merged_y


def get_coreset(x_coresets, y_coresets, single_head, coreset_size = 5000, gans = None, task_id=0):
    if gans is not None:
        if single_head:
            merged_x, merged_y = gans[0].generate_samples(coreset_size, task_id)
            for i in range(1, len(gans)):
                new_x, new_y = gans[i].generate_samples(coreset_size, task_id)
                merged_x = np.vstack((merged_x,new_x))
                merged_y = np.hstack((merged_y,new_y))
            return merged_x, merged_y
        else:
            return gans.generate_samples(coreset_size, task_id)[:coreset_size]
    else:
        if single_head:
            return merge_coresets(x_coresets, y_coresets)
        else:
            return x_coresets, y_coresets


def get_scores(model, x_testsets, y_testsets, no_epochs, single_head,  x_coresets, y_coresets, batch_size=None, just_vanilla = False, gans = None):

    acc = []
    if single_head:
        if len(x_coresets) > 0 or gans is not None:
            x_train, y_train = get_coreset(x_coresets, y_coresets, single_head, coreset_size = 6000, gans = gans, task_id=0)

            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            x_train = torch.Tensor(x_train)
            y_train = torch.Tensor(y_train)
            model.train(x_train, y_train, 0, no_epochs, bsize)

    for i in range(len(x_testsets)):
        if not single_head:
            if len(x_coresets)>0 or gans is not None:
                model.load_weights()
                gan_i = None
                if gans is not None:
                    gan_i = gans[i]
                    x_train, y_train = get_coreset(None, None, single_head, coreset_size = 6000, gans= gan_i, task_id=i)
                else:
                    x_train, y_train = get_coreset(x_coresets[i], y_coresets[i], single_head, coreset_size = 6000, gans= None, task_id=i)
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