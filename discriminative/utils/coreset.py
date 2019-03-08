####Code by  nvcuong ###
#
#
# Nothing to change here to make it work on pytorch

import sklearn.decomposition as decomp
import numpy as np

""" Random coreset selection """
def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    idx = np.random.choice(x_train.shape[0], coreset_size, False)
    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train    

""" K-center coreset selection """
def k_center(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    dists = np.full(x_train.shape[0], np.inf)
    current_id = 0
    dists = update_distance(dists, x_train, current_id)
    idx = [ current_id ]

    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train, current_id)
        idx.append(current_id)

    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)

    return x_coreset, y_coreset, x_train, y_train


""" K-center performed on reduced data by pca coreset selection """
def pca_k_center(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    pca = decomp.PCA(20)
    pca.fit(x_train)
    x_train_reduced = pca.transform(x_train)
    dists = np.full(x_train_reduced.shape[0], np.inf)
    current_id = 0
    dists = update_distance(dists, x_train_reduced, current_id)
    idx = [ current_id ]

    for i in range(1, coreset_size):
        current_id = np.argmax(dists)
        dists = update_distance(dists, x_train_reduced, current_id)
        idx.append(current_id)

    x_coreset.append(x_train[idx,:])
    y_coreset.append(y_train[idx])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)

    return x_coreset, y_coreset, x_train, y_train



def attention_like_coreset(x_coreset, y_coreset, x_train, y_train, coreset_size):
    """TODO: learning a subnetwork that chooses the coreset (attention-like) """
    return x_coreset, y_coreset, x_train, y_train


def uncertainty_like_coreset(x_coreset, y_coreset, x_train, y_train, coreset_size):
    """TODO: Keeping only the instances that were not classified w.h. certainty """
    return x_coreset, y_coreset, x_train, y_train


def update_distance(dists, x_train, current_id):
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists
