import numpy as np
#import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def filterLabels(filename, dirname='', min_count=20, save_filename=None, return_result=0):
    df = np.load(dirname+filename)
    X = df['data']
    y = df['label']
    y = np.array(map(int, y))
    ind = y > 0
    X, y = X[ind], y[ind]
    labels, counts = np.unique(y, return_counts=1)
    ind = counts < min_count
    for value in labels[ind]:
        y[y == value] = -1
    ind = y >= -1
    X, y = X[ind], y[ind]
    if save_filename is not None:
        np.savez_compressed(dirname+save_filename, data=X, label=y)
    if return_result:
        return X, y



def concatenateSeparateToOneDF(filenames, dirname, save_filename=None, return_result=0):
    X, y = [], []
    for k in xrange(len(filenames)):
        filename = filenames[k]
        df = np.load(dirname+filename)
        X.append( df['data'] )
        y.append( df['label'] )
    X = np.vstack(X)
    y = np.hstack(y)
    if save_filename is not None:
        np.savez_compressed(dirname+save_filename, data=X, label=y)
    if return_result:
        return X, y

def generateRandomizedKFoldedSet(
    X, y, n_splits=5, n_repeats=10, random_state=235, filename='kfold_partition',
    dirname='', return_result=0
):
    '''
    # generate indices for training/validation
    # Repeated randomized K-fold with K
    # (stratified - to take into account unbalanced number of representatives per class)
    '''
    kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    train_indices = []
    test_indices = []
    for train_index, test_index in kfold.split(X, y):
        train_indices.append(train_index)
        test_indices.append(test_index)
    if filename is not None:
        np.savez_compressed(
            dirname+filename, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
            train_indices=train_indices, test_indices=test_indices
        )
    if return_result:
        return train_indices, test_indices

if __name__ == '__main__':
    pass

