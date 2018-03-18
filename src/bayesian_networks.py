import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import os
from pomegranate import BayesianNetwork
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import networkx as nx # 2.1 !
from sklearn.metrics import accuracy_score

import re
def stringSplitByNumbers(x):
    '''
    from comment here
    http://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    '''
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

def vec(A):
    return A.flatten(order='F')

def reshape(A, shape):
    return np.reshape(A, shape, order='F')

def checkData(data, prefix):
    p = data.columns.values
    rv = filter(lambda x: x.startswith(prefix), p)
    return rv

def loadMatrix(filename, classes=None, one_node=True, offset=100, ignore_negative=1):
    idColumnName = 'identity'
    df = np.load(filename)
    X, labels = df['data'], df['label']
    # ignore negative
    if ignore_negative:
        labels = np.array(map(int, labels))
        ind = labels >= 0
        X = X[ind]
        labels = labels[ind]
    # matrix case
    Nsamples, Nmz, Npol = X.shape
    X = reshape(X, [Nsamples, -1])
    S = [] 
    for k in xrange(Nmz*Npol):
        if k < Nmz:
            num = offset + k
            S.append('+P' + str(num))
        else:
            num = k - Nmz + offset
            S.append('-P' + str(num))
    labels = reshape(labels, [-1, 1])
    X = np.hstack([labels, X])
    S = [idColumnName] + S
    data = pd.DataFrame(X, columns=S)
    data[[idColumnName]].astype(int, inplace=True)
    if not one_node:
        n_classes = len(np.unique(labels))
        data = pd.get_dummies(data, columns=[idColumnName], drop_first=0)
        colnames = list(data_dummy.columns.values)
        data = data[colnames[-n_classes:] + colnames[:-n_classes]]
    return data

def thresholdMatrix(data, left_fraction=30, one_node=1, cut_const_cols=1):
    assert one_node, "Unsupported: multiple roots"
    assert 0 < left_fraction < 100
    ind = 1
    X = data.iloc[:, ind:].values
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    perc = 100 - left_fraction
    tau = np.percentile(X, perc, axis=1, keepdims=1)
    mask = X > tau
    X *= mask
    #data.iloc[:, ind:] = X
    data.iloc[:, ind:] = mask+0
    #data = data.loc[:, data.any()] # remove only zeo columns
    if cut_const_cols:
        data = data.loc[:, (data != data.iloc[0]).any()] # remove all constant columns
    #data[[idColumnName]].astype(int, inplace=True)
    data.astype(int, inplace=True)
    return data, tau



def produceModelsForValidationToJSON(data, train_indices, dirname='./', filename_base='model_bn_'):
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    state_names = data.columns.values
    model_estimating_times = []
    model_fitting_time = []
    index = 0
    for train_index in train_indices:
        X_train = X[train_index, :].copy()
        y_train = y[train_index].copy()
        y_train = reshape(y_train, [-1, 1])
        X_train = np.hstack([y_train, X_train])
        dummy = np.ones([2, X_train.shape[1]])
        dummy[:, 0] = -1 ### all 
        dummy[1, 1:] = 0
        X_train = np.vstack([X_train, dummy])
        X_train = X_train.astype(int)
        #Learning structure
        print "Learning..."
        tic = time.time(); tic2 = time.clock()
        model = BayesianNetwork.from_samples(
            X_train, root=0, state_names=state_names, algorithm='chow-liu', n_jobs=8
        )
        toc2 = time.clock(); toc = time.time()
        model_estimating_times.append([toc2-tic2, toc-tic])
        print 'Model estimated in %.5f clock, %.5f time' % (toc2-tic2, toc-tic)
        tic = time.time(); tic2 = time.clock()
        model.fit(X_train, pseudocount=1, verbose=True)
        toc2 = time.clock(); toc = time.time()
        model_fitting_time.append([toc2-tic2, toc-tic])
        print 'Model fitted in %.5f clock, %.5f time' % (toc2-tic2, toc-tic)
        #model.bake()
        #print 'Model was baked'
        string = model.to_json()
        model_filename = dirname+filename_base + str(index) + '.json'
        with open(model_filename,'w+') as f:
            f.writelines(string)
        index += 1
        np.savez_compressed(
            dirname+filename_base+'times', model_estimating_times=model_estimating_times,
            model_fitting_time=model_fitting_time
        )


    
if __name__ == '__main__':
    pass

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
