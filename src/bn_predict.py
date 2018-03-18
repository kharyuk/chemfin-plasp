import multiprocessing
import time
import operator
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import os
from pomegranate import BayesianNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from bayesian_networks import loadMatrix, thresholdMatrix

import re
def stringSplitByNumbers(x):
    '''
    from comment here
    http://code.activestate.com/recipes/135435-sort-a-string-using-numeric-order/
    '''
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

_GLOBAL_filename_base = '_predicted_test.npz'

def vec(A):
    return A.flatten(order='F')

def reshape(A, shape):
    return np.reshape(A, shape, order='F')

def gatherResults(numOfPartition, save_dirname, postfix='', filename_base=_GLOBAL_filename_base):
    fnms = os.listdir(save_dirname)
    fnms = filter(lambda x: x.endswith(filename_base), fnms)
    Njobs = len(fnms)
    labels_pred = []
    labels_true = []
    stats = []
    times = np.empty([1, 0])
    for k in xrange(Njobs):
        filename = str(k) + filename_base
        df = np.load(save_dirname+filename)
        stats += df['stats'].tolist()
        times = np.hstack([times, reshape(df['times'], [1, -1])])
        labels_pred += df['y_pred'].tolist()
        labels_true += df['y_part'].tolist()
        os.remove(save_dirname+filename)

    predictionTable = pd.DataFrame(stats)
    #nrows = len(predictionTable)
    predictionTable['TRUE LABELS'] = labels_true#[:nrows]
    print "Accuracy: %.4f" % (accuracy_score(labels_true, labels_pred))
    filename = save_dirname+'validation_results_on_model_' + str(numOfPartition) + postfix
    np.savez_compressed(filename, table=predictionTable, times=times)
    predictionTable.to_csv(filename+'.csv')
    

def launcher(N_jobs, model_filename, X_test, y_test, model_dirname='', save_dirname=''):
    model = BayesianNetwork.from_json(model_dirname+model_filename)
    model.freeze()
    jobs = []
    Nsamples_test = y_test.size
    Nsamples_per_process = Nsamples_test / N_jobs
    Number_of_hard_workers = Nsamples_test % N_jobs
    ind = 0
    for i in xrange(N_jobs):
        offset = Nsamples_per_process
        if i < Number_of_hard_workers:
            offset += 1
        X_part = X_test[ind:ind+offset, :].copy()
        y_part = y_test[ind:ind+offset].copy()
        if len(y_part) == 0:
            break
        p = multiprocessing.Process(target=worker, args=(i, model, X_part, y_part, save_dirname, 1))
        jobs.append(p)
        p.start()
        print "process %d with %d samples (%d-%d)" % (i+1, y_part.size, ind+1, ind+y_part.size)
        ind += offset
    for p in jobs:
        p.join()
    print "========================================================"
    print "Launcher has successfully finished his work"

def worker(numThread, model, X_part, y_part, save_dirname='', verbose=0, filename_base=_GLOBAL_filename_base):
    Nsamp_test = y_part.size
    stats = []
    y_pred = []
    accuracy = []
    times = []
    for k in xrange(Nsamp_test):
        tic = time.time()
        y = model.predict_proba(X_part[k, :])
        toc = time.time()
        times.append(toc-tic)
        result_k = y[0].parameters[0]
        stats.append(result_k)
        label_k = max(result_k.iteritems(), key=operator.itemgetter(1))[0]
        y_pred.append( label_k )
        accuracy.append( y_pred[k] == y_part[k] )
        if verbose:
            percentage = 100.*float(k+1)/Nsamp_test
            current_accuracy = np.sum(accuracy) / float(len(accuracy))
            average_time = np.mean(times)
            print "Thread %d: %.2f%% , accuracy=%.5f, averaged time: %.3f s/sample" % (
                numThread, percentage, current_accuracy, average_time
            )
        np.savez_compressed(
            save_dirname+str(numThread)+filename_base, k=k, times=times, stats=stats, y_pred=y_pred, accuracy=accuracy, y_part=y_part
        )
    return

if __name__ == '__main__':
    pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
