import numpy as np
import copy
import os
import time
import torch
from torch import nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from computational_utils import reshape
from io_work import stringSplitByNumbers

random_state = 150
torch.manual_seed(random_state);

def getStats(model, A):
    X = Variable(torch.from_numpy(A.copy()))
    Y = model.encode(X)
    Z = model.decode(Y)
    rres_l2_all = (1./A.shape[0])*np.linalg.norm(Z.data.numpy() - A, 2) / np.linalg.norm(A, 2)
    stat = np.linalg.norm(Z.data.numpy() - A, 2, axis=1) / np.linalg.norm(A, 2, axis=1) 
    print 'min=%.3e / mean=%.3e / median=%.3e / max=%.3e' % (
        np.min(stat), np.mean(stat), np.median(stat), np.max(stat)
    )
    return rres_l2_all, stat

def checkRelRes(T, train_indices, test_indices, sizes_in, model_dirname, model_filename_prefix):
    N = T.shape[1]
    sizes = [N] + sizes_in
    
    nls = [nn.ReLU()]+[nn.Sigmoid()]*(len(sizes)-1)
    optimizer = lambda params: torch.optim.Adam(params)
    
    model_fname_list = os.listdir(model_dirname)
    model_fname_list = filter(lambda x: x.startswith(model_filename_prefix), model_fname_list)
    model_fname_list = sorted(model_fname_list, key=stringSplitByNumbers)
    for k in xrange(len(model_fname_list)):
        model_fname = model_fname_list[k]
        print '========================'
        print model_fname
        autoencoder = AutoEncoder(sizes, nls, optimizer=optimizer)
        autoencoder.load_state_dict(torch.load(model_dirname+model_fname))
        print "Training set:"
        getStats(autoencoder, T[train_indices[k]])
        print "Validation set:"
        getStats(autoencoder, T[test_indices[k]])

def encodeDataset(df, sizes_in, model_dirname, model_filename_prefix,
                  save_filename=None, num_workers=8, return_result=0):
    resultsTime = []
    # optimizer parameters
    learning_rate = 0.0025
    betas = (0.9, 0.999)
    eps = 1e-5
    optimizer = lambda params: torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)
    
    T, y = df['data'], df['label']
    # unfold into matrix
    T = reshape(T, [T.shape[0], -1])
    # normalize among samples
    T /= np.linalg.norm(T, axis=1, keepdims=1)
    
    N = T.shape[1]
    sizes = [N] + sizes_in
    
    nls = [nn.ReLU()]+[nn.Sigmoid()]*(len(sizes)-1)
    
    model_fname_list = os.listdir(model_dirname)
    model_fname_list = filter(lambda x: x.startswith(model_filename_prefix), model_fname_list)
    model_fname_list = sorted(model_fname_list, key=stringSplitByNumbers)
    Z = []
    for k in xrange(len(model_fname_list)):
        model_fname = model_fname_list[k]
        autoencoder = AutoEncoder(sizes, nls, optimizer=optimizer, loss=nn.SmoothL1Loss)
        autoencoder.load_state_dict(torch.load(model_dirname+model_fname))
        
        X = Variable(torch.from_numpy(T.copy()))
        tic = time.clock()
        Y = autoencoder.encode(X)
        toc = time.clock()
        X = Y.data.numpy()
        Z.append(X.copy())
        resultsTime.append(toc-tic)
        print "encoded with %s. Comp.time=%.5f" % (model_fname, resultsTime[-1])
    if save_filename is not None:
        np.savez_compressed(save_filename, data=Z, label=y, time=resultsTime)
    if return_result:
        return X, y, resultsTime
        
class AutoEncoder(nn.Module):
    def __init__(self, sizes=None, nls=None, optimizer=None, loss=nn.MSELoss, use_gpu=False, random_state=None):
        super(AutoEncoder, self).__init__()
        assert isinstance(sizes, (list, tuple, np.ndarray))
        assert isinstance(nls, (list, tuple))
        assert len(nls) == len(sizes)
        assert optimizer is not None
        
        self.random_state = random_state
        if random_state is not None:
            torch.manual_seed(random_state) # making reulsts reproducable
        
        encoderParameters = []
        for k in xrange(len(nls)-1):
            encoderParameters.append(nn.Linear(sizes[k], sizes[k+1]))
            if nls[k+1] is not None:
                encoderParameters.append(nls[k+1])
        encoderParameters = tuple(encoderParameters)
        self.Encoder = nn.Sequential(*encoderParameters)
        decoderParameters = []
        for k in xrange(len(nls)-1, -1, -1):
            if k > 0:
                decoderParameters.append(nn.Linear(sizes[k], sizes[k-1]))
            if nls[k] is not None:
                decoderParameters.append(nls[k])
        decoderParameters = tuple(decoderParameters)
        self.Decoder = nn.Sequential(*decoderParameters)
        self.loss = loss
        self.use_gpu = use_gpu
        self.optimizer = optimizer
        self.double()
        
    def encode(self, X):
        return self.Encoder(X)
    
    def decode(self, Y):
        return self.Decoder(Y)
    
    def forward(self, X):
        Y = self.encode(X)
        Z = self.decode(Y)
        return Z
    
    def fit(self, X, nEpoch=100, delta=1e-5, verbose=0):
        if self.use_gpu:
            self.gpu()
        lossFunction = self.loss(size_average=False)
        optimizer = self.optimizer(self.parameters())
        loss_values = []
        for epoch in xrange(nEpoch):
            for partX in X:
                var = Variable(partX)
                var.cpu()
                if self.use_gpu:
                    var.gpu()
                loss_val = [0.]
                def closure(loss_val=loss_val):
                    optimizer.zero_grad()
                    output = self(var)
                    loss = lossFunction(output, var)
                    loss.backward()
                    loss_val[0] = (loss.data.numpy())[0]
                    return loss
                optimizer.step(closure)
            #print loss.data
            if verbose:
                print "Epoch: %d/%d \t Loss: %.5f" % (epoch+1, nEpoch, loss_val[0])
            loss_values += loss_val
            #torch.save(self.state_dict(), filename)
            #if loss.data[0] < delta:
            #    break
        return loss_values
            
        
def buildAutoencoderModels(T, train_indices, test_indices, sizes_in, dirname='./', nEpoch=100, batch_size=200, num_workers=8,
                model_filename_prefix='testApproach'):
    # optimizer parameters
    
    learning_rate = 0.0025
    betas = (0.9, 0.999)
    eps = 1e-5
    optimizer = lambda params: torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps) # Sparse
    N = T.shape[1]
    # AE structure + instance
    nls = [nn.ReLU()]+[nn.Sigmoid()]*len(sizes_in)
    sizes = [N] + sizes_in

    # Ntrain/Ntest, Nlevels, l1/l2
    train_stats_integral = np.zeros([len(train_indices), len(sizes)-1])
    test_stats_integral = np.zeros([len(train_indices), len(sizes)-1])
    # N samples, Nlevels, l1/l2
    sample_stats = np.zeros([len(train_indices), len(sizes)-1, T.shape[0]])
    # Ntrain, Nlevels, nEpoch
    loss_values = np.zeros([len(train_indices), len(sizes)-1, max(nEpoch)])
    times = np.zeros([len(train_indices), len(sizes)-1, 2])
    for k_cv in xrange(len(train_indices)):
        train_index, test_index = train_indices[k_cv], test_indices[k_cv]
        dataset = torch.from_numpy(T[train_index].copy())
        data = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        print "============= CV %d / %d ================" % (k_cv+1, len(train_indices))
        for k in xrange(len(sizes)-1):
            autoencoder = AutoEncoder(sizes=sizes[:k+2], nls=nls[:k+2], optimizer=optimizer, loss=nn.SmoothL1Loss)
            if k > 0:
                for i in xrange(k):
                    autoencoder.Encoder[2*i].weight = weights[2*i]
                    autoencoder.Encoder[2*i].bias = biases[2*i]
                    autoencoder.Decoder[2*(i+1)].weight = weights[2*i+1]
                    autoencoder.Decoder[2*(i+1)].bias = biases[2*i+1]
            t1_time = time.time(); t1_clock = time.clock()
            loss_values_level = autoencoder.fit(data, nEpoch[k], verbose=0)
            t2_clock = time.clock(); t2_time = time.time()
            times[k_cv, k, 0] = t2_time-t1_time
            times[k_cv, k, 1] = t2_clock-t1_clock
            torch.save(autoencoder.state_dict(), dirname+model_filename_prefix+str(k_cv))
            print '(%d) Errors on training set (%d samples): ' % (k+1, T[train_index].shape[0])
            train_stats_level = getStats(autoencoder, T[train_index])
            print '(%d) Errors on validation set (%d samples): ' % (k+1, T[test_index].shape[0])
            test_stats_level = getStats(autoencoder, T[test_index])


            train_stats_integral[k_cv, k] = train_stats_level[0]
            sample_stats[k_cv, k, train_index] = train_stats_level[1]
            test_stats_integral[k_cv, k] = test_stats_level[0]
            sample_stats[k_cv, k, test_index] = test_stats_level[1]
            loss_values[k_cv, k, :nEpoch[k]] = loss_values_level
            np.savez_compressed(
                dirname+'ae_cv', train_stats_integral=train_stats_integral,
                test_stats_integral=test_stats_integral, loss_values=loss_values,
                test_indices=test_indices, train_indices=train_indices,
                sample_stats=sample_stats, nEpoch=nEpoch, times=times
            )
            # test on train/valid. sets
            if k < (len(sizes)-2):
                weights, biases = [], []
                for i in xrange(k+1):
                    weights.append( copy.deepcopy(autoencoder.Encoder[2*i].weight) )
                    weights.append( copy.deepcopy(autoencoder.Decoder[2*i].weight) )
                    biases.append( copy.deepcopy(autoencoder.Encoder[2*i].bias) )
                    biases.append( copy.deepcopy(autoencoder.Decoder[2*i].bias) )

def investigateLastLayerTrain(T, train_indices, test_indices, sizes_in, sizes_ll,
        model_dirname, nEpoch, batch_size, num_workers, model_filename_prefix):

    buildAutoencoderModels(
        T, train_indices, test_indices, sizes_in[:-1], model_dirname, nEpoch[:-1],
        batch_size, num_workers, model_filename_prefix
    )

    
    
    model_fname_list = os.listdir(model_dirname)
    model_fname_list = filter(lambda x: x.startswith(model_filename_prefix), model_fname_list)
    model_fname_list = sorted(model_fname_list, key=stringSplitByNumbers)
    tms = []
    for k in xrange(len(model_fname_list)):
        train_index = train_indices[k]
        test_index = test_indices[k]
        model_fname = model_fname_list[k]
        resultsTime = []
        # optimizer parameters
        learning_rate = 0.0025
        betas = (0.9, 0.999)
        eps = 1e-5
        optimizer = lambda params: torch.optim.Adam(params, lr=learning_rate, betas=betas, eps=eps)
        N = T.shape[1]
        sizes = [N] + sizes_in[:-1]
        nls = [nn.ReLU()]+[nn.Sigmoid()]*(len(sizes))
        autoencoder_base = AutoEncoder(sizes, nls[:-1], optimizer=optimizer, loss=nn.SmoothL1Loss)
        autoencoder_base.load_state_dict(torch.load(model_dirname+model_fname))
        for k in xrange(len(sizes_ll)):
            print '========= %d / %d =========' % (k+1, len(sizes_ll))
            autoencoder = AutoEncoder(sizes+[sizes_ll[k]], nls, optimizer=optimizer, loss=nn.SmoothL1Loss)
            for i in xrange(len(sizes)-1):
                autoencoder.Encoder[2*i].weight = copy.deepcopy(autoencoder_base.Encoder[2*i].weight)
                autoencoder.Encoder[2*i].bias = copy.deepcopy(autoencoder_base.Encoder[2*i].bias)
                autoencoder.Decoder[2*(i+1)].weight = copy.deepcopy(autoencoder_base.Decoder[2*i].weight)
                autoencoder.Decoder[2*(i+1)].bias = copy.deepcopy(autoencoder_base.Decoder[2*i].bias)

            dataset = torch.from_numpy(T[train_index].copy())
            data = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            tic=time.time(); tic2=time.clock();
            autoencoder.fit(data, nEpoch[-1], verbose=0)
            toc2=time.clock(); toc=time.time();
            cus_pref = 'll=%d_' % (sizes_ll[k])
            torch.save(autoencoder.state_dict(), model_dirname+cus_pref+model_fname)
            resultsTime.append([toc2-tic2, toc-tic])
            np.savez_compressed(model_dirname+cus_pref+'times', tms=tms)
        tms.append(resultsTime)
                    
                  

if __name__ == '__main__':
    pass
