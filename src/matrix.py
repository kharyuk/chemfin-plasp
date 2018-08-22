import numpy as np
import pandas as pd
import copy
import time

from sklearn.decomposition import NMF

from cca import cca, corr

def vec(x):
    return x.flatten(order='F')

def reshape(a, shape):
    return np.reshape(a, shape, order = 'F')

def fast_svd(x, R=None, tol=None, scale=1.5):
    [m, n] = x.shape
    if m >= scale*n:
        tmp = np.dot(x.T, x)
        _, s, vt = np.linalg.svd(tmp)
        if tol is not None:
            csumdiff = np.diff(_np.cumsum(s))
            Rtol = csumdiff[csumdiff >= tol].sum()
        if R is not None:
            Rs = R
            if tol is not None:
                Rs = min(Rtol, Rs)
            s = s[:Rs]
            vt = vt[:Rs, :]
        s = s**0.5
        u = np.dot(x, (vt.T / s))
    elif n >= scale*m:
        tmp = np.dot(x, x.T)
        u, s, _ = np.linalg.svd(tmp)
        if tol is not None:
            csumdiff = _np.diff(np.cumsum(s))
            Rtol = csumdiff[csumdiff >= tol].sum()
        if R is not None:
            Rs = R
            if tol is not None:
                Rs = min(Rtol, Rs)
            s = s[:Rs]
            u = u[:, :Rs]
        s = s**0.5
        vt = np.dot(u.T, x).T / s
    else:
        u, s, vt = np.linalg.svd(x)
        if tol is not None:
            csumdiff = np.diff(np.cumsum(s**2.))
            Rtol = csumdiff[csumdiff >= tol].sum()
        if R is not None:
            Rs = R
            if tol is not None:
                Rs = min(Rtol, Rs)
            u = u[:, :Rs]
            s = s[:Rs]
            vt = vt[:Rs, :]
    return u, s, vt
    
    
def functional(T, A, B, Lambda, AtA, BtB, separate=False):
    # NNLS part
    normT = np.linalg.norm(T)
    d = T.ndim
    # assumption: n_rows > n_cols (n_samples > n_features)
    partABAB = np.dot(B, AtA)
    partABAB = np.einsum('ij,ij', partABAB, B)
    partABT = np.dot(A.T, T).T
    partABT = np.einsum('ij,ij', partABT, B)
    rv = 0.5*(normT**2. + partABAB - 2*partABT)
    # sparse part
    nrms1 = [
        np.linalg.norm(vec(A), 1),
        np.linalg.norm(vec(B), 1)
    ]
    if separate:
        return rv, np.array(nrms1)*Lambda
    rv += np.sum(np.array(nrms1)*Lambda)
    return rv

def update_L_factor(CtC):
    _, s, _ = np.linalg.svd(CtC)
    L = s[0]
    L = np.maximum(L, 1.)
    return L

def update_t(t):
    t2 = 0.5*(1+np.sqrt(1 + 4*np.power(t, 2)))
    return t2

def update_w(L, Lnew, t, tnew, alpha=0.9999):
    what = (t - 1) / tnew
    w = np.minimum(what, alpha*np.sqrt(L/Lnew))
    return w

def gradient_factor_A(T, A, B, BtB):    
    rv = np.dot(A, BtB)
    rv -= np.dot(T, B)
    return rv

def gradient_factor_B(T, A, B, AtA):    
    rv = np.dot(B, AtA)
    rv -= np.dot(T.T, A)
    return rv    

def proximalAG(T, r, Lambda, Gamma, maxitnum=100, margin=1e-3, alpha=0.9999, verbose=1, tol=1e-5, init='rand'):
    d = T.ndim # 2
    assert d == 2, 'current function is designed for matrix decomposition'
    n = T.shape
    
    L = np.ones(d)
    infT = np.abs(T).max()
    eps = np.spacing(1.)
    if init == 'svd':
        u, s, v = fast_svd(T)
        if verbose:
            print 'mode %d, svd_rank: %d, claimed rank: %d' % (k+1, u.shape[1], r)
        A = u[:, :r]*np.sqrt(s[:r])
        B = (v[:, :r])*np.sqrt(s[:r])
    else:
        A = np.random.uniform(0, 1, size=[n[0], r]) 
        A /= np.linalg.norm(A)
        B = np.random.uniform(0, 1, size=[n[1], r]) 
        B /= np.linalg.norm(B)
        
    AtA = np.dot(A.T, A)
    BtB = np.dot(B.T, B)
    
    Aold, Bold = None, None
    
    t, tnew = np.ones(d), np.ones(d)
    L, Lnew = np.ones(d), np.ones(d)
    w, wnew = np.ones(d), np.ones(d)
    flist = [functional(T, A, B, Lambda, AtA, BtB)]
    rl = functional(T, A, B, Lambda, AtA, BtB, separate=True)
    rlist = [rl[0]]
    for itnum in xrange(maxitnum):
        # factor-matrix A 
        Lnew[0] = update_L_factor(BtB) # A excluded
        tnew[0] = update_t(t[0])
        wnew[0] = update_w(L[0], Lnew[0], t[0], tnew[0], alpha)
        if Aold is not None:
            Anew = (1 + wnew[0])*A - Aold
        else:
            Anew = A.copy()
        Anew -= (gradient_factor_A(T, A, B, BtB) + Lambda[0]) / Lnew[0]
        if Gamma[0]:
            Anew = np.maximum(margin, Anew)
        AtA_new = np.dot(Anew.T, Anew)
        # factor-matrix B
        Lnew[1] = update_L_factor(AtA_new) # B excluded
        tnew[1] = update_t(t[1])
        wnew[1] = update_w(L[1], Lnew[1], t[1], tnew[1], alpha)
        if Bold is not None:
            Bnew = (1 + wnew[1])*B - Bold
        else:
            Bnew = B.copy()
        Bnew -= (gradient_factor_B(T, Anew, B, AtA_new) + Lambda[1]) / Lnew[1]
        if Gamma[1]:
            Bnew = np.maximum(margin, Bnew)
        BtB_new = np.dot(Bnew.T, Bnew)

        funval = functional(T, Anew, Bnew, Lambda, AtA_new, BtB_new)

        if funval > flist[-1]:
            Anew = (gradient_factor_A(T, A, B, BtB) + Lambda[0]) / Lnew[0]
            if Gamma[0]:
                Anew = np.maximum(margin, Anew)
            AtA_new = np.dot(Anew.T, Anew)

            Bnew = (gradient_factor_B(T, Anew, B, AtA_new) + Lambda[1]) / Lnew[1]
            if Gamma[1]:
                Bnew = np.maximum(margin, Bnew)
            BtB_new = np.dot(Bnew.T, Bnew)
            funval = functional(T, Anew, Bnew, Lambda, AtA_new, BtB_new)
        AtA = AtA_new.copy()
        BtB = BtB_new.copy()
        Aold = A.copy()
        Bold = B.copy()
        A = Anew.copy()
        B = Bnew.copy()
        flist.append(funval)
        t[0], L[0] = tnew[0], Lnew[0]
        t[1], L[1] = tnew[1], Lnew[1]
        rl = functional(T, A, B, Lambda, AtA, BtB, separate=True)
        rlist.append(rl[0])
        if verbose:
            print "itnum %d\t funval:%.5f" % (itnum+1, flist[-1]) 
        if np.abs(flist[-1] - flist[-2]) < tol:
            break
    return A, B, flist

def estimateMzPolarityFactors_old(X, y, r=1, classes=None, Lambda=None, Gamma=None, maxitnum=1000, init='svd', dirname='', prefix='', savedata=1, verbose=0):
    T = X.copy()
    n = T.shape
    d = len(n) 
    assert d == 2, "this is a function for matrix-related problem"
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    
    if Lambda is None:
        Lambda = 0.00003*np.ones(d)
    if Gamma is None:
        Gamma = np.ones(d)
    if classes is None:
        classes = np.unique(y)
    tucks = []
    times = []
    for ind in xrange(len(classes)):
        which = np.where(y == classes[ind])[0]
        y_local = y[which].copy()
        Z = T[which].copy()
        normZ = np.linalg.norm(Z)
        Z /= normZ
        
        t1 = time.clock()
        A, B, flist = proximalAG(
            Z, r, Lambda, Gamma, margin=0., alpha=0.9999, maxitnum=maxitnum, init=init,
            verbose=verbose
        )
        t2 = time.clock()
        Zhat = np.dot(A, B.T)
        stats = []
        for k in xrange(Z.shape[0]):
            stats.append( np.linalg.norm(Z[k] - Zhat[k]) / np.linalg.norm(Z[k]) )
        stats = np.array(stats)
        if verbose:
            print "Class: %d  Time=%.2fs RRes: min=%.2e mean=%.2e med=%.2e max=%.2e" % (
                classes[ind], t2-t1, np.min(stats), np.mean(stats), np.median(stats), np.max(stats)
            )
        times.append(t2-t1)
        tucks.append(copy.deepcopy(B))
        if savedata:
            filename = prefix + 'matrix_decomposition_rank_%d' % (r)
            np.savez_compressed(dirname+filename, tucks=tucks, errors=stats, classes=classes, times=times)
    return tucks, times, stats

def estimateMzPolarityFactors(X, y, r=1, classes=None, Lambda=None, maxitnum=1000, init='svd', dirname='', prefix='', savedata=1, verbose=0):
    T = X.copy()
    n = T.shape
    d = len(n) 
    assert d == 2, "this is a function for matrix-related problem"
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    
    Lambda = 0.00003
    if classes is None:
        classes = np.unique(y)
    tucks = []
    times = []
    for ind in xrange(len(classes)):
        which = np.where(y == classes[ind])[0]
        y_local = y[which].copy()
        Z = T[which].copy()
        normZ = np.linalg.norm(Z)
        Z /= normZ
        
        transform = NMF(
            n_components=min(r, len(which)), init=None, solver='cd', beta_loss='frobenius',
            tol=0.0001, max_iter=maxitnum, random_state=None, alpha=Lambda, l1_ratio=0.0,
            verbose=0, shuffle=False
        )
        
        t1 = time.clock()
        A = transform.fit_transform(Z)
        t2 = time.clock()
        B = transform.components_.T
        Zhat = np.dot(A, B.T)
        stats = []
        for k in xrange(Z.shape[0]):
            stats.append( np.linalg.norm(Z[k] - Zhat[k]) / np.linalg.norm(Z[k]) )
        stats = np.array(stats)
        if verbose:
            print "Class: %d  Time=%.2fs RRes: min=%.2e mean=%.2e med=%.2e max=%.2e" % (
                classes[ind], t2-t1, np.min(stats), np.mean(stats), np.median(stats), np.max(stats)
            )
        times.append(t2-t1)
        tucks.append(copy.deepcopy(B))
        if savedata:
            filename = prefix + 'matrix_decomposition_rank_%d' % (r)
            np.savez_compressed(dirname+filename, tucks=tucks, errors=stats, classes=classes, times=times)
    return tucks, times, stats

class MatrixClassifierLCMS():
    
    def __init__(self, Nmz, rank, Lambda=None, Gamma=None, maxitnum=1000, init='svd'):
        self.Lambda = Lambda
        self.Gamma = Gamma
        self.maxitnum = maxitnum
        self.init = init
        self.Nmz = Nmz
        self.rank = rank
        
    def saveParameters(self, filename):
        try:
            np.savez_compressed(
                filename, classes=self.classes, FeatureSpaces=self.FeatureSpaces
            )
        except:
            print "No parameters to save"
    
    def loadParameters(self, filename):
        try:
            df = np.load(filename)
            self.classes = df['classes']
            self.FeatureSpaces = df['FeatureSpaces'].item()
        except:
            print "No parameters to load"
        
        
    def fit(self, X, y, verbose=0):
        T = X.copy()
        T /= np.linalg.norm(T, axis=1, keepdims=1)
        self.classes = np.unique(y)
        B, times, stats = estimateMzPolarityFactors(
            X, y, r=self.rank, classes=self.classes, Lambda=self.Lambda,
            maxitnum=1000, init='svd', savedata=0,
            verbose=verbose
        )
        self.FeatureSpaces = {}
        for i in xrange(len(self.classes)):
            self.FeatureSpaces[self.classes[i]] = B[i]
        return
    
    def predict(self, X, return_all=0, metric=None):
        T = X.copy()
        T /= np.linalg.norm(T, axis=1, keepdims=1)
        resultTable = np.zeros([T.shape[0], len(self.classes)])
        for i in xrange(len(self.classes)):
            currentClass = self.classes[i]
            for k in xrange(T.shape[0]):
                if metric is None:
                    a, b = cca(T[k:k+1].T, self.FeatureSpaces[currentClass], N=1)
                    resultTable[k, i] = corr(a, b)[0, 1]
                else:
                    resultTable[k, i] = metric(T[k], self.FeatureSpaces[currentClass])
        ind = np.argmax(resultTable, axis=1)
        if return_all:
            data = pd.DataFrame(resultTable, columns=self.classes)
            return self.classes[ind], data
        return self.classes[ind]




if __name__ == '__main__':
    pass
