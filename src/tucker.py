import numpy as np
import copy
import time

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

def unfold(x, mode, return_sigma=False, reverse=False):
    n = x.shape
    d = x.ndim
    if not reverse:
        sigma = [mode] + range(mode) + range(mode+1, d)
    else:
        sigma = [mode] + range(d-1, mode, -1) + range(mode-1, -1, -1)
    tmp = np.transpose(x, sigma)
    tmp = reshape(tmp, [n[mode], -1])
    if return_sigma:
        return tmp, sigma
    return tmp

def prodTenMat(T, M, mode_t, mode_m=1):
    assert M.ndim == 2, "Second operand must be a matrix"
    subT = range(T.ndim)
    subR = range(T.ndim)
    subR[mode_t] = T.ndim
    subM = [T.ndim, T.ndim]
    subM[mode_m] = subT[mode_t]
    result = np.einsum(T, subT, M, subM, subR)
    return result

def recover(A, G, exclude_modes=[]):
    d = len(A)
    assert G.ndim == d
    rv = G.copy()
    for k in xrange(d):
        if k in exclude_modes:
            continue
        rv = prodTenMat(rv, A[k], k)
    return rv
    
    
def functional(k, T, A, G, Lambda, AtA, exclude_modes, separate=False):
    # NNLS part
    normT = np.linalg.norm(T)
    d = T.ndim
    partTuck = G.copy()
    partTD = T.copy()
    for k in xrange(d):
        if k in exclude_modes:
            continue
        partTuck = prodTenMat(partTuck, AtA[k], k)
        partTD = prodTenMat(partTD, A[k].T, k)
    axes = range(d)
    rv = np.einsum(partTuck - 2*partTD, axes, G, axes)
    rv = 0.5*(normT**2. + rv)
    # sparse part
    tmp = set(range(d)).difference(set(exclude_modes))
    tmp = np.array(list(tmp))
    nrms1 = [np.linalg.norm(vec(G), 1)]
    nrms1 = [np.linalg.norm(vec(A[k]), 1) for k in tmp] + nrms1
    if separate:
        return rv, np.array(nrms1)[tmp]*Lambda[tmp]
    rv += np.sum(np.array(nrms1)[tmp]*Lambda[tmp])
    return rv

def update_L_factor(k, AtA, G, exclude_modes=[]):
    d = len(AtA)
    tmp = G.copy()
    for p in xrange(d):
        if (p == k) or (p in exclude_modes):
            continue
        tmp = prodTenMat(tmp, AtA[p], p)
    axes1 = range(d)
    axes1[k] = d+k
    axes2 = range(d)
    axes2[k] = d+k+1
    tmp = np.einsum(tmp, axes1, G, axes2)
    #L = np.linalg.norm(tmp, ntype)
    _, s, _ = np.linalg.svd(tmp)
    L = s[0]
    L = np.maximum(L, 1.)
    return L

def update_L_core(AtA, exclude_modes=[]):
    d = len(AtA)
    L = 1.
    for k in xrange(d):
        if k in exclude_modes:
            continue
        _, s, _ = np.linalg.svd(AtA[k])
        L *= s[0]
    L = np.maximum(L, 1.)
    return L

def update_t(t):
    t2 = 0.5*(1+np.sqrt(1 + 4*np.power(t, 2)))
    return t2

def update_w(L, Lnew, t, tnew, alpha=0.9999):
    what = (t - 1) / tnew
    w = np.minimum(what, alpha*np.sqrt(L/Lnew))
    return w

def gradient_core(T, A, G, AtA, exclude_modes=[]):
    d = T.ndim
    partTuck = G.copy()
    partTD = T.copy()
    for k in xrange(d):
        if k in exclude_modes:
            continue
        partTuck = prodTenMat(partTuck, AtA[k], k)
        partTD = prodTenMat(partTD, A[k].T, k)    
    return partTuck - partTD

def gradient_factor(k, T, A, G, AtA, exclude_modes=[]):
    d = T.ndim
    partTuck = G.copy()
    partTD = T.copy()
    for p in xrange(d):
        if (p == k) or (p in exclude_modes):
            continue
        partTuck = prodTenMat(partTuck, AtA[p], p)
        partTD = prodTenMat(partTD, A[p].T, p)    
    partTuck = prodTenMat(partTuck, A[k], k)
    axes1 = range(k) + [d] + range(k+1, d)
    axes2 = range(k) + [d+1] + range(k+1, d)
    rv = np.einsum(partTuck - partTD, axes1, G, axes2)
    return rv
    

    #https://arxiv.org/pdf/1302.2559.pdf
def proximalALS(T, r, Lambda, Gamma, maxitnum=100, margin=1e-3, alpha=0.9999, verbose=1, tol=1e-5, init='rand',
        exclude_modes=None):
    d = T.ndim
    n = T.shape
    
    L = np.ones(d+1)
    infT = np.abs(T).max()
    eps = np.spacing(1.)
    AtA = [[]]*d
    A = [[]]*d
    Aold = [[]]*d
    if exclude_modes is None:
        exclude_modes = []
    for k in xrange(d):
        if k in exclude_modes:
            continue
        if init == 'svd':
            tmp,_, _ = fast_svd(unfold(T, k))
            if verbose:
                print 'mode %d, svd_rank: %d, claimed rank: %d' % (k+1, tmp.shape[1], r[k])
            tmp = tmp[:, :r[k]]
            A[k] = tmp.copy()
        else:
            A[k] = np.random.uniform(0, 1, size=[n[k], r[k]]) 
            A[k] /= np.linalg.norm(A[k])
        AtA[k] = np.dot(A[k].T, A[k])
    if init == 'svd':
        axes = range(d)
        operands = [T, axes]
        tmp = set(range(d)).difference(set(exclude_modes))
        tmp = list(tmp)
        op2 = [[A[p], [p, p+d]] for p in tmp]
        op2 = reduce(lambda x, y: x+y, op2)
        operands += op2
        G = np.einsum(*tuple(operands))
    else:
        G = np.random.uniform(0, 1, size=r)
        G /= np.linalg.norm(G)
    
    t, tnew = np.ones(d+1), np.ones(d+1)
    L, Lnew = np.ones(d+1), np.ones(d+1)
    w, wnew = np.ones(d+1), np.ones(d+1)
    Gold = None
    flist = [functional(k, T, A, G, Lambda, AtA, exclude_modes)]
    rl = functional(k, T, A, G, Lambda, AtA, exclude_modes, separate=True)
    rlist = [rl[0]]
    for itnum in xrange(maxitnum):
        for k in xrange(d):
            if k in exclude_modes:
                continue
            Lnew[d] = update_L_core(AtA, exclude_modes)
            tnew[d] = update_t(t[d])
            wnew[d] = update_w(L[d], Lnew[d], t[d], tnew[d], alpha)
            if Gold is not None:
                Gnew = (1 + wnew[d])*G - Gold
            else:
                Gnew = G.copy()
            Gnew -= (gradient_core(T, A, Gnew, AtA, exclude_modes) + Lambda[d]) / Lnew[d]
            if Gamma[d]:
                Gnew = np.maximum(margin, Gnew)
            # update factor
            Lnew[k] = update_L_factor(k, AtA, Gnew, exclude_modes)
            tnew[k] = update_t(t[k])
            wnew[k] = update_w(L[k], Lnew[k], t[k], tnew[k], alpha)
            if itnum > 0:
                Anew_k = (1+wnew[k])*A[k] - Aold[k]
            else:
                Anew_k = A[k].copy()
            AtA_k_new = np.dot(Anew_k.T, Anew_k)
            Anew_k -= (gradient_factor(k, T, A[:k]+[Anew_k]+A[k+1:], Gnew, AtA[:k]+[AtA_k_new]+AtA[k+1:],
                                      exclude_modes) + Lambda[k]) / Lnew[k]
            if Gamma[k]:
                Anew_k = np.maximum(margin, Anew_k)
            AtA_k_new = np.dot(Anew_k.T, Anew_k)
            funval = functional(
                k, T, A[:k]+[Anew_k]+A[k+1:], Gnew, Lambda, AtA[:k]+[AtA_k_new]+AtA[k+1:],
                exclude_modes
            )
            if funval > flist[-1]:
                Gnew = G - (gradient_core(T, A, G, AtA, exclude_modes) + Lambda[d]) / Lnew[d]
                if Gamma[d]:
                    Gnew = np.maximum(margin, Gnew)
                
                Anew_k = A[k] - (gradient_factor(k, T, A, Gnew, AtA, exclude_modes) +\
                            Lambda[k]) / Lnew[k]
                if Gamma[k]:
                    Anew_k = np.maximum(margin, Anew_k)
                AtA_k_new = np.dot(Anew_k.T, Anew_k)
                funval = functional(
                    k, T, A[:k]+[Anew_k]+A[k+1:], Gnew, Lambda, AtA[:k]+[AtA_k_new]+AtA[k+1:],
                    exclude_modes
                )
            AtA[k] = AtA_k_new.copy()
            Aold[k] = A[k].copy()
            A[k] = Anew_k.copy()
            Gold = G.copy()
            G = Gnew.copy()
            flist.append(funval)
            t[k], L[k] = tnew[k], Lnew[k]
            t[d], L[d] = tnew[d], Lnew[d]
            rl = functional(k, T, A, Gnew, Lambda, AtA, exclude_modes, separate=True)
            rlist.append(rl[0])
            if verbose:
                print "itnum %d\t sweep %d/%d\t funval:%.5f" % (itnum+1, k+1, d, flist[-1]) 
        if np.abs(flist[-1] - flist[-1-d+len(exclude_modes)]) < tol:
            break
    return A, G, flist

def estimateMzPolarityFactors(X, y, r=1, classes=None, Lambda=None, Gamma=None, maxitnum=1000, init='svd', dirname='', prefix='', savedata=1, verbose=0):
    T = X.copy()
    n = T.shape
    d = len(n)
    T = unfold(T, 0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    T = reshape(T, n)
  
    if Lambda is None:
        Lambda = 0.00003*np.ones(d+1)
    if Gamma is None:
        Gamma = np.ones(d+1)
    if classes is None:
        classes = np.unique(y)
    if isinstance(r, int):
        r = [r, 2]
    elif isinstance(r, (list, tuple, np.ndarray)):
        assert len(r) == 2
        r = list(r)
    tucks = []
    times = []
    for ind in xrange(len(classes)):
        which = np.where(y == classes[ind])[0]
        y_local = y[which].copy()
        Z = T[which].copy()
        normZ = np.linalg.norm(Z)
        Z /= normZ
        
        r_local = [Z.shape[0]] + r
        t1 = time.clock()
        A, G, flist = proximalALS(
            Z, r_local, Lambda, Gamma, margin=0., alpha=0.9999, maxitnum=maxitnum, init=init,
            exclude_modes=[0], verbose=verbose
        )
        t2 = time.clock()
        Zhat = recover(A, G, exclude_modes=[0])
        stats = []
        for k in xrange(Z.shape[0]):
            stats.append( np.linalg.norm(Z[k] - Zhat[k]) / np.linalg.norm(Z[k]) )
        stats = np.array(stats)
        if verbose:
            print "Class: %d  Time=%.2fs RRes: min=%.2e mean=%.2e med=%.2e max=%.2e" % (
                classes[ind], t2-t1, np.min(stats), np.mean(stats), np.median(stats), np.max(stats)
            )
        times.append(t2-t1)
        tucks.append(copy.deepcopy(A))
        if savedata:
            filename = prefix + 'tucker_factors_%dmz_%dpol' % (r[0], r[1])
            np.savez_compressed(dirname+filename, tucks=tucks, errors=stats, classes=classes, times=times)
    return tucks, times, stats



def sparse_ratio(X, tau=1e-3):
    sizeX = float(X.size)
    if isinstance(tau, float):
        tau = [tau]
    sr = np.zeros(len(tau))
    for i in xrange(len(tau)):
        sr[i] = X[np.abs(X) < tau[i]].size / sizeX
    return sr

def sparsity_approximation_tradeoff(T, tucker_dict, tau=1e-3, dims=None, use_core=True):
    d = T.ndim
    n = T.shape
    if dims is None:
        dims = range(d)
    normT = np.linalg.norm(T)
    sizeX = float(X.size)
    if isinstance(tau, float):
        tau = [tau]
    sr = np.zeros(len(tau))
    ap = np.zeros(len(tau))
    
    for i in xrange(len(tau)):
        tD = copy.deepcopy(tucker_dict)
        sizeX = 0.
        for k in xrange(len(dims)):
            Ak = tD['A'][k]
            sizeX += float(Ak.size)
            ind = np.abs(Ak) < tau[i]
            sr[i] += Ak[ind].size
            tD['A'][k][ind] = 0.
        if use_core:
            G = tD['G']
            sizeX += float(G.size)
            ind = np.abs(G) < tau[i]
            sr[i] += G[ind].size
            tD['G'][ind] = 0.
        sr[i] /= sizeX                    
        ap[i] = np.linalg.norm(recover(n, tD) - T) / normT
    return sr, ap

class TuckerClassifierLCMS():
    
    def __init__(self, Nmz, ranks, Lambda=None, Gamma=None, maxitnum=1000, init='svd'):
        self.Lambda = Lambda
        self.Gamma = Gamma
        self.maxitnum = maxitnum
        self.init = init
        self.Nmz = Nmz
        self.ranks = ranks
        
    def saveParameters(self, filename):
        try:
            np.savez_compressed(
                filename, classes=self.classes, MZspaces=self.MZspaces, Polspaces_inv=self.Polspaces_inv
            )
        except:
            print "No parameters to save"
    
    def loadParameters(self, filename):
        try:
            df = np.load(filename)
            self.classes = df['classes']
            self.MZspaces = df['MZspaces'].item()
            self.Polspaces_inv = df['Polspaces_inv'].item()
        except:
            print "No parameters to load"
        
        
    def fit(self, X, y, verbose=0):
        T = X.copy()
        if T.ndim > 2:
            T = unfold(T, 0)
        T /= np.linalg.norm(T, axis=1, keepdims=1)
        n = [T.shape[0], self.Nmz, 2]
        T = reshape(T, n)
        self.classes = np.unique(y)
        A, times, stats = estimateMzPolarityFactors(
            X, y, r=self.ranks, classes=self.classes, Lambda=None, Gamma=None, maxitnum=1000, init='svd', savedata=0,
            verbose=verbose
        )
        self.Polspaces_inv = {}
        self.MZspaces = {}
        for i in xrange(len(self.classes)):
            self.MZspaces[self.classes[i]] = A[i][1]
            self.Polspaces_inv[self.classes[i]] = np.linalg.pinv(A[i][2])
        return
    
    def predict(self, X, return_all=0, metric=None):
        T = X.copy()
        if T.ndim > 2:
            T = unfold(T,0)
        T /= np.linalg.norm(T, axis=1, keepdims=1)
        n = [T.shape[0], self.Nmz, 2]
        T = reshape(T, n)
        resultTable = np.zeros([T.shape[0], len(self.classes)])
        for i in xrange(len(self.classes)):
            currentClass = self.classes[i]
            Z = prodTenMat(T, self.Polspaces_inv[currentClass], 2)
            for k in xrange(T.shape[0]):
                if metric is None:
                    a, b = cca(Z[k], self.MZspaces[currentClass], N=1)
                    resultTable[k, i] = corr(a, b)[0, 1]
                else:
                    resultTable[k, i] = metric(Z[k], self.MZspaces[currentClass])
        if return_all:
            data = pd.DataFrame(resultTable, columns=self.classes)
            return data
        ind = np.argmax(resultTable, axis=1)
        return self.classes[ind]




if __name__ == '__main__':
    pass
