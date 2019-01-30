"""
 Copyright 2018 - by Jerome Tubiana (jertubiana@@gmail.com)
     All rights reserved
     
     Permission is granted for anyone to copy, use, or modify this
     software for any uncommercial purposes, provided this copyright 
     notice is retained, and note is made of any changes that have 
     been made. This software is distributed without any warranty, 
     express or implied. In no event shall the author or contributors be 
     liable for any damage arising out of the use of this software.
     
     The publication of research using this software, modified or not, must include 
     appropriate citations to:
"""
    


#%% Useful functions
import numpy as np
import numbers
from scipy.special import erf,erfinv
from scipy.sparse import csr_matrix
import cy_utilities
import itertools

def check_random_state(seed):
    if seed==None or seed==np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def gen_even_slices(n, n_packs, n_samples=None):
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples <> None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


def logistic(x, out=None): ## from SKlearn
    if out==None:
        out = np.empty(np.atleast_1d(x).shape, dtype=np.float64)
    out[:] = x

    # 1 / (1 + exp(-x)) = (1 + tanh(x / 2)) / 2
    # This way of computing the logistic==both fast and stable.
    out *= .5
    np.tanh(out, out)
    out += 1
    out *= .5

    return out.reshape(np.shape(x))

def log_logistic(X, out=None): ## from SKLearn
    is_1d = X.ndim == 1


    if out==None:
        out = np.empty_like(X)

    out = -np.logaddexp(0,-X)

    if is_1d:
        return np.squeeze(out)
    return out

def logsumexp(a, axis=None, b=None, keepdims=False): ## from scipy
    # This==a more elegant implementation, requiring NumPy >= 1.7.0
    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b <> None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        out = np.log(np.sum(tmp, axis=axis, keepdims=keepdims))

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)

    out += a_max

    return out

def erf_times_gauss(X):
    m = np.zeros(X.shape)
    tmp = X< 6
    m[tmp] = np.exp(0.5 * X[tmp]**2) * (1- erf(X[tmp]/np.sqrt(2))) * np.sqrt(np.pi/2);
    m[~tmp] = ( 1/X[~tmp] - 1/X[~tmp]**3 + 3/X[~tmp]**5);
    return m

def log_erf_times_gauss2(X):
    m = np.zeros(X.shape)
    tmp = X< 6
    m[tmp] = 0.5 * X[tmp]**2 + np.log(1- erf(X[tmp]/np.sqrt(2)))  + 0.5 * np.log(np.pi/2)
    m[~tmp] = -np.log(X[~tmp]) + np.log(1- 1/X[~tmp]**2 + 3/X[~tmp]**4)
    return m

def log_erf_times_gauss(X):
    m = np.zeros(X.shape)
    tmp = X < 6
    m[tmp] = (0.5 * X[tmp]**2 + np.log(1- erf(X[tmp]/np.sqrt(2))  )- np.log(2))
    m[~tmp] = (0.5 * np.log(2/ np.pi) - np.log(2) - np.log(X[~tmp]) + np.log( 1 - 1/X[~tmp]**2 + 3/X[~tmp]**4) )
    return m

def softmax(X):
    X -= X.max(-1)[:,:,np.newaxis]    
    out = np.exp(X)
    if X.ndim ==1:
        out/= (out.sum(-1))
    elif X.ndim == 2:
        out/= (out.sum(-1))[:,np.newaxis]
    elif X.ndim == 3:
        out/= (out.sum(-1))[:,:,np.newaxis]
    return out


def invert_softmax(mu, eps=1e-6, gauge = 'zerosum'):
    n_c = mu.shape[1]
    fields = np.log( (1- eps) * mu + eps/n_c )
    if gauge=='zerosum':
        fields -= fields.sum(1)[:,np.newaxis]/n_c
    return fields





def cumulative_probabilities(X,maxi=1e9):
    X -= X.max(-1)[:,:,np.newaxis]
    out = np.exp(X)
    # out[out>maxi] = maxi # For numerical stability.
    out = np.cumsum(out,axis=-1)
    out/= out[:,:,-1][:,:,np.newaxis]
    return out



def kronecker(X1,c1):
    l = []
    for color in range(c1):
        l.append( np.asarray(X1 == color,dtype=float))
    return np.rollaxis(np.array(l),0, X1.ndim+1)

def saturate(x,xmax):
    return np.sign(x) * np.minimum(np.abs(x),xmax)


#def average(X,c=1):
#    if (c==1):
#        return X.mean(0)
#    else:
#        out = np.zeros([X.shape[1],c])
#        for color in range(c):
#            out[:,color] = csr_matrix(X==color).mean(0)
#        return out

def average(X,c=1,weights=None):
    if (c==1):
        if weights is None:
            return X.mean(0)
        else:
            if X.ndim ==1:
                return (X * weights).sum(0)/weights.sum()
            elif X.ndim ==2:
                return (X * weights[:,np.newaxis]).sum(0)/weights.sum()
            elif X.ndim ==3:
                return (X * weights[:,np.newaxis,np.newaxis]).sum(0)/weights.sum()
    else:
        if weights is None:
            if X.ndim == 1:
                X = X[np.newaxis,:]
                return cy_utilities.average_C(X,c)[0]
            elif X.ndim == 2:
                return cy_utilities.average_C(X,c)
            elif X.ndim == 3:
                l = []
                for i in range(X.shape[0]):
                    l.append(cy_utilities.average_C(X[i],c))
                return np.array(l)
        else: ### Vector of Weights; over the second index only (!)
            if X.ndim ==1:
                X = X[np.newaxis,:]
                return cy_utilities.average_C(X,weights,c)[0]
            elif X.ndim==2:
                return cy_utilities.weighted_average_C(X,weights,c)
            elif X.ndim==3:
                l = []
                for i in range(X.shape[0]):
                    l.append(cy_utilities.weighted_average_C(X[i],weights,c))
                return np.array(l)         
                


#def average_product(X1, X2, c1 = 1,c2=1, mean1 = False, mean2= False):
#    if (c1 == 1) & (c2 == 1):
#        out = np.dot(X1.T,X2)
#    elif (c1 == 1) & (c2 <> 1):
#        if mean2: # X2 in format [data,site,color]; each conditional mean for each color.
#            out = np.tensordot(X1,X2,axes=([0],[0]))
#        else:
#            out = np.zeros([X1.shape[1], X2.shape[1],c2])
#            for color in range(c2):
#                out[:,:,color] = csr_matrix(X2 == color).T.dot(X1).T  #sparse version of np.dot(X1.T, (X2 == color))
#    elif (c1 <>1) & (c2 ==1):
#        if mean1: # X1 in format [data,site,color]; each conditional mean for each color.
#            out = np.swapaxes( np.tensordot(X1,X2,axes=([0],[0])),1,2)
#        else:
#            out = np.zeros([X1.shape[1], X2.shape[1],c1])
#            for color in range(c1):
#                out[:,:,color] =  csr_matrix(X1== color,dtype=int).T.dot(X2)
#    elif (c1 <>1) & (c2 <>1):
#        if mean1 & mean2:
#            out = np.swapaxes(np.tensordot(X1,X2,axes = ([0],[0])), 1,2)
#        elif mean1 & (~mean2):
#            out = np.zeros([X1.shape[1], X2.shape[1],c1,c2])
#            for color2 in range(c2):
#                out[:,:,:,color2] = np.swapaxes(np.tensordot(X1, (X2 == color2), axes = ([0],[0]) ), 1,2 )
#        elif (~mean1) & mean2:
#            out = np.zeros([X1.shape[1], X2.shape[1],c1,c2])
#            for color1 in range(c1):
#                out[:,:,color1,:] = np.tensordot( (X1 == color1), X2, axes = ([0],[0]))
#        else:
#            out = np.zeros([X1.shape[1], X2.shape[1],c1,c2])
#            for color1 in range(c1):
#                for color2 in range(c2):
#                    out[:,:,color1,color2] = (csr_matrix(X1== color1,dtype=int).T.dot(csr_matrix(X2== color2,dtype=int) )).toarray()
#
#    return out/X1.shape[0]

def average_product(X1, X2, c1 = 1,c2=1, mean1 = False, mean2= False,weights =None):
    if (c1 == 1) & (c2 == 1):
        if weights is None:
            return np.dot(X1.T,np.asarray(X2,dtype=float))/float(X1.shape[0])
        else:
            return (X1[:,:,np.newaxis] * X2[:,np.newaxis,:] * weights[:,np.newaxis,np.newaxis]).sum(0)/weights.sum()
    elif (c1 == 1) & (c2 <> 1):
        if mean2: # X2 in format [data,site,color]; each conditional mean for each color.
            if weights is None:
                return np.tensordot(X1,X2,axes=([0],[0]))/float(X1.shape[0])
            else:
                return np.tensordot(X1 * weights[:,np.newaxis],X2,axes=([0],[0]))/weights.sum()
        else:
            if weights is None:
                return cy_utilities.average_product_FxP_C(np.asarray(X1,dtype=float),X2,c2)
            else:
                return cy_utilities.average_product_FxP_C(np.asarray(X1*weights[:,np.newaxis],dtype=float),X2,c2) * weights.shape[0]/weights.sum()
    elif (c1 <>1) & (c2 ==1):
        if mean1: # X1 in format [data,site,color]; each conditional mean for each color.
            if weights is None:
                return np.swapaxes( np.tensordot(X1,X2,axes=([0],[0])),1,2)/float(X1.shape[0])
            else:
                return np.swapaxes( np.tensordot(X1,weights[:,np.newaxis,:]*X2,axes=([0],[0])),1,2)/weights.sum()
        else:
            if weights is None:
                return np.swapaxes(cy_utilities.average_product_FxP_C(np.asarray(X2,dtype=float),X1,c1),0,1)
            else:
                return np.swapaxes(cy_utilities.average_product_FxP_C(np.asarray(X2 * weights[:,np.newaxis],dtype=float),X1,c1),0,1) * weights.shape[0]/weights.sum()
    elif (c1 <>1) & (c2 <>1):
        if mean1 & mean2:
            if weights is None:
                return np.swapaxes(np.tensordot(X1,X2,axes = ([0],[0])), 1,2)/float(X1.shape[0])
            else:
                return np.swapaxes(np.tensordot(X1 * weights[:,np.newaxis,np.newaxis],X2,axes = ([0],[0])), 1,2)/weights.sum()
        elif mean1 & (~mean2):
            out = np.zeros([X1.shape[1], X2.shape[1],c1,c2])
            for color2 in range(c2):
                if weights is None:
                    out[:,:,:,color2] = np.swapaxes(np.tensordot(X1, (X2 == color2), axes = ([0],[0]) ), 1,2 )/float(X1.shape[0])
                else:
                    out[:,:,:,color2] = np.swapaxes(np.tensordot(X1 * weights[:,np.newaxis,np.newaxis], (X2 == color2), axes = ([0],[0]) ), 1,2 )/weights.sum()
            return out
        elif (~mean1) & mean2:
            out = np.zeros([X1.shape[1], X2.shape[1],c1,c2])
            for color1 in range(c1):
                if weights is None:
                    out[:,:,color1,:] = np.tensordot( (X1 == color1), X2, axes = ([0],[0]))/float(X1.shape[0])
                else:
                    out[:,:,color1,:] = np.tensordot( (X1 == color1), weights[:,np.newaxis,np.newaxis]* X2, axes = ([0],[0]))/weights.sum()
            return out

        else:
            if weights is None:
                return cy_utilities.average_product_PxP_C(X1,X2,c1,c2)
            else:
                return cy_utilities.weighted_average_product_PxP_C(X1,X2,weights,c1,c2)


def covariance(X1, X2, c1 = 1,c2=1, mean1 = False, mean2= False,weights =None):
    if mean1:
        mu1 = average(X1,weights=weights)
    else:
        mu1 = average(X1,c=c1,weights=weights)
    if mean2:
        mu2 = average(X2,weights=weights)
    else:
        mu2 = average(X2,c=c2,weights=weights)

    prod = average_product(X1,X2,c1=c1,c2=c2,mean1=mean1,mean2=mean2,weights=weights)

    if (c1>1) & (c2>1):
        covariance = prod - mu1[:,np.newaxis,:,np.newaxis] * mu2[np.newaxis,:,np.newaxis,:]
    elif (c1>1) & (c2==1):
        covariance = prod - mu1[:,np.newaxis,:] * mu2[np.newaxis,:,np.newaxis]
    elif (c1==1) & (c2>1):
        covariance = prod - mu1[:,np.newaxis,np.newaxis] * mu2[np.newaxis,:,:]
    else:
        covariance = prod - mu1[:,np.newaxis] * mu2[np.newaxis,:]
    return covariance


#def bilinear_form(W, X1,X2,c1=1,c2=1):
#    if (c1==1) & (c2==1):
#        return np.sum( X1 * np.dot(W,X2.T).T,1)
#    elif (c1 ==1) & (c2>1):
#        bil = np.zeros(X1.shape[0])
#        for color in range(c2):
#            A = csr_matrix(X2 == color)
#            bil += np.sum(A.dot(W[:,:,color].T) * X1,1)
#        return bil
#    elif (c1>1) & (c2 ==1):
#        bil = np.zeros(X1.shape[0])
#        for color in range(c1):
#            A = csr_matrix(X1==color)
#            bil += np.sum(A.dot(W[:,:,color]) * X2, 1)
#        return bil
#    elif (c1>1) & (c2>1):
#        bil = np.zeros([X1.shape[0],1])
#        for color1 in range(c1):
#            A = csr_matrix(X1==color1)
#            for color2 in range(c2):
#                B = csr_matrix(X2==color2)
#                bil += B.multiply(  A.dot(W[:,:,color1,color2]) ).sum(1)
#        return bil[:,0]


def bilinear_form(W, X1,X2,c1=1,c2=1):
    if (c1==1) & (c2==1):
        return np.sum( X1 * np.dot(W,X2.T).T,1)
    elif (c1 ==1) & (c2>1):
        return np.sum(X1 * cy_utilities.compute_output_C(X2.shape[1],c2,X2,W) ,1)
    elif (c1>1) & (c2 ==1):
        return cy_utilities.dot_Potts2_C(X1.shape[1],c1,X1, np.tensordot(X2,W,(1,1) ) )
    elif (c1>1) & (c2>1):
        return cy_utilities.dot_Potts2_C(X1.shape[1], c1, X1, cy_utilities.compute_output_Potts_C(X2.shape[1],c2,c1,X2,W) )


def copy_config(config,N_PT=1,record_replica=False):
    if type(config)==tuple:
        if N_PT>1:
            if record_replica:
                return config[0].copy()
            else:
                return config[0][0].copy()
        else:
            return config[0].copy()
    else:
        if N_PT>1:
            if record_replica:
                return config.copy()
            else:
                return config[0].copy()
        else:
            return config.copy()


def make_all_discrete_configs(N,nature,c=1):
    if nature == 'Bernoulli':
        string = ','.join(['[0,1]' for _ in range(N)])
        exec('iter_configurations=itertools.product(%s)'%string)
    elif nature =='Spin':
        string = ','.join(['[-1,1]' for _ in range(N)])
        exec('iter_configurations=itertools.product(%s)'%string)
    elif nature =='Potts':
        liste_configs = '[' + ','.join([str(c) for c in range(c)]) + ']'
        string = ','.join([liste_configs for _ in range(N)])
        exec('iter_configurations=itertools.product(%s)'%string)
    else:
        print 'no supported'
    configurations = np.array([config for config in iter_configurations])
    return configurations
