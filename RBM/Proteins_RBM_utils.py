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
    

import numpy as np
import sys
import os
import pickle
sys.path.append('../RBM/')
import pandas as pd
import rbm,layer,utilities
import copy
import types
from multiprocessing import Pool
import itertools
from functools import partial
    



def saveRBM(filename,RBM):
    pickle.dump(RBM,open(filename,'wb'))

def loadRBM(filename):
    return pickle.load(open(filename,'r'))




def get_sparsity(W,a=3,include_gaps=True):
    if not include_gaps:
        W_ = W[:,:,:-1]
    else:
        W_ = W
    tmp = np.sqrt((W_**2).sum(-1))
    p = ((tmp**a).sum(1))**2/(tmp**(2*a)).sum(1)
    return p/W.shape[1]

def get_beta(W,include_gaps=True):
    if not include_gaps:
        W_ = W[:,:,:-1]
    else:
        W_ = W
    return  np.sqrt( (W_**2).sum(-1).sum(-1) )

def get_theta(RBM):
    if RBM.hidden == 'ReLU':
        return (RBM.hlayer.theta_plus+ RBM.hlayer.theta_minus)/(2*RBM.hlayer.a)
    elif RBM.hidden == 'ReLU+':
        return (RBM.hlayer.theta_plus - RBM.hlayer.b)/RBM.hlayer.a
    elif RBM.hidden =='dReLU':
        return (1-RBM.hlayer.eta**2)/2 * (RBM.hlayer.theta_plus + RBM.hlayer.theta_minus)/RBM.hlayer.a
    else:
        print 'get_theta not supported for hidden %s'%RBM.hidden

def get_beta_gaps(W):
    return np.sqrt( (W**2)[:,:,-1].sum(-1) )

def get_hidden_input(data,RBM,normed=False,offset=True):
    if normed:
        mu = utilities.average(data,c=21)
        norm_null = np.sqrt(  ((RBM.weights**2 * mu).sum(-1) - (RBM.weights*mu).sum(-1)**2).sum(-1) )
        return (RBM.vlayer.compute_output(data,RBM.weights) - RBM.hlayer.b[np.newaxis,:])/norm_null[np.newaxis,:]
    else:
        if offset:
            return (RBM.vlayer.compute_output(data,RBM.weights) - RBM.hlayer.b[np.newaxis,:])
        else:
            return (RBM.vlayer.compute_output(data,RBM.weights) )




def conditioned_RBM(RBM, conditions):
    num_conditions = len(conditions)
    l_hh = np.array( [condition[0] for condition in conditions] )
    l_value = np.array( [condition[1] for condition in conditions] )
    
    remaining = np.array( [x for x in range(RBM.n_h) if not x in l_hh] )
    tmp_RBM = rbm.RBM(n_v = RBM.n_v, n_h = RBM.n_h-num_conditions, n_cv = RBM.n_cv, n_ch = RBM.n_ch, visible = RBM.visible, hidden= RBM.hidden)
    tmp_RBM.vlayer.fields = RBM.vlayer.fields.copy()
    tmp_RBM.vlayer.fields0 = RBM.vlayer.fields0.copy()
    tmp_RBM.weights = RBM.weights[remaining,:]
    if RBM.hidden in ['Bernoulli','Spin']:
        tmp_RBM.hlayer.fields = RBM.hlayer.fields[remaining].copy()
        tmp_RBM.hlayer.fields0 = RBM.hlayer.fields0[remaining].copy()                         
    elif RBM.hidden == 'Gaussian':
        tmp_RBM.hlayer.a = RBM.hlayer.a[remaining].copy()
        tmp_RBM.hlayer.a0 = RBM.hlayer.a0[remaining].copy()                         
        tmp_RBM.hlayer.b = RBM.hlayer.b[remaining].copy()
        tmp_RBM.hlayer.b0 = RBM.hlayer.b0[remaining].copy()                         
    elif RBM.hidden == 'ReLU':
        tmp_RBM.hlayer.a = RBM.hlayer.a[remaining].copy()
        tmp_RBM.hlayer.a0 = RBM.hlayer.a0[remaining].copy()                         
        tmp_RBM.hlayer.theta_plus = RBM.hlayer.theta_plus[remaining].copy()
        tmp_RBM.hlayer.theta_plus0 = RBM.hlayer.theta_plus0[remaining].copy()                         
        tmp_RBM.hlayer.theta_minus = RBM.hlayer.theta_minus[remaining].copy()         
        tmp_RBM.hlayer.theta_minus0 = RBM.hlayer.theta_minus0[remaining].copy() 
    elif RBM.hidden == 'dReLU':
        tmp_RBM.hlayer.a = RBM.hlayer.a[remaining].copy()
        tmp_RBM.hlayer.a_plus = RBM.hlayer.a_plus[remaining].copy()                         
        tmp_RBM.hlayer.a_plus0 = RBM.hlayer.a_plus0[remaining].copy()      
        tmp_RBM.hlayer.a_minus = RBM.hlayer.a_minus[remaining].copy()                         
        tmp_RBM.hlayer.a_minus0 = RBM.hlayer.a_minus0[remaining].copy()
        tmp_RBM.hlayer.theta_plus = RBM.hlayer.theta_plus[remaining].copy()
        tmp_RBM.hlayer.theta_plus0 = RBM.hlayer.theta_plus0[remaining].copy()
        tmp_RBM.hlayer.theta_minus = RBM.hlayer.theta_minus[remaining].copy()
        tmp_RBM.hlayer.theta_minus0 = RBM.hlayer.theta_minus0[remaining].copy()

    elif RBM.hidden == 'ReLU+':
        tmp_RBM.hlayer.a = RBM.hlayer.a[remaining].copy()
        tmp_RBM.hlayer.a0 = RBM.hlayer.a0[remaining].copy()                         
        tmp_RBM.hlayer.theta_plus = RBM.hlayer.theta_plus[remaining].copy()
        tmp_RBM.hlayer.theta_plus0 = RBM.hlayer.theta_plus0[remaining].copy()                         
    tmp_RBM.vlayer.fields += (RBM.weights[l_hh] * l_value[:,np.newaxis,np.newaxis]).sum(0)
    tmp_RBM.vlayer.fields0 += (RBM.weights[l_hh] * l_value[:,np.newaxis,np.newaxis]).sum(0)

    return tmp_RBM




def gen_data_lowT(RBM, beta=1, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    if which == 'joint':
        tmp_RBM = copy.deepcopy(RBM)
        tmp_RBM.vlayer.fields *= beta
        tmp_RBM.weights *= beta
        if RBM.hidden in ['Bernoulli','Spin']:
            tmp_RBM.hlayer.fields *= beta
        elif RBM.hidden == 'Gaussian':
            tmp_RBM.hlayer.a *= beta
            tmp_RBM.hlayer.b *= beta
        elif RBM.hidden == 'ReLU+':
            tmp_RBM.hlayer.a *= beta
            tmp_RBM.hlayer.theta_plus *= beta
        elif RBM.hidden == 'ReLU':
            tmp_RBM.hlayer.a *= beta
            tmp_RBM.hlayer.theta_plus *= beta
            tmp_RBM.hlayer.theta_minus *= beta
        elif RBM.hidden == 'dReLU':
            tmp_RBM.hlayer.a_plus *= beta
            tmp_RBM.hlayer.a_minus *= beta
            tmp_RBM.hlayer.theta_plus *= beta
            tmp_RBM.hlayer.theta_minus *= beta
    elif which == 'marginal':
        if type(beta) == int:
            tmp_RBM = rbm.RBM(n_v=RBM.n_v, n_h = beta* RBM.n_h,visible=RBM.visible,hidden=RBM.hidden, n_cv = RBM.n_cv, n_ch = RBM.n_ch)
            tmp_RBM.vlayer.fields = beta * RBM.vlayer.fields
            tmp_RBM.vlayer.fields0 = RBM.vlayer.fields0
            tmp_RBM.weights = np.repeat(RBM.weights,beta,axis=0)
            if RBM.hidden in ['Bernoulli','Spin']:
                tmp_RBM.hlayer.fields = np.repeat(RBM.hlayer.fields,beta,axis=0)
                tmp_RBM.hlayer.fields0 = np.repeat(RBM.hlayer.fields0,beta,axis=0)
            elif RBM.hidden == 'Gaussian':
                tmp_RBM.hlayer.a = np.repeat(RBM.hlayer.a,beta,axis=0)
                tmp_RBM.hlayer.a0 = np.repeat(RBM.hlayer.a0,beta,axis=0)
                tmp_RBM.hlayer.b = np.repeat(RBM.hlayer.b,beta,axis=0)
                tmp_RBM.hlayer.b0 = np.repeat(RBM.hlayer.b0,beta,axis=0)
            elif RBM.hidden == 'ReLU+':
                tmp_RBM.hlayer.a = np.repeat(RBM.hlayer.a,beta,axis=0)
                tmp_RBM.hlayer.a0 = np.repeat(RBM.hlayer.a0,beta,axis=0)
                tmp_RBM.hlayer.theta_plus = np.repeat(RBM.hlayer.theta_plus,beta,axis=0)
                tmp_RBM.hlayer.theta_plus0 = np.repeat(RBM.hlayer.theta_plus0,beta,axis=0) 
            elif RBM.hidden == 'ReLU':
                tmp_RBM.hlayer.a = np.repeat(RBM.hlayer.a,beta,axis=0)
                tmp_RBM.hlayer.a0 = np.repeat(RBM.hlayer.a0,beta,axis=0)
                tmp_RBM.hlayer.theta_plus = np.repeat(RBM.hlayer.theta_plus,beta,axis=0)
                tmp_RBM.hlayer.theta_plus0 = np.repeat(RBM.hlayer.theta_plus0,beta,axis=0) 
                tmp_RBM.hlayer.theta_minus = np.repeat(RBM.hlayer.theta_minus,beta,axis=0)
                tmp_RBM.hlayer.theta_minus0 = np.repeat(RBM.hlayer.theta_minus0,beta,axis=0)
            elif RBM.hidden == 'dReLU':
                tmp_RBM.hlayer.a_plus = np.repeat(RBM.hlayer.a_plus,beta,axis=0)
                tmp_RBM.hlayer.a_plus0 = np.repeat(RBM.hlayer.a_plus0,beta,axis=0)
                tmp_RBM.hlayer.a_minus = np.repeat(RBM.hlayer.a_minus,beta,axis=0)
                tmp_RBM.hlayer.a_minus0 = np.repeat(RBM.hlayer.a_minus0,beta,axis=0)                
                tmp_RBM.hlayer.theta_plus = np.repeat(RBM.hlayer.theta_plus,beta,axis=0)
                tmp_RBM.hlayer.theta_plus0 = np.repeat(RBM.hlayer.theta_plus0,beta,axis=0) 
                tmp_RBM.hlayer.theta_minus = np.repeat(RBM.hlayer.theta_minus,beta,axis=0)
                tmp_RBM.hlayer.theta_minus0 = np.repeat(RBM.hlayer.theta_minus0,beta,axis=0)
    return tmp_RBM.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)





def gen_data_zeroT(RBM, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    tmp_RBM = copy.deepcopy(RBM)
    if which == 'joint':
        tmp_RBM.markov_step = types.MethodType(markov_step_zeroT_joint, tmp_RBM)
    elif which == 'marginal':
        tmp_RBM.markov_step = types.MethodType(markov_step_zeroT_marginal, tmp_RBM)
    return tmp_RBM.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)




def markov_step_zeroT_joint(self,(v,h),beta=1):
    I = self.vlayer.compute_output(v,self.weights,direction='up')
    h = self.hlayer.transform(I)
    I = self.hlayer.compute_output(h,self.weights,direction='down')
    v = self.vlayer.transform(I)
    return (v,h)

def markov_step_zeroT_marginal(self,(v,h),beta=1):
    I = self.vlayer.compute_output(v,self.weights,direction='up')
    h = self.hlayer.mean_from_inputs(I)
    I = self.hlayer.compute_output(h,self.weights,direction='down')
    v = self.vlayer.transform(I)
    return (v,h)


def couplings_to_contacts(couplings,with_gaps=True): # From N x N x n_c x n_c to Average Product Correction.
    if with_gaps:
        W = np.sqrt( (couplings**2).sum(-1).sum(-1) )
    else:
        W = np.sqrt( (couplings[:,:,:-1,:-1]**2).sum(-1).sum(-1) )
    tmp = W
    tmp2 = tmp.sum(1)
    F = tmp - tmp2[np.newaxis,:] * tmp2[:,np.newaxis]/tmp2.sum()
    return F

def compare_couplings_contacts(F,contact_map, distant = False,return_list=False,filter_alignment =None):

    F2 = F.copy()
    contact_map = contact_map.copy()
    if filter_alignment is not None:
        F2 = F2[filter_alignment,:][:,filter_alignment]
        contact_map = contact_map[filter_alignment,:][:,filter_alignment]

    n_sites = F2.shape[0]    
    if distant:
        for i in range(n_sites):
            for j in range(n_sites):
                if np.abs(i-j)<5:
                    F2[i,j] = -1
                    contact_map[i,j] = 0

    n_contacts = int(contact_map.sum()/2)
    contact_map = contact_map.flatten()
    for i in range(n_sites):
        for j in range(i,n_sites):
            F2[i,j]= -1
    l = np.argsort(F2.flatten())[::-1]
    corrects = np.zeros(n_contacts)
    if return_list:
        I,J = np.unravel_index(  l, (n_sites,n_sites) )
        liste_pairs = zip(I,J)

    for i in range(n_contacts):
        if contact_map[l[i]]:
            corrects[i]=1

    if return_list:
        is_correct = corrects.copy()
            

    corrects = np.cumsum(np.array(corrects))/ np.arange(1,n_contacts+1)

    if return_list:
        return liste_pairs, is_correct, corrects
    else:
        return corrects

def weights_to_couplings_approx(RBM,data,weights=None):
    psi = RBM.vlayer.compute_output(data,RBM.weights)
    var = RBM.hlayer.var_from_inputs(psi)
    mean_var = utilities.average(var,weights=weights)
    J_eff = np.tensordot( RBM.weights, RBM.weights * mean_var[:,np.newaxis,np.newaxis], axes= [0,0] )
    J_eff = np.swapaxes(J_eff,1,2)
    return J_eff

def weights_to_couplings_exact(RBM,data,weights=None,nbins=10,subset = None,pool=None):
    N = RBM.n_v
    M = RBM.n_h
    c = RBM.n_cv
    J = np.zeros([N,N,c,c])
    if pool is None:
        pool = Pool()

    inputs = []
    if subset is not None:
        for i,j in subset:
            inputs.append(i*N+j)
    else:
        for i in range(N):
            for j in range(i,N):
                inputs.append( i*N + j)

    partial_Ker = partial(_Ker_weights_to_couplings_exact,RBM=RBM,data=data,weights=weights,nbins=nbins)
    res = pool.map(partial_Ker,inputs)
    for x in range( len(inputs) ):
        i = inputs[x]/N
        j = inputs[x]%N
        J[i,j] = res[x]
        if j <>i:
            J[j,i] = res[x].T
    pool.close()
    return J


def _Ker_weights_to_couplings_exact(x,RBM,data,weights=None,nbins=10):
    N = RBM.n_v
    M = RBM.n_h
    c = RBM.n_cv
    Jij = np.zeros([c,c])
    i = x/N
    j = x%N
    L = layer.Layer(N=1,nature=RBM.hidden)
    tmpW = RBM.weights.copy()
    subsetW = tmpW[:,[i,j],:].copy()
    tmpW[:,[i,j],:] *= 0
    psi_restr = RBM.vlayer.compute_output(data,tmpW)
    for m in range(M):
        count,hist = np.histogram(psi_restr[:,m],bins=nbins,weights=weights)
        hist = (hist[:-1]+hist[1:])/2
        hist_mod = (hist[:,np.newaxis,np.newaxis] + subsetW[m,0][np.newaxis,:,np.newaxis] + subsetW[m,1][np.newaxis,np.newaxis,:]).reshape([nbins*c**2,1])
        if RBM.hidden == 'Gaussian':
            L.a[0] = RBM.hlayer.a[m]
            L.b[0] = RBM.hlayer.b[m]                    
        elif RBM.hidden == 'dReLU':
            L.a_plus[0] = RBM.hlayer.a_plus[m]                                 
            L.a_minus[0] = RBM.hlayer.a_minus[m]
            L.theta_plus[0] = RBM.hlayer.theta_plus[m]
            L.theta_minus[0] = RBM.hlayer.theta_minus[m]                
        Phi = utilities.average(L.logpartition(hist_mod).reshape([nbins,c,c]),weights=count)
        Jij += (Phi[:,:,np.newaxis,np.newaxis] + Phi[np.newaxis,np.newaxis,:,:] - Phi[np.newaxis,:,:,np.newaxis].T - Phi[:,np.newaxis,np.newaxis,:]).sum(-1).sum(-1)/c**2
    return Jij







